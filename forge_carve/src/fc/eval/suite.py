from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from fc.adv.mutator import ProgramMutator
from fc.dsl.codec import decode_program
from fc.dsl.program import Program
from fc.eval.metrics import (
    attack_success_rate,
    flip_sensitivity_score,
    flip_output_pass,
    orbit_invariance_pass_rate,
    orbit_output_pass,
    proof_validity_correlation,
    repair_success_rate,
    selective_accuracy,
    verified_accuracy,
)
from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.dsl.tokens import TokenVocab
from fc.train.data import Example, TextVocab
from fc.util.jsonl import read_jsonl
from fc.util.vocab_identity import assert_vocab_match, vocab_identity
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.opcode_repair import repair_invalid_opcodes
from fc.util.decode_limits import resolve_max_prog_len
from fc.util.runtime_solve import runtime_solve
from fc.util.tags import domain_from_tag
from fc.verify.mesh import VerifierMesh, set_orbit_parallelism


def _load_text_vocab(mapping: dict[str, int]) -> TextVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TextVocab(token_to_id=mapping, id_to_token=id_to_token)

def _load_prog_vocab(mapping: dict[str, int]) -> TokenVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TokenVocab(token_to_id=mapping, id_to_token=id_to_token)


def _truncate_at_eos(ids: list[int], vocab: TokenVocab) -> list[int]:
    eos_id = vocab.token_to_id.get("<EOS>")
    if eos_id is None:
        return ids
    if eos_id in ids:
        return ids[: ids.index(eos_id) + 1]
    return ids


def run_eval(
    data_path: str,
    ckpt_path: str,
    out_path: str,
    device: str | torch.device | None = None,
    log_every: int = 50,
    repair_op: bool = False,
    constrained_op: bool = True,
    min_proof_tokens: int = 0,
    max_prog_len: int | None = None,
) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    rows = [Example.model_validate(r) for r in read_jsonl(data_path)]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    text_vocab = _load_text_vocab(ckpt["text_vocab"])
    prog_vocab = ckpt["prog_vocab"]
    prog_vocab_obj = _load_prog_vocab(prog_vocab)
    cfg = ckpt["config"]
    if "prog_vocab_sha256" in ckpt:
        ckpt_id = vocab_identity(prog_vocab)
        if ckpt["prog_vocab_sha256"] != ckpt_id.sha256:
            raise ValueError(
                "ckpt prog_vocab_sha256 mismatch "
                f"expected={ckpt_id.sha256} got={ckpt['prog_vocab_sha256']}"
            )
    ckpt_max_prog_len = ckpt.get("max_prog_len")
    if ckpt_max_prog_len is not None and cfg.get("max_prog_len") not in (None, ckpt_max_prog_len):
        raise ValueError(
            "ckpt max_prog_len mismatch "
            f"ckpt={ckpt_max_prog_len} cfg={cfg.get('max_prog_len')}"
        )
    prog_vocab_path = Path(ckpt_path).resolve().parent / "prog_vocab.json"
    if prog_vocab_path.exists():
        disk_map = json.loads(prog_vocab_path.read_text(encoding="utf-8"))
        if isinstance(disk_map, dict):
            assert_vocab_match(
                prog_vocab,
                disk_map,
                expected_label="ckpt.prog_vocab",
                actual_label=str(prog_vocab_path),
            )

    from fc.model.forge import ForgeModel, ModelConfig
    from fc.model.backbone import BackboneConfig
    from fc.model.primal_dual import PrimalDualConfig
    from fc.model.slots import SlotConfig

    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(
        vocab_size=len(prog_vocab),
        max_prog_len=cfg["max_prog_len"],
        backbone=bcfg,
        slots=scfg,
        primal_dual=pcfg,
    )
    model = ForgeModel(mcfg)
    model.load_state_dict(ckpt["model"])
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.eval()

    interp = Interpreter()
    mesh = VerifierMesh()
    mutator = ProgramMutator()

    correct = []
    valid = []
    orbit_pass = []
    flip_pass = []
    repaired = []
    confs = []
    adv_success = 0
    adv_total = 0

    total = len(rows)
    max_tokens = max(1, resolve_max_prog_len(max_prog_len, cfg, ckpt=ckpt))
    min_tokens = max(0, int(min_proof_tokens))
    eos_pos_counts: dict[int, int] = {}
    trunc_count = 0

    def _record_decode_stats(stats: Any) -> None:
        nonlocal trunc_count
        if stats is None:
            return
        eos_pos = getattr(stats, "eos_pos", None)
        if eos_pos is None:
            trunc_count += 1
            return
        eos_pos_counts[eos_pos] = eos_pos_counts.get(eos_pos, 0) + 1

    def _predict(text: str, constraints: list[dict[str, Any]] | None, domain_tag: str | None) -> tuple[Program, Any]:
        input_ids = torch.tensor(
            [text_vocab.encode(text, cfg["train"]["max_text_len"])],
            dtype=torch.long,
            device=torch_device,
        )
        with torch.inference_mode():
            base_out = model(input_ids)
        logits_seq = base_out["logits"][0].detach()
        domain_hint = domain_tag or domain_from_tag(text)
        pred_ids, decode_stats = greedy_decode_with_opcode_mask(
            logits_seq,
            prog_vocab_obj,
            enforce_opcode=constrained_op,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            domain_tag=domain_hint,
            return_stats=True,
        )
        pred_ids = _truncate_at_eos(pred_ids, prog_vocab_obj)
        _record_decode_stats(decode_stats)
        prog = decode_program(pred_ids, prog_vocab_obj)
        out, _, _ = interp.execute(prog, text)
        if out is None and repair_op:
            proof_tokens = [prog_vocab_obj.decode(i) for i in pred_ids]
            runtime_out, failure = runtime_solve(text, constraints or [], proof_tokens, return_error=True)
            if failure is not None and failure.code == "PARSE_FAIL":
                repaired_ids, did_repair = repair_invalid_opcodes(
                    pred_ids,
                    logits_seq,
                    prog_vocab_obj,
                    log_fn=(lambda msg: logger.info(msg)),
                )
                if did_repair:
                    repaired_tokens = [prog_vocab_obj.decode(i) for i in repaired_ids]
                    runtime_out, _ = runtime_solve(text, constraints or [], repaired_tokens, return_error=True)
                    if runtime_out is not None:
                        out = runtime_out
                        prog = decode_program(repaired_ids, prog_vocab_obj)
        return prog, out

    for idx, ex in enumerate(rows):
        if log_every > 0 and (idx % log_every == 0 or idx + 1 == total):
            logger.info("eval progress=%d/%d data=%s", idx + 1, total, data_path)
        constraints = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (ex.constraints or [])
        ]
        prog, out = _predict(ex.x, constraints, ex.domain_tag)
        orbits = [o.x for o in ex.orbit]
        flips = [f.x for f in ex.flips]
        report = mesh.run(
            ex.x,
            prog,
            out,
            domain=ex.domain,
            expected=ex.y,
            orbits=orbits,
            flips=flips,
            mutator=mutator,
            constraints=ex.constraints,
        )
        c_sum = sum(report.c)
        conf = float(1.0 / (1.0 + c_sum))
        confs.append(conf)

        is_correct = outputs_equivalent(out, ex.y)
        correct.append(is_correct)

        formal_ok = (
            report.c[mesh.constraint_names.index("schema")]
            + report.c[mesh.constraint_names.index("arith")]
            + report.c[mesh.constraint_names.index("csp")]
            == 0.0
        )
        valid.append(formal_ok)

        orbit_execs = []
        for otext in orbits:
            _, orbit_exec = _predict(otext, constraints, ex.domain_tag)
            orbit_execs.append(orbit_exec)
        orbit_pass.append(orbit_output_pass(out, orbit_execs))

        flip_execs = []
        flip_ys = []
        for fvar in ex.flips:
            _, flip_exec = _predict(fvar.x, constraints, ex.domain_tag)
            flip_execs.append(flip_exec)
            flip_ys.append(fvar.y)
        flip_pass.append(flip_output_pass(out, ex.y, flip_execs, flip_ys))

        # Repair attempt (bounded) for formal failures only.
        if not formal_ok:
            rep = mesh.run(
                ex.x,
                prog,
                out,
                domain=ex.domain,
                constraints=ex.constraints,
                repair=True,
                max_repairs=2,
            )
            repaired.append(bool(rep.meta.get("repair", {}).get("success", False)))
        else:
            repaired.append(True)

        # Adversarial success count
        mutants = mutator.mutate(prog)
        for mprog in mutants:
            adv_total += 1
            mout, _, _ = interp.execute(mprog, ex.x)
            rep = mesh.run(ex.x, mprog, mout, domain=ex.domain, constraints=ex.constraints)
            if sum(rep.c[:3]) == 0:
                adv_success += 1

    logger.info(
        "eval decode max_prog_len=%d min_proof_tokens=%d",
        max_tokens,
        min_tokens,
    )
    if eos_pos_counts:
        ordered = sorted(eos_pos_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        dist = ", ".join([f"{pos}:{count}" for pos, count in ordered])
        logger.info(
            "eval_decode eos_pos_top=%s trunc_by_maxlen=%d min_tokens=%d max_tokens=%d",
            dist,
            trunc_count,
            min_tokens,
            max_tokens,
        )
    report = {
        "verified_accuracy": verified_accuracy(correct),
        "orbit_invariance_pass_rate": orbit_invariance_pass_rate(orbit_pass),
        "flip_sensitivity_score": flip_sensitivity_score(flip_pass),
        "proof_validity_correlation": proof_validity_correlation(valid, correct),
        "repair_success_rate": repair_success_rate(repaired),
        "attack_success_rate": attack_success_rate(adv_success, adv_total),
        "selective_accuracy": selective_accuracy(confs, correct, thresholds=[0.2, 0.4, 0.6, 0.8]),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def load_eval_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_eval_suite(
    config_path: str,
    device: str | torch.device | None = None,
    *,
    repair_op: bool | None = None,
    constrained_op: bool | None = None,
    min_proof_tokens: int | None = None,
    max_prog_len: int | None = None,
) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    set_orbit_parallelism(bool(cfg.get("parallel_orbits", False)))
    log_every = int(cfg.get("log_every", 50))
    ckpt = cfg.get("ckpt", "out/ckpt.pt")
    schema_path = cfg.get("schema_path", "out/data/schema.jsonl")
    math_path = cfg.get("math_path", "out/data/math.jsonl")
    csp_path = cfg.get("csp_path", "out/data/csp.jsonl")
    out_path = cfg.get("out_path", "out/report.json")
    repair_flag = bool(cfg.get("repair_op", False)) if repair_op is None else bool(repair_op)
    constrained_flag = bool(cfg.get("constrained_op", True)) if constrained_op is None else bool(constrained_op)
    min_tokens = int(cfg.get("min_proof_tokens", 0)) if min_proof_tokens is None else int(min_proof_tokens)
    max_tokens = resolve_max_prog_len(max_prog_len, cfg)
    report = {
        "schema": run_eval(
            schema_path,
            ckpt,
            out_path="out/schema_report.json",
            device=device,
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
        ),
        "math": run_eval(
            math_path,
            ckpt,
            out_path="out/math_report.json",
            device=device,
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
        ),
        "csp": run_eval(
            csp_path,
            ckpt,
            out_path="out/csp_report.json",
            device=device,
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
        ),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
