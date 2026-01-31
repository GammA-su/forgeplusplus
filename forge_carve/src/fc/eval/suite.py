from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
from fc.util.repair_math import repair_math_apply_arith_operator
from fc.util.repair_math_cegis import repair_math_cegis
from fc.util.vocab_identity import assert_vocab_match, vocab_identity
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.opcode_repair import repair_invalid_opcodes
from fc.util.decode_limits import resolve_max_prog_len
from fc.util.runtime_solve import runtime_solve
from fc.util.tags import domain_from_tag
from fc.verify.mesh import VerifierMesh, set_orbit_parallelism
from fc.verify.arithmetic import ArithmeticVerifier


@dataclass(frozen=True)
class EvalState:
    ckpt_path: str
    ckpt: dict[str, Any]
    cfg: dict[str, Any]
    text_vocab: TextVocab
    prog_vocab: dict[str, int]
    prog_vocab_obj: TokenVocab
    model: Any
    device: torch.device


def load_eval_state(
    ckpt_path: str,
    *,
    device: str | torch.device | None = None,
) -> EvalState:
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
    return EvalState(
        ckpt_path=ckpt_path,
        ckpt=ckpt,
        cfg=cfg,
        text_vocab=text_vocab,
        prog_vocab=prog_vocab,
        prog_vocab_obj=prog_vocab_obj,
        model=model,
        device=torch_device,
    )


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
    decode_overrides: dict[str, dict[str, Any]] | None = None,
    min_proof_tokens: int = 0,
    max_prog_len: int | None = None,
    batch_size: int = 64,
    max_orbits: int | None = None,
    max_flips: int | None = None,
    max_mutants: int | None = None,
    max_examples: int | None = None,
    cegis_ms: int | None = None,
    cegis_mode: str | None = None,
    state: EvalState | None = None,
) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    if max_examples is None:
        rows = [Example.model_validate(r) for r in read_jsonl(data_path)]
    else:
        rows = []
        limit = max(0, int(max_examples))
        for idx, row in enumerate(read_jsonl(data_path)):
            if idx >= limit:
                break
            rows.append(Example.model_validate(row))
    if state is None:
        state = load_eval_state(ckpt_path, device=device)
    ckpt = state.ckpt
    cfg = state.cfg
    text_vocab = state.text_vocab
    prog_vocab_obj = state.prog_vocab_obj
    model = state.model
    torch_device = state.device

    interp = Interpreter()
    arith_verifier = ArithmeticVerifier()
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
    batch_size = max(1, int(batch_size))

    def _record_decode_stats(stats: Any) -> None:
        nonlocal trunc_count
        if stats is None:
            return
        eos_pos = getattr(stats, "eos_pos", None)
        if eos_pos is None:
            trunc_count += 1
            return
        eos_pos_counts[eos_pos] = eos_pos_counts.get(eos_pos, 0) + 1

    def _predict_batch(
        texts: list[str],
        constraints_list: list[list[dict[str, Any]] | None],
        domain_tags: list[str | None],
    ) -> list[tuple[Program, Any, dict[str, Any]]]:
        if not texts:
            return []
        if len(texts) != len(constraints_list) or len(texts) != len(domain_tags):
            raise ValueError("predict_batch input lengths must match")
        results: list[tuple[Program, Any, dict[str, Any]]] = []
        max_text_len = cfg["train"]["max_text_len"]
        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            input_ids = torch.tensor(
                [text_vocab.encode(text, max_text_len) for text in chunk_texts],
                dtype=torch.long,
                device=torch_device,
            )
            with torch.inference_mode():
                base_out = model(input_ids)
            logits_batch = base_out["logits"].detach()
            for idx, text in enumerate(chunk_texts):
                logits_seq = logits_batch[idx]
                domain_hint = domain_tags[start + idx] or domain_from_tag(text)
                resolved_domain = domain_from_tag(domain_hint) or domain_from_tag(text) or domain_hint
                effective_constrained = constrained_op
                if decode_overrides:
                    domain_overrides = decode_overrides.get(resolved_domain, {})
                    if "constrained_op" in domain_overrides:
                        override_val = domain_overrides.get("constrained_op")
                        if override_val is not None:
                            effective_constrained = bool(override_val)
                pred_ids, decode_stats = greedy_decode_with_opcode_mask(
                    logits_seq,
                    prog_vocab_obj,
                    enforce_opcode=effective_constrained,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    domain_tag=domain_hint,
                    return_stats=True,
                )
                pred_ids = _truncate_at_eos(pred_ids, prog_vocab_obj)
                _record_decode_stats(decode_stats)
                prog = decode_program(pred_ids, prog_vocab_obj)
                out, _, _ = interp.execute(prog, text)
                semantic_meta = {
                    "success": False,
                    "edits": 0,
                    "candidates": 0,
                    "kind": "",
                    "repair_semantic_kind": "",
                    "candidate_rank": 0,
                    "cegis_candidates_tried": 0,
                    "cegis_depth": 0,
                    "cegis_found": False,
                    "repair_cegis_candidates_tried": 0,
                    "repair_cegis_depth": 0,
                    "repair_cegis_found": False,
                    "repair_cegis_mode": "",
                    "repair_cegis_kind": "",
                    "repair_cegis_states": 0,
                    "repair_cegis_nums_used": 0,
                    "repair_cegis_subsets_built": 0,
                    "repair_cegis_values_kept": 0,
                    "repair_cegis_time_ms": 0,
                    "repair_cegis_wallclock_ms": 0,
                }
                if resolved_domain == "math":
                    constraints = constraints_list[start + idx] or []
                    res = arith_verifier.verify(text, prog, out, constraints=constraints)
                    if not res.valid:
                        proof_tokens = [prog_vocab_obj.decode(i) for i in pred_ids]
                        candidates = repair_math_apply_arith_operator(
                            proof_tokens,
                            max_edits=2,
                            allowed_tokens=prog_vocab_obj.token_to_id.keys(),
                        )
                        semantic_meta["candidates"] = len(candidates)
                        for rank, (cand_tokens, kind) in enumerate(candidates, start=1):
                            cand_ids = [prog_vocab_obj.encode(tok) for tok in cand_tokens]
                            cand_prog = decode_program(cand_ids, prog_vocab_obj)
                            cand_out, _, _ = interp.execute(cand_prog, text)
                            res2 = arith_verifier.verify(text, cand_prog, cand_out, constraints=constraints)
                            if res2.valid:
                                prog = cand_prog
                                out = cand_out
                                semantic_meta["success"] = True
                                semantic_meta["kind"] = kind
                                semantic_meta["repair_semantic_kind"] = kind
                                semantic_meta["candidate_rank"] = rank
                                semantic_meta["edits"] = sum(
                                    1 for a, b in zip(proof_tokens, cand_tokens) if a != b
                                )
                                break
                        if not semantic_meta["success"]:
                            cegis = repair_math_cegis(
                                text,
                                constraints,
                                program=prog,
                                max_nums=6,
                                depth=4,
                                limit=20000,
                                max_seconds=0.08,
                                wallclock_ms=cegis_ms,
                                cegis_mode=cegis_mode or "brute",
                            )
                            if cegis is not None:
                                cegis_tokens, meta = cegis
                                cegis_ids = [prog_vocab_obj.encode(tok) for tok in cegis_tokens]
                                cegis_prog = decode_program(cegis_ids, prog_vocab_obj)
                                cegis_out, _, _ = interp.execute(cegis_prog, text)
                                res3 = arith_verifier.verify(text, cegis_prog, cegis_out, constraints=constraints)
                                semantic_meta["cegis_candidates_tried"] = int(meta.get("candidates_tried", 0))
                                semantic_meta["cegis_depth"] = int(meta.get("depth", 0))
                                semantic_meta["cegis_found"] = bool(meta.get("found", False))
                                semantic_meta["repair_cegis_candidates_tried"] = semantic_meta[
                                    "cegis_candidates_tried"
                                ]
                                semantic_meta["repair_cegis_depth"] = semantic_meta["cegis_depth"]
                                semantic_meta["repair_cegis_found"] = semantic_meta["cegis_found"]
                                semantic_meta["repair_cegis_mode"] = str(meta.get("repair_cegis_mode", ""))
                                semantic_meta["repair_cegis_kind"] = str(meta.get("repair_cegis_kind", ""))
                                semantic_meta["repair_cegis_states"] = int(meta.get("repair_cegis_states", 0))
                                semantic_meta["repair_cegis_nums_used"] = int(
                                    meta.get("repair_cegis_nums_used", 0)
                                )
                                semantic_meta["repair_cegis_subsets_built"] = int(
                                    meta.get("repair_cegis_subsets_built", 0)
                                )
                                semantic_meta["repair_cegis_values_kept"] = int(
                                    meta.get("repair_cegis_values_kept", 0)
                                )
                                semantic_meta["repair_cegis_time_ms"] = int(
                                    meta.get("repair_cegis_time_ms", 0)
                                )
                                semantic_meta["repair_cegis_wallclock_ms"] = int(
                                    meta.get("repair_cegis_wallclock_ms", 0)
                                )
                                if res3.valid:
                                    prog = cegis_prog
                                    out = cegis_out
                                    semantic_meta["success"] = True
                                    semantic_meta["kind"] = "cegis"
                                    semantic_meta["repair_semantic_kind"] = "cegis"
                if out is None and repair_op:
                    constraints = constraints_list[start + idx] or []
                    proof_tokens = [prog_vocab_obj.decode(i) for i in pred_ids]
                    runtime_out, failure = runtime_solve(text, constraints, proof_tokens, return_error=True)
                    if failure is not None and failure.code == "PARSE_FAIL":
                        repaired_ids, did_repair = repair_invalid_opcodes(
                            pred_ids,
                            logits_seq,
                            prog_vocab_obj,
                            log_fn=(lambda msg: logger.info(msg)),
                        )
                        if did_repair:
                            repaired_tokens = [prog_vocab_obj.decode(i) for i in repaired_ids]
                            runtime_out, _ = runtime_solve(text, constraints, repaired_tokens, return_error=True)
                            if runtime_out is not None:
                                out = runtime_out
                                prog = decode_program(repaired_ids, prog_vocab_obj)
                results.append((prog, out, semantic_meta))
        return results

    for idx, ex in enumerate(rows):
        if log_every > 0 and (idx % log_every == 0 or idx + 1 == total):
            logger.info("eval progress=%d/%d data=%s", idx + 1, total, data_path)
        constraints = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (ex.constraints or [])
        ]
        orbit_variants = list(ex.orbit)
        flip_variants = list(ex.flips)
        if max_orbits is not None:
            orbit_variants = orbit_variants[: max(0, int(max_orbits))]
        if max_flips is not None:
            flip_variants = flip_variants[: max(0, int(max_flips))]
        orbits = [o.x for o in orbit_variants]
        flips = [f.x for f in flip_variants]
        texts = [ex.x] + orbits + flips
        domain_tag = ex.domain_tag if ex.domain_tag else None
        domain_tags = [domain_tag for _ in texts]
        constraints_list = [constraints for _ in texts]
        preds = _predict_batch(texts, constraints_list, domain_tags)
        prog, out, semantic_meta = preds[0]
        orbit_execs = [out for _, out, _ in preds[1 : 1 + len(orbits)]]
        flip_execs = [out for _, out, _ in preds[1 + len(orbits) :]]
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
        report.meta["repair_semantic"] = dict(semantic_meta)
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

        orbit_pass.append(orbit_output_pass(out, orbit_execs))

        flip_ys = [fvar.y for fvar in flip_variants]
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
        if max_mutants is not None:
            mutants = mutants[: max(0, int(max_mutants))]
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


_DEFAULT_EVAL_SUITE_CONFIG: dict[str, Any] = {
    "ckpt": "out/ckpt.pt",
    "schema_path": "out/data/schema.jsonl",
    "math_path": "out/data/math.jsonl",
    "csp_path": "out/data/csp.jsonl",
    "out_path": "out/report.json",
    "parallel_orbits": False,
    "log_every": 50,
    "batch_size": 64,
    "cegis_ms": None,
    "cegis_mode": "brute",
}


def merge_eval_suite_config(
    cfg: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(_DEFAULT_EVAL_SUITE_CONFIG)
    if cfg:
        merged.update(cfg)
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            merged[key] = value
    return merged


def run_eval_suite(
    config_path: str,
    device: str | torch.device | None = None,
    *,
    ckpt: str | None = None,
    schema_path: str | None = None,
    math_path: str | None = None,
    csp_path: str | None = None,
    out_path: str | None = None,
    log_every: int | None = None,
    parallel_orbits: bool | None = None,
    repair_op: bool | None = None,
    constrained_op: bool | None = None,
    decode_overrides: dict[str, dict[str, Any]] | None = None,
    min_proof_tokens: int | None = None,
    max_prog_len: int | None = None,
    batch_size: int | None = None,
    max_orbits: int | None = None,
    max_flips: int | None = None,
    max_mutants: int | None = None,
    max_examples: int | None = None,
    cegis_ms: int | None = None,
    cegis_mode: str | None = None,
) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    merged = merge_eval_suite_config(
        cfg,
        {
            "ckpt": ckpt,
            "schema_path": schema_path,
            "math_path": math_path,
            "csp_path": csp_path,
            "out_path": out_path,
            "log_every": log_every,
            "parallel_orbits": parallel_orbits,
            "batch_size": batch_size,
            "max_orbits": max_orbits,
            "max_flips": max_flips,
            "max_mutants": max_mutants,
            "max_examples": max_examples,
            "cegis_ms": cegis_ms,
            "cegis_mode": cegis_mode,
        },
    )
    set_orbit_parallelism(bool(merged.get("parallel_orbits", False)))
    log_every = int(merged.get("log_every", 50))
    ckpt = str(merged.get("ckpt", "out/ckpt.pt"))
    schema_path = str(merged.get("schema_path", "out/data/schema.jsonl"))
    math_path = str(merged.get("math_path", "out/data/math.jsonl"))
    csp_path = str(merged.get("csp_path", "out/data/csp.jsonl"))
    resolved_out_path = str(merged.get("out_path", "out/report.json"))
    repair_flag = bool(merged.get("repair_op", False)) if repair_op is None else bool(repair_op)
    constrained_flag = bool(merged.get("constrained_op", True)) if constrained_op is None else bool(constrained_op)
    min_tokens = int(merged.get("min_proof_tokens", 0)) if min_proof_tokens is None else int(min_proof_tokens)
    max_tokens = resolve_max_prog_len(max_prog_len, merged)
    batch_flag = int(merged.get("batch_size", 64)) if batch_size is None else int(batch_size)
    max_orbits_flag = merged.get("max_orbits") if max_orbits is None else max_orbits
    max_flips_flag = merged.get("max_flips") if max_flips is None else max_flips
    max_mutants_flag = merged.get("max_mutants") if max_mutants is None else max_mutants
    max_examples_flag = merged.get("max_examples") if max_examples is None else max_examples
    cegis_ms_flag = merged.get("cegis_ms") if cegis_ms is None else cegis_ms
    cegis_mode_flag = str(merged.get("cegis_mode", "brute")) if cegis_mode is None else str(cegis_mode)
    state = load_eval_state(ckpt, device=device)
    report = {
        "schema": run_eval(
            schema_path,
            ckpt,
            out_path="out/schema_report.json",
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            decode_overrides=decode_overrides,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
            batch_size=batch_flag,
            max_orbits=max_orbits_flag,
            max_flips=max_flips_flag,
            max_mutants=max_mutants_flag,
            max_examples=max_examples_flag,
            cegis_ms=cegis_ms_flag,
            cegis_mode=cegis_mode_flag,
            state=state,
        ),
        "math": run_eval(
            math_path,
            ckpt,
            out_path="out/math_report.json",
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            decode_overrides=decode_overrides,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
            batch_size=batch_flag,
            max_orbits=max_orbits_flag,
            max_flips=max_flips_flag,
            max_mutants=max_mutants_flag,
            max_examples=max_examples_flag,
            cegis_ms=cegis_ms_flag,
            cegis_mode=cegis_mode_flag,
            state=state,
        ),
        "csp": run_eval(
            csp_path,
            ckpt,
            out_path="out/csp_report.json",
            log_every=log_every,
            repair_op=repair_flag,
            constrained_op=constrained_flag,
            decode_overrides=decode_overrides,
            min_proof_tokens=min_tokens,
            max_prog_len=max_tokens,
            batch_size=batch_flag,
            max_orbits=max_orbits_flag,
            max_flips=max_flips_flag,
            max_mutants=max_mutants_flag,
            max_examples=max_examples_flag,
            cegis_ms=cegis_ms_flag,
            cegis_mode=cegis_mode_flag,
            state=state,
        ),
    }
    out_path_obj = Path(resolved_out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(json.dumps(report, indent=2))
    return report
