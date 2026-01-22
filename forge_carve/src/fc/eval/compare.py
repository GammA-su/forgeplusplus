from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from fc.dsl.codec import decode_program
from fc.dsl.tokens import TokenVocab
from fc.interp.core import Interpreter
from fc.model.backbone import BackboneConfig
from fc.model.baseline import BaselineConfig, BaselineModel
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig
from fc.morph.equiv import outputs_equivalent
from fc.eval.metrics import flip_output_pass, orbit_output_pass
from fc.train.answer import AnswerVocab, parse_answer
from fc.train.data import Example, TextVocab
from fc.util.jsonl import read_jsonl
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.opcode_repair import repair_invalid_opcodes
from fc.util.decode_limits import resolve_max_prog_len
from fc.util.tags import domain_from_tag
from fc.util.runtime_solve import runtime_solve
from fc.util.vocab_identity import assert_vocab_match, vocab_identity
from fc.verify.mesh import VerifierMesh

CONF_THRESHOLD = 0.8

logger = logging.getLogger(__name__)


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


def _load_answer_vocab(mapping: dict[str, int]) -> AnswerVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return AnswerVocab(token_to_id=mapping, id_to_token=id_to_token)


def _formal_confidence(mesh: VerifierMesh, report: Any) -> float:
    indices = [
        mesh.constraint_names.index("schema"),
        mesh.constraint_names.index("arith"),
        mesh.constraint_names.index("csp"),
        mesh.constraint_names.index("code"),
    ]
    c_sum = sum(report.c[i] for i in indices)
    return float(1.0 / (1.0 + c_sum))


def _log_eval_progress(stage: str, idx: int, total: int, domain: str) -> None:
    if not total:
        return
    if idx % 10 == 0 or idx + 1 == total:
        logger.info("%s %s progress %d/%d", stage, domain, idx + 1, total)


def _mean(vals: Iterable[bool]) -> float:
    items = list(vals)
    if not items:
        return 0.0
    return float(sum(1 for v in items if v) / len(items))


def _confident_error_rate(confs: list[float], correct: list[bool]) -> float:
    if not confs:
        return 0.0
    total = 0
    errors = 0
    for conf, ok in zip(confs, correct):
        if conf >= CONF_THRESHOLD:
            total += 1
            if not ok:
                errors += 1
    if total == 0:
        return 0.0
    return float(errors / total)


def _load_examples(schema_path: str, math_path: str, csp_path: str) -> list[Example]:
    rows = []
    for path in (schema_path, math_path, csp_path):
        rows.extend(read_jsonl(path))
    return [Example.model_validate(r) for r in rows]


def _split_by_domain(examples: list[Example]) -> dict[str, list[Example]]:
    grouped = {"schema": [], "math": [], "csp": []}
    for ex in examples:
        if ex.domain in grouped:
            grouped[ex.domain].append(ex)
    return grouped


def _predict_baseline(
    model: BaselineModel,
    text_vocab: TextVocab,
    answer_vocab: AnswerVocab,
    text: str,
    max_text_len: int,
    device: torch.device,
) -> Any:
    input_ids = torch.tensor([text_vocab.encode(text, max_text_len)], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids)
    pred_ids = outputs["answer_ids"][0].detach().cpu().tolist()
    pred_text = answer_vocab.decode(pred_ids)
    return parse_answer(pred_text)


def evaluate_baseline(
    examples: list[Example],
    ckpt_path: str,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    text_vocab = _load_text_vocab(ckpt["text_vocab"])
    answer_vocab = _load_answer_vocab(ckpt["answer_vocab"])
    max_text_len = cfg["train"]["max_text_len"]
    max_answer_len = cfg.get("max_answer_len") or cfg["train"].get("max_answer_len") or cfg.get("max_prog_len", 256)
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    mcfg = BaselineConfig(
        vocab_size=cfg["text_vocab_size"],
        answer_vocab_size=len(answer_vocab.token_to_id),
        max_answer_len=max_answer_len,
        backbone=bcfg,
    )
    model = BaselineModel(mcfg)
    model.load_state_dict(ckpt["model"])
    device_str = str(device) if device is not None else "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.eval()
    logger.info("baseline evaluation start rows=%d device=%s", len(examples), torch_device)

    mesh = VerifierMesh()
    correct = []
    orbit_ok = []
    flip_ok = []
    confs = []

    for idx, ex in enumerate(examples):
        _log_eval_progress("baseline", idx, len(examples), ex.domain)
        pred = _predict_baseline(model, text_vocab, answer_vocab, ex.x, max_text_len, torch_device)
        report = mesh.run(ex.x, program=None, output=pred, domain=ex.domain, constraints=ex.constraints)
        confs.append(_formal_confidence(mesh, report))
        correct.append(outputs_equivalent(pred, ex.y))

        orbit_preds = []
        for orb in ex.orbit:
            o_pred = _predict_baseline(model, text_vocab, answer_vocab, orb.x, max_text_len, torch_device)
            orbit_preds.append(o_pred)
        orbit_ok.append(orbit_output_pass(pred, orbit_preds))

        flip_preds = []
        flip_ys = []
        for flip in ex.flips:
            f_pred = _predict_baseline(model, text_vocab, answer_vocab, flip.x, max_text_len, torch_device)
            flip_preds.append(f_pred)
            flip_ys.append(flip.y)
        flip_ok.append(flip_output_pass(pred, ex.y, flip_preds, flip_ys))

    return {
        "verified_accuracy": _mean(correct),
        "orbit_invariance": _mean(orbit_ok),
        "flip_sensitivity": _mean(flip_ok),
        "confident_error_rate": _confident_error_rate(confs, correct),
    }


def _predict_forge(
    model: ForgeModel,
    text_vocab: TextVocab,
    prog_vocab: TokenVocab,
    text: str,
    max_text_len: int,
    device: torch.device,
    interp: Interpreter,
    constraints: list[dict[str, Any]] | None = None,
    *,
    repair_op: bool = False,
    constrained_op: bool = True,
    min_proof_tokens: int = 0,
    max_prog_len: int | None = None,
    domain_tag: str | None = None,
    record_decode_stats: Callable[[Any], None] | None = None,
    log_repair: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    input_ids = torch.tensor([text_vocab.encode(text, max_text_len)], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids)
    logits_seq = outputs["logits"][0].detach()
    domain_hint = domain_tag or domain_from_tag(text)
    pred_ids, decode_stats = greedy_decode_with_opcode_mask(
        logits_seq,
        prog_vocab,
        enforce_opcode=constrained_op,
        min_tokens=min_proof_tokens,
        max_tokens=max_prog_len,
        domain_tag=domain_hint,
        return_stats=True,
    )
    pred_ids = _truncate_at_eos(pred_ids, prog_vocab)
    if record_decode_stats is not None:
        record_decode_stats(decode_stats)
    program = decode_program(pred_ids, prog_vocab)
    out, _, _ = interp.execute(program, text)
    if out is None and repair_op:
        proof_tokens = [prog_vocab.decode(i) for i in pred_ids]
        runtime_out, failure = runtime_solve(text, constraints or [], proof_tokens, return_error=True)
        if failure is not None and failure.code == "PARSE_FAIL":
            repaired_ids, did_repair = repair_invalid_opcodes(
                pred_ids,
                logits_seq,
                prog_vocab,
                log_fn=log_repair,
            )
            if did_repair:
                repaired_tokens = [prog_vocab.decode(i) for i in repaired_ids]
                runtime_out, _ = runtime_solve(text, constraints or [], repaired_tokens, return_error=True)
                if runtime_out is not None:
                    out = runtime_out
                    program = decode_program(repaired_ids, prog_vocab)
    return program, out


def evaluate_forge(
    examples: list[Example],
    ckpt_path: str,
    device: str | torch.device | None = None,
    *,
    repair_op: bool = False,
    constrained_op: bool = True,
    min_proof_tokens: int = 0,
    max_prog_len: int | None = None,
) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    text_vocab = _load_text_vocab(ckpt["text_vocab"])
    prog_vocab_map = ckpt["prog_vocab"]
    prog_vocab = _load_prog_vocab(prog_vocab_map)
    if "prog_vocab_sha256" in ckpt:
        ckpt_id = vocab_identity(prog_vocab_map)
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
                prog_vocab_map,
                disk_map,
                expected_label="ckpt.prog_vocab",
                actual_label=str(prog_vocab_path),
            )
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(
        vocab_size=len(prog_vocab.token_to_id),
        max_prog_len=cfg["max_prog_len"],
        backbone=bcfg,
        slots=scfg,
        primal_dual=pcfg,
    )
    model = ForgeModel(mcfg)
    model.load_state_dict(ckpt["model"])
    device_str = str(device) if device is not None else "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.eval()
    logger.info("forge evaluation start rows=%d device=%s", len(examples), torch_device)

    interp = Interpreter()
    mesh = VerifierMesh()
    correct = []
    orbit_ok = []
    flip_ok = []
    confs = []
    max_text_len = cfg["train"]["max_text_len"]
    max_tokens = resolve_max_prog_len(max_prog_len, cfg, ckpt=ckpt)
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

    for idx, ex in enumerate(examples):
        _log_eval_progress("forge", idx, len(examples), ex.domain)
        constraints = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (ex.constraints or [])
        ]
        log_repair = (lambda msg: logger.info(msg)) if repair_op else None
        prog, out = _predict_forge(
            model,
            text_vocab,
            prog_vocab,
            ex.x,
            max_text_len,
            torch_device,
            interp,
            constraints,
            repair_op=repair_op,
            constrained_op=constrained_op,
            min_proof_tokens=min_proof_tokens,
            max_prog_len=max_tokens,
            domain_tag=ex.domain_tag,
            record_decode_stats=_record_decode_stats,
            log_repair=log_repair,
        )
        report = mesh.run(ex.x, prog, out, domain=ex.domain, constraints=ex.constraints)
        confs.append(_formal_confidence(mesh, report))
        correct.append(outputs_equivalent(out, ex.y))

        orbit_execs = []
        for orb in ex.orbit:
            _, o_out = _predict_forge(
                model,
                text_vocab,
                prog_vocab,
                orb.x,
                max_text_len,
                torch_device,
                interp,
                constraints,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
                domain_tag=ex.domain_tag,
                record_decode_stats=_record_decode_stats,
                log_repair=log_repair,
            )
            orbit_execs.append(o_out)
        orbit_ok.append(orbit_output_pass(out, orbit_execs))

        flip_execs = []
        flip_ys = []
        for flip in ex.flips:
            _, f_out = _predict_forge(
                model,
                text_vocab,
                prog_vocab,
                flip.x,
                max_text_len,
                torch_device,
                interp,
                constraints,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
                domain_tag=ex.domain_tag,
                record_decode_stats=_record_decode_stats,
                log_repair=log_repair,
            )
            flip_execs.append(f_out)
            flip_ys.append(flip.y)
        flip_ok.append(flip_output_pass(out, ex.y, flip_execs, flip_ys))

    if eos_pos_counts:
        ordered = sorted(eos_pos_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        dist = ", ".join([f"{pos}:{count}" for pos, count in ordered])
        logger.info(
            "eval_compare_decode eos_pos_top=%s trunc_by_maxlen=%d min_tokens=%d max_tokens=%s",
            dist,
            trunc_count,
            min_proof_tokens,
            str(max_tokens),
        )
    return {
        "verified_accuracy": _mean(correct),
        "orbit_invariance": _mean(orbit_ok),
        "flip_sensitivity": _mean(flip_ok),
        "confident_error_rate": _confident_error_rate(confs, correct),
    }


def run_compare(
    schema_path: str,
    math_path: str,
    csp_path: str,
    out_path: str,
    baseline_ckpt: str | None = None,
    forge_ckpt: str | None = None,
    ablation_ckpt: str | None = None,
    device: str | torch.device | None = None,
    repair_op: bool = False,
    constrained_op: bool = True,
    min_proof_tokens: int = 0,
    max_prog_len: int | None = None,
) -> dict[str, Any]:
    examples = _load_examples(schema_path, math_path, csp_path)
    grouped = _split_by_domain(examples)
    logger.info("run_compare loaded=%d schema=%d math=%d csp=%d", len(examples), len(grouped["schema"]), len(grouped["math"]), len(grouped["csp"]))
    report: dict[str, Any] = {}
    if baseline_ckpt:
        baseline_overall = evaluate_baseline(examples, baseline_ckpt, device=device)
        report["baseline"] = {
            "overall": baseline_overall,
            "schema": evaluate_baseline(grouped["schema"], baseline_ckpt, device=device),
            "math": evaluate_baseline(grouped["math"], baseline_ckpt, device=device),
            "csp": evaluate_baseline(grouped["csp"], baseline_ckpt, device=device),
        }
    if forge_ckpt:
        forge_overall = evaluate_forge(
            examples,
            forge_ckpt,
            device=device,
            repair_op=repair_op,
            constrained_op=constrained_op,
            min_proof_tokens=min_proof_tokens,
            max_prog_len=max_prog_len,
        )
        report["forge"] = {
            "overall": forge_overall,
            "schema": evaluate_forge(
                grouped["schema"],
                forge_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
            "math": evaluate_forge(
                grouped["math"],
                forge_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
            "csp": evaluate_forge(
                grouped["csp"],
                forge_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
        }
    if ablation_ckpt:
        ablation_overall = evaluate_forge(
            examples,
            ablation_ckpt,
            device=device,
            repair_op=repair_op,
            constrained_op=constrained_op,
            min_proof_tokens=min_proof_tokens,
            max_prog_len=max_prog_len,
        )
        report["ablation"] = {
            "overall": ablation_overall,
            "schema": evaluate_forge(
                grouped["schema"],
                ablation_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
            "math": evaluate_forge(
                grouped["math"],
                ablation_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
            "csp": evaluate_forge(
                grouped["csp"],
                ablation_ckpt,
                device=device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_prog_len,
            ),
        }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
