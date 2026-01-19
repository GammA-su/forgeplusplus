from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import typer

from fc.dsl.codec import decode_program
from fc.dsl.tokens import TokenVocab
from fc.interp.core import Interpreter
from fc.train.answer import AnswerVocab, parse_answer
from fc.train.data import Example, TextVocab
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.jsonl import read_jsonl
from fc.util.opcode_repair import repair_invalid_opcodes
from fc.util.proof_repair import longest_valid_prefix, repair_tokens, scan_tokens
from fc.util.vocab_identity import assert_vocab_match, vocab_identity
from fc.util.tags import domain_from_tag
from fc.verify.mesh import VerifierMesh

app = typer.Typer(add_completion=False)


class Domain(str, Enum):
    schema = "schema"
    math = "math"
    csp = "csp"


def _load_text_vocab(mapping: dict[str, int]) -> TextVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TextVocab(token_to_id=mapping, id_to_token=id_to_token)


def _load_prog_vocab(mapping: dict[str, int]) -> TokenVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TokenVocab(token_to_id=mapping, id_to_token=id_to_token)


def _load_answer_vocab(mapping: dict[str, int]) -> AnswerVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return AnswerVocab(token_to_id=mapping, id_to_token=id_to_token)


def _resolve_device(device: str) -> torch.device:
    if device:
        device_str = device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return str(value)


def _fmt_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)


def _top_violations(violations: dict[str, float], limit: int = 3) -> list[str]:
    ordered = sorted(violations.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return [f"{key}={float(val):.3f}" for key, val in ordered[:limit]]


def _hash_vocab(mapping: dict[str, int]) -> str:
    payload = json.dumps(mapping, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _special_token_ids(mapping: dict[str, int]) -> dict[str, int | None]:
    return {tok: mapping.get(tok) for tok in ("<PAD>", "<UNK>", "<BOS>", "<EOS>")}


@app.command()
def main(
    domain: Domain = typer.Option(..., "--domain"),
    data: str = typer.Option("", "--data"),
    ckpt: str = typer.Option("out/ckpt.pt", "--ckpt"),
    n: int = typer.Option(20, "--n"),
    device: str = typer.Option("", "--device"),
    out: str = typer.Option("", "--out"),
    repair_op: bool = typer.Option(False, "--repair-op"),
    constrained_op: bool = typer.Option(True, "--constrained-op/--no-constrained-op"),
    repair: bool = typer.Option(False, "--repair"),
    max_repairs: int = typer.Option(5, "--max-repairs"),
    repair_k: int = typer.Option(30, "--repair-k"),
    max_proof_tokens: int = typer.Option(0, "--max-proof-tokens"),
    min_proof_tokens: int = typer.Option(0, "--min-proof-tokens"),
) -> None:
    data_path = data or f"out/data/{domain}.jsonl"
    rows = [Example.model_validate(r) for r in read_jsonl(data_path)]
    rows = rows[: max(0, n)]

    ckpt_data = torch.load(ckpt, map_location="cpu")
    mode = ckpt_data.get("mode", "forge" if "prog_vocab" in ckpt_data else "baseline")
    cfg = ckpt_data["config"]
    text_vocab = _load_text_vocab(ckpt_data["text_vocab"])
    prog_vocab_map = ckpt_data.get("prog_vocab")
    prog_vocab: TokenVocab | None = _load_prog_vocab(prog_vocab_map) if isinstance(prog_vocab_map, dict) else None
    torch_device = _resolve_device(device)

    model: torch.nn.Module
    answer_vocab: AnswerVocab | None = None
    interp = Interpreter()
    max_text_len = cfg["train"]["max_text_len"]

    if prog_vocab_map:
        ckpt_id = vocab_identity(prog_vocab_map)
        print(
            "ckpt_prog_vocab size=%d sha256=%s specials=%s"
            % (len(prog_vocab_map), ckpt_id.sha256, _special_token_ids(prog_vocab_map))
        )
        on_disk = Path(ckpt).resolve().parent / "prog_vocab.json"
        if on_disk.exists():
            disk_map = json.loads(on_disk.read_text(encoding="utf-8"))
            if isinstance(disk_map, dict):
                disk_hash = _hash_vocab(disk_map)
                print(
                    "out/prog_vocab.json size=%d sha256=%s specials=%s"
                    % (len(disk_map), disk_hash, _special_token_ids(disk_map))
                )
                assert_vocab_match(
                    prog_vocab_map,
                    disk_map,
                    expected_label="ckpt.prog_vocab",
                    actual_label="out/prog_vocab.json",
                )

    if mode == "baseline" or "answer_vocab" in ckpt_data:
        from fc.model.backbone import BackboneConfig
        from fc.model.baseline import BaselineConfig, BaselineModel

        answer_vocab = _load_answer_vocab(ckpt_data["answer_vocab"])
        max_answer_len = cfg.get("max_answer_len") or cfg["train"].get("max_answer_len") or cfg.get("max_prog_len", 64)
        bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
        mcfg = BaselineConfig(
            vocab_size=cfg["text_vocab_size"],
            answer_vocab_size=len(answer_vocab.token_to_id),
            max_answer_len=max_answer_len,
            backbone=bcfg,
        )
        model = BaselineModel(mcfg)
        model.load_state_dict(ckpt_data["model"])
    else:
        from fc.model.backbone import BackboneConfig
        from fc.model.forge import ForgeModel, ModelConfig
        from fc.model.primal_dual import PrimalDualConfig
        from fc.model.slots import SlotConfig

        if prog_vocab is None:
            raise SystemExit("Missing prog_vocab for forge checkpoint")
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
        model.load_state_dict(ckpt_data["model"])

    model.to(torch_device)
    model.eval()

    mesh = VerifierMesh()
    out_handle = None
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")

    stats_fail_pos: dict[int, int] = {}
    stats_eos_pos: dict[int, int] = {}
    stats_early_eos = 0
    stats_trunc_maxlen = 0

    for idx, ex in enumerate(rows):
        constraints = [
            c.model_dump() if hasattr(c, "model_dump") else c for c in (ex.constraints or [])
        ]
        input_ids = torch.tensor(
            [text_vocab.encode(ex.x, max_text_len)],
            dtype=torch.long,
            device=torch_device,
        )
        with torch.inference_mode():
            outputs = model(input_ids)

        program = None
        pred: Any
        if answer_vocab is not None:
            pred_ids = outputs["answer_ids"][0].detach().cpu().tolist()
            pred_text = answer_vocab.decode(pred_ids)
            pred = parse_answer(pred_text)
            if pred is None:
                from fc.util.runtime_solve import runtime_solve

                head = pred_ids[:32]
                print(f"  pred_ids_head={head}")
                if prog_vocab is None:
                    print("  runtime_solve_error=VOCAB_MISMATCH: missing prog_vocab in checkpoint")
                else:
                    proof_tokens = [prog_vocab.decode(i) for i in pred_ids]
                    unk = sum(1 for t in proof_tokens if t == "<UNK>")
                    print(f"  proof_tokens_len={len(proof_tokens)} unk={unk}")
                    print(f"  proof_tokens_head={proof_tokens[:40]}")
                    pred, failure = runtime_solve(ex.x, constraints, proof_tokens, return_error=True)
                    print(f"  runtime_solve_pred={_fmt_value(pred)}")
                    if failure is not None:
                        print(f"  runtime_solve_error={failure.code}:{failure.detail}")
        else:
            if prog_vocab is None:
                raise SystemExit("Missing prog_vocab for forge checkpoint")
            logits_seq = outputs["logits"][0].detach()
            max_tokens = max_proof_tokens if max_proof_tokens > 0 else cfg["max_prog_len"]
            min_tokens = max(0, min_proof_tokens)
            pred_ids, decode_stats = greedy_decode_with_opcode_mask(
                logits_seq,
                prog_vocab,
                enforce_opcode=constrained_op,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                return_stats=True,
            )
            if decode_stats.eos_pos is None:
                stats_trunc_maxlen += 1
            else:
                stats_eos_pos[decode_stats.eos_pos] = stats_eos_pos.get(decode_stats.eos_pos, 0) + 1
                if decode_stats.eos_pos < min_tokens:
                    stats_early_eos += 1
            program = decode_program(pred_ids, prog_vocab)
            pred, _, errors = interp.execute(program, ex.x)
            if pred is None:
                from fc.util.runtime_solve import runtime_solve

                head = pred_ids[:32]
                print(f"  pred_ids_head={head}")
                proof_tokens = [prog_vocab.decode(i) for i in pred_ids]
                unk = sum(1 for t in proof_tokens if t == "<UNK>")
                print(f"  proof_tokens_len={len(proof_tokens)} unk={unk}")
                print(f"  proof_tokens_head={proof_tokens[:40]}")
                ok, parse_failure, _ = scan_tokens(proof_tokens)
                if not ok and parse_failure is not None:
                    stats_fail_pos[parse_failure.pos] = stats_fail_pos.get(parse_failure.pos, 0) + 1
                    if parse_failure.reason == "invalid_opcode":
                        print(f"  parse_fail_reason=INVALID_OPCODE pos={parse_failure.pos}")
                    if "<EOS>" in proof_tokens and proof_tokens.index("<EOS>") < len(proof_tokens) - 1:
                        stats_early_eos += 1
                    if parse_failure.reason == "eof" and len(pred_ids) >= cfg["max_prog_len"]:
                        stats_trunc_maxlen += 1

                runtime_pred, failure = runtime_solve(ex.x, constraints, proof_tokens, return_error=True)
                if runtime_pred is not None:
                    pred = runtime_pred
                print(f"  runtime_solve_pred={_fmt_value(runtime_pred)}")
                if failure is not None:
                    print(f"  runtime_solve_error={failure.code}:{failure.detail}")
                    if repair_op and failure.code == "PARSE_FAIL":
                        repaired_ids, did_repair = repair_invalid_opcodes(
                            pred_ids,
                            logits_seq,
                            prog_vocab,
                            log_fn=print,
                        )
                        if did_repair:
                            repaired_tokens = [prog_vocab.decode(i) for i in repaired_ids]
                            repaired_pred, repaired_failure = runtime_solve(
                                ex.x,
                                constraints,
                                repaired_tokens,
                                return_error=True,
                            )
                            if repaired_pred is not None:
                                pred = repaired_pred
                            print(f"  op_repair_pred={_fmt_value(repaired_pred)}")
                            if repaired_failure is not None:
                                print(
                                    f"  op_repair_error={repaired_failure.code}:{repaired_failure.detail}"
                                )
                    if repair and failure.code == "PARSE_FAIL":
                        repair_result = repair_tokens(
                            proof_tokens,
                            logits_seq,
                            prog_vocab,
                            max_repairs=max_repairs,
                            k=repair_k,
                            log_fn=print,
                        )
                        if repair_result.success:
                            repaired_pred, repaired_failure = runtime_solve(
                                ex.x,
                                constraints,
                                repair_result.tokens,
                                return_error=True,
                            )
                            if repaired_pred is not None:
                                pred = repaired_pred
                            print(f"  repair_pred={_fmt_value(repaired_pred)}")
                            if repaired_failure is not None:
                                print(
                                    f"  repair_error={repaired_failure.code}:{repaired_failure.detail}"
                                )
                        else:
                            prefix = longest_valid_prefix(repair_result.tokens)
                            if prefix:
                                prefix_pred, prefix_failure = runtime_solve(
                                    ex.x,
                                    constraints,
                                    prefix,
                                    return_error=True,
                                )
                                if prefix_pred is not None:
                                    pred = prefix_pred
                                print("  prefix_reason=VALID_PREFIX")
                                print(f"  prefix_pred={_fmt_value(prefix_pred)}")
                                if prefix_failure is not None:
                                    print(
                                        f"  prefix_error={prefix_failure.code}:{prefix_failure.detail}"
                                    )
                            else:
                                print("  prefix_reason=NO_VALID_PREFIX")
                if errors:
                    print(f"  interp_errors={errors[:5]}")

        report = mesh.run(ex.x, program, pred, domain=ex.domain, constraints=ex.constraints)
        formal = report.meta.get("formal", {})
        violations = formal.get("violations", {}) or {}
        c_hard = float(sum(float(v) for v in violations.values())) if violations else 0.0
        status = "PASS" if formal.get("valid", False) else "FAIL"
        top = _top_violations(violations)
        tag = ex.domain_tag or domain_from_tag(ex.x) or ""
        snippet = ex.x[:16]

        print(
            f"[{idx}] tag={tag} snippet={snippet} y_true={_fmt_value(ex.y)} y_pred={_fmt_value(pred)} "
            f"status={status} c_hard={c_hard:.3f} top_violations={top}"
        )
        if program is not None:
            print(f"  program={program.skeleton()}")

        if out_handle is not None:
            row = {
                "x": ex.x,
                "y_true": _json_safe(ex.y),
                "y_pred": _json_safe(pred),
                "violations": {str(k): float(v) for k, v in violations.items()},
                "c_hard": c_hard,
            }
            out_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            out_handle.write("\n")

    if out_handle is not None:
        out_handle.close()
    if stats_fail_pos:
        ordered = sorted(stats_fail_pos.items(), key=lambda kv: kv[0])
        dist = ", ".join([f"{pos}:{count}" for pos, count in ordered[:10]])
        print(f"parse_fail_pos_top={dist}")
    if stats_eos_pos:
        ordered = sorted(stats_eos_pos.items(), key=lambda kv: kv[1], reverse=True)[:10]
        dist = ", ".join([f"{pos}:{count}" for pos, count in ordered])
        print(f"eos_pos_top={dist}")
    print(f"early_eos_count={stats_early_eos} trunc_by_maxlen_count={stats_trunc_maxlen}")


if __name__ == "__main__":
    app()
