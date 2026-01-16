from __future__ import annotations

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
from fc.util.jsonl import read_jsonl
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


@app.command()
def main(
    domain: Domain = typer.Option(..., "--domain"),
    data: str = typer.Option("", "--data"),
    ckpt: str = typer.Option("out/ckpt.pt", "--ckpt"),
    n: int = typer.Option(20, "--n"),
    device: str = typer.Option("", "--device"),
    out: str = typer.Option("", "--out"),
) -> None:
    data_path = data or f"out/data/{domain}.jsonl"
    rows = [Example.model_validate(r) for r in read_jsonl(data_path)]
    rows = rows[: max(0, n)]

    ckpt_data = torch.load(ckpt, map_location="cpu")
    mode = ckpt_data.get("mode", "forge" if "prog_vocab" in ckpt_data else "baseline")
    cfg = ckpt_data["config"]
    text_vocab = _load_text_vocab(ckpt_data["text_vocab"])
    torch_device = _resolve_device(device)

    model: torch.nn.Module
    answer_vocab: AnswerVocab | None = None
    prog_vocab: TokenVocab | None = None
    interp = Interpreter()
    max_text_len = cfg["train"]["max_text_len"]

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

        prog_vocab = _load_prog_vocab(ckpt_data["prog_vocab"])
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

    for idx, ex in enumerate(rows):
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
        else:
            if prog_vocab is None:
                raise SystemExit("Missing prog_vocab for forge checkpoint")
            pred_ids = torch.argmax(outputs["logits"], dim=-1)[0].detach().cpu().tolist()
            program = decode_program(pred_ids, prog_vocab)
            pred, _, _ = interp.execute(program, ex.x)

        report = mesh.run(ex.x, program, pred, domain=ex.domain, constraints=ex.constraints)
        formal = report.meta.get("formal", {})
        violations = formal.get("violations", {}) or {}
        c_hard = float(sum(float(v) for v in violations.values())) if violations else 0.0
        status = "PASS" if formal.get("valid", False) else "FAIL"
        top = _top_violations(violations)

        print(
            f"[{idx}] y_true={_fmt_value(ex.y)} y_pred={_fmt_value(pred)} "
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


if __name__ == "__main__":
    app()
