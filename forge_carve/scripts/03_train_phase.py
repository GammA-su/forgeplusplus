from __future__ import annotations

from pathlib import Path
import json
from collections import Counter
from typing import Any

import torch

import typer

from fc.train.trainer import load_config, train_from_dataset
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


def _read_proof_tokens(row: dict[str, Any], proof_source: str) -> list[str]:
    value = row.get(proof_source)
    row_id = row.get("id", "<no-id>")
    if isinstance(value, list):
        return [str(tok) for tok in value]
    if isinstance(value, str):
        return value.split()
    if isinstance(value, dict) and "tokens" in value:
        tokens = value.get("tokens")
        if isinstance(tokens, list):
            return [str(tok) for tok in tokens]
        raise ValueError(f"proof tokens not list id={row_id} type={type(tokens).__name__}")
    raise ValueError(f"unsupported proof type id={row_id} type={type(value).__name__}")


def _load_prog_vocab(path: str) -> dict[str, int]:
    ckpt = torch.load(path, map_location="cpu")
    raw = ckpt.get("prog_vocab")
    if raw is None:
        raise ValueError("init ckpt missing prog_vocab")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {tok: idx for idx, tok in enumerate(raw)}
    raise ValueError("init ckpt prog_vocab must be dict or list")


def _scan_preflight(
    data_path: str,
    proof_source: str,
    vocab: dict[str, int],
    limit: int,
) -> tuple[Counter[str], tuple[int, str, list[str]] | None]:
    missing_counts: Counter[str] = Counter()
    first_bad: tuple[int, str, list[str]] | None = None
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = _read_proof_tokens(row, proof_source)
            missing = [tok for tok in tokens if tok not in vocab]
            if missing:
                missing_counts.update(missing)
                if first_bad is None:
                    row_id = str(row.get("id", idx))
                    first_bad = (idx, row_id, missing[:50])
    return missing_counts, first_bad


def _preflight_or_raise(
    data_path: str,
    proof_source: str,
    vocab: dict[str, int],
    limit: int,
    logger: Any | None = None,
) -> None:
    missing_counts, first_bad = _scan_preflight(
        data_path=data_path,
        proof_source=proof_source,
        vocab=vocab,
        limit=limit,
    )
    if not missing_counts:
        return
    if logger is not None:
        top_missing = missing_counts.most_common(20)
        logger.error("proof preflight found missing tokens in vocab")
        if first_bad is not None:
            idx, row_id, row_missing = first_bad
            logger.error("first bad row idx=%s id=%s missing=%s", idx, row_id, row_missing)
        logger.error("top missing tokens=%s", top_missing)
    raise ValueError("Unknown proof tokens detected during preflight")


def _apply_train_overrides(
    cfg: dict[str, Any],
    *,
    precision: str | None,
    microbatch: int | None,
    grad_accum: int | None,
    grad_checkpoint: bool | None,
    optimizer: str | None,
) -> dict[str, Any]:
    train_cfg = cfg.setdefault("train", {})
    if precision:
        train_cfg["precision"] = precision
    else:
        train_cfg.setdefault("precision", "auto")
    if microbatch is not None:
        train_cfg["microbatch"] = int(microbatch)
    else:
        train_cfg.setdefault("microbatch", 1)
    if grad_accum is not None:
        train_cfg["grad_accum"] = int(grad_accum)
    else:
        train_cfg.setdefault("grad_accum", 1)
    if grad_checkpoint is not None:
        train_cfg["grad_checkpoint"] = bool(grad_checkpoint)
    else:
        train_cfg.setdefault("grad_checkpoint", True)
    if optimizer:
        train_cfg["optimizer"] = optimizer
    else:
        train_cfg.setdefault("optimizer", "adafactor")
    return cfg


@app.command()
def main(
    config: str = typer.Option("configs/train/phase3_headtune.yaml", "--config"),
    data: str = typer.Option("out/data/math.jsonl", "--data"),
    init: str = typer.Option("", "--init"),
    steps: int | None = typer.Option(None, "--steps"),
    out: str = typer.Option("out/ckpt_phase3.pt", "--out"),
    device: str = typer.Option("", "--device"),
    max_prog_len: int | None = typer.Option(None, "--max-prog-len"),
    include_variants: bool = typer.Option(True, "--include-variants/--no-include-variants"),
    proof_source: str = typer.Option("", "--proof-source"),
    proof_preflight: bool = typer.Option(True, "--proof-preflight/--no-proof-preflight"),
    proof_preflight_n: int = typer.Option(5000, "--proof-preflight-n"),
    proof_preflight_only: bool = typer.Option(False, "--proof-preflight-only"),
    precision: str = typer.Option("", "--precision"),
    microbatch: int = typer.Option(1, "--microbatch"),
    grad_accum: int = typer.Option(1, "--grad-accum"),
    grad_checkpoint: bool = typer.Option(True, "--grad-checkpoint/--no-grad-checkpoint"),
    optimizer: str = typer.Option("adafactor", "--optimizer"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    cfg = load_config(config)
    _apply_train_overrides(
        cfg,
        precision=precision or None,
        microbatch=microbatch,
        grad_accum=grad_accum,
        grad_checkpoint=grad_checkpoint,
        optimizer=optimizer,
    )
    resolved_proof_source = proof_source or cfg.get("train", {}).get("proof_supervision_source", "proof")
    out_path = Path(out)
    if out_path.suffix == ".pt":
        out_dir = str(out_path.parent)
    else:
        out_dir = str(out_path)
    init_path = init or None
    device_override = device or str(runtime.device)
    logger.info(
        "train_phase config=%s data=%s init=%s steps=%s out=%s max_prog_len=%s proof_source=%s",
        config,
        data,
        init_path,
        steps,
        out,
        max_prog_len,
        resolved_proof_source,
    )
    logger.info(
        "train_phase vram precision=%s microbatch=%s grad_accum=%s grad_checkpoint=%s",
        cfg.get("train", {}).get("precision"),
        cfg.get("train", {}).get("microbatch"),
        cfg.get("train", {}).get("grad_accum"),
        cfg.get("train", {}).get("grad_checkpoint"),
    )
    logger.info("train_phase optimizer=%s", cfg.get("train", {}).get("optimizer"))
    if proof_preflight and init_path:
        vocab = _load_prog_vocab(init_path)
        _preflight_or_raise(
            data_path=data,
            proof_source=resolved_proof_source,
            vocab=vocab,
            limit=proof_preflight_n,
            logger=logger,
        )
        logger.info("proof preflight ok rows_scanned=%s", proof_preflight_n)
        if proof_preflight_only:
            logger.info("proof preflight only requested; exiting")
            return
    ckpt_path = train_from_dataset(
        config_path=config,
        data_path=data,
        out_dir=out_dir,
        device=device_override,
        init_ckpt_path=init_path,
        max_prog_len=max_prog_len,
        steps=steps,
        include_variants=include_variants,
        proof_source=resolved_proof_source,
        precision=cfg.get("train", {}).get("precision"),
        microbatch=cfg.get("train", {}).get("microbatch"),
        grad_accum=cfg.get("train", {}).get("grad_accum"),
        grad_checkpoint=cfg.get("train", {}).get("grad_checkpoint"),
        optimizer=cfg.get("train", {}).get("optimizer"),
    )
    final_ckpt = Path(ckpt_path)
    if out_path.suffix == ".pt" and final_ckpt != out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_ckpt.replace(out_path)
        final_ckpt = out_path
    logger.info("train_phase complete ckpt=%s", final_ckpt)


if __name__ == "__main__":
    app()
