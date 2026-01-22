from __future__ import annotations

from pathlib import Path

import typer

from fc.train.trainer import load_config, train_from_dataset
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


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
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    cfg = load_config(config)
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
    )
    final_ckpt = Path(ckpt_path)
    if out_path.suffix == ".pt" and final_ckpt != out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_ckpt.replace(out_path)
        final_ckpt = out_path
    logger.info("train_phase complete ckpt=%s", final_ckpt)


if __name__ == "__main__":
    app()
