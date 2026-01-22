from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from fc.eval.suite import run_eval, run_eval_suite
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = "",
    ckpt: str = "out/ckpt.pt",
    schema_path: str = "out/data/schema.jsonl",
    math_path: str = "out/data/math.jsonl",
    csp_path: str = "out/data/csp.jsonl",
    out_path: str = "out/report.json",
    repair_op: bool = typer.Option(False, "--repair-op"),
    constrained_op: bool = typer.Option(True, "--constrained-op/--no-constrained-op"),
    max_prog_len: int = typer.Option(256, "--max-prog-len"),
    min_proof_tokens: int = typer.Option(0, "--min-proof-tokens"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    if config:
        with open(config, "r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        ckpt = cfg.get("ckpt", ckpt)
        schema_path = cfg.get("schema_path", schema_path)
        math_path = cfg.get("math_path", math_path)
        csp_path = cfg.get("csp_path", csp_path)
        out_path = cfg.get("out_path", out_path)
    logger.info("eval_suite ckpt=%s schema=%s math=%s csp=%s out=%s", ckpt, schema_path, math_path, csp_path, out_path)
    if config:
        report = run_eval_suite(
            config,
            device=runtime.device,
            repair_op=repair_op,
            constrained_op=constrained_op,
            min_proof_tokens=min_proof_tokens if min_proof_tokens > 0 else None,
            max_prog_len=max_prog_len,
        )
    else:
        max_tokens = max_prog_len
        report = {
            "schema": run_eval(
                schema_path,
                ckpt,
                out_path="out/schema_report.json",
                device=runtime.device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
            ),
            "math": run_eval(
                math_path,
                ckpt,
                out_path="out/math_report.json",
                device=runtime.device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
            ),
            "csp": run_eval(
                csp_path,
                ckpt,
                out_path="out/csp_report.json",
                device=runtime.device,
                repair_op=repair_op,
                constrained_op=constrained_op,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
            ),
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
    logger.info("eval_suite complete out=%s", out_path)


if __name__ == "__main__":
    app()
