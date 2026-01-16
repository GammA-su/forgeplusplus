from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from fc.eval.compare import run_compare
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = "",
    baseline_ckpt: str = "out/baseline_ckpt.pt",
    forge_ckpt: str = "out/ckpt.pt",
    ablation_ckpt: str = "",
    schema_path: str = "out/data/schema.jsonl",
    math_path: str = "out/data/math.jsonl",
    csp_path: str = "out/data/csp.jsonl",
    out: str = typer.Option("out/compare.json", "--out"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    out_path = out
    if config:
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        baseline_ckpt = cfg.get("baseline_ckpt", baseline_ckpt)
        forge_ckpt = cfg.get("forge_ckpt", forge_ckpt)
        ablation_ckpt = cfg.get("ablation_ckpt", ablation_ckpt)
        schema_path = cfg.get("schema_path", schema_path)
        math_path = cfg.get("math_path", math_path)
        csp_path = cfg.get("csp_path", csp_path)
        if out == "out/compare.json":
            out_path = cfg.get("out_path", out_path)
    logger.info(
        "eval_compare baseline=%s forge=%s ablation=%s schema=%s math=%s csp=%s out=%s",
        baseline_ckpt,
        forge_ckpt,
        ablation_ckpt or "none",
        schema_path,
        math_path,
        csp_path,
        out_path,
    )
    try:
        report = run_compare(
            schema_path=schema_path,
            math_path=math_path,
            csp_path=csp_path,
            out_path=out_path,
            baseline_ckpt=baseline_ckpt if baseline_ckpt else None,
            forge_ckpt=forge_ckpt if forge_ckpt else None,
            ablation_ckpt=ablation_ckpt if ablation_ckpt else None,
            device=runtime.device,
        )
    except Exception:
        logger.exception("eval_compare failed")
        raise SystemExit(1)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True)
    out_file.write_text(payload)
    print(payload)
    if not report:
        logger.error("eval_compare produced an empty report")
        raise SystemExit(1)
    logger.info("eval_compare complete out=%s", out_path)


if __name__ == "__main__":
    app()
