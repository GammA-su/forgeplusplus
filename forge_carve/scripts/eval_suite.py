from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from fc.eval.suite import load_eval_state, merge_eval_suite_config, run_eval, run_eval_suite
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = "",
    ckpt: str = "out/ckpt.pt",
    schema_path: str | None = typer.Option(None, "--schema-path"),
    math_path: str | None = typer.Option(None, "--math-path"),
    csp_path: str | None = typer.Option(None, "--csp-path"),
    out_path: str | None = typer.Option(
        None,
        "--out",
        "--out-path",
        help="Aggregate report output path (overrides config).",
    ),
    repair_op: bool = typer.Option(False, "--repair-op"),
    constrained_op: bool = typer.Option(True, "--constrained-op/--no-constrained-op"),
    math_constrained_op: bool | None = typer.Option(
        None,
        "--math-constrained-op/--math-no-constrained-op",
        help="Override constrained opcode decoding for math domain only.",
    ),
    cegis_mode: str = typer.Option(
        "brute",
        "--cegis-mode",
        help="CEGIS search mode for math repair: brute or dp.",
    ),
    max_prog_len: int = typer.Option(256, "--max-prog-len"),
    min_proof_tokens: int = typer.Option(0, "--min-proof-tokens"),
    batch_size: int = typer.Option(-1, "--batch-size"),
    max_orbits: int = typer.Option(-1, "--max-orbits"),
    max_flips: int = typer.Option(-1, "--max-flips"),
    max_mutants: int = typer.Option(-1, "--max-mutants"),
    max_examples: int = typer.Option(-1, "--max-examples"),
    cegis_ms: int = typer.Option(-1, "--cegis-ms"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    cfg: dict[str, Any] = {}
    if config:
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    batch_size_opt = None if batch_size < 0 else batch_size
    max_orbits_opt = None if max_orbits < 0 else max_orbits
    max_flips_opt = None if max_flips < 0 else max_flips
    max_mutants_opt = None if max_mutants < 0 else max_mutants
    max_examples_opt = None if max_examples < 0 else max_examples
    cegis_ms_opt = None if cegis_ms < 0 else cegis_ms
    decode_overrides = None
    if math_constrained_op is not None:
        decode_overrides = {"math": {"constrained_op": bool(math_constrained_op)}}
    merged = merge_eval_suite_config(
        cfg,
        {
            "ckpt": ckpt,
            "schema_path": schema_path,
            "math_path": math_path,
            "csp_path": csp_path,
            "out_path": out_path,
            "batch_size": batch_size_opt,
            "max_orbits": max_orbits_opt,
            "max_flips": max_flips_opt,
            "max_mutants": max_mutants_opt,
            "max_examples": max_examples_opt,
            "cegis_ms": cegis_ms_opt,
            "cegis_mode": cegis_mode,
        },
    )
    ckpt = str(merged.get("ckpt", ckpt))
    schema_path = str(merged.get("schema_path", "out/data/schema.jsonl"))
    math_path = str(merged.get("math_path", "out/data/math.jsonl"))
    csp_path = str(merged.get("csp_path", "out/data/csp.jsonl"))
    resolved_out_path = str(merged.get("out_path", "out/report.json"))
    logger.info(
        "eval_suite ckpt=%s schema=%s math=%s csp=%s out=%s",
        ckpt,
        schema_path,
        math_path,
        csp_path,
        resolved_out_path,
    )
    if config:
        report = run_eval_suite(
            config,
            device=runtime.device,
            ckpt=ckpt,
            schema_path=schema_path,
            math_path=math_path,
            csp_path=csp_path,
            out_path=resolved_out_path,
            repair_op=repair_op,
            constrained_op=constrained_op,
            decode_overrides=decode_overrides,
            min_proof_tokens=min_proof_tokens if min_proof_tokens > 0 else None,
            max_prog_len=max_prog_len,
            batch_size=batch_size_opt,
            max_orbits=max_orbits_opt,
            max_flips=max_flips_opt,
            max_mutants=max_mutants_opt,
            max_examples=max_examples_opt,
            cegis_ms=cegis_ms_opt,
            cegis_mode=str(merged.get("cegis_mode", cegis_mode)),
        )
    else:
        max_tokens = max_prog_len
        state = load_eval_state(ckpt, device=runtime.device)
        batch_size_resolved = 64 if batch_size_opt is None else batch_size_opt
        report = {
            "schema": run_eval(
                schema_path,
                ckpt,
                out_path="out/schema_report.json",
                repair_op=repair_op,
                constrained_op=constrained_op,
                decode_overrides=decode_overrides,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
                batch_size=batch_size_resolved,
                max_orbits=max_orbits_opt,
                max_flips=max_flips_opt,
                max_mutants=max_mutants_opt,
                max_examples=max_examples_opt,
                cegis_ms=cegis_ms_opt,
                cegis_mode=str(merged.get("cegis_mode", cegis_mode)),
                state=state,
            ),
            "math": run_eval(
                math_path,
                ckpt,
                out_path="out/math_report.json",
                repair_op=repair_op,
                constrained_op=constrained_op,
                decode_overrides=decode_overrides,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
                batch_size=batch_size_resolved,
                max_orbits=max_orbits_opt,
                max_flips=max_flips_opt,
                max_mutants=max_mutants_opt,
                max_examples=max_examples_opt,
                cegis_ms=cegis_ms_opt,
                cegis_mode=str(merged.get("cegis_mode", cegis_mode)),
                state=state,
            ),
            "csp": run_eval(
                csp_path,
                ckpt,
                out_path="out/csp_report.json",
                repair_op=repair_op,
                constrained_op=constrained_op,
                decode_overrides=decode_overrides,
                min_proof_tokens=min_proof_tokens,
                max_prog_len=max_tokens,
                batch_size=batch_size_resolved,
                max_orbits=max_orbits_opt,
                max_flips=max_flips_opt,
                max_mutants=max_mutants_opt,
                max_examples=max_examples_opt,
                cegis_ms=cegis_ms_opt,
                cegis_mode=str(merged.get("cegis_mode", cegis_mode)),
                state=state,
            ),
        }
        Path(resolved_out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(resolved_out_path).write_text(json.dumps(report, indent=2))
    print(f"writing aggregate report to: {resolved_out_path}")
    logger.info("eval_suite complete out=%s", resolved_out_path)


if __name__ == "__main__":
    app()
