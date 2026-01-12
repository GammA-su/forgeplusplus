from __future__ import annotations

import typer

from fc.train.trainer import train_from_paths
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = "configs/train_mvp.yaml",
    schema_path: str = "out/data/schema.jsonl",
    math_path: str = "out/data/math.jsonl",
    csp_path: str = "out/data/csp.jsonl",
    out_dir: str = "out",
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    runtime = configure_runtime(logger)
    logger.info("train_mvp config=%s schema=%s math=%s csp=%s out=%s", config, schema_path, math_path, csp_path, out_dir)
    train_from_paths(config_path=config, out_dir=out_dir, schema_path=schema_path, math_path=math_path, csp_path=csp_path, device=runtime.device)
    logger.info("train_mvp complete out=%s", out_dir)


if __name__ == "__main__":
    app()
