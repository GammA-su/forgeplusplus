from __future__ import annotations

import typer

from fc.train.data import generate_dataset, save_dataset
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(n: int = 50, seed: int = 123, out: str = "out/data/schema.jsonl") -> None:
    configure_logging()
    logger = get_logger(__name__)
    configure_runtime(logger)
    logger.info("data_gen domain=schema n=%d seed=%d out=%s", n, seed, out)
    examples = generate_dataset("schema", n=n, seed=seed)
    save_dataset(out, examples)
    logger.info("data_gen complete domain=schema rows=%d", len(examples))


if __name__ == "__main__":
    app()
