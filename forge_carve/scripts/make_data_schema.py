from __future__ import annotations

import typer

from fc.train.data import generate_dataset, save_dataset
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n: int = typer.Option(50, "--n"),
    orbits: int = typer.Option(-1, "--orbits"),
    flips: int = typer.Option(-1, "--flips"),
    seed: int = typer.Option(123, "--seed"),
    out: str = typer.Option("out/data/schema.jsonl", "--out"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    configure_runtime(logger)
    orbits_count = None if orbits < 0 else orbits
    flips_count = None if flips < 0 else flips
    logger.info(
        "data_gen domain=schema n=%d orbits=%s flips=%s seed=%d out=%s",
        n,
        str(orbits_count),
        str(flips_count),
        seed,
        out,
    )
    examples = generate_dataset("schema", n=n, seed=seed, orbits=orbits_count, flips=flips_count)
    save_dataset(out, examples)
    logger.info("data_gen complete domain=schema rows=%d", len(examples))


if __name__ == "__main__":
    app()
