from __future__ import annotations

import typer

from fc.dsl.program import Program
from fc.train.data import generate_dataset, program_to_proof, save_dataset
from fc.util.tags import DOMAIN_TAGS, apply_domain_tag
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)


def _tag_example(domain: str, ex):
    tag = DOMAIN_TAGS[domain]
    tagged_orbit = [o.model_copy(update={"x": apply_domain_tag(domain, o.x)}) for o in ex.orbit]
    tagged_flips = [f.model_copy(update={"x": apply_domain_tag(domain, f.x)}) for f in ex.flips]
    return ex.model_copy(
        update={
            "x": apply_domain_tag(domain, ex.x),
            "orbit": tagged_orbit,
            "flips": tagged_flips,
            "domain_tag": tag,
        }
    )


def _ensure_proof_tokens(ex):
    if isinstance(ex.proof, dict) and ex.proof.get("dsl") == "PTv1":
        return ex
    return ex.model_copy(update={"proof": program_to_proof(Program.from_dict(ex.proof))})


@app.command()
def main(
    n: int = typer.Option(50, "--n"),
    orbits: int = typer.Option(-1, "--orbits"),
    flips: int = typer.Option(-1, "--flips"),
    seed: int = typer.Option(123, "--seed"),
    out: str = typer.Option("out/data/math.jsonl", "--out"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    configure_runtime(logger)
    orbits_count = None if orbits < 0 else orbits
    flips_count = None if flips < 0 else flips
    logger.info(
        "data_gen domain=math n=%d orbits=%s flips=%s seed=%d out=%s",
        n,
        str(orbits_count),
        str(flips_count),
        seed,
        out,
    )
    examples = generate_dataset("math", n=n, seed=seed, orbits=orbits_count, flips=flips_count)
    examples = [_tag_example("math", ex) for ex in examples]
    examples = [_ensure_proof_tokens(ex) for ex in examples]
    save_dataset(out, examples)
    logger.info("data_gen complete domain=math rows=%d", len(examples))


if __name__ == "__main__":
    app()
