from __future__ import annotations

import typer

from fc.dsl.program import Program
from fc.dsl.tokens import build_default_vocab
from fc.train.data import generate_dataset, program_to_proof, proof_to_program, save_dataset
from fc.util.tags import DOMAIN_TAGS, apply_domain_tag
from fc.util.logging import configure_logging, get_logger
from fc.util.runtime import configure_runtime

app = typer.Typer(add_completion=False)
_PROG_VOCAB = build_default_vocab()


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
        proof = ex.proof
    else:
        proof = program_to_proof(Program.from_dict(ex.proof))
    program = proof_to_program(proof, _PROG_VOCAB)
    ops = program.skeleton()
    if len(ops) < 3 or ops[-3:] != ["APPLY_TOPO", "APPLY_CUMSUM", "EMIT_SCHEDULE"]:
        raise ValueError(f"csp proof must end with APPLY_TOPO/APPLY_CUMSUM/EMIT_SCHEDULE, got={ops}")
    return ex.model_copy(update={"proof": proof})


@app.command()
def main(
    n: int = typer.Option(50, "--n"),
    orbits: int = typer.Option(-1, "--orbits"),
    flips: int = typer.Option(-1, "--flips"),
    seed: int = typer.Option(123, "--seed"),
    out: str = typer.Option("out/data/csp.jsonl", "--out"),
) -> None:
    configure_logging()
    logger = get_logger(__name__)
    configure_runtime(logger)
    orbits_count = None if orbits < 0 else orbits
    flips_count = None if flips < 0 else flips
    logger.info(
        "data_gen domain=csp n=%d orbits=%s flips=%s seed=%d out=%s",
        n,
        str(orbits_count),
        str(flips_count),
        seed,
        out,
    )
    examples = generate_dataset("csp", n=n, seed=seed, orbits=orbits_count, flips=flips_count)
    examples = [_tag_example("csp", ex) for ex in examples]
    examples = [_ensure_proof_tokens(ex) for ex in examples]
    save_dataset(out, examples)
    logger.info("data_gen complete domain=csp rows=%d", len(examples))


if __name__ == "__main__":
    app()
