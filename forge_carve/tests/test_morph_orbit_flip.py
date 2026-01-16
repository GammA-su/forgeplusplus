from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.morph.flips import generate_flips
from fc.morph.orbit import generate_orbits
from fc.train.data import generate_dataset, proof_to_program
from fc.dsl.tokens import build_default_vocab


def _expected_output(domain: str, y: object) -> object:
    if domain == "math" and not isinstance(y, dict):
        return {"result": y}
    return y


def test_orbits_preserve_truth() -> None:
    for domain in ["schema", "math", "csp"]:
        ex = generate_dataset(domain, n=1, seed=7)[0]
        expected = _expected_output(domain, ex.y)
        program = proof_to_program(ex.proof, build_default_vocab())
        orbits = generate_orbits(domain, ex.x)
        assert orbits
        for otext in orbits:
            out, _, errors = Interpreter().execute(program, otext)
            assert not errors
            assert outputs_equivalent(out, expected)


def test_flips_change_truth() -> None:
    for domain in ["schema", "math", "csp"]:
        ex = generate_dataset(domain, n=1, seed=9)[0]
        expected = _expected_output(domain, ex.y)
        program = proof_to_program(ex.proof, build_default_vocab())
        flips = generate_flips(domain, ex.x)
        assert flips
        for ftext in flips:
            out, _, errors = Interpreter().execute(program, ftext)
            assert not errors
            assert not outputs_equivalent(out, expected)
