from fc.dsl.tokens import build_default_vocab
from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.train.data import generate_dataset, proof_to_program
from fc.verify.csp import CSPVerifier


def test_csp_proof_exec() -> None:
    ex = generate_dataset("csp", n=1, seed=7)[0]
    program = proof_to_program(ex.proof, build_default_vocab())
    out, _, errors = Interpreter().execute(program, ex.x)
    assert not errors
    constraints = [c.model_dump() for c in ex.constraints]
    res = CSPVerifier().verify(ex.x, program, out, constraints=constraints)
    assert res.valid
    assert outputs_equivalent(out, ex.y)
