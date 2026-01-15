from fc.dsl.codec import decode_program
from fc.dsl.program import Program
from fc.dsl.tokens import build_default_vocab
from fc.interp.core import Interpreter
from fc.train.data import generate_dataset, program_to_proof
from fc.verify.csp import CSPVerifier


def _decode_proof_tokens(proof: dict) -> Program:
    vocab = build_default_vocab()
    token_ids = []
    for tok in proof.get("tokens", []):
        if isinstance(tok, int):
            token_ids.append(tok)
        else:
            token_ids.append(vocab.encode(str(tok)))
    return decode_program(token_ids, vocab)


def test_csp_proof_exec() -> None:
    ex = generate_dataset("csp", n=1, seed=7)[0]
    proof = program_to_proof(Program.from_dict(ex.proof))
    program = _decode_proof_tokens(proof)
    out, _, errors = Interpreter().execute(program, ex.x)
    assert not errors
    constraints = [c.model_dump() for c in ex.constraints]
    res = CSPVerifier().verify(ex.x, program, out, constraints=constraints)
    assert res.valid
