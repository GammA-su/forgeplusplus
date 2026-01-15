from fc.dsl.codec import decode_program
from fc.dsl.program import Program
from fc.dsl.tokens import build_default_vocab
from fc.interp.core import Interpreter
from fc.train.data import generate_dataset, program_to_proof
from fc.verify.arithmetic import ArithmeticVerifier


def _decode_proof_tokens(proof: dict) -> Program:
    vocab = build_default_vocab()
    token_ids = []
    for tok in proof.get("tokens", []):
        if isinstance(tok, int):
            token_ids.append(tok)
        else:
            token_ids.append(vocab.encode(str(tok)))
    return decode_program(token_ids, vocab)


def test_math_proof_exec() -> None:
    ex = generate_dataset("math", n=1, seed=5)[0]
    proof = program_to_proof(Program.from_dict(ex.proof))
    program = _decode_proof_tokens(proof)
    out, _, errors = Interpreter().execute(program, ex.x)
    assert not errors
    constraints = [c.model_dump() for c in ex.constraints]
    res = ArithmeticVerifier().verify(ex.x, program, out, constraints=constraints)
    assert res.valid
