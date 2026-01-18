from fc.dsl.program import Instruction, Program
from fc.train.data import Example, audit_proof_tokens, build_program_vocab_from_examples, program_to_proof


def test_vocab_audit_passes_and_includes_ops() -> None:
    math_prog = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="APPLY_ARITH", args={"a": "a", "b": "b", "op": "+"}, dest="result"),
            Instruction(opcode="EMIT_NUM", args={"value": "result"}),
        ]
    )
    csp_prog = Program(
        [
            Instruction(opcode="APPLY_TOPO", args={}, dest="order"),
            Instruction(opcode="APPLY_CUMSUM", args={}, dest="schedule"),
            Instruction(opcode="EMIT_SCHEDULE", args={}),
        ]
    )
    examples = [
        Example(
            id="math_0",
            domain="math",
            x="[MATH] Compute: 2 + 3.",
            y=5,
            proof=program_to_proof(math_prog),
        ),
        Example(
            id="csp_0",
            domain="csp",
            x="[CSP] Tasks: A=2,B=1. Constraints: A<B.",
            y={"schedule": {"A": 0, "B": 2}, "status": "ok"},
            proof=program_to_proof(csp_prog),
        ),
    ]
    audit_proof_tokens(examples)
    vocab = build_program_vocab_from_examples(examples)
    for tok in ("APPLY_ARITH", "EMIT_NUM", "APPLY_TOPO", "APPLY_CUMSUM", "EMIT_SCHEDULE"):
        assert tok in vocab.token_to_id
