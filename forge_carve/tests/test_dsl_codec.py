from fc.dsl.codec import alignment_distance, decode_program, encode_program
from fc.dsl.program import Instruction, Program
from fc.dsl.tokens import build_default_vocab


def test_encode_decode_roundtrip() -> None:
    vocab = build_default_vocab()
    prog = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="ADD", args={"a": "a", "b": "b"}, dest="result"),
            Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
        ]
    )
    ids = encode_program(prog, vocab)
    decoded = decode_program(ids, vocab)
    assert decoded.to_dict() == prog.to_dict()


def test_alignment_distance() -> None:
    a = Program([Instruction(opcode="EXTRACT_INT", args={}, dest="a")])
    b = Program([Instruction(opcode="EXTRACT_INT", args={}, dest="a"), Instruction(opcode="ADD", args={})])
    dist = alignment_distance(a, b)
    assert dist == 1
