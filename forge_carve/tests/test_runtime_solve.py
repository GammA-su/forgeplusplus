from __future__ import annotations

from fractions import Fraction

from fc.dsl.codec import program_to_tokens
from fc.dsl.program import Instruction, Program
from fc.util.runtime_solve import runtime_solve


def test_runtime_solve_returns_value_without_emit() -> None:
    x = "[MATH] Compute: ((7063 - 4345) / (1168 plus (-2141)))."
    prog = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="EXTRACT_INT", args={"index": 2}, dest="c"),
            Instruction(opcode="EXTRACT_INT", args={"index": 3}, dest="d"),
            Instruction(opcode="APPLY_ARITH", args={"a": "a", "b": "b", "op": "-"}, dest="t0"),
            Instruction(opcode="APPLY_ARITH", args={"a": "t0", "b": "c", "op": "/"}, dest="result"),
        ]
    )
    tokens = program_to_tokens(prog)
    out = runtime_solve(x, [], tokens)
    assert out is not None
    assert Fraction(out) == Fraction(7063 - 4345, 1168)
