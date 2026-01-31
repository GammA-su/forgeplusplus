from __future__ import annotations

from fractions import Fraction

from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter


def test_arithmetic_division_true() -> None:
    prog = Program(
        instructions=[
            Instruction(opcode="BIND", args={"value": 5}, dest="a"),
            Instruction(opcode="BIND", args={"value": 5193}, dest="b"),
            Instruction(opcode="APPLY_ARITH", args={"a": "a", "b": "b", "op": "/"}, dest="c"),
            Instruction(opcode="BIND", args={"value": 9027}, dest="d"),
            Instruction(opcode="APPLY_ARITH", args={"a": "c", "b": "d", "op": "/"}, dest="result"),
            Instruction(opcode="EMIT_NUM", args={"value": "result"}),
        ]
    )
    out, _, _ = Interpreter().execute(prog, "[MATH]")
    assert out not in (0, 0.0)
    assert isinstance(out, (float, Fraction, int))
    if isinstance(out, Fraction):
        got = float(out)
    else:
        got = float(out)
    assert abs(got - 1.0666e-7) < 5e-10
