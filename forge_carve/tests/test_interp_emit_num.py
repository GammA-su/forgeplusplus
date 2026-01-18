from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter


def test_interp_emit_num_scalar() -> None:
    program = Program(
        [
            Instruction(opcode="BIND", args={"value": 7}, dest="result"),
            Instruction(opcode="EMIT_NUM", args={}),
        ]
    )
    out, _, errors = Interpreter().execute(program, "ignored")
    assert not errors
    assert out == 7
