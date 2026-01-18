from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter


def test_interp_emit_schedule_dict() -> None:
    program = Program(
        [
            Instruction(opcode="APPLY_TOPO", args={}, dest="order"),
            Instruction(opcode="APPLY_CUMSUM", args={}, dest="schedule"),
            Instruction(opcode="EMIT_SCHEDULE", args={}),
        ]
    )
    text = "Tasks: A=2,B=1. Constraints: A<B."
    out, _, errors = Interpreter().execute(program, text)
    assert not errors
    assert isinstance(out, dict)
    assert out.get("status") == "ok"
    assert out.get("schedule") == {"A": 0, "B": 2}
