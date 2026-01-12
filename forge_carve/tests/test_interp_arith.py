from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter


def test_interpreter_arithmetic() -> None:
    program = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="MUL", args={"a": "a", "b": "b"}, dest="result"),
            Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
        ]
    )
    text = "Compute: 3 * 4."
    out, _, errors = Interpreter().execute(program, text)
    assert not errors
    assert out["result"] == 12


def test_interpreter_multiline_schema_extract() -> None:
    program = Program(
        [
            Instruction(opcode="EXTRACT_STR", args={"key": "name"}, dest="name"),
            Instruction(opcode="EXTRACT_INT", args={"key": "age"}, dest="age"),
            Instruction(opcode="EXTRACT_STR", args={"key": "city"}, dest="city"),
            Instruction(
                opcode="EMIT",
                args={"schema": "person", "fields": {"name": "name", "age": "age", "city": "city"}},
            ),
        ]
    )
    text = "- name: Alice\n- age: 30\n- city: Paris"
    out, _, errors = Interpreter().execute(program, text)
    assert not errors
    assert out["name"] == "Alice"
    assert out["age"] == 30
    assert out["city"] == "Paris"
