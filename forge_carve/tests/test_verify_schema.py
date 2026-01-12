from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter
from fc.verify.arithmetic import ArithmeticVerifier
from fc.verify.csp import CSPVerifier
from fc.verify.schema import SchemaVerifier


def test_schema_verifier() -> None:
    verifier = SchemaVerifier()
    ok = {"name": "Alice", "age": 30, "city": "Paris"}
    res = verifier.verify("", None, ok)
    assert res.valid
    bad = {"name": "Alice", "age": "old"}
    res2 = verifier.verify("", None, bad)
    assert not res2.valid
    extra = {"name": "Alice", "age": 30, "city": "Paris", "extra": "x"}
    res3 = verifier.verify("", None, extra)
    assert not res3.valid
    assert "extra" in res3.meta.get("extra_keys", [])
    missing = {"name": "Alice", "age": 30}
    res4 = verifier.verify("", None, missing)
    assert not res4.valid
    assert "city" in res4.meta.get("missing_keys", [])


def test_schema_verifier_multiline_extract() -> None:
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
    res = SchemaVerifier().verify(text, program, out)
    assert res.valid


def test_arithmetic_verifier() -> None:
    program = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="ADD", args={"a": "a", "b": "b"}, dest="result"),
            Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
        ]
    )
    text = "Compute: 5 7."
    out, _, errors = Interpreter().execute(program, text)
    assert not errors
    constraints = [{"type": "arithmetic", "args": {"op": "+"}}]
    res = ArithmeticVerifier().verify(text, program, out, constraints=constraints)
    assert res.valid


def test_csp_verifier() -> None:
    program = Program(
        [
            Instruction(opcode="SOLVE_CSP", args={}, dest="schedule"),
            Instruction(opcode="EMIT", args={"schema": "schedule", "fields": {"schedule": "schedule", "status": "status"}}),
        ]
    )
    text = "Tasks: A=1,B=2,C=1. Constraints: A<B,B<C."
    out, _, errors = Interpreter().execute(program, text)
    assert not errors
    constraints = [{"type": "csp", "args": {"tasks": {"A": 1, "B": 2, "C": 1}, "constraints": [("A", "B"), ("B", "C")]}}]
    res = CSPVerifier().verify(text, program, out, constraints=constraints)
    assert res.valid
