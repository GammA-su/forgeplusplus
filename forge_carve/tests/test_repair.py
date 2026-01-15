from typing import Any

from fc.dsl.program import Instruction, Program
from fc.dsl.repair import repair_program
from fc.interp.core import Interpreter
from fc.verify.schema import SchemaVerifier
from fc.verify.mesh import VerifierMesh


def test_repair_schema_missing_key() -> None:
    program = Program(
        [
            Instruction(opcode="EXTRACT_STR", args={"key": "name"}, dest="name"),
            Instruction(opcode="EXTRACT_INT", args={"key": "age"}, dest="age"),
            Instruction(opcode="EXTRACT_STR", args={"key": "city"}, dest="city"),
            Instruction(opcode="EMIT", args={"schema": "person", "fields": {"name": "name", "age": "age"}}),
        ]
    )
    text = "Record: name=Alice; age=30; city=Paris."
    interp = Interpreter()
    out, _, errors = interp.execute(program, text)
    assert not errors
    mesh = VerifierMesh()
    report = mesh.run(text, program, out, domain="schema", repair=True, max_repairs=2)
    assert report.meta.get("repair", {}).get("success") is True
    assert report.meta.get("repair", {}).get("steps", 0) <= 2


def test_repair_schema_passes_verifier() -> None:
    program = Program(
        [
            Instruction(opcode="EXTRACT_STR", args={"key": "name"}, dest="name"),
            Instruction(opcode="EXTRACT_INT", args={"key": "age"}, dest="age"),
            Instruction(opcode="EXTRACT_STR", args={"key": "city"}, dest="city"),
            Instruction(opcode="EMIT", args={"schema": "person", "fields": {"name": "name", "age": "age"}}),
        ]
    )
    text = "Record: name=Alice; age=30; city=Paris."
    interp = Interpreter()
    verifier = SchemaVerifier()

    def evaluator(prog: Program) -> tuple[float, dict[str, Any]]:
        out, _, _ = interp.execute(prog, text)
        res = verifier.verify(text, prog, out)
        score = float(sum(res.violations.values())) if res.violations else 0.0
        meta = dict(res.meta)
        meta["violations"] = res.violations
        return score, meta

    best_prog, best_score, steps, _ = repair_program(program, evaluator, max_steps=2)
    out, _, _ = interp.execute(best_prog, text)
    res = verifier.verify(text, best_prog, out)
    assert res.valid
    assert best_score <= 0.0
    assert steps <= 2
