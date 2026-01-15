from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter
from fc.verify.mesh import VerifierMesh, set_orbit_parallelism


def test_orbit_parallelism_matches() -> None:
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
    text = "Record: name=Alice; age=30; city=Paris."
    orbits = [
        "Record: age=30; name=Alice; city=Paris.",
        "Record: city=Paris; name=Alice; age=30.",
    ]
    interp = Interpreter()
    out, _, errors = interp.execute(program, text)
    assert not errors
    mesh = VerifierMesh()
    try:
        set_orbit_parallelism(False)
        seq_report = mesh.run(text, program, out, domain="schema", orbits=orbits)
        set_orbit_parallelism(True)
        par_report = mesh.run(text, program, out, domain="schema", orbits=orbits)
    finally:
        set_orbit_parallelism(False)
    assert seq_report.model_dump() == par_report.model_dump()
