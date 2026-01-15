from fc.adv.attacks import attack_success_rate
from fc.adv.mutator import ProgramMutator
from fc.dsl.program import Instruction, Program
from fc.interp.core import Interpreter
from fc.verify.mesh import VerifierMesh


def test_mutator_generates_variants() -> None:
    prog = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="ADD", args={"a": "a", "b": "b"}, dest="result"),
            Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
        ]
    )
    mutants = ProgramMutator().mutate(prog)
    assert mutants
    assert all(m.instructions for m in mutants)


def test_mesh_flags_mutants() -> None:
    prog = Program(
        [
            Instruction(opcode="EXTRACT_INT", args={"index": 0}, dest="a"),
            Instruction(opcode="EXTRACT_INT", args={"index": 1}, dest="b"),
            Instruction(opcode="SUB", args={"a": "a", "b": "b"}, dest="result"),
            Instruction(opcode="EMIT", args={"schema": "math", "fields": {"result": "result"}}),
        ]
    )
    text = "Compute: 8 - 3."
    mesh = VerifierMesh()
    interp = Interpreter()
    out, _, errors = interp.execute(prog, text)
    assert not errors
    report = mesh.run(text, prog, out, domain="math", mutator=ProgramMutator())
    assert report.c[mesh.constraint_names.index("adv_caught")] in (0.0, 1.0)
    rate = attack_success_rate([text], [prog], domain="math", mesh=mesh)
    assert rate <= 0.5
