from fc.adv.mutator import ProgramMutator
from fc.dsl.program import Instruction, Program


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
