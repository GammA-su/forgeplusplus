from fc.adv.mutator import ProgramMutator
from fc.dsl.program import Instruction, Program


def test_mutator_skips_non_numeric_index() -> None:
    prog = Program([Instruction(opcode="EXTRACT_INT", args={"index": "a"}, dest="x")])
    mutants = ProgramMutator().mutate(prog)
    assert mutants == []
