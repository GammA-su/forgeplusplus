from __future__ import annotations

from typing import Iterable

from fc.dsl.program import Instruction, Program


class ProgramMutator:
    def mutate(self, program: Program) -> list[Program]:
        mutants: list[Program] = []
        insts = program.instructions
        # Swap operands in arithmetic ops
        for i, inst in enumerate(insts):
            if inst.opcode in {"ADD", "SUB", "MUL", "DIV"}:
                args = dict(inst.args)
                a = args.get("a")
                b = args.get("b")
                if a is not None and b is not None:
                    args["a"], args["b"] = b, a
                    new_inst = Instruction(opcode=inst.opcode, args=args, dest=inst.dest)
                    new_prog = Program(insts[:i] + [new_inst] + insts[i + 1 :])
                    mutants.append(new_prog)
        # Increment extract index
        for i, inst in enumerate(insts):
            if inst.opcode == "EXTRACT_INT" and "index" in inst.args:
                args = dict(inst.args)
                args["index"] = int(args["index"]) + 1
                new_inst = Instruction(opcode=inst.opcode, args=args, dest=inst.dest)
                mutants.append(Program(insts[:i] + [new_inst] + insts[i + 1 :]))
        # Force division by zero edge case
        for i, inst in enumerate(insts):
            if inst.opcode == "DIV":
                args = dict(inst.args)
                args["b"] = 0
                new_inst = Instruction(opcode=inst.opcode, args=args, dest=inst.dest)
                mutants.append(Program(insts[:i] + [new_inst] + insts[i + 1 :]))
        # Drop a field from EMIT
        for i, inst in enumerate(insts):
            if inst.opcode == "EMIT" and isinstance(inst.args.get("fields"), dict):
                fields = dict(inst.args.get("fields", {}))
                if fields:
                    key = sorted(fields.keys())[0]
                    fields.pop(key, None)
                    new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                    mutants.append(Program(insts[:i] + [new_inst] + insts[i + 1 :]))
        return mutants


def mutate_batch(programs: Iterable[Program]) -> list[Program]:
    mutator = ProgramMutator()
    out: list[Program] = []
    for prog in programs:
        out.extend(mutator.mutate(prog))
    return out
