from __future__ import annotations

from typing import Iterable

from fc.dsl.program import Instruction, Program


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
        return None
    return None


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
        # Swap extracted numbers (indices) if present
        extract_idxs = [i for i, inst in enumerate(insts) if inst.opcode == "EXTRACT_INT" and "index" in inst.args]
        if len(extract_idxs) >= 2:
            i1, i2 = extract_idxs[0], extract_idxs[1]
            inst1 = insts[i1]
            inst2 = insts[i2]
            args1 = dict(inst1.args)
            args2 = dict(inst2.args)
            args1["index"], args2["index"] = args2["index"], args1["index"]
            new_inst1 = Instruction(opcode=inst1.opcode, args=args1, dest=inst1.dest)
            new_inst2 = Instruction(opcode=inst2.opcode, args=args2, dest=inst2.dest)
            new_insts = list(insts)
            new_insts[i1] = new_inst1
            new_insts[i2] = new_inst2
            mutants.append(Program(new_insts))
        # Increment extract index
        for i, inst in enumerate(insts):
            if inst.opcode == "EXTRACT_INT" and "index" in inst.args:
                args = dict(inst.args)
                idx_val = _coerce_int(args.get("index"))
                if idx_val is None:
                    continue
                args["index"] = idx_val + 1
                new_inst = Instruction(opcode=inst.opcode, args=args, dest=inst.dest)
                mutants.append(Program(insts[:i] + [new_inst] + insts[i + 1 :]))
        # Drop a constraint-related step (e.g., solver or arithmetic op)
        for i, inst in enumerate(insts):
            if inst.opcode in {"SOLVE_CSP", "ADD", "SUB", "MUL", "DIV"}:
                new_insts = list(insts)
                new_insts.pop(i)
                if new_insts:
                    mutants.append(Program(new_insts))
                break
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
        # Empty/NaN-like edge cases in EMIT
        for i, inst in enumerate(insts):
            if inst.opcode == "EMIT" and isinstance(inst.args.get("fields"), dict):
                fields = dict(inst.args.get("fields", {}))
                if fields:
                    key = sorted(fields.keys())[0]
                    fields[key] = float("nan")
                    new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                    mutants.append(Program(insts[:i] + [new_inst] + insts[i + 1 :]))
                break
        return mutants


def mutate_batch(programs: Iterable[Program]) -> list[Program]:
    mutator = ProgramMutator()
    out: list[Program] = []
    for prog in programs:
        out.extend(mutator.mutate(prog))
    return out
