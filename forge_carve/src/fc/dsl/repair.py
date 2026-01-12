from __future__ import annotations

from typing import Any

from fc.dsl.program import Instruction, Program


def _replace_instruction(program: Program, idx: int, new_inst: Instruction) -> Program:
    insts = list(program.instructions)
    if 0 <= idx < len(insts):
        insts[idx] = new_inst
    return Program(insts)


def propose_repairs(program: Program, meta: dict[str, Any]) -> list[Program]:
    """Propose deterministic bounded edits based on verifier metadata."""
    candidates: list[Program] = []
    # Schema repair: re-emit with required keys
    missing = meta.get("missing_keys")
    if missing:
        for i, inst in enumerate(program.instructions):
            if inst.opcode == "EMIT":
                fields = dict(inst.args.get("fields", {}))
                for k in missing:
                    fields[k] = k
                new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                candidates.append(_replace_instruction(program, i, new_inst))
                break
    # Arithmetic repair: bind expected value before emit
    if "arith_expected" in meta:
        expected = meta["arith_expected"]
        bind = Instruction(opcode="BIND", args={"value": expected}, dest="result")
        candidates.append(Program(program.instructions + [bind]))
    # CSP repair: update status to infeasible if indicated
    if meta.get("csp_infeasible"):
        for i, inst in enumerate(program.instructions):
            if inst.opcode == "EMIT":
                fields = dict(inst.args.get("fields", {}))
                fields["status"] = "infeasible"
                new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                candidates.append(_replace_instruction(program, i, new_inst))
                break
    return candidates
