from __future__ import annotations

from typing import Any, Callable

from fc.dsl.program import Instruction, Program

ARITH_OPS = ("ADD", "SUB", "MUL", "DIV")
EXTRACT_OPS = ("EXTRACT_STR", "EXTRACT_INT")
STOP_NEWLINE = "\\n"


def _replace_instruction(program: Program, idx: int, new_inst: Instruction) -> Program:
    insts = list(program.instructions)
    if 0 <= idx < len(insts):
        insts[idx] = new_inst
    return Program(insts)


def _update_emit_fields(fields: dict[str, Any], missing: list[str], extra: list[str]) -> dict[str, Any]:
    out = dict(fields)
    for key in extra:
        out.pop(key, None)
    for key in missing:
        out[key] = key
    return out


def _extract_opcode_for_key(key: str) -> str:
    if key in {"age"}:
        return "EXTRACT_INT"
    return "EXTRACT_STR"


def _reextract_with_stop(program: Program, key: str) -> Program | None:
    opcode = _extract_opcode_for_key(key)
    idx = None
    for i, inst in enumerate(program.instructions):
        if inst.dest == key and inst.opcode in EXTRACT_OPS:
            idx = i
            break
    if idx is None:
        for i, inst in enumerate(program.instructions):
            if inst.opcode in EXTRACT_OPS:
                idx = i
                break
    if idx is None:
        return None
    new_inst = Instruction(opcode=opcode, args={"key": key, "stop": STOP_NEWLINE}, dest=key)
    return _replace_instruction(program, idx, new_inst)


def _fix_bound_value(program: Program, expected: Any) -> list[Program]:
    candidates: list[Program] = []
    for i, inst in enumerate(program.instructions):
        if inst.opcode == "BIND" and inst.dest == "result":
            new_inst = Instruction(opcode="BIND", args={"value": expected}, dest="result")
            candidates.append(_replace_instruction(program, i, new_inst))
            return candidates
    for i, inst in enumerate(program.instructions):
        if inst.opcode == "BIND" and inst.dest:
            new_inst = Instruction(opcode="BIND", args={"value": expected}, dest=inst.dest)
            candidates.append(_replace_instruction(program, i, new_inst))
            return candidates
    for i, inst in enumerate(program.instructions):
        if inst.opcode == "EMIT":
            fields = dict(inst.args.get("fields", {}))
            if "result" in fields:
                fields["result"] = expected
                new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                candidates.append(_replace_instruction(program, i, new_inst))
                return candidates
    bind = Instruction(opcode="BIND", args={"value": expected}, dest="result")
    candidates.append(Program(program.instructions + [bind]))
    return candidates


def propose_repairs(program: Program, meta: dict[str, Any]) -> list[Program]:
    """Propose deterministic bounded edits based on verifier metadata."""
    candidates: list[Program] = []
    missing = sorted(meta.get("missing_keys") or [])
    extra = sorted(meta.get("extra_keys") or [])
    violations = set((meta.get("violations") or {}).keys())

    # Re-EMIT with correct keys.
    if missing or extra:
        for i, inst in enumerate(program.instructions):
            if inst.opcode == "EMIT":
                fields = dict(inst.args.get("fields", {}))
                new_fields = _update_emit_fields(fields, missing, extra)
                new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": new_fields}, dest=inst.dest)
                candidates.append(_replace_instruction(program, i, new_inst))
                break

    # Re-EXTRACT span with newline stop.
    for key in missing:
        candidate = _reextract_with_stop(program, key)
        if candidate is not None:
            candidates.append(candidate)

    # Fix a bound value using expected arithmetic result.
    if "arith_expected" in meta and (not violations or "arith" in violations or "arith_parse" in violations):
        candidates.extend(_fix_bound_value(program, meta["arith_expected"]))

    # Swap opcode on arithmetic ops.
    if not violations or "arith" in violations or "arith_parse" in violations:
        for i, inst in enumerate(program.instructions):
            if inst.opcode in ARITH_OPS:
                for alt in ARITH_OPS:
                    if alt == inst.opcode:
                        continue
                    new_inst = Instruction(opcode=alt, args=dict(inst.args), dest=inst.dest)
                    candidates.append(_replace_instruction(program, i, new_inst))
                break

    # CSP repair: update status to infeasible if indicated.
    if meta.get("csp_infeasible"):
        for i, inst in enumerate(program.instructions):
            if inst.opcode == "EMIT":
                fields = dict(inst.args.get("fields", {}))
                fields["status"] = "infeasible"
                new_inst = Instruction(opcode="EMIT", args={**inst.args, "fields": fields}, dest=inst.dest)
                candidates.append(_replace_instruction(program, i, new_inst))
                break
    return candidates


def repair_program(
    program: Program,
    evaluator: Callable[[Program], tuple[float, dict[str, Any]]],
    meta: dict[str, Any] | None = None,
    max_steps: int = 2,
) -> tuple[Program, float, int, dict[str, Any]]:
    best_prog = program
    best_score, best_meta = evaluator(program)
    steps = 0
    while steps < max_steps:
        candidates = propose_repairs(best_prog, best_meta)
        if not candidates:
            break
        improved = False
        batch_eval = getattr(evaluator, "batch", None)
        if callable(batch_eval):
            scored = batch_eval(candidates)
            eval_pairs = zip(candidates, scored)
        else:
            eval_pairs = ((cand, evaluator(cand)) for cand in candidates)
        for cand, (score, cand_meta) in eval_pairs:
            if score < best_score:
                best_prog = cand
                best_score = score
                best_meta = cand_meta
                improved = True
                if best_score <= 0:
                    steps += 1
                    return best_prog, best_score, steps, best_meta
        if not improved:
            break
        steps += 1
    return best_prog, best_score, steps, best_meta
