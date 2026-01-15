from __future__ import annotations

import re
from typing import Any

from fc.dsl.program import Instruction, Program


def _normalize_stop(stop: str | None) -> str | None:
    if stop is None:
        return None
    if stop in {"\\n", "newline", "line"}:
        return "\n"
    return stop


def _extract_by_key(text: str, key: str, stop: str | None = None) -> str | None:
    stop = _normalize_stop(stop)
    if stop:
        stop_pat = re.escape(stop)
        pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^{stop_pat}]+)"
    else:
        pattern = rf"{re.escape(key)}\s*(?:[:=]|is)\s*([^\n;,.]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_int(text: str, key: str | None, index: int | None, stop: str | None) -> int | None:
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        digits = re.findall(r"-?\d+", val)
        if not digits:
            return None
        return int(digits[0])
    ints = re.findall(r"-?\d+", text)
    if index is None or index >= len(ints):
        return None
    return int(ints[index])


def _extract_str(text: str, key: str | None, stop: str | None) -> str | None:
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        return val.strip()
    return None


def _parse_tasks(text: str) -> dict[str, int]:
    tasks: dict[str, int] = {}
    for name, dur in re.findall(r"([A-Z])\s*=\s*(\d+)", text):
        tasks[name] = int(dur)
    return tasks


def _parse_constraints(text: str) -> list[tuple[str, str]]:
    return re.findall(r"([A-Z])\s*<\s*([A-Z])", text)


def _solve_csp(tasks: dict[str, int], constraints: list[tuple[str, str]]) -> tuple[dict[str, int], bool]:
    # Kahn's algorithm for precedence; schedule sequentially.
    preds: dict[str, set[str]] = {t: set() for t in tasks}
    succs: dict[str, set[str]] = {t: set() for t in tasks}
    for a, b in constraints:
        if a in tasks and b in tasks:
            preds[b].add(a)
            succs[a].add(b)
    queue = [t for t, ps in preds.items() if not ps]
    order: list[str] = []
    while queue:
        node = sorted(queue)[0]
        queue.remove(node)
        order.append(node)
        for nxt in list(succs[node]):
            preds[nxt].discard(node)
            if not preds[nxt]:
                queue.append(nxt)
    if len(order) != len(tasks):
        return {}, False
    schedule: dict[str, int] = {}
    t = 0
    for task in order:
        schedule[task] = t
        t += tasks[task]
    return schedule, True


class Interpreter:
    def execute(self, program: Program, text: str) -> tuple[Any, list[dict[str, Any]], list[str]]:
        state: dict[str, Any] = {}
        output: Any = None
        trace: list[dict[str, Any]] = []
        errors: list[str] = []
        for inst in program.instructions:
            try:
                output = self._exec_inst(inst, state, text)
            except Exception as exc:
                errors.append(f"exec_error:{inst.opcode}:{exc}")
                output = None
            trace.append(
                {
                    "opcode": inst.opcode,
                    "dest": inst.dest,
                    "args": dict(inst.args),
                    "value": output,
                }
            )
            if output is None:
                errors.append(f"null_result:{inst.opcode}")
        if output is None and "output" in state:
            output = state["output"]
        return output, trace, errors

    def _exec_inst(self, inst: Instruction, state: dict[str, Any], text: str) -> Any:
        op = inst.opcode
        if op == "EXTRACT_INT":
            key = inst.args.get("key")
            index = inst.args.get("index")
            stop = inst.args.get("stop")
            val = _extract_int(text, key, index, stop)
            if inst.dest:
                state[inst.dest] = val
            return val
        if op == "EXTRACT_STR":
            key = inst.args.get("key")
            stop = inst.args.get("stop")
            val = _extract_str(text, key, stop)
            if inst.dest:
                state[inst.dest] = val
            return val
        if op == "BIND":
            val = inst.args.get("value")
            if inst.dest:
                state[inst.dest] = val
            return val
        if op in {"ADD", "SUB", "MUL", "DIV"}:
            a = inst.args.get("a")
            b = inst.args.get("b")
            aval = state.get(a, a)
            bval = state.get(b, b)
            if op == "ADD":
                res = aval + bval
            elif op == "SUB":
                res = aval - bval
            elif op == "MUL":
                res = aval * bval
            else:
                res = aval / bval if bval != 0 else float("nan")
            if inst.dest:
                state[inst.dest] = res
            return res
        if op == "SOLVE_CSP":
            tasks = _parse_tasks(text)
            constraints = _parse_constraints(text)
            schedule, ok = _solve_csp(tasks, constraints)
            state["schedule"] = schedule
            state["status"] = "ok" if ok else "infeasible"
            return schedule
        if op == "EMIT":
            fields = inst.args.get("fields", {})
            out: dict[str, Any] = {}
            for key, val in fields.items():
                if isinstance(val, str) and val in state:
                    out[key] = state[val]
                else:
                    out[key] = val
            state["output"] = out
            return out
        return None
