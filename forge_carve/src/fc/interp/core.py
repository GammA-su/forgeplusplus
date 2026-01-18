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


def _extract_float(text: str, key: str | None, index: int | None, stop: str | None) -> float | None:
    pattern = r"-?\d+(?:\.\d+)?"
    if key is not None:
        val = _extract_by_key(text, key, stop=stop)
        if val is None:
            return None
        nums = re.findall(pattern, val)
        if not nums:
            return None
        return float(nums[0])
    nums = re.findall(pattern, text)
    if index is None or index >= len(nums):
        return None
    return float(nums[index])


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


def _topo_order(tasks: dict[str, int], constraints: list[tuple[str, str]]) -> tuple[list[str], bool]:
    # Kahn's algorithm for precedence ordering.
    # Tie-break alphabetically so orbits that reorder tasks stay invariant.
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
    ok = len(order) == len(tasks)
    return order, ok


def _build_schedule(order: list[str], tasks: dict[str, int]) -> dict[str, int]:
    schedule: dict[str, int] = {}
    t = 0
    for task in order:
        if task not in tasks:
            continue
        schedule[task] = t
        t += tasks[task]
    return schedule


def _solve_csp(tasks: dict[str, int], constraints: list[tuple[str, str]]) -> tuple[dict[str, int], bool]:
    order, ok = _topo_order(tasks, constraints)
    if not ok:
        return {}, False
    return _build_schedule(order, tasks), True


def _resolve_value(state: dict[str, Any], value: Any) -> Any:
    if isinstance(value, str) and value in state:
        return state[value]
    return value


def _coerce_number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return None
    return None


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
        if op == "EXTRACT_FLOAT":
            key = inst.args.get("key")
            index = inst.args.get("index")
            stop = inst.args.get("stop")
            val = _extract_float(text, key, index, stop)
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
        if op == "APPLY_ARITH":
            aval = _coerce_number(_resolve_value(state, inst.args.get("a")))
            bval = _coerce_number(_resolve_value(state, inst.args.get("b")))
            op_val = _resolve_value(state, inst.args.get("op"))
            if aval is None or bval is None or op_val is None:
                return None
            op_str = str(op_val).lower()
            if op_str in {"+", "add", "plus"}:
                res = aval + bval
            elif op_str in {"-", "sub", "minus"}:
                res = aval - bval
            elif op_str in {"*", "mul", "times", "x"}:
                res = aval * bval
            elif op_str in {"/", "div", "divide"}:
                res = aval / bval if bval != 0 else float("nan")
            else:
                return None
            if inst.dest:
                state[inst.dest] = res
            return res
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
        if op == "APPLY_TOPO":
            tasks = _parse_tasks(text)
            constraints = _parse_constraints(text)
            order, ok = _topo_order(tasks, constraints)
            state["tasks"] = tasks
            state["constraints"] = constraints
            state["order"] = order
            state["status"] = "ok" if ok else "infeasible"
            if inst.dest:
                state[inst.dest] = order
            return order
        if op == "APPLY_CUMSUM":
            tasks = state.get("tasks")
            if not isinstance(tasks, dict):
                tasks = _parse_tasks(text)
            constraints = state.get("constraints")
            if not isinstance(constraints, list):
                constraints = _parse_constraints(text)
            order = state.get("order")
            if not isinstance(order, list):
                order, ok = _topo_order(tasks, constraints)
                state["order"] = order
                state["status"] = "ok" if ok else "infeasible"
            status = state.get("status", "ok")
            if status != "ok":
                schedule = {}
            else:
                schedule = _build_schedule(order, tasks)
            state["schedule"] = schedule
            if inst.dest:
                state[inst.dest] = schedule
            return schedule
        if op == "SOLVE_CSP":
            tasks = _parse_tasks(text)
            constraints = _parse_constraints(text)
            schedule, ok = _solve_csp(tasks, constraints)
            state["schedule"] = schedule
            state["status"] = "ok" if ok else "infeasible"
            return schedule
        if op == "EMIT_NUM":
            val = inst.args.get("value")
            if val is None and "result" in state:
                val = state["result"]
            val = _resolve_value(state, val)
            state["output"] = val
            return val
        if op == "EMIT_SCHEDULE":
            schedule = inst.args.get("schedule")
            if schedule is None:
                schedule = state.get("schedule")
            else:
                schedule = _resolve_value(state, schedule)
            status = inst.args.get("status")
            if status is None:
                status = state.get("status", "ok")
            else:
                status = _resolve_value(state, status)
            if schedule is None:
                schedule = {}
            out = {"schedule": schedule, "status": status}
            state["output"] = out
            return out
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
