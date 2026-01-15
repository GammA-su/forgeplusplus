from __future__ import annotations

import re
from typing import Any

from fc.verify.base import VerifierResult, normalize_constraints


def _parse_tasks(text: str) -> dict[str, int]:
    tasks: dict[str, int] = {}
    for name, dur in re.findall(r"([A-Z])\s*=\s*(\d+)", text):
        tasks[name] = int(dur)
    return tasks


def _parse_constraints(text: str) -> list[tuple[str, str]]:
    return re.findall(r"([A-Z])\s*<\s*([A-Z])", text)


def _is_acyclic(tasks: dict[str, int], constraints: list[tuple[str, str]]) -> bool:
    preds: dict[str, set[str]] = {t: set() for t in tasks}
    succs: dict[str, set[str]] = {t: set() for t in tasks}
    for a, b in constraints:
        if a in tasks and b in tasks:
            preds[b].add(a)
            succs[a].add(b)
    queue = [t for t, ps in preds.items() if not ps]
    seen: list[str] = []
    while queue:
        node = sorted(queue)[0]
        queue.remove(node)
        seen.append(node)
        for nxt in list(succs[node]):
            preds[nxt].discard(node)
            if not preds[nxt]:
                queue.append(nxt)
    return len(seen) == len(tasks)


class CSPVerifier:
    name = "csp"

    def _load_constraints(
        self, text: str, constraints: list[dict[str, Any]] | None
    ) -> tuple[dict[str, int], list[tuple[str, str]]]:
        task_map = _parse_tasks(text)
        prec = _parse_constraints(text)
        for constraint in normalize_constraints(constraints):
            if constraint.get("type") == "csp":
                args = constraint.get("args", {})
                task_map = args.get("tasks", task_map)
                prec = args.get("constraints", prec)
                break
        return task_map, prec

    def _validate_output(
        self, output: Any, task_map: dict[str, int], prec: list[tuple[str, str]]
    ) -> VerifierResult:
        violations: dict[str, float] = {}
        meta: dict[str, Any] = {}
        acyclic = _is_acyclic(task_map, prec)
        schedule = None
        status = None
        if isinstance(output, dict):
            schedule = output.get("schedule")
            status = output.get("status")
        if not acyclic:
            if status != "infeasible":
                violations["csp_infeasible"] = 1.0
                meta["csp_infeasible"] = True
            return VerifierResult(valid=not violations, violations=violations, meta=meta)
        if not isinstance(schedule, dict):
            violations["csp"] = 1.0
            return VerifierResult(valid=False, violations=violations, meta=meta)
        # Precedence and non-overlap check
        for a, b in prec:
            if a in schedule and b in schedule:
                if schedule[a] + task_map[a] > schedule[b]:
                    violations["csp_prec"] = 1.0
                    break
        if not violations:
            # Non-overlap (single resource) check by sorting starts
            ordered = sorted(schedule.items(), key=lambda kv: kv[1])
            t = 0
            for task, start in ordered:
                if start < t:
                    violations["csp_overlap"] = 1.0
                    break
                t = start + task_map.get(task, 0)
        valid = not violations
        return VerifierResult(valid=valid, violations=violations, meta=meta)

    def verify(
        self,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> VerifierResult:
        task_map, prec = self._load_constraints(text, constraints)
        return self._validate_output(output, task_map, prec)

    def verify_batch(
        self,
        text: str,
        programs: list[Any],
        outputs: list[Any],
        constraints: list[dict[str, Any]] | None = None,
    ) -> list[VerifierResult]:
        task_map, prec = self._load_constraints(text, constraints)
        return [self._validate_output(output, task_map, prec) for output in outputs]
