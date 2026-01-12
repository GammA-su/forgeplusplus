from __future__ import annotations

from typing import Any


def _num_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _is_csp_output(obj: Any) -> bool:
    return isinstance(obj, dict) and "status" in obj and "schedule" in obj


def _csp_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    status_a = a.get("status")
    status_b = b.get("status")
    if status_a != status_b:
        return False
    if status_a == "infeasible":
        return True
    sched_a = a.get("schedule")
    sched_b = b.get("schedule")
    if not isinstance(sched_a, dict) or not isinstance(sched_b, dict):
        return False
    if set(sched_a.keys()) != set(sched_b.keys()):
        return False
    for key in sched_a:
        try:
            aval = float(sched_a[key])
            bval = float(sched_b[key])
        except (TypeError, ValueError):
            return False
        if not _num_close(aval, bval):
            return False
    return True


def outputs_equivalent(a: Any, b: Any) -> bool:
    if _is_csp_output(a) and _is_csp_output(b):
        return _csp_equivalent(a, b)
    if type(a) != type(b):
        # allow numeric cross-types
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _num_close(float(a), float(b))
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a:
            if not outputs_equivalent(a[k], b[k]):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(outputs_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)):
        return _num_close(float(a), float(b))
    return a == b
