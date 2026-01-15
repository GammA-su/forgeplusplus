from __future__ import annotations

import json
import re
from typing import Any


def _num_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _is_csp_output(obj: Any) -> bool:
    return isinstance(obj, dict) and "status" in obj and "schedule" in obj


def _coerce_number(value: Any, allow_str: bool) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if allow_str and isinstance(value, str):
        cleaned = value.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
            try:
                return float(cleaned)
            except ValueError:
                return None
    return None


def _unwrap_math_result(obj: Any) -> Any:
    if isinstance(obj, dict) and set(obj.keys()) == {"result"}:
        return obj.get("result")
    return obj


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
        aval = _coerce_number(sched_a[key], allow_str=True)
        bval = _coerce_number(sched_b[key], allow_str=True)
        if aval is None or bval is None:
            return False
        if not _num_close(aval, bval):
            return False
    return True


def _is_math_output(obj: Any) -> bool:
    return isinstance(obj, dict) and set(obj.keys()) == {"result"}


def _math_equivalent(a: Any, b: Any) -> bool:
    aval = _unwrap_math_result(a)
    bval = _unwrap_math_result(b)
    a_num = _coerce_number(aval, allow_str=True)
    b_num = _coerce_number(bval, allow_str=True)
    if a_num is None or b_num is None:
        return aval == bval
    return _num_close(a_num, b_num)


def _json_equivalent(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a:
            if not _json_equivalent(a[k], b[k]):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_json_equivalent(x, y) for x, y in zip(a, b))
    return a == b


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return value
    return value


def outputs_equivalent(a: Any, b: Any) -> bool:
    norm_a = _maybe_parse_json(a)
    norm_b = _maybe_parse_json(b)
    if _is_csp_output(norm_a) and _is_csp_output(norm_b):
        return _csp_equivalent(norm_a, norm_b)
    if _is_math_output(norm_a) or _is_math_output(norm_b):
        return _math_equivalent(norm_a, norm_b)
    if _coerce_number(norm_a, allow_str=True) is not None and _coerce_number(norm_b, allow_str=True) is not None:
        return _math_equivalent(norm_a, norm_b)
    return _json_equivalent(norm_a, norm_b)
