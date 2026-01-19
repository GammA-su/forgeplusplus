from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import numcanon


def _is_csp_output(obj: Any) -> bool:
    return isinstance(obj, dict) and "status" in obj and "schedule" in obj


def _num_equal(a: Any, b: Any) -> bool:
    return numcanon.json_equal(a, b)


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
    return _num_equal(sched_a, sched_b)


def _is_math_output(obj: Any) -> bool:
    return isinstance(obj, dict) and set(obj.keys()) == {"result"}


def _math_equivalent(a: Any, b: Any) -> bool:
    aval = _unwrap_math_result(a)
    bval = _unwrap_math_result(b)
    return _num_equal(aval, bval)


def _json_equivalent(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _num_equal(a, b)
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


def _numeric_like(value: Any) -> bool:
    if isinstance(value, (int, float, Fraction)):
        return True
    if isinstance(value, str):
        coerced = numcanon.canon_json(value)
        return isinstance(coerced, (int, float))
    return False


def outputs_equivalent(a: Any, b: Any) -> bool:
    norm_a = _maybe_parse_json(a)
    norm_b = _maybe_parse_json(b)
    if _is_csp_output(norm_a) and _is_csp_output(norm_b):
        return _csp_equivalent(norm_a, norm_b)
    if _is_math_output(norm_a) or _is_math_output(norm_b):
        return _math_equivalent(norm_a, norm_b)
    if _numeric_like(norm_a) and _numeric_like(norm_b):
        return _num_equal(norm_a, norm_b)
    return _json_equivalent(norm_a, norm_b)
