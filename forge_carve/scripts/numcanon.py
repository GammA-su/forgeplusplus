import math
import re
from fractions import Fraction
from typing import Any

# Tight enough to only catch float representation noise.
_ABS = 1e-9
_REL = 1e-12

_FRACTION_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_DECIMAL_RE = re.compile(r"^\s*([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))\s*$")


def _fraction_string_to_fraction(text: str) -> Fraction | None:
    match = _FRACTION_RE.match(text)
    if not match:
        return None
    n, d = match.groups()
    return Fraction(int(n), int(d))


def _decimal_string_to_fraction(text: str) -> Fraction | None:
    match = _DECIMAL_RE.match(text)
    if not match:
        return None
    sign, whole, frac_a, frac_b = match.groups()
    frac = frac_b if frac_b is not None else (frac_a or "")
    whole = whole or "0"
    if not frac:
        val = int(whole)
        return Fraction(-val if sign == "-" else val, 1)
    num = int(f"{whole}{frac}")
    if sign == "-":
        num = -num
    return Fraction(num, 10 ** len(frac))


def canon_json(v: Any) -> Any:
    """Canonicalize JSON-ish values; mainly fixes float noise."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, str):
        frac = _fraction_string_to_fraction(v)
        if frac is None:
            frac = _decimal_string_to_fraction(v)
        if frac is None:
            return v
        v = frac
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return int(v.numerator)
        v = float(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return v
        r = round(v)
        # If it's basically an int, make it an int (prevents 41610.00000000001)
        if math.isclose(v, r, rel_tol=_REL, abs_tol=_ABS):
            return int(r)
        # Otherwise keep a stable float (optional; keeps repr controlled)
        return float(f"{v:.12g}")
    if isinstance(v, list):
        return [canon_json(x) for x in v]
    if isinstance(v, dict):
        return {k: canon_json(v[k]) for k in v}
    return v

def json_equal(a: Any, b: Any) -> bool:
    """Deep equality with numeric tolerance for float noise."""
    a = canon_json(a)
    b = canon_json(b)

    # numeric scalar compare
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=_REL, abs_tol=_ABS)

    # structured compare
    if type(a) != type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a.keys())
    return a == b
