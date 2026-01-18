import math
from typing import Any

# Canonical numeric policy:
# - If a float is effectively an integer, store it as int.
# - Otherwise, keep a stable decimal float with fixed significant digits.
# This keeps JSON labels deterministic while remaining tolerant to runtime noise.
_ABS = 1e-9
_REL = 1e-12
_SIGFIGS = 12

def canon_json(v: Any) -> Any:
    """Canonicalize JSON-ish values; mainly fixes float noise."""
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return v
        r = round(v)
        # If it's basically an int, make it an int (prevents 41610.00000000001)
        if math.isclose(v, r, rel_tol=_REL, abs_tol=_ABS):
            return int(r)
        # Otherwise keep a stable float (keeps repr controlled)
        return float(f"{v:.{_SIGFIGS}g}")
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
