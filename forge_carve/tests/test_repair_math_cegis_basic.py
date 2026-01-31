from __future__ import annotations

from fc.util.repair_math_cegis import repair_math_cegis
from fc.util.runtime_solve import runtime_solve
from fc.verify.arithmetic import ArithmeticVerifier


def test_repair_math_cegis_basic() -> None:
    text = "[MATH] What is 2 + 3?"
    constraints = [{"id": "arith", "type": "arithmetic", "args": {}}]
    result = repair_math_cegis(
        text,
        constraints,
        max_nums=6,
        depth=4,
        limit=20000,
        max_seconds=0.08,
        dp_value_cap_per_subset=64,
    )
    assert result is not None
    tokens, meta = result
    assert meta.get("found") is True
    assert meta.get("repair_cegis_mode") == "brute"
    out = runtime_solve(text, constraints, tokens)
    verifier = ArithmeticVerifier()
    res = verifier.verify(text, None, out, constraints=constraints)
    assert res.valid


def test_repair_math_cegis_fractional_intermediate() -> None:
    text = "[MATH] Compute: (6 / 4) * 8"
    constraints = [{"id": "arith", "type": "arithmetic", "args": {}}]
    result = repair_math_cegis(
        text,
        constraints,
        max_nums=6,
        depth=4,
        limit=20000,
        max_seconds=0.08,
        dp_value_cap_per_subset=64,
        cegis_mode="dp",
    )
    assert result is not None
    tokens, meta = result
    assert meta.get("found") is True
    assert meta.get("repair_cegis_mode") == "dp"
    subsets = int(meta.get("repair_cegis_subsets_built", 0))
    values_kept = int(meta.get("repair_cegis_values_kept", 0))
    assert values_kept <= subsets * 64
    out = runtime_solve(text, constraints, tokens)
    verifier = ArithmeticVerifier()
    res = verifier.verify(text, None, out, constraints=constraints)
    assert res.valid


def test_repair_math_cegis_percent_case() -> None:
    text = "[MATH] What is 20% of 50?"
    constraints = [{"id": "arith", "type": "arithmetic", "args": {}}]
    result = repair_math_cegis(
        text,
        constraints,
        max_nums=8,
        depth=4,
        limit=20000,
        max_seconds=0.08,
        dp_value_cap_per_subset=64,
        cegis_mode="dp",
    )
    assert result is not None
    tokens, meta = result
    assert meta.get("found") is True
    assert meta.get("repair_cegis_mode") == "dp"
    subsets = int(meta.get("repair_cegis_subsets_built", 0))
    values_kept = int(meta.get("repair_cegis_values_kept", 0))
    assert values_kept <= subsets * 64
    out = runtime_solve(text, constraints, tokens)
    verifier = ArithmeticVerifier()
    res = verifier.verify(text, None, out, constraints=constraints)
    assert res.valid
