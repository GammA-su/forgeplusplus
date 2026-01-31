from __future__ import annotations

from fc.util.repair_math_cegis import repair_math_cegis
from fc.util.runtime_solve import runtime_solve
from fc.verify.arithmetic import ArithmeticVerifier


def test_repair_math_cegis_frac_depth4() -> None:
    text = "[MATH] What is 1/2 + 1/2?"
    constraints = [{"id": "arith", "type": "arithmetic", "args": {}}]
    result = repair_math_cegis(
        text,
        constraints,
        max_nums=6,
        depth=4,
        limit=20000,
        max_seconds=0.08,
        cegis_mode="brute",
    )
    assert result is not None
    tokens, meta = result
    assert meta.get("repair_cegis_mode") == "brute"
    assert meta.get("repair_cegis_kind") == "frac_depth4"
    assert meta.get("repair_cegis_depth") == 4
    assert meta.get("repair_cegis_hit") is True
    out = runtime_solve(text, constraints, tokens)
    verifier = ArithmeticVerifier()
    res = verifier.verify(text, None, out, constraints=constraints)
    assert res.valid
