from __future__ import annotations

from fc.util.repair_math_cegis import repair_math_cegis


def test_repair_math_cegis_wallclock_override() -> None:
    text = "Compute 2 + 3."
    res = repair_math_cegis(
        text,
        constraints=[],
        max_nums=4,
        depth=3,
        limit=2000,
        wallclock_ms=10,
    )
    assert res is not None
    _, meta = res
    assert meta.get("repair_cegis_wallclock_ms") == 10
