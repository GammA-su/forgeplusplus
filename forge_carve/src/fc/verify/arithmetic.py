from __future__ import annotations

import re
from typing import Any

from fc.interp.core import Interpreter
from fc.verify.base import VerifierResult, normalize_constraints


class ArithmeticVerifier:
    name = "arithmetic"

    def __init__(self) -> None:
        self.interp = Interpreter()

    def _parse_expected(self, text: str, op_override: str | None = None) -> float | None:
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
        if len(nums) < 2:
            return None
        if op_override == "+":
            return nums[0] + nums[1]
        if op_override == "-":
            return nums[0] - nums[1]
        if op_override == "*":
            return nums[0] * nums[1]
        if op_override == "/":
            if nums[1] == 0:
                return None
            return nums[0] / nums[1]
        if re.search(r"\+|plus|add", text, flags=re.IGNORECASE):
            return nums[0] + nums[1]
        if re.search(r"-|minus|subtract", text, flags=re.IGNORECASE):
            return nums[0] - nums[1]
        if re.search(r"\*|times|multiply", text, flags=re.IGNORECASE):
            return nums[0] * nums[1]
        if re.search(r"/|divided", text, flags=re.IGNORECASE):
            if nums[1] == 0:
                return None
            return nums[0] / nums[1]
        return None

    def verify(
        self,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> VerifierResult:
        violations: dict[str, float] = {}
        meta: dict[str, Any] = {}
        op_override = None
        for constraint in normalize_constraints(constraints):
            if constraint.get("type") == "arithmetic":
                op_override = constraint.get("args", {}).get("op")
                break
        expected = self._parse_expected(text, op_override=op_override)
        if expected is None:
            violations["arith_parse"] = 1.0
            return VerifierResult(valid=False, violations=violations, meta=meta)
        exec_out, _, _ = self.interp.execute(program, text)
        if isinstance(exec_out, dict):
            got = exec_out.get("result")
        else:
            got = exec_out
        if got is None or abs(float(got) - float(expected)) > 1e-6:
            violations["arith"] = 1.0
            meta["arith_expected"] = expected
            meta["arith_got"] = got
        valid = not violations
        return VerifierResult(valid=valid, violations=violations, meta=meta)
