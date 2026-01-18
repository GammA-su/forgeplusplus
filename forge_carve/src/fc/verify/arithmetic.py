from __future__ import annotations

import math
import re
from typing import Any

from fc.interp.core import Interpreter
from fc.util.math_expr import eval_math_expression
from fc.verify.base import VerifierResult, normalize_constraints


class ArithmeticVerifier:
    name = "arithmetic"

    def __init__(self) -> None:
        self.interp = Interpreter()

    def _parse_expected(self, text: str, op_override: str | None = None) -> float | None:
        expr_val = eval_math_expression(text)
        if expr_val is not None:
            return expr_val
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

    def _expected_from_constraints(self, text: str, constraints: list[dict[str, Any]] | None) -> float | None:
        op_override = None
        for constraint in normalize_constraints(constraints):
            if constraint.get("type") == "arithmetic":
                op_override = constraint.get("args", {}).get("op")
                break
        return self._parse_expected(text, op_override=op_override)

    def _validate_output(self, expected: float | None, output: Any) -> VerifierResult:
        violations: dict[str, float] = {}
        meta: dict[str, Any] = {}
        if expected is None:
            violations["arith_parse"] = 1.0
            return VerifierResult(valid=False, violations=violations, meta=meta)
        if isinstance(output, dict):
            got = output.get("result")
        else:
            got = output
        if got is None:
            violations["arith"] = 1.0
            meta["arith_expected"] = expected
            meta["arith_got"] = got
            return VerifierResult(valid=False, violations=violations, meta=meta)
        try:
            got_val = float(got)
        except (TypeError, ValueError):
            violations["arith"] = 1.0
            meta["arith_expected"] = expected
            meta["arith_got"] = got
            return VerifierResult(valid=False, violations=violations, meta=meta)
        if not math.isfinite(got_val) or abs(got_val - float(expected)) > 1e-6:
            violations["arith"] = 1.0
            meta["arith_expected"] = expected
            meta["arith_got"] = got
        valid = not violations
        return VerifierResult(valid=valid, violations=violations, meta=meta)

    def verify(
        self,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> VerifierResult:
        expected = self._expected_from_constraints(text, constraints)
        exec_out = output
        if exec_out is None:
            exec_out, _, _ = self.interp.execute(program, text)
        return self._validate_output(expected, exec_out)

    def verify_batch(
        self,
        text: str,
        programs: list[Any],
        outputs: list[Any],
        constraints: list[dict[str, Any]] | None = None,
    ) -> list[VerifierResult]:
        expected = self._expected_from_constraints(text, constraints)
        results: list[VerifierResult] = []
        for program, output in zip(programs, outputs):
            exec_out = output
            if exec_out is None:
                exec_out, _, _ = self.interp.execute(program, text)
            results.append(self._validate_output(expected, exec_out))
        return results
