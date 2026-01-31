from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any

from fc.interp.core import Interpreter
from fc.util.math_expr import eval_math_expression
from fc.verify.base import VerifierResult, normalize_constraints

_FRAC_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_DECIMAL_RE = re.compile(r"^\s*([+-]?)(?:(\d+)(?:\.(\d*))?|\.(\d+))\s*$")
_NON_INT_TOL = 1e-9
_NON_INT_REL = 1e-12


def _to_fraction(v: Any) -> Fraction | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v, 1)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return Fraction(v)
    if isinstance(v, str):
        match = _FRAC_RE.match(v)
        if match:
            n, d = match.groups()
            return Fraction(int(n), int(d))
        match = _DECIMAL_RE.match(v)
        if match:
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
    return None


def _expected_meta(expected: Fraction) -> Any:
    if expected.denominator == 1:
        return int(expected.numerator)
    return float(expected)


class ArithmeticVerifier:
    name = "arithmetic"

    def __init__(self) -> None:
        self.interp = Interpreter()

    def _parse_expected(self, text: str, op_override: str | None = None) -> Fraction | None:
        expr_val = eval_math_expression(text)
        if expr_val is not None:
            return expr_val
        percent_match = re.search(
            r"([+-]?\d+(?:\.\d+)?)\s*%\s*(?:of\s*)?([+-]?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if percent_match:
            try:
                pct = Fraction(percent_match.group(1))
                base = Fraction(percent_match.group(2))
                return pct / 100 * base
            except (ValueError, ZeroDivisionError):
                pass
        nums = [int(x) for x in re.findall(r"-?\d+", text)]
        if len(nums) < 2:
            return None
        if op_override == "+":
            return Fraction(nums[0] + nums[1], 1)
        if op_override == "-":
            return Fraction(nums[0] - nums[1], 1)
        if op_override == "*":
            return Fraction(nums[0] * nums[1], 1)
        if op_override == "/":
            if nums[1] == 0:
                return None
            return Fraction(nums[0], nums[1])
        if re.search(r"\+|plus|add", text, flags=re.IGNORECASE):
            return Fraction(nums[0] + nums[1], 1)
        if re.search(r"-|minus|subtract", text, flags=re.IGNORECASE):
            return Fraction(nums[0] - nums[1], 1)
        if re.search(r"\*|times|multiply", text, flags=re.IGNORECASE):
            return Fraction(nums[0] * nums[1], 1)
        if re.search(r"/|divided", text, flags=re.IGNORECASE):
            if nums[1] == 0:
                return None
            return Fraction(nums[0], nums[1])
        return None

    def _expected_from_constraints(self, text: str, constraints: list[dict[str, Any]] | None) -> Fraction | None:
        op_override = None
        for constraint in normalize_constraints(constraints):
            if constraint.get("type") == "arithmetic":
                op_override = constraint.get("args", {}).get("op")
                break
        return self._parse_expected(text, op_override=op_override)

    def _validate_output(self, expected: Fraction | None, output: Any) -> VerifierResult:
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
            meta["arith_expected"] = _expected_meta(expected)
            meta["arith_got"] = got
            return VerifierResult(valid=False, violations=violations, meta=meta)
        got_frac = _to_fraction(got)
        if got_frac is None:
            violations["arith"] = 1.0
            meta["arith_expected"] = _expected_meta(expected)
            meta["arith_got"] = got
            return VerifierResult(valid=False, violations=violations, meta=meta)
        if expected.denominator == 1:
            if got_frac.denominator != 1 or got_frac.numerator != expected.numerator:
                violations["arith"] = 1.0
                meta["arith_expected"] = _expected_meta(expected)
                meta["arith_got"] = got
        else:
            got_val = float(got_frac)
            exp_val = float(expected)
            if not math.isfinite(got_val) or not math.isclose(
                got_val,
                exp_val,
                rel_tol=_NON_INT_REL,
                abs_tol=_NON_INT_TOL,
            ):
                violations["arith"] = 1.0
                meta["arith_expected"] = _expected_meta(expected)
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
