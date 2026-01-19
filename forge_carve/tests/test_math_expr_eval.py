from __future__ import annotations

from fractions import Fraction

from fc.train.data import math_prompt_to_proof_tokens
from fc.util.math_expr import eval_math_expression
from fc.util.runtime_solve import runtime_solve
from fc.verify.arithmetic import ArithmeticVerifier


def _eval(text: str) -> Fraction:
    val = eval_math_expression(text)
    assert val is not None
    return val


def test_math_expr_decimal_division() -> None:
    text = "[MATH] Compute exactly: (-327.8) / ((-4374) / ((-3700) plus (1464 - 5766)))."
    got = _eval(text)
    expected = Fraction(-1639, 5) * Fraction(4001, 2187)
    assert got == expected


def test_math_expr_large_integer() -> None:
    text = "[MATH] Compute exactly: ((-9874) plus ((-6549) + (-3718))) - (3863 times 3197)."
    got = _eval(text)
    assert got == Fraction(-12370152, 1)


def test_math_expr_divided_by_negative() -> None:
    text = "[MATH] Compute exactly: ((-4501) - (-4455)) divided by 6688."
    got = _eval(text)
    expected = Fraction(-46, 6688)
    assert got == expected
    assert abs(float(got) - (-0.00687799043062)) < 1e-12


def test_math_expr_simple_division() -> None:
    text = "[MATH] Calculate the value of 6271 / (7847 minus 7292)."
    got = _eval(text)
    assert got == Fraction(6271, 555)


def test_arith_verifier_accepts_expected_and_rejects_offsets() -> None:
    verifier = ArithmeticVerifier()
    cases = [
        (
            "[MATH] Compute exactly: (-327.8) / ((-4374) / ((-3700) plus (1464 - 5766))).",
            Fraction(-1639, 5) * Fraction(4001, 2187),
        ),
        (
            "[MATH] Compute exactly: ((-9874) plus ((-6549) + (-3718))) - (3863 times 3197).",
            Fraction(-12370152, 1),
        ),
        (
            "[MATH] Compute exactly: ((-4501) - (-4455)) divided by 6688.",
            Fraction(-46, 6688),
        ),
    ]
    for text, expected in cases:
        ok = verifier.verify(text, program=None, output=_expected_value(expected))
        assert ok.valid
        bad_up = verifier.verify(text, program=None, output=_expected_value(expected + 1))
        assert not bad_up.valid
        bad_down = verifier.verify(text, program=None, output=_expected_value(expected - 1))
        assert not bad_down.valid


def test_math_prompt_to_proof_tokens_executes() -> None:
    text = "[MATH] Compute: ((7063 - 4345) / (1168 plus (-2141)))."
    tokens = math_prompt_to_proof_tokens(text)
    assert tokens is not None
    out = runtime_solve(text, [], tokens)
    expected = eval_math_expression(text)
    assert expected is not None
    if isinstance(out, Fraction):
        got = out
    elif isinstance(out, int):
        got = Fraction(out, 1)
    elif isinstance(out, float):
        got = Fraction(out).limit_denominator(10**12)
    else:
        raise AssertionError(f"unexpected runtime output: {out!r}")
    assert got == expected


def _expected_value(value: Fraction) -> object:
    if value.denominator == 1:
        return int(value.numerator)
    return float(value)
