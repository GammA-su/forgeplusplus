from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from fc.dsl.codec import program_to_tokens
from fc.dsl.program import Instruction, Program
from fc.util.math_expr import eval_math_expression
from fc.util.runtime_solve import runtime_solve
from fc.verify.arithmetic import ArithmeticVerifier
from prooftape.ptv1 import _extract_int

_INT_RE = re.compile(r"(?<!\w)[+-]?\d+(?!\w)")
_DECIMAL_RE = re.compile(r"(?<!\w)[+-]?\d+\.\d+(?!\w)")
_FRACTION_RE = re.compile(r"(?<!\w)[+-]?\d+\s*/\s*\d+(?!\w)")
_PERCENT_RE = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?%(?!\w)")
_WORD_NUM_RE = re.compile(
    r"\b("
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|double|twice"
    r")\b",
    flags=re.IGNORECASE,
)
_MAX_ABS = 10**9
_MAX_DENOM = 10**6
_DEFAULT_DEPTH = 4
_DEFAULT_LIMIT = 20000
_DEFAULT_MAX_SECONDS = 0.08
_DEFAULT_DP_CAP = 64
_DEFAULT_BRUTE_CAP = 512
_OPS = ("+", "-", "*", "/")

_WORD_NUMS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

_SPECIAL_WORDS = {"half", "double", "twice"}


@dataclass(frozen=True)
class _Expr:
    value: Fraction
    ast: Any


@dataclass(frozen=True)
class _Operand:
    value: Fraction
    literal: int | str


def extract_ints_from_text(text: str, *, max_nums: int = 6) -> list[int]:
    vals = [int(tok) for tok in _INT_RE.findall(text)]
    return vals[: max(0, int(max_nums))]


def extract_ints_from_exec_env(
    text: str,
    program: Program,
    *,
    max_nums: int = 6,
) -> list[int]:
    vals: list[int] = []
    for inst in program.instructions:
        if inst.opcode != "EXTRACT_INT":
            continue
        key = inst.args.get("key")
        index = inst.args.get("index")
        stop = inst.args.get("stop")
        value = _extract_int(text, key, index, stop)
        if value is None:
            continue
        vals.append(int(value))
        if len(vals) >= max(0, int(max_nums)):
            break
    return vals


def _fraction_from_decimal(token: str) -> Fraction | None:
    if "." not in token:
        return None
    sign = -1 if token.strip().startswith("-") else 1
    token = token.lstrip("+-")
    whole, frac = token.split(".", 1)
    if not whole:
        whole = "0"
    if not frac:
        return Fraction(sign * int(whole), 1)
    numerator = int(whole + frac)
    denom = 10 ** len(frac)
    return Fraction(sign * numerator, denom)


def _fraction_to_decimal_str(value: Fraction) -> str | None:
    if value.denominator == 1:
        return str(int(value.numerator))
    denom = value.denominator
    two_count = 0
    five_count = 0
    while denom % 2 == 0:
        denom //= 2
        two_count += 1
    while denom % 5 == 0:
        denom //= 5
        five_count += 1
    if denom != 1:
        return None
    scale = max(two_count, five_count)
    num = value.numerator
    if two_count < scale:
        num *= 2 ** (scale - two_count)
    if five_count < scale:
        num *= 5 ** (scale - five_count)
    sign = "-" if num < 0 else ""
    num = abs(num)
    digits = f"{num:0{scale + 1}d}"
    if scale == 0:
        return f"{sign}{digits}"
    int_part = digits[:-scale] or "0"
    frac_part = digits[-scale:].rstrip("0")
    if not frac_part:
        return f"{sign}{int_part}"
    return f"{sign}{int_part}.{frac_part}"


def _add_operand(
    out: list[_Operand],
    seen: set[Fraction],
    value: Fraction,
    literal: int | str | None,
    *,
    max_nums: int,
) -> None:
    if literal is None:
        return
    if value in seen:
        return
    out.append(_Operand(value=value, literal=literal))
    seen.add(value)
    if len(out) >= max_nums:
        raise StopIteration


def extract_operands(
    text: str,
    program: Program | None,
    *,
    max_nums: int = 8,
) -> list[_Operand]:
    seen: set[Fraction] = set()
    operands: list[_Operand] = []
    if program is not None:
        for value in extract_ints_from_exec_env(text, program, max_nums=max_nums):
            frac = Fraction(int(value), 1)
            _add_operand(operands, seen, frac, int(value), max_nums=max_nums)
            if len(operands) >= max_nums:
                return operands

    pattern = re.compile(
        r"(?<!\w)([+-]?\d+\s*/\s*\d+)(?!\w)"
        r"|(?<!\w)([+-]?\d+\.\d+)(?!\w)"
        r"|(?<!\w)([+-]?\d+(?:\.\d+)?%)(?!\w)"
        r"|(?<!\w)([+-]?\d+)(?!\w)"
        r"|\b("
        r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|double|twice"
        r")\b",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        if len(operands) >= max_nums:
            break
        frac_token = match.group(1)
        dec_token = match.group(2)
        pct_token = match.group(3)
        int_token = match.group(4)
        word_token = match.group(5)

        try:
            if frac_token:
                parts = frac_token.replace(" ", "").split("/", 1)
                if len(parts) == 2:
                    num = int(parts[0])
                    den = int(parts[1])
                    if den != 0:
                        _add_operand(operands, seen, Fraction(num, 1), num, max_nums=max_nums)
                        _add_operand(operands, seen, Fraction(den, 1), den, max_nums=max_nums)
                        frac_val = Fraction(num, den)
                        literal = _fraction_to_decimal_str(frac_val)
                        _add_operand(operands, seen, frac_val, literal, max_nums=max_nums)
                continue
            if pct_token:
                token = pct_token.strip().rstrip("%")
                frac_val = _fraction_from_decimal(token) or Fraction(int(token), 1)
                _add_operand(
                    operands,
                    seen,
                    frac_val,
                    _fraction_to_decimal_str(frac_val),
                    max_nums=max_nums,
                )
                pct_val = frac_val / 100
                _add_operand(
                    operands,
                    seen,
                    pct_val,
                    _fraction_to_decimal_str(pct_val),
                    max_nums=max_nums,
                )
                continue
            if dec_token:
                frac_val = _fraction_from_decimal(dec_token)
                if frac_val is not None:
                    _add_operand(
                        operands,
                        seen,
                        frac_val,
                        _fraction_to_decimal_str(frac_val),
                        max_nums=max_nums,
                    )
                continue
            if int_token:
                num = int(int_token)
                _add_operand(operands, seen, Fraction(num, 1), num, max_nums=max_nums)
                continue
            if word_token:
                word = word_token.lower()
                if word in _SPECIAL_WORDS:
                    if word == "half":
                        frac_val = Fraction(1, 2)
                        _add_operand(
                            operands,
                            seen,
                            frac_val,
                            _fraction_to_decimal_str(frac_val),
                            max_nums=max_nums,
                        )
                    else:
                        _add_operand(operands, seen, Fraction(2, 1), 2, max_nums=max_nums)
                else:
                    value = _WORD_NUMS.get(word)
                    if value is not None:
                        _add_operand(operands, seen, Fraction(value, 1), value, max_nums=max_nums)
        except StopIteration:
            break
    return operands


def _combine(a: _Expr, b: _Expr, op: str) -> _Expr | None:
    av = a.value
    bv = b.value
    if op == "+":
        val = av + bv
    elif op == "-":
        val = av - bv
    elif op == "*":
        val = av * bv
    else:
        if bv == 0:
            return None
        val = Fraction(av, bv)
    if abs(val.numerator) > _MAX_ABS or val.denominator > _MAX_DENOM:
        return None
    return _Expr(value=val, ast=(op, a.ast, b.ast))


def _priority_key(value: Fraction) -> tuple[int, int]:
    return (int(value.denominator), abs(int(value.numerator)))


def _prune_values(values: dict[Fraction, _Expr], cap: int) -> dict[Fraction, _Expr]:
    if len(values) <= cap:
        return values
    ordered = sorted(values.items(), key=lambda item: _priority_key(item[0]))
    kept = dict(ordered[:cap])
    return kept


def _prune_exprs(exprs: dict[Fraction, _Expr], cap: int) -> dict[Fraction, _Expr]:
    if len(exprs) <= cap:
        return exprs
    ordered = sorted(exprs.items(), key=lambda item: _priority_key(item[0]))
    return dict(ordered[:cap])


def _brute_frac_depth4(
    operands: list[_Operand],
    expected: Fraction | None,
    *,
    depth: int,
    limit: int,
    max_seconds: float,
    cap_per_depth: int,
    meta: dict[str, Any],
) -> _Expr | None:
    start = time.perf_counter()
    levels: list[dict[Fraction, _Expr]] = []
    level0: dict[Fraction, _Expr] = {}
    for idx, operand in enumerate(operands):
        level0[operand.value] = _Expr(value=operand.value, ast=("leaf", idx))
    level0 = _prune_exprs(level0, cap_per_depth)
    levels.append(level0)
    meta["repair_cegis_states"] = len(level0)
    for expr in level0.values():
        meta["candidates_tried"] += 1
        if limit and meta["candidates_tried"] > limit:
            return None
        if _matches_target(expected, expr.value):
            return expr
        if time.perf_counter() - start > max_seconds:
            return None
    for d in range(2, max(2, depth + 1)):
        next_level: dict[Fraction, _Expr] = {}
        for left_depth in range(1, d):
            right_depth = d - left_depth
            lefts = levels[left_depth - 1]
            rights = levels[right_depth - 1]
            for aexpr in lefts.values():
                for bexpr in rights.values():
                    if time.perf_counter() - start > max_seconds:
                        return None
                    for op in ("+", "*"):
                        combined = _combine(aexpr, bexpr, op)
                        if combined is None:
                            continue
                        next_level.setdefault(combined.value, combined)
                    for op in ("-", "/"):
                        combined = _combine(aexpr, bexpr, op)
                        if combined is not None:
                            next_level.setdefault(combined.value, combined)
                        combined = _combine(bexpr, aexpr, op)
                        if combined is not None:
                            next_level.setdefault(combined.value, combined)
        next_level = _prune_exprs(next_level, cap_per_depth)
        levels.append(next_level)
        meta["repair_cegis_states"] += len(next_level)
        for expr in next_level.values():
            meta["candidates_tried"] += 1
            if limit and meta["candidates_tried"] > limit:
                return None
            if _matches_target(expected, expr.value):
                return expr
            if time.perf_counter() - start > max_seconds:
                return None
    return None


def enumerate_expr_dp(
    operands: list[_Operand],
    *,
    cap_per_subset: int = _DEFAULT_DP_CAP,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> tuple[list[dict[Fraction, _Expr]], dict[str, int]]:
    start = time.perf_counter()
    n = len(operands)
    dp: list[dict[Fraction, _Expr]] = [dict() for _ in range(1 << n)]
    meta = {
        "subsets_built": 0,
        "values_kept": 0,
    }
    for i, operand in enumerate(operands):
        dp[1 << i][operand.value] = _Expr(value=operand.value, ast=("leaf", i))
    for mask in range(1, 1 << n):
        if time.perf_counter() - start > max_seconds:
            break
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other and dp[sub] and dp[other]:
                for aval, aexpr in dp[sub].items():
                    for bval, bexpr in dp[other].items():
                        if time.perf_counter() - start > max_seconds:
                            break
                        for op in ("+", "*"):
                            combined = _combine(aexpr, bexpr, op)
                            if combined is None:
                                continue
                            dp[mask].setdefault(combined.value, combined)
                            if len(dp[mask]) > cap_per_subset:
                                dp[mask] = _prune_values(dp[mask], cap_per_subset)
                        for op in ("-", "/"):
                            combined = _combine(aexpr, bexpr, op)
                            if combined is not None:
                                dp[mask].setdefault(combined.value, combined)
                                if len(dp[mask]) > cap_per_subset:
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
                            combined = _combine(bexpr, aexpr, op)
                            if combined is not None:
                                dp[mask].setdefault(combined.value, combined)
                                if len(dp[mask]) > cap_per_subset:
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
            sub = (sub - 1) & mask
    for mask in range(1, 1 << n):
        if dp[mask]:
            meta["subsets_built"] += 1
            if len(dp[mask]) > cap_per_subset:
                dp[mask] = _prune_values(dp[mask], cap_per_subset)
            meta["values_kept"] += len(dp[mask])
    return dp, meta


def search_dp_target(
    operands: list[_Operand],
    expected: Fraction,
    *,
    cap_per_subset: int = _DEFAULT_DP_CAP,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> tuple[_Expr | None, dict[str, int]]:
    start = time.perf_counter()
    n = len(operands)
    dp: list[dict[Fraction, _Expr]] = [dict() for _ in range(1 << n)]
    for i, operand in enumerate(operands):
        expr = _Expr(value=operand.value, ast=("leaf", i))
        dp[1 << i][operand.value] = expr
        if _matches_target(expected, operand.value):
            meta = {"subsets_built": 1, "values_kept": 1}
            return expr, meta
    for mask in range(1, 1 << n):
        if time.perf_counter() - start > max_seconds:
            break
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other and dp[sub] and dp[other]:
                for aval, aexpr in dp[sub].items():
                    for bval, bexpr in dp[other].items():
                        if time.perf_counter() - start > max_seconds:
                            break
                        for op in ("+", "*"):
                            combined = _combine(aexpr, bexpr, op)
                            if combined is None:
                                continue
                            dp[mask].setdefault(combined.value, combined)
                            if _matches_target(expected, combined.value):
                                dp[mask] = _prune_values(dp[mask], cap_per_subset)
                                meta = {
                                    "subsets_built": 0,
                                    "values_kept": 0,
                                }
                                for mmask in range(1, 1 << n):
                                    if dp[mmask]:
                                        meta["subsets_built"] += 1
                                        if len(dp[mmask]) > cap_per_subset:
                                            dp[mmask] = _prune_values(dp[mmask], cap_per_subset)
                                        meta["values_kept"] += len(dp[mmask])
                                return combined, meta
                            if len(dp[mask]) > cap_per_subset:
                                dp[mask] = _prune_values(dp[mask], cap_per_subset)
                        for op in ("-", "/"):
                            combined = _combine(aexpr, bexpr, op)
                            if combined is not None:
                                dp[mask].setdefault(combined.value, combined)
                                if _matches_target(expected, combined.value):
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
                                    meta = {"subsets_built": 0, "values_kept": 0}
                                    for mmask in range(1, 1 << n):
                                        if dp[mmask]:
                                            meta["subsets_built"] += 1
                                            if len(dp[mmask]) > cap_per_subset:
                                                dp[mmask] = _prune_values(dp[mmask], cap_per_subset)
                                            meta["values_kept"] += len(dp[mmask])
                                    return combined, meta
                                if len(dp[mask]) > cap_per_subset:
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
                            combined = _combine(bexpr, aexpr, op)
                            if combined is not None:
                                dp[mask].setdefault(combined.value, combined)
                                if _matches_target(expected, combined.value):
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
                                    meta = {"subsets_built": 0, "values_kept": 0}
                                    for mmask in range(1, 1 << n):
                                        if dp[mmask]:
                                            meta["subsets_built"] += 1
                                            if len(dp[mmask]) > cap_per_subset:
                                                dp[mmask] = _prune_values(dp[mmask], cap_per_subset)
                                            meta["values_kept"] += len(dp[mmask])
                                    return combined, meta
                                if len(dp[mask]) > cap_per_subset:
                                    dp[mask] = _prune_values(dp[mask], cap_per_subset)
            sub = (sub - 1) & mask
    meta = {"subsets_built": 0, "values_kept": 0}
    for mmask in range(1, 1 << n):
        if dp[mmask]:
            meta["subsets_built"] += 1
            if len(dp[mmask]) > cap_per_subset:
                dp[mmask] = _prune_values(dp[mmask], cap_per_subset)
            meta["values_kept"] += len(dp[mmask])
    return None, meta


def _expected_from_constraints(text: str, constraints: list[dict[str, Any]] | None) -> Fraction | None:
    verifier = ArithmeticVerifier()
    return verifier._expected_from_constraints(text, constraints)  # type: ignore[attr-defined]


def _matches_target(expected: Fraction | None, candidate: Fraction) -> bool:
    if expected is None:
        return True
    if expected.denominator == 1:
        return candidate.denominator == 1 and candidate.numerator == expected.numerator
    cand_val = float(candidate)
    exp_val = float(expected)
    if not math.isfinite(cand_val) or not math.isfinite(exp_val):
        return False
    if math.isclose(cand_val, exp_val, rel_tol=1e-12, abs_tol=1e-9):
        return True
    return False


def _enumerate_expr_brute(
    operands: list[_Operand],
    *,
    depth: int,
    limit: int,
    max_seconds: float,
    meta: dict[str, Any],
) -> Iterable[_Expr]:
    start = time.perf_counter()
    leaves = [_Expr(value=op.value, ast=("leaf", idx)) for idx, op in enumerate(operands)]
    levels: list[list[_Expr]] = [leaves]
    for expr in leaves:
        meta["candidates_tried"] += 1
        yield expr
        if meta["candidates_tried"] >= limit or time.perf_counter() - start > max_seconds:
            return
    for d in range(2, max(2, depth + 1)):
        next_level: list[_Expr] = []
        for left_depth in range(1, d):
            right_depth = d - left_depth
            lefts = levels[left_depth - 1]
            rights = levels[right_depth - 1]
            for aexpr in lefts:
                for bexpr in rights:
                    if time.perf_counter() - start > max_seconds or meta["candidates_tried"] >= limit:
                        return
                    for op in ("+", "*"):
                        combined = _combine(aexpr, bexpr, op)
                        if combined is None:
                            continue
                        next_level.append(combined)
                        meta["candidates_tried"] += 1
                        yield combined
                        if meta["candidates_tried"] >= limit:
                            return
                    for op in ("-", "/"):
                        combined = _combine(aexpr, bexpr, op)
                        if combined is not None:
                            next_level.append(combined)
                            meta["candidates_tried"] += 1
                            yield combined
                            if meta["candidates_tried"] >= limit:
                                return
                        combined = _combine(bexpr, aexpr, op)
                        if combined is not None:
                            next_level.append(combined)
                            meta["candidates_tried"] += 1
                            yield combined
                            if meta["candidates_tried"] >= limit:
                                return
        levels.append(next_level)


def _brute_search(
    operands: list[_Operand],
    expected: Fraction | None,
    *,
    depth: int,
    limit: int,
    max_seconds: float,
    meta: dict[str, Any],
) -> _Expr | None:
    for expr in _enumerate_expr_brute(
        operands,
        depth=depth,
        limit=limit,
        max_seconds=max_seconds,
        meta=meta,
    ):
        if _matches_target(expected, expr.value):
            return expr
    return None


def build_ptv1_proof_from_expr(
    expr: _Expr,
    operands: list[_Operand],
    *,
    base_program: Program | None = None,
) -> list[str]:
    insts: list[Instruction] = []
    temp_counter = 0

    if base_program is not None:
        for inst in base_program.instructions:
            if inst.opcode == "EXTRACT_INT":
                insts.append(inst)

    def _leaf_literal(idx: int) -> int | str:
        return operands[idx].literal

    def _new_temp() -> str:
        nonlocal temp_counter
        temp_counter += 1
        return f"t{temp_counter}"

    def _compile(node: Any, *, is_root: bool = False) -> int | str:
        if node[0] == "leaf":
            return _leaf_literal(int(node[1]))
        op, left, right = node
        a = _compile(left)
        b = _compile(right)
        dest = "result" if is_root else _new_temp()
        insts.append(Instruction(opcode="APPLY_ARITH", args={"a": a, "b": b, "op": op}, dest=dest))
        return dest

    result_ref = _compile(expr.ast, is_root=True)
    insts.append(Instruction(opcode="EMIT_NUM", args={"value": result_ref}))
    return program_to_tokens(Program(insts))


def repair_math_cegis(
    text: str,
    constraints: list[dict[str, Any]] | None,
    program: Program | None = None,
    *,
    max_nums: int = 8,
    depth: int = _DEFAULT_DEPTH,
    limit: int = _DEFAULT_LIMIT,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
    wallclock_ms: int | None = None,
    dp_value_cap_per_subset: int = _DEFAULT_DP_CAP,
    cegis_mode: str = "brute",
) -> tuple[list[str], dict[str, Any]] | None:
    start = time.perf_counter()
    operands = extract_operands(text, program, max_nums=max_nums)
    if not operands:
        return None
    mode = (cegis_mode or "brute").strip().lower()
    if mode not in {"brute", "dp"}:
        raise ValueError(f"Unknown cegis_mode: {cegis_mode}")
    expected = _expected_from_constraints(text, constraints)
    if expected is None:
        expected_val = eval_math_expression(text)
        if expected_val is not None:
            expected = expected_val
    meta = {
        "repair_cegis_mode": mode,
        "repair_cegis_kind": "",
        "repair_cegis_depth": int(depth),
        "repair_cegis_states": 0,
        "repair_cegis_nums_used": len(operands),
        "candidates_tried": 0,
        "depth": int(depth),
        "found": False,
        "repair_cegis_hit": False,
        "repair_cegis_subsets_built": 0,
        "repair_cegis_values_kept": 0,
        "repair_cegis_time_ms": 0,
        "repair_cegis_wallclock_ms": int(wallclock_ms) if wallclock_ms is not None else 0,
    }
    if wallclock_ms is not None and wallclock_ms >= 0:
        max_seconds = float(wallclock_ms) / 1000.0
    verifier = ArithmeticVerifier()
    if mode == "brute":
        meta["repair_cegis_kind"] = "frac_depth4"
        meta["repair_cegis_depth"] = int(depth)
        expr = _brute_frac_depth4(
            operands,
            expected,
            depth=depth,
            limit=limit,
            max_seconds=max_seconds,
            cap_per_depth=_DEFAULT_BRUTE_CAP,
            meta=meta,
        )
        if expr is not None:
            tokens = build_ptv1_proof_from_expr(expr, operands, base_program=program)
            cand_out = runtime_solve(text, constraints or [], tokens)
            if cand_out is not None:
                res = verifier.verify(text, None, cand_out, constraints=constraints)
                if res.valid:
                    meta["found"] = True
                    meta["repair_cegis_hit"] = True
                    meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
                    return tokens, meta
        meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
        return None

    if expected is not None:
        expr, dp_meta = search_dp_target(
            operands,
            expected,
            cap_per_subset=dp_value_cap_per_subset,
            max_seconds=max_seconds,
        )
        meta["repair_cegis_subsets_built"] = int(dp_meta.get("subsets_built", 0))
        meta["repair_cegis_values_kept"] = int(dp_meta.get("values_kept", 0))
        if expr is not None:
            tokens = build_ptv1_proof_from_expr(expr, operands, base_program=program)
            cand_out = runtime_solve(text, constraints or [], tokens)
            if cand_out is not None:
                res = verifier.verify(text, None, cand_out, constraints=constraints)
                if res.valid:
                    meta["found"] = True
                    meta["repair_cegis_hit"] = True
                    meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
                    return tokens, meta
    dp, dp_meta = enumerate_expr_dp(
        operands,
        cap_per_subset=dp_value_cap_per_subset,
        max_seconds=max_seconds,
    )
    meta["repair_cegis_subsets_built"] = int(dp_meta.get("subsets_built", 0))
    meta["repair_cegis_values_kept"] = int(dp_meta.get("values_kept", 0))
    for mask in range(1, len(dp)):
        if time.perf_counter() - start > max_seconds:
            break
        values = dp[mask]
        if not values:
            continue
        ordered = sorted(values.items(), key=lambda item: _priority_key(item[0]))
        for value, expr in ordered:
            if time.perf_counter() - start > max_seconds:
                break
            meta["candidates_tried"] += 1
            if limit and meta["candidates_tried"] > limit:
                break
            if not _matches_target(expected, value):
                continue
            tokens = build_ptv1_proof_from_expr(expr, operands)
            cand_out = runtime_solve(text, constraints or [], tokens)
            if cand_out is None:
                continue
            res = verifier.verify(text, None, cand_out, constraints=constraints)
            if res.valid:
                meta["found"] = True
                meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
                return tokens, meta
        if limit and meta["candidates_tried"] > limit:
            break
    meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
    return None


def _target_to_fraction(target: Any) -> Fraction | None:
    if target is None:
        return None
    if isinstance(target, Fraction):
        return target
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        return Fraction(target, 1)
    if isinstance(target, float):
        if not math.isfinite(target):
            return None
        return Fraction(target).limit_denominator(_MAX_DENOM)
    if isinstance(target, str):
        text = target.strip()
        if _FRACTION_RE.match(text):
            parts = text.replace(" ", "").split("/", 1)
            if len(parts) == 2:
                try:
                    num = int(parts[0])
                    den = int(parts[1])
                except ValueError:
                    return None
                if den != 0:
                    return Fraction(num, den)
        if _DECIMAL_RE.match(text):
            frac = _fraction_from_decimal(text)
            if frac is not None:
                return frac
        if _INT_RE.match(text):
            try:
                return Fraction(int(text), 1)
            except ValueError:
                return None
    return None


def repair_math_cegis_target(
    text: str,
    target: Any,
    program: Program | None = None,
    *,
    max_nums: int = 8,
    depth: int = _DEFAULT_DEPTH,
    limit: int = _DEFAULT_LIMIT,
    max_seconds: float = _DEFAULT_MAX_SECONDS,
) -> tuple[list[str], dict[str, Any]] | None:
    operands = extract_operands(text, program, max_nums=max_nums)
    if not operands:
        return None
    expected = _target_to_fraction(target)
    if expected is None:
        return None
    meta = {
        "repair_cegis_mode": "brute",
        "repair_cegis_kind": "frac_depth4",
        "repair_cegis_depth": int(depth),
        "repair_cegis_states": 0,
        "repair_cegis_nums_used": len(operands),
        "candidates_tried": 0,
        "found": False,
        "repair_cegis_hit": False,
        "repair_cegis_time_ms": 0,
    }
    start = time.perf_counter()
    expr = _brute_frac_depth4(
        operands,
        expected,
        depth=depth,
        limit=limit,
        max_seconds=max_seconds,
        cap_per_depth=_DEFAULT_BRUTE_CAP,
        meta=meta,
    )
    if expr is None:
        meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
        return None
    tokens = build_ptv1_proof_from_expr(expr, operands, base_program=program)
    meta["found"] = True
    meta["repair_cegis_hit"] = True
    meta["repair_cegis_time_ms"] = int((time.perf_counter() - start) * 1000)
    return tokens, meta
