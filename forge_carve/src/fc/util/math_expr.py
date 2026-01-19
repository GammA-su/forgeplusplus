from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

_DOMAIN_TAG_RE = re.compile(r"^\s*\[[A-Z]+\]\s*")
_NON_EXPR_RE = re.compile(r"[^0-9\.\+\-\*/\(\)\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    lowered = _DOMAIN_TAG_RE.sub("", text.strip()).lower()
    lowered = re.sub(r"\b(divided by|divide by|divided|divide|over)\b", "/", lowered)
    lowered = re.sub(r"\b(times|multiply|multiplied by)\b", "*", lowered)
    lowered = re.sub(r"\b(plus|add)\b", "+", lowered)
    lowered = re.sub(r"\b(minus|subtract)\b", "-", lowered)
    # Strip sentence punctuation periods while preserving decimal points.
    lowered = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", lowered)
    cleaned = _NON_EXPR_RE.sub(" ", lowered)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned


_OP_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
_PREFIX_PRECEDENCE = 3


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            j = i + 1
            saw_dot = ch == "."
            while j < len(expr):
                nxt = expr[j]
                if nxt.isdigit():
                    j += 1
                    continue
                if nxt == "." and not saw_dot:
                    saw_dot = True
                    j += 1
                    continue
                break
            tokens.append(expr[i:j])
            i = j
            continue
        return []
    return tokens


def _parse_number(tok: str) -> Fraction:
    if tok.count(".") > 1:
        raise ValueError("bad number")
    if tok.startswith("."):
        whole = "0"
        frac = tok[1:]
    elif tok.endswith("."):
        whole = tok[:-1]
        frac = ""
    elif "." in tok:
        whole, frac = tok.split(".", 1)
    else:
        whole, frac = tok, ""
    if not whole and not frac:
        raise ValueError("bad number")
    if not whole:
        whole = "0"
    if not frac:
        return Fraction(int(whole), 1)
    return Fraction(int(whole + frac), 10 ** len(frac))


def _parse_expr(tokens: list[str]) -> Fraction:
    idx = 0

    def parse_expression(min_prec: int = 0) -> Fraction:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end")
        tok = tokens[idx]
        if tok in ("+", "-"):
            idx += 1
            val = parse_expression(_PREFIX_PRECEDENCE)
            left = val if tok == "+" else -val
        elif tok == "(":
            idx += 1
            left = parse_expression(0)
            if idx >= len(tokens) or tokens[idx] != ")":
                raise ValueError("missing )")
            idx += 1
        else:
            left = _parse_number(tok)
            idx += 1
        while idx < len(tokens):
            op = tokens[idx]
            prec = _OP_PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            idx += 1
            right = parse_expression(prec + 1)
            if op == "+":
                left = left + right
            elif op == "-":
                left = left - right
            elif op == "*":
                left = left * right
            elif op == "/":
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                left = left / right
        return left

    result = parse_expression(0)
    if idx != len(tokens):
        raise ValueError("trailing tokens")
    return result


def eval_math_expression(text: str) -> Fraction | None:
    expr = _normalize_text(text)
    if not expr:
        return None


def parse_math_expression_ast(text: str) -> dict[str, Any] | None:
    expr = _normalize_text(text)
    if not expr:
        return None
    tokens = _tokenize(expr)
    if not tokens:
        return None
    idx = 0
    num_idx = 0

    def parse_number_node(sign: int = 1) -> dict[str, Any]:
        nonlocal idx, num_idx
        if idx >= len(tokens):
            raise ValueError("unexpected end")
        tok = tokens[idx]
        if tok in "+-":
            raise ValueError("unexpected sign")
        val = _parse_number(tok)
        if sign < 0:
            val = -val
        node = {"kind": "leaf", "idx": num_idx, "value": val}
        num_idx += 1
        idx += 1
        return node

    def parse_expression(min_prec: int = 0) -> dict[str, Any]:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end")
        tok = tokens[idx]
        if tok in ("+", "-"):
            sign = -1 if tok == "-" else 1
            idx += 1
            if idx >= len(tokens):
                raise ValueError("dangling unary")
            if tokens[idx] == "(":
                raise ValueError("unary on subexpression unsupported")
            left = parse_number_node(sign)
        elif tok == "(":
            idx += 1
            left = parse_expression(0)
            if idx >= len(tokens) or tokens[idx] != ")":
                raise ValueError("missing )")
            idx += 1
        else:
            left = parse_number_node(1)
        while idx < len(tokens):
            op = tokens[idx]
            prec = _OP_PRECEDENCE.get(op)
            if prec is None or prec < min_prec:
                break
            idx += 1
            right = parse_expression(prec + 1)
            left = {"kind": "op", "op": op, "left": left, "right": right}
        return left

    try:
        ast = parse_expression(0)
        if idx != len(tokens):
            return None
        return ast
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    try:
        tokens = _tokenize(expr)
        if not tokens:
            return None
        return _parse_expr(tokens)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
