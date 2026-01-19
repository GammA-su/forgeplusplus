from __future__ import annotations

import ast
import re
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


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported math expression")


def eval_math_expression(text: str) -> float | None:
    expr = _normalize_text(text)
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    try:
        return _eval_node(tree.body)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
