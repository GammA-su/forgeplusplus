from __future__ import annotations

from fc.util.repair_math import repair_math_apply_arith_operator


def test_repair_math_apply_arith_operator_generates_candidates_for_val_slots() -> None:
    tokens = [
        "<BOS>",
        "BEGIN",
        "OP",
        "EXTRACT_INT",
        "DEST",
        "STR:a",
        "ARG",
        "index",
        "VAL",
        "INT:0",
        "OP",
        "EXTRACT_INT",
        "DEST",
        "STR:b",
        "ARG",
        "index",
        "VAL",
        "INT:1",
        "OP",
        "APPLY_ARITH",
        "DEST",
        "STR:result",
        "ARG",
        "a",
        "VAL",
        "STR:a",
        "ARG",
        "b",
        "VAL",
        "STR:b",
        "ARG",
        "op",
        "VAL",
        "STR:-",
        "OP",
        "EMIT_NUM",
        "ARG",
        "value",
        "VAL",
        "STR:result",
        "END",
        "<EOS>",
    ]
    candidates = repair_math_apply_arith_operator(
        tokens,
        max_edits=2,
        max_candidates=32,
        allowed_tokens={"STR:+", "STR:-", "ADD"},
    )
    assert len(candidates) <= 32
    last_apply = max(i for i, tok in enumerate(tokens) if tok == "APPLY_ARITH")
    val_slots = [
        i + 1
        for i in range(last_apply + 1, len(tokens))
        if tokens[i] == "VAL" and i + 1 < len(tokens)
    ]
    touched = set()
    for cand, _ in candidates:
        diffs = [idx for idx, (a, b) in enumerate(zip(tokens, cand)) if a != b]
        assert 1 <= len(diffs) <= 2
        touched.update(diffs)
    assert set(val_slots[:4]).issuperset(touched)


def test_repair_math_apply_arith_operator_can_swap_operand() -> None:
    tokens = [
        "<BOS>",
        "BEGIN",
        "OP",
        "APPLY_ARITH",
        "DEST",
        "STR:result",
        "ARG",
        "a",
        "VAL",
        "a",
        "ARG",
        "b",
        "VAL",
        "b",
        "ARG",
        "op",
        "VAL",
        "STR:+",
        "END",
        "<EOS>",
    ]
    candidates = repair_math_apply_arith_operator(
        tokens,
        max_edits=2,
        max_candidates=64,
        allowed_tokens={"a", "b", "c", "STR:+", "STR:-"},
    )
    assert len(candidates) <= 64
    assert any(cand[0][9] != "a" or cand[0][13] != "b" for cand in candidates)
