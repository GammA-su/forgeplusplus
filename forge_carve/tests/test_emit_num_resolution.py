from fc.util.runtime_solve import runtime_solve


def _emit_num_with_env_tokens(dest: str = "result") -> list[str]:
    return [
        "<BOS>",
        "BEGIN",
        "OP",
        "APPLY_ARITH",
        "DEST",
        f"STR:{dest}",
        "ARG",
        "a",
        "VAL",
        "INT:1",
        "ARG",
        "b",
        "VAL",
        "INT:2",
        "ARG",
        "op",
        "VAL",
        "STR:+",
        "OP",
        "EMIT_NUM",
        "ARG",
        "value",
        "VAL",
        "STR:result",
        "END",
        "<EOS>",
    ]


def test_emit_num_uses_env_value() -> None:
    tokens = _emit_num_with_env_tokens(dest="result")
    out, failure = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
    assert failure is None
    assert out == 3


def test_emit_num_falls_back_to_last_num() -> None:
    tokens = _emit_num_with_env_tokens(dest="tmp")
    out, failure = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
    assert failure is None
    assert out == 3


def test_emit_num_fails_without_numeric() -> None:
    tokens = [
        "<BOS>",
        "BEGIN",
        "OP",
        "EMIT_NUM",
        "ARG",
        "value",
        "VAL",
        "STR:result",
        "END",
        "<EOS>",
    ]
    out, failure = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
    assert out is None
    assert failure is not None
    assert failure.code in {"MISSING_VALUE", "RUNTIME_ERROR"}


def test_runtime_solve_ignores_non_numeric_index() -> None:
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
        "STR:a",
        "OP",
        "EMIT_NUM",
        "ARG",
        "value",
        "VAL",
        "STR:result",
        "END",
        "<EOS>",
    ]
    out, failure = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
    assert out is None
    assert failure is not None
