from fc.dsl.tokens import build_default_vocab


def test_prog_vocab_contains_ops() -> None:
    vocab = build_default_vocab()
    assert "EMIT_NUM" in vocab.token_to_id
    assert "EMIT_SCHEDULE" in vocab.token_to_id
