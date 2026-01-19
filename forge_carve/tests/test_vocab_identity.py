from __future__ import annotations

import pytest

from fc.util.vocab_identity import assert_vocab_match, vocab_identity


def test_vocab_identity_mismatch_raises() -> None:
    a = {"<PAD>": 0, "OP": 1}
    b = {"<PAD>": 0, "OP": 2}
    with pytest.raises(ValueError) as excinfo:
        assert_vocab_match(a, b, expected_label="ckpt", actual_label="disk")
    msg = str(excinfo.value)
    assert "vocab_mismatch" in msg
    assert "first_diff_index" in msg


def test_vocab_identity_hash_stable() -> None:
    mapping = {"<PAD>": 0, "OP": 1}
    ident = vocab_identity(mapping)
    assert ident.sha256
    assert ident.tokens_by_id[0] == "<PAD>"
