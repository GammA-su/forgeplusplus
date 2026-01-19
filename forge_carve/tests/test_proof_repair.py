from __future__ import annotations

import torch

from fc.dsl.tokens import build_default_vocab
from fc.util.proof_repair import repair_tokens, scan_tokens


def test_repair_tokens_fixes_invalid_opcode() -> None:
    vocab = build_default_vocab()
    tokens = ["<BOS>", "BEGIN", "OP", "b", "END", "<EOS>"]
    ok, failure, _ = scan_tokens(tokens)
    assert not ok
    assert failure is not None
    vocab_size = len(vocab.token_to_id)
    logits = torch.full((len(tokens), vocab_size), -10.0)
    bad_id = vocab.encode("b")
    good_id = vocab.encode("EXTRACT_INT")
    logits[3, bad_id] = 5.0
    logits[3, good_id] = 4.0
    result = repair_tokens(tokens, logits, vocab, max_repairs=1, k=5)
    assert result.success
    assert result.tokens[3] == "EXTRACT_INT"
