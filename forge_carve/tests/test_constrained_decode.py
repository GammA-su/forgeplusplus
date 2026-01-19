from __future__ import annotations

import torch

from fc.dsl.tokens import OPCODES, build_default_vocab
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.proof_repair import scan_tokens
from fc.util.runtime_solve import runtime_solve


def test_opcode_mask_forces_valid_opcode() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    # Sequence: OP then an opcode position.
    op_id = vocab.encode("OP")
    bad_id = vocab.encode("b")
    good_id = vocab.encode("EXTRACT_INT")
    logits = torch.full((2, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, bad_id] = 5.0
    logits[1, good_id] = 4.0

    decoded = greedy_decode_with_opcode_mask(logits, vocab, enforce_opcode=True)
    assert vocab.decode(decoded[0]) == "OP"
    assert vocab.decode(decoded[1]) in OPCODES
    assert vocab.decode(decoded[1]) == "EXTRACT_INT"


def test_constrained_decode_emits_parseable_programs() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    rng = torch.Generator().manual_seed(123)
    for _ in range(100):
        logits = torch.randn(32, vocab_size, generator=rng)
        ids = greedy_decode_with_opcode_mask(
            logits,
            vocab,
            enforce_opcode=True,
            min_tokens=2,
            max_tokens=32,
        )
        tokens = [vocab.decode(i) for i in ids]
        ok, failure, _ = scan_tokens(tokens)
        assert ok, failure
        _, err = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
        assert err is None or err.code != "PARSE_FAIL"
