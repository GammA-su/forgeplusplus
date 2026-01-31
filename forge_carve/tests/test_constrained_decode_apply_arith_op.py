from __future__ import annotations

import torch

from fc.dsl.tokens import TokenVocab
from fc.util import constrained_decode as cd


def _vocab_from_tokens(tokens: list[str]) -> TokenVocab:
    token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return TokenVocab(token_to_id=token_to_id, id_to_token=id_to_token)


def test_constrained_decode_apply_arith_op_mask(monkeypatch) -> None:
    monkeypatch.setenv("FC_MATH_APPLY_ARITH_OP_MASK", "1")
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "BEGIN",
        "END",
        "OP",
        "DEST",
        "ARG",
        "VAL",
        "APPLY_ARITH",
        "ADD",
        "a",
        "b",
        "op",
        "STR:tmp",
        "STR:a",
        "STR:b",
        "STR:+",
        "STR:noop",
    ]
    vocab = _vocab_from_tokens(tokens)
    vocab_size = len(vocab.token_to_id)

    seq = [
        "OP",
        "APPLY_ARITH",
        "DEST",
        "STR:tmp",
        "ARG",
        "a",
        "VAL",
        None,  # operator value slot per masking rule
        "ARG",
        "END",
        "<EOS>",
    ]
    logits = torch.full((len(seq), vocab_size), -10.0)
    for idx, tok in enumerate(seq):
        if tok is None:
            continue
        logits[idx, vocab.token_to_id[tok]] = 5.0
    op_idx = seq.index(None)
    logits[op_idx, vocab.token_to_id["STR:noop"]] = 7.0
    logits[op_idx, vocab.token_to_id["ADD"]] = 9.0
    logits[op_idx, vocab.token_to_id["STR:+"]] = 6.0
    logits[op_idx + 1, vocab.token_to_id["ARG"]] = 4.0
    logits[op_idx + 1, vocab.token_to_id["ADD"]] = 8.0

    decoded = cd.greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=len(seq),
        domain_tag="[MATH]",
    )
    assert vocab.decode(decoded[op_idx]) == "STR:+"
    assert vocab.decode(decoded[op_idx + 1]) == "ARG"

    token_sets = cd._build_token_sets(vocab)
    allowed = cd._resolve_apply_arith_op_ids(token_sets)
    assert allowed == {vocab.token_to_id["STR:+"]}


def test_apply_arith_mask_off_by_default() -> None:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "BEGIN",
        "END",
        "OP",
        "DEST",
        "ARG",
        "VAL",
        "APPLY_ARITH",
        "ADD",
        "a",
        "STR:tmp",
        "STR:+",
    ]
    vocab = _vocab_from_tokens(tokens)
    vocab_size = len(vocab.token_to_id)
    seq = ["OP", "APPLY_ARITH", "DEST", "STR:tmp", "ARG", "a", "VAL", None, "ARG", "END", "<EOS>"]
    logits = torch.full((len(seq), vocab_size), -10.0)
    for idx, tok in enumerate(seq):
        if tok is None:
            continue
        logits[idx, vocab.token_to_id[tok]] = 5.0
    op_idx = seq.index(None)
    logits[op_idx, vocab.token_to_id["ADD"]] = 7.0
    logits[op_idx, vocab.token_to_id["STR:+"]] = 6.0
    decoded = cd.greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=len(seq),
        domain_tag="[MATH]",
    )
    assert vocab.decode(decoded[op_idx]) == "STR:+"
