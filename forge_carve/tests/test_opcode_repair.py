from __future__ import annotations

import torch

from fc.dsl.tokens import OPCODES, build_default_vocab
from fc.util.opcode_repair import repair_invalid_opcodes


def test_repair_invalid_opcode_selects_valid_op() -> None:
    vocab = build_default_vocab()
    tokens = ["<BOS>", "BEGIN", "OP", "b", "END", "<EOS>"]
    pred_ids = [vocab.encode(tok) for tok in tokens]
    vocab_size = len(vocab.token_to_id)
    logits = torch.full((len(pred_ids), vocab_size), -10.0)
    bad_id = vocab.encode("b")
    good_id = vocab.encode("EXTRACT_INT")
    logits[3, bad_id] = 5.0
    logits[3, good_id] = 4.0

    repaired_ids, did_repair = repair_invalid_opcodes(pred_ids, logits, vocab, k=5)

    assert did_repair
    assert vocab.decode(repaired_ids[3]) in OPCODES
    assert vocab.decode(repaired_ids[3]) == "EXTRACT_INT"
