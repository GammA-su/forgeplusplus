from __future__ import annotations

from fc.dsl.tokens import build_default_vocab
from fc.train.data import Example, TextVocab, collate_batch


def test_collate_batch_uses_gold_tokens() -> None:
    vocab = build_default_vocab()
    text_vocab = TextVocab.build(["[MATH] Compute: 1 + 2."])
    proof_tokens = ["<BOS>", "BEGIN", "OP", "EXTRACT_INT", "DEST", "STR:a", "END", "<EOS>"]
    gold_tokens = ["<BOS>", "BEGIN", "END", "<EOS>"]
    ex = Example(
        id="math_0",
        domain="math",
        domain_tag="[MATH]",
        x="[MATH] Compute: 1 + 2.",
        y=3,
        proof={"dsl": "PTv1", "tokens": proof_tokens},
        proof_tokens_gold=gold_tokens,
    )
    batch = [ex]
    proof_batch = collate_batch(batch, text_vocab, vocab, max_text_len=16, max_prog_len=8, proof_source="proof")
    gold_batch = collate_batch(batch, text_vocab, vocab, max_text_len=16, max_prog_len=8, proof_source="gold")
    proof_ids = proof_batch["program_ids"][0].tolist()
    gold_ids = gold_batch["program_ids"][0].tolist()
    assert proof_ids[2] == vocab.encode("OP")
    assert gold_ids[2] == vocab.encode("END")
