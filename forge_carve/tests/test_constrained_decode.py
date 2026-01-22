from __future__ import annotations

import torch

from fc.dsl.tokens import OPCODES, build_default_vocab
from fc.util.constrained_decode import greedy_decode_with_opcode_mask
from fc.util.proof_repair import scan_tokens
from fc.util.runtime_solve import runtime_solve


def test_opcode_mask_forces_valid_opcode() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    # Sequence: OP then an opcode position, followed by a minimal valid program.
    op_id = vocab.encode("OP")
    bad_id = vocab.encode("b")
    good_id = vocab.encode("EXTRACT_INT")
    dest_id = vocab.encode("DEST")
    str_a = vocab.encode("STR:a")
    end_id = vocab.encode("END")
    eos_id = vocab.encode("<EOS>")
    logits = torch.full((6, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, bad_id] = 5.0
    logits[1, good_id] = 4.0
    logits[2, dest_id] = 5.0
    logits[3, str_a] = 5.0
    logits[4, end_id] = 5.0
    logits[5, eos_id] = 5.0

    decoded = greedy_decode_with_opcode_mask(logits, vocab, enforce_opcode=True)
    assert vocab.decode(decoded[0]) == "OP"
    assert vocab.decode(decoded[1]) in OPCODES
    assert vocab.decode(decoded[1]) == "EXTRACT_INT"


def test_domain_constrains_emit_opcode() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    op_id = vocab.encode("OP")
    emit_id = vocab.encode("EMIT")
    emit_num_id = vocab.encode("EMIT_NUM")
    arg_id = vocab.encode("ARG")
    key_id = vocab.encode("value")
    val_id = vocab.encode("VAL")
    str_res = vocab.encode("STR:result")
    end_id = vocab.encode("END")
    logits = torch.full((7, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, emit_id] = 5.0
    logits[1, emit_num_id] = 4.0
    logits[2, arg_id] = 5.0
    logits[3, key_id] = 5.0
    logits[4, val_id] = 5.0
    logits[5, str_res] = 5.0
    logits[6, end_id] = 5.0

    decoded = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=7,
        domain_tag="[MATH]",
    )
    assert vocab.decode(decoded[1]) == "EMIT_NUM"


def test_math_domain_blocks_csp_opcodes() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    op_id = vocab.encode("OP")
    topo_id = vocab.encode("APPLY_TOPO")
    bind_id = vocab.encode("BIND")
    arg_id = vocab.encode("ARG")
    key_id = vocab.encode("value")
    val_id = vocab.encode("VAL")
    str_res = vocab.encode("STR:result")
    end_id = vocab.encode("END")
    logits = torch.full((7, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, topo_id] = 5.0
    logits[1, bind_id] = 4.0
    logits[2, arg_id] = 5.0
    logits[3, key_id] = 5.0
    logits[4, val_id] = 5.0
    logits[5, str_res] = 5.0
    logits[6, end_id] = 5.0
    decoded = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=7,
        domain_tag="[MATH]",
    )
    assert vocab.decode(decoded[1]) == "BIND"


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
        assert ok or (failure is not None and failure.reason == "eof"), failure
        _, err = runtime_solve("[MATH] Compute: 1 + 2.", [], tokens, return_error=True)
        if err is not None and err.code == "PARSE_FAIL":
            assert "reason=eof" in err.detail


def test_decode_stops_after_emit() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    op_id = vocab.encode("OP")
    emit_id = vocab.encode("EMIT")
    logits = torch.full((8, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, emit_id] = 5.0
    decoded = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        min_tokens=0,
        max_tokens=8,
        stop_on_emit=True,
    )
    assert len(decoded) >= 2
    assert vocab.decode(decoded[1]) == "EMIT"


def test_max_tokens_controls_decode_length() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    logits = torch.randn(128, vocab_size)
    out_64 = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=False,
        min_tokens=0,
        max_tokens=64,
        stop_on_emit=False,
    )
    out_128 = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=False,
        min_tokens=0,
        max_tokens=128,
        stop_on_emit=False,
    )
    assert len(out_64) <= 64
    assert len(out_128) <= 128
    assert len(out_64) > 0
    assert len(out_128) > 0


def test_decode_terminates_at_eos() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    eos_id = vocab.encode("<EOS>")
    logits = torch.full((6, vocab_size), -10.0)
    logits[0, eos_id] = 5.0
    out = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=False,
        max_tokens=6,
    )
    assert len(out) == 1
    assert vocab.decode(out[0]) == "<EOS>"


def test_schema_emit_requires_fields_dict() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    op_id = vocab.encode("OP")
    emit_id = vocab.encode("EMIT")
    arg_id = vocab.encode("ARG")
    fields_id = vocab.encode("fields")
    val_id = vocab.encode("VAL")
    dict_start_id = vocab.encode("DICT_START")
    dict_end_id = vocab.encode("DICT_END")
    end_id = vocab.encode("END")
    eos_id = vocab.encode("<EOS>")
    logits = torch.full((8, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, emit_id] = 5.0
    logits[2, end_id] = 5.0
    logits[2, arg_id] = 4.0
    logits[3, fields_id] = 5.0
    logits[4, val_id] = 5.0
    logits[5, dict_start_id] = 4.0
    logits[5, eos_id] = 5.0
    logits[6, dict_end_id] = 5.0
    logits[7, end_id] = 5.0

    decoded = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=8,
        domain_tag="[SCHEMA]",
    )
    tokens = [vocab.decode(i) for i in decoded]
    assert tokens[:6] == ["OP", "EMIT", "ARG", "fields", "VAL", "DICT_START"]


def test_csp_allows_schedule_ops() -> None:
    vocab = build_default_vocab()
    vocab_size = len(vocab.token_to_id)
    op_id = vocab.encode("OP")
    topo_id = vocab.encode("APPLY_TOPO")
    dest_id = vocab.encode("DEST")
    str_id = vocab.encode("STR:a")
    end_id = vocab.encode("END")
    logits = torch.full((5, vocab_size), -10.0)
    logits[0, op_id] = 5.0
    logits[1, topo_id] = 5.0
    logits[2, dest_id] = 5.0
    logits[3, str_id] = 5.0
    logits[4, end_id] = 5.0
    decoded = greedy_decode_with_opcode_mask(
        logits,
        vocab,
        enforce_opcode=True,
        max_tokens=5,
        domain_tag="[CSP]",
    )
    assert vocab.decode(decoded[1]) == "APPLY_TOPO"
