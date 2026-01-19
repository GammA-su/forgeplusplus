from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from fc.dsl.tokens import ARG_KEYS, OPCODES, TokenVocab


@dataclass(frozen=True)
class ParseFailure:
    pos: int
    expected_class: str
    expected_tokens: list[str]
    got_token: str | None
    reason: str


@dataclass(frozen=True)
class RepairResult:
    tokens: list[str]
    repairs: int
    success: bool
    failure: ParseFailure | None


def _is_value_start(tok: str) -> bool:
    return (
        tok.startswith("STR:")
        or tok.startswith("INT:")
        or tok.startswith("FLOAT:")
        or tok.startswith("BOOL:")
        or tok in {"LIST_START", "DICT_START"}
    )


def _parse_value(tokens: list[str], idx: int) -> tuple[int, ParseFailure | None]:
    if idx >= len(tokens):
        return idx, ParseFailure(idx, "VALUE", [], None, "eof")
    tok = tokens[idx]
    if tok.startswith(("STR:", "INT:", "BOOL:")):
        return idx + 1, None
    if tok == "LIST_START":
        idx += 1
        while True:
            if idx >= len(tokens):
                return idx, ParseFailure(idx, "LIST_END", ["LIST_END"], None, "eof")
            if tokens[idx] == "LIST_END":
                return idx + 1, None
            if tokens[idx] == "SEP":
                idx += 1
                continue
            idx, failure = _parse_value(tokens, idx)
            if failure is not None:
                return idx, failure
    if tok == "DICT_START":
        idx += 1
        while True:
            if idx >= len(tokens):
                return idx, ParseFailure(idx, "DICT_END", ["DICT_END"], None, "eof")
            if tokens[idx] == "DICT_END":
                return idx + 1, None
            if tokens[idx] == "SEP":
                idx += 1
                continue
            idx, failure = _parse_value(tokens, idx)
            if failure is not None:
                return idx, failure
            if idx < len(tokens) and tokens[idx] == "SEP":
                idx += 1
            idx, failure = _parse_value(tokens, idx)
            if failure is not None:
                return idx, failure
            if idx < len(tokens) and tokens[idx] == "SEP":
                idx += 1
        # unreachable
    return idx, ParseFailure(idx, "VALUE", [], tok, "bad_value")


def scan_tokens(tokens: list[str]) -> tuple[bool, ParseFailure | None, int]:
    i = 0
    n = len(tokens)
    last_good_end = 0
    while i < n and tokens[i] in ("<BOS>", "BEGIN"):
        i += 1
    last_good_end = i
    while i < n:
        tok = tokens[i]
        if tok in ("END", "<EOS>"):
            return True, None, i + 1
        if tok != "OP":
            return (
                False,
                ParseFailure(i, "OP_OR_END", ["OP", "END", "<EOS>"], tok, "expected_op"),
                last_good_end,
            )
        if i + 1 >= n:
            return (
                False,
                ParseFailure(i + 1, "OPCODE", list(OPCODES), None, "eof"),
                last_good_end,
            )
        op_tok = tokens[i + 1]
        if op_tok not in OPCODES:
            return (
                False,
                ParseFailure(i + 1, "OPCODE", list(OPCODES), op_tok, "invalid_opcode"),
                last_good_end,
            )
        i += 2
        if i < n and tokens[i] == "DEST":
            if i + 1 >= n:
                return (
                    False,
                    ParseFailure(i + 1, "DEST_VALUE", [], None, "eof"),
                    last_good_end,
                )
            if not tokens[i + 1].startswith("STR:"):
                return (
                    False,
                    ParseFailure(i + 1, "DEST_VALUE", [], tokens[i + 1], "bad_dest"),
                    last_good_end,
                )
            i += 2
        while i < n and tokens[i] == "ARG":
            if i + 1 >= n:
                return (
                    False,
                    ParseFailure(i + 1, "ARG_KEY", list(ARG_KEYS), None, "eof"),
                    last_good_end,
                )
            key = tokens[i + 1]
            if key not in ARG_KEYS:
                return (
                    False,
                    ParseFailure(i + 1, "ARG_KEY", list(ARG_KEYS), key, "bad_key"),
                    last_good_end,
                )
            if i + 2 >= n:
                return (
                    False,
                    ParseFailure(i + 2, "VAL", ["VAL"], None, "eof"),
                    last_good_end,
                )
            if tokens[i + 2] != "VAL":
                return (
                    False,
                    ParseFailure(i + 2, "VAL", ["VAL"], tokens[i + 2], "bad_val"),
                    last_good_end,
                )
            val_idx = i + 3
            if val_idx >= n:
                return (
                    False,
                    ParseFailure(val_idx, "VALUE", [], None, "eof"),
                    last_good_end,
                )
            next_idx, failure = _parse_value(tokens, val_idx)
            if failure is not None:
                return False, failure, last_good_end
            i = next_idx
        last_good_end = i
    return True, None, last_good_end


def _token_matches_class(tok: str, cls: str) -> bool:
    if cls == "OPCODE":
        return tok in OPCODES
    if cls == "OP_OR_END":
        return tok in {"OP", "END", "<EOS>"}
    if cls == "ARG_KEY":
        return tok in ARG_KEYS
    if cls == "VAL":
        return tok == "VAL"
    if cls in {"DEST_VALUE", "VALUE"}:
        return _is_value_start(tok)
    if cls == "LIST_END":
        return tok == "LIST_END"
    if cls == "DICT_END":
        return tok == "DICT_END"
    return False


def repair_tokens(
    tokens: list[str],
    logits: torch.Tensor,
    vocab: TokenVocab,
    *,
    max_repairs: int = 5,
    k: int = 30,
    log_fn: Callable[[str], None] | None = None,
) -> RepairResult:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [seq_len, vocab]")
    n = len(tokens)
    logits_cpu = logits.detach().cpu()
    repairs = 0
    current = list(tokens)
    while repairs < max_repairs:
        ok, failure, _ = scan_tokens(current)
        if ok:
            return RepairResult(tokens=current, repairs=repairs, success=True, failure=None)
        if failure is None or failure.pos >= n:
            return RepairResult(tokens=current, repairs=repairs, success=False, failure=failure)
        pos = failure.pos
        row = logits_cpu[pos]
        k_eff = max(1, min(int(k), row.numel()))
        probs = torch.softmax(row, dim=-1)
        topk = torch.topk(probs, k_eff)
        candidates: list[tuple[str, float, int]] = []
        for prob, tok_id in zip(topk.values.tolist(), topk.indices.tolist()):
            tok = vocab.decode(int(tok_id))
            if _token_matches_class(tok, failure.expected_class):
                candidates.append((tok, float(prob), int(tok_id)))
        if log_fn:
            got = failure.got_token
            log_fn(f"  repair_pos={pos} expected={failure.expected_class} got={got}")
            cand_str = ", ".join([f"{tok}:{prob:.4g}" for tok, prob, _ in candidates]) if candidates else "<none>"
            log_fn(f"  repair_candidates={cand_str}")
        if not candidates:
            return RepairResult(tokens=current, repairs=repairs, success=False, failure=failure)
        tok, prob, tok_id = candidates[0]
        current[pos] = tok
        repairs += 1
        if log_fn:
            log_fn(f"  repair_chosen={tok} prob={prob:.4g}")
    ok, failure, _ = scan_tokens(current)
    return RepairResult(tokens=current, repairs=repairs, success=ok, failure=failure)


def longest_valid_prefix(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    for end in range(len(tokens), 0, -1):
        ok, _, _ = scan_tokens(tokens[:end])
        if ok:
            return tokens[:end]
    return []
