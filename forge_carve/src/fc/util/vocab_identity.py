from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VocabIdentity:
    token_to_id: dict[str, int]
    tokens_by_id: list[str]
    sha256: str


@dataclass(frozen=True)
class VocabMismatch:
    index: int | None
    expected_token: str | None
    actual_token: str | None
    expected_hash: str
    actual_hash: str


def _stable_json(mapping: dict[str, int]) -> str:
    return json.dumps(mapping, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def vocab_identity(mapping: dict[str, int]) -> VocabIdentity:
    payload = _stable_json(mapping)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    max_id = max(mapping.values(), default=-1)
    tokens_by_id = ["<MISSING>"] * (max_id + 1)
    for tok, idx in mapping.items():
        if 0 <= idx < len(tokens_by_id):
            tokens_by_id[idx] = tok
    return VocabIdentity(token_to_id=mapping, tokens_by_id=tokens_by_id, sha256=digest)


def compare_vocab(expected: dict[str, int], actual: dict[str, int]) -> VocabMismatch | None:
    expected_id = vocab_identity(expected)
    actual_id = vocab_identity(actual)
    if expected_id.sha256 == actual_id.sha256:
        return None
    min_len = min(len(expected_id.tokens_by_id), len(actual_id.tokens_by_id))
    diff_idx = None
    for i in range(min_len):
        if expected_id.tokens_by_id[i] != actual_id.tokens_by_id[i]:
            diff_idx = i
            break
    if diff_idx is None:
        diff_idx = min_len if len(expected_id.tokens_by_id) != len(actual_id.tokens_by_id) else None
    exp_tok = expected_id.tokens_by_id[diff_idx] if diff_idx is not None and diff_idx < len(expected_id.tokens_by_id) else None
    act_tok = actual_id.tokens_by_id[diff_idx] if diff_idx is not None and diff_idx < len(actual_id.tokens_by_id) else None
    return VocabMismatch(
        index=diff_idx,
        expected_token=exp_tok,
        actual_token=act_tok,
        expected_hash=expected_id.sha256,
        actual_hash=actual_id.sha256,
    )


def assert_vocab_match(
    expected: dict[str, int],
    actual: dict[str, int],
    *,
    expected_label: str,
    actual_label: str,
) -> None:
    mismatch = compare_vocab(expected, actual)
    if mismatch is None:
        return
    detail = (
        f"vocab_mismatch expected={expected_label} actual={actual_label} "
        f"expected_sha256={mismatch.expected_hash} actual_sha256={mismatch.actual_hash}"
    )
    if mismatch.index is not None:
        detail += (
            f" first_diff_index={mismatch.index} "
            f"expected_token={mismatch.expected_token!r} actual_token={mismatch.actual_token!r}"
        )
    raise ValueError(detail)
