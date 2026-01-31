from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fc.util.constrained_decode import _build_token_sets, _resolve_apply_arith_op_ids


def _find_apply_arith_val_slots(tokens: list[str]) -> list[int]:
    last_apply = -1
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] == "APPLY_ARITH":
            last_apply = idx
            break
    if last_apply < 0:
        return []
    end_idx = len(tokens)
    for idx in range(last_apply + 1, len(tokens)):
        if tokens[idx] in {"END", "<EOS>"}:
            end_idx = idx
            break
    slots: list[int] = []
    for idx in range(last_apply + 1, end_idx):
        if tokens[idx] == "VAL" and idx + 1 < end_idx:
            slots.append(idx + 1)
    return slots


def _select_operand_slots(tokens: list[str], val_slots: list[int]) -> list[int]:
    operand_slots: list[int] = []
    for slot in val_slots:
        if slot - 2 >= 0 and tokens[slot - 2] == "ARG":
            arg_key = tokens[slot - 1]
            if arg_key in {"a", "b"}:
                operand_slots.append(slot)
    if operand_slots:
        return operand_slots
    # Fallback: skip the first VAL slot (often operator), use remaining.
    return val_slots[1:]


def _collect_operand_candidates(tokens: list[str], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    sym_candidates: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in seen:
            continue
        if tok.isalpha() and len(tok) == 1:
            if tok in allowed_set:
                sym_candidates.append(tok)
                seen.add(tok)
    literal_candidates: list[str] = []
    for lit in ("STR:0", "STR:1", "STR:2", "STR:3", "STR:4", "STR:5", "STR:10"):
        if lit in allowed_set and lit not in seen:
            literal_candidates.append(lit)
            seen.add(lit)
    out = sym_candidates + literal_candidates
    return out[:8]


@dataclass(frozen=True)
class _MiniVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]


def _resolve_operator_tokens(allowed: Iterable[str]) -> list[str]:
    token_to_id = {tok: idx for idx, tok in enumerate(sorted(set(allowed)))}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    vocab = _MiniVocab(token_to_id=token_to_id, id_to_token=id_to_token)
    token_sets = _build_token_sets(vocab)
    op_ids = _resolve_apply_arith_op_ids(token_sets)
    ops = [id_to_token[i] for i in sorted(op_ids) if i in id_to_token]
    preferred = ["STR:+", "STR:-", "STR:*", "STR:/"]
    ordered = [tok for tok in preferred if tok in ops]
    ordered.extend(tok for tok in ops if tok not in ordered)
    return ordered


def repair_math_apply_arith_operator(
    tokens: list[str],
    *,
    max_edits: int = 2,
    max_candidates: int = 32,
    allowed_tokens: Iterable[str] | None = None,
) -> list[tuple[list[str], str]]:
    if max_edits <= 0 or max_candidates <= 0:
        return []
    if allowed_tokens is None:
        allowed = list(tokens)
    else:
        allowed = list(allowed_tokens)
    op_tokens = _resolve_operator_tokens(allowed)
    if not op_tokens:
        return []
    slots = _find_apply_arith_val_slots(tokens)
    if not slots:
        return []
    slots = slots[:4]
    op_tokens = op_tokens[:8]
    operand_slots = _select_operand_slots(tokens, slots)
    operand_tokens = _collect_operand_candidates(tokens, allowed)
    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def _add_candidate(updated: list[str], kind: str) -> None:
        key = tuple(updated)
        if key in seen:
            return
        seen.add(key)
        candidates.append((updated, kind))

    for slot in slots:
        current = tokens[slot]
        for op in op_tokens:
            if op == current:
                continue
            updated = list(tokens)
            updated[slot] = op
            _add_candidate(updated, "op")
            if len(candidates) >= max_candidates:
                return candidates

    for slot in operand_slots:
        current = tokens[slot]
        for op in operand_tokens:
            if op == current:
                continue
            updated = list(tokens)
            updated[slot] = op
            _add_candidate(updated, "operand")
            if len(candidates) >= max_candidates:
                return candidates

    if max_edits >= 2:
        for op_slot in slots:
            op_current = tokens[op_slot]
            for op in op_tokens:
                if op == op_current:
                    continue
                for val_slot in operand_slots:
                    val_current = tokens[val_slot]
                    for val in operand_tokens:
                        if val == val_current:
                            continue
                        updated = list(tokens)
                        updated[op_slot] = op
                        updated[val_slot] = val
                        _add_candidate(updated, "op+operand")
                        if len(candidates) >= max_candidates:
                            return candidates
    return candidates
