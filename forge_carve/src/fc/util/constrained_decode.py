from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import torch

from fc.dsl.tokens import ARG_KEYS, OPCODES, TokenVocab
from fc.util.tags import domain_from_tag

try:
    from prooftape.ptv1 import (
        ARITH_OP_STRINGS as _PTV1_ARITH_OP_STRINGS,
        get_apply_arith_op_tokens as _get_ptv1_apply_arith_op_tokens,
    )
except Exception:  # pragma: no cover - optional dependency
    _get_ptv1_apply_arith_op_tokens = None
    _PTV1_ARITH_OP_STRINGS = None

_DEST_REQUIRED = {
    "EXTRACT_INT",
    "EXTRACT_FLOAT",
    "EXTRACT_STR",
    "APPLY_ARITH",
    "APPLY_TOPO",
    "APPLY_CUMSUM",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "BIND",
}

_OP_REQUIRED_ARGS: dict[str, set[str]] = {
    "APPLY_ARITH": {"a", "b", "op"},
    "ADD": {"a", "b"},
    "SUB": {"a", "b"},
    "MUL": {"a", "b"},
    "DIV": {"a", "b"},
    "BIND": {"value"},
}

_EMIT_OPS = {"EMIT", "EMIT_NUM", "EMIT_SCHEDULE"}
_DISALLOWED_TOKENS = {"<PAD>", "<UNK>"}
_EMIT_BY_DOMAIN = {
    "math": "EMIT_NUM",
    "csp": "EMIT_SCHEDULE",
    "schema": "EMIT",
}
_OPCODES_BY_DOMAIN = {
    "math": {"EXTRACT_INT", "APPLY_ARITH", "ADD", "SUB", "MUL", "DIV", "BIND", "EMIT_NUM"},
    "schema": {"EXTRACT_STR", "EXTRACT_INT", "EMIT"},
    "csp": {"APPLY_TOPO", "APPLY_CUMSUM", "EMIT_SCHEDULE"},
}
_ARITH_OP_SUBSTRINGS = ("ADD", "SUB", "MUL", "DIV", "MOD", "POW")


def _normalize_domain(domain_tag: str | None) -> str | None:
    if not domain_tag:
        return None
    if domain_tag in _EMIT_BY_DOMAIN:
        return domain_tag
    tagged = domain_from_tag(domain_tag)
    return tagged


@dataclass
class DecodeStats:
    eos_pos: int | None
    eos_token: str | None
    truncated: bool
    min_tokens: int
    max_tokens: int
    apply_arith_op_fallback: int = 0


@dataclass
class _ValueContext:
    kind: str
    phase: str


@dataclass
class _TokenSets:
    op_id: int | None
    dest_id: int | None
    arg_id: int | None
    val_id: int | None
    begin_id: int | None
    bos_id: int | None
    end_id: int | None
    eos_id: int | None
    sep_id: int | None
    list_start_id: int | None
    list_end_id: int | None
    dict_start_id: int | None
    dict_end_id: int | None
    opcode_ids: list[int]
    arg_key_ids: list[int]
    arg_key_map: dict[str, int]
    value_ids: list[int]
    str_ids: list[int]
    token_id_map: dict[str, int]


def _build_token_sets(vocab: TokenVocab) -> _TokenSets:
    def tid(tok: str) -> int | None:
        return vocab.token_to_id.get(tok)

    opcode_ids = [vocab.token_to_id[tok] for tok in OPCODES if tok in vocab.token_to_id]
    arg_key_ids = [vocab.token_to_id[tok] for tok in ARG_KEYS if tok in vocab.token_to_id]
    arg_key_map = {tok: vocab.token_to_id[tok] for tok in ARG_KEYS if tok in vocab.token_to_id}
    value_ids: list[int] = []
    str_ids: list[int] = []
    for tok_id, tok in vocab.id_to_token.items():
        if tok in _DISALLOWED_TOKENS:
            continue
        if tok.startswith("STR:"):
            value_ids.append(tok_id)
            str_ids.append(tok_id)
        elif tok.startswith(("INT:", "FLOAT:", "BOOL:")):
            value_ids.append(tok_id)
    return _TokenSets(
        op_id=tid("OP"),
        dest_id=tid("DEST"),
        arg_id=tid("ARG"),
        val_id=tid("VAL"),
        begin_id=tid("BEGIN"),
        bos_id=tid("<BOS>"),
        end_id=tid("END"),
        eos_id=tid("<EOS>"),
        sep_id=tid("SEP"),
        list_start_id=tid("LIST_START"),
        list_end_id=tid("LIST_END"),
        dict_start_id=tid("DICT_START"),
        dict_end_id=tid("DICT_END"),
        opcode_ids=opcode_ids,
        arg_key_ids=arg_key_ids,
        arg_key_map=arg_key_map,
        value_ids=value_ids,
        str_ids=str_ids,
        token_id_map=dict(vocab.token_to_id),
    )


class _Ptv1DecodeState:
    def __init__(self, domain: str | None = None) -> None:
        self.mode = "START"
        self.current_op: str | None = None
        self.missing_args: set[str] = set()
        self.dest_required = False
        self.dest_seen = False
        self.args_started = False
        self.done = False
        self.value_stack: list[_ValueContext] = []
        self.arg_key: str | None = None
        self.in_value = False
        self.domain = domain
        self.emit_required_arg: str | None = None
        self.emit_value_start: str | None = None
        self.emit_force_end = False
        self.emit_value_pending = False
        self.arg_index = -1

    def clone(self) -> "_Ptv1DecodeState":
        clone = _Ptv1DecodeState(domain=self.domain)
        clone.mode = self.mode
        clone.current_op = self.current_op
        clone.missing_args = set(self.missing_args)
        clone.dest_required = self.dest_required
        clone.dest_seen = self.dest_seen
        clone.args_started = self.args_started
        clone.done = self.done
        clone.value_stack = [_ValueContext(kind=ctx.kind, phase=ctx.phase) for ctx in self.value_stack]
        clone.arg_key = self.arg_key
        clone.in_value = self.in_value
        clone.emit_required_arg = self.emit_required_arg
        clone.emit_value_start = self.emit_value_start
        clone.emit_force_end = self.emit_force_end
        clone.emit_value_pending = self.emit_value_pending
        clone.arg_index = self.arg_index
        return clone

    def _allowed_structural(self, tokens: _TokenSets) -> set[int]:
        allowed: set[int] = set()
        if self.done:
            if tokens.eos_id is not None:
                allowed.add(tokens.eos_id)
            return allowed
        if self.mode == "START":
            for tok_id in (tokens.bos_id, tokens.begin_id, tokens.op_id, tokens.end_id, tokens.eos_id):
                if tok_id is not None:
                    allowed.add(tok_id)
            return allowed
        if self.mode == "EXPECT_OP":
            for tok_id in (tokens.op_id, tokens.end_id, tokens.eos_id):
                if tok_id is not None:
                    allowed.add(tok_id)
            return allowed
        if self.mode == "EXPECT_OPCODE":
            allowed.update(tokens.opcode_ids)
            return allowed
        if self.mode == "EXPECT_DEST":
            if tokens.dest_id is not None:
                allowed.add(tokens.dest_id)
            return allowed
        if self.mode == "EXPECT_DEST_VALUE":
            allowed.update(tokens.str_ids)
            return allowed
        if self.mode == "EXPECT_ARG_KEY":
            if self.missing_args:
                allowed.update(
                    tok_id for key, tok_id in tokens.arg_key_map.items() if key in self.missing_args
                )
            else:
                allowed.update(tokens.arg_key_ids)
            return allowed
        if self.mode == "EXPECT_VAL":
            if tokens.val_id is not None:
                allowed.add(tokens.val_id)
            return allowed
        if self.mode == "EXPECT_ARG_OR_END":
            if self.emit_force_end:
                for tok_id in (tokens.end_id, tokens.eos_id):
                    if tok_id is not None:
                        allowed.add(tok_id)
                return allowed
            if not self.args_started and not self.dest_seen and not self.dest_required:
                if tokens.dest_id is not None:
                    allowed.add(tokens.dest_id)
            if self.missing_args:
                if tokens.arg_id is not None:
                    allowed.add(tokens.arg_id)
                return allowed
            if tokens.arg_id is not None:
                allowed.add(tokens.arg_id)
            if tokens.op_id is not None:
                allowed.add(tokens.op_id)
            for tok_id in (tokens.end_id, tokens.eos_id):
                if tok_id is not None:
                    allowed.add(tok_id)
            return allowed
        return allowed

    def _allowed_value(self, tokens: _TokenSets) -> set[int]:
        allowed: set[int] = set()
        if self.emit_value_pending and not self.value_stack and self.emit_value_start:
            forced_id = tokens.token_id_map.get(self.emit_value_start)
            if forced_id is not None:
                return {forced_id}
        if not self.value_stack:
            allowed.update(tokens.value_ids)
            for tok_id in (tokens.list_start_id, tokens.dict_start_id):
                if tok_id is not None:
                    allowed.add(tok_id)
            return allowed
        top = self.value_stack[-1]
        if top.kind == "LIST":
            if top.phase == "EXPECT_VALUE_OR_END":
                allowed.update(tokens.value_ids)
                if tokens.list_start_id is not None:
                    allowed.add(tokens.list_start_id)
                if tokens.dict_start_id is not None:
                    allowed.add(tokens.dict_start_id)
                if tokens.list_end_id is not None:
                    allowed.add(tokens.list_end_id)
                if tokens.sep_id is not None:
                    allowed.add(tokens.sep_id)
            elif top.phase == "EXPECT_SEP_OR_END":
                if tokens.sep_id is not None:
                    allowed.add(tokens.sep_id)
                if tokens.list_end_id is not None:
                    allowed.add(tokens.list_end_id)
        elif top.kind == "DICT":
            if top.phase == "EXPECT_KEY_OR_END":
                allowed.update(tokens.value_ids)
                if tokens.list_start_id is not None:
                    allowed.add(tokens.list_start_id)
                if tokens.dict_start_id is not None:
                    allowed.add(tokens.dict_start_id)
                if tokens.dict_end_id is not None:
                    allowed.add(tokens.dict_end_id)
                if tokens.sep_id is not None:
                    allowed.add(tokens.sep_id)
            elif top.phase == "EXPECT_VAL":
                allowed.update(tokens.value_ids)
                if tokens.list_start_id is not None:
                    allowed.add(tokens.list_start_id)
                if tokens.dict_start_id is not None:
                    allowed.add(tokens.dict_start_id)
                if tokens.sep_id is not None:
                    allowed.add(tokens.sep_id)
            elif top.phase == "EXPECT_SEP_OR_END":
                if tokens.sep_id is not None:
                    allowed.add(tokens.sep_id)
                if tokens.dict_end_id is not None:
                    allowed.add(tokens.dict_end_id)
        return allowed

    def allowed_ids(self, tokens: _TokenSets) -> set[int]:
        if self.in_value:
            return self._allowed_value(tokens)
        return self._allowed_structural(tokens)

    def _consume_value_token(self, tok: str) -> None:
        if not self.value_stack:
            return
        top = self.value_stack[-1]
        if tok == "SEP":
            if top.kind == "LIST":
                if top.phase == "EXPECT_SEP_OR_END":
                    top.phase = "EXPECT_VALUE_OR_END"
            elif top.kind == "DICT":
                if top.phase == "EXPECT_SEP_OR_END":
                    top.phase = "EXPECT_KEY_OR_END"
            return
        if top.kind == "LIST" and top.phase == "EXPECT_VALUE_OR_END":
            top.phase = "EXPECT_SEP_OR_END"
            return
        if top.kind == "DICT":
            if top.phase == "EXPECT_KEY_OR_END":
                top.phase = "EXPECT_VAL"
                return
            if top.phase == "EXPECT_VAL":
                top.phase = "EXPECT_SEP_OR_END"
                return

    def _complete_value(self) -> None:
        if not self.value_stack:
            self._complete_arg_value()
            return
        top = self.value_stack[-1]
        if top.kind == "LIST" and top.phase == "EXPECT_VALUE_OR_END":
            top.phase = "EXPECT_SEP_OR_END"
        elif top.kind == "DICT":
            if top.phase == "EXPECT_KEY_OR_END":
                top.phase = "EXPECT_VAL"
            elif top.phase == "EXPECT_VAL":
                top.phase = "EXPECT_SEP_OR_END"

    def consume(self, tok: str) -> None:
        if self.done:
            return
        if self.in_value:
            if self.emit_value_pending and tok == self.emit_value_start:
                self.emit_value_pending = False
            if tok == "LIST_START":
                self.value_stack.append(_ValueContext(kind="LIST", phase="EXPECT_VALUE_OR_END"))
                return
            if tok == "DICT_START":
                self.value_stack.append(_ValueContext(kind="DICT", phase="EXPECT_KEY_OR_END"))
                return
            if tok == "LIST_END":
                if self.value_stack and self.value_stack[-1].kind == "LIST":
                    self.value_stack.pop()
                    self._complete_value()
                return
            if tok == "DICT_END":
                if self.value_stack and self.value_stack[-1].kind == "DICT":
                    self.value_stack.pop()
                    self._complete_value()
                return
            if tok == "SEP":
                self._consume_value_token(tok)
                return
            if tok.startswith(("STR:", "INT:", "FLOAT:", "BOOL:")):
                self._complete_value()
                return
            return

        if self.mode == "START":
            if tok in ("<BOS>", "BEGIN"):
                return
            if tok == "OP":
                self.mode = "EXPECT_OPCODE"
                return
            if tok in ("END", "<EOS>"):
                self.done = True
                return
            self.mode = "EXPECT_OP"
            return
        if self.mode == "EXPECT_OP":
            if tok == "OP":
                self.mode = "EXPECT_OPCODE"
                return
            if tok in ("END", "<EOS>"):
                self.done = True
                return
            return
        if self.mode == "EXPECT_OPCODE":
            self.current_op = tok
            self.dest_required = tok in _DEST_REQUIRED
            self.dest_seen = False
            self.args_started = False
            self.arg_index = -1
            self.missing_args = set(_OP_REQUIRED_ARGS.get(tok, set()))
            self.emit_required_arg = None
            self.emit_value_start = None
            self.emit_force_end = False
            self.emit_value_pending = False
            if self.domain == "schema" and tok == "EMIT":
                self.missing_args = {"fields"}
                self.emit_required_arg = "fields"
                self.emit_value_start = "DICT_START"
            elif self.domain == "math" and tok == "EMIT_NUM":
                self.missing_args = {"value"}
                self.emit_required_arg = "value"
                self.emit_value_start = "STR:result"
            elif self.domain == "csp" and tok == "EMIT_SCHEDULE":
                self.emit_force_end = True
            if self.dest_required:
                self.mode = "EXPECT_DEST"
            else:
                self.mode = "EXPECT_ARG_OR_END"
            return
        if self.mode == "EXPECT_DEST":
            if tok == "DEST":
                self.mode = "EXPECT_DEST_VALUE"
            return
        if self.mode == "EXPECT_DEST_VALUE":
            if tok.startswith("STR:"):
                self.dest_seen = True
                self.mode = "EXPECT_ARG_OR_END"
            return
        if self.mode == "EXPECT_ARG_OR_END":
            if tok == "DEST":
                self.mode = "EXPECT_DEST_VALUE"
                return
            if tok == "ARG":
                self.args_started = True
                self.arg_index += 1
                self.mode = "EXPECT_ARG_KEY"
                return
            if tok in ("END", "<EOS>"):
                self.done = True
                return
            if tok == "OP":
                self.mode = "EXPECT_OPCODE"
                return
            return
        if self.mode == "EXPECT_ARG_KEY":
            self.arg_key = tok
            self.mode = "EXPECT_VAL"
            return
        if self.mode == "EXPECT_VAL":
            if tok == "VAL":
                self.value_stack = []
                self.in_value = True
                if self.emit_value_start and (self.emit_required_arg is None or self.arg_key == self.emit_required_arg):
                    self.emit_value_pending = True
            return

    def _complete_arg_value(self) -> None:
        if self.arg_key and self.arg_key in self.missing_args:
            self.missing_args.discard(self.arg_key)
        if self.emit_required_arg and self.arg_key == self.emit_required_arg:
            self.emit_value_start = None
            self.emit_value_pending = False
        self.arg_key = None
        self.in_value = False
        self.mode = "EXPECT_ARG_OR_END"


def _mask_logits(row: torch.Tensor, allowed: Iterable[int]) -> torch.Tensor:
    mask = torch.full_like(row, float("-inf"))
    for tok_id in allowed:
        if 0 <= int(tok_id) < mask.numel():
            mask[int(tok_id)] = row[int(tok_id)]
    return mask


def _advance_parent_after_value(ctx: _ValueContext) -> None:
    if ctx.kind == "LIST" and ctx.phase == "EXPECT_VALUE_OR_END":
        ctx.phase = "EXPECT_SEP_OR_END"
    elif ctx.kind == "DICT":
        if ctx.phase == "EXPECT_KEY_OR_END":
            ctx.phase = "EXPECT_VAL"
        elif ctx.phase == "EXPECT_VAL":
            ctx.phase = "EXPECT_SEP_OR_END"


def _min_tokens_to_close_value(stack: list[_ValueContext]) -> int:
    if not stack:
        return 1
    tokens = 0
    work = [_ValueContext(kind=ctx.kind, phase=ctx.phase) for ctx in stack]
    while work:
        top = work[-1]
        if top.kind == "LIST":
            tokens += 1
            work.pop()
            if work:
                _advance_parent_after_value(work[-1])
            continue
        if top.kind == "DICT":
            if top.phase == "EXPECT_VAL":
                tokens += 1
                top.phase = "EXPECT_SEP_OR_END"
            tokens += 1
            work.pop()
            if work:
                _advance_parent_after_value(work[-1])
            continue
        tokens += 1
        work.pop()
    return tokens


def _min_tokens_to_complete(state: _Ptv1DecodeState) -> int:
    if state.done:
        return 0
    base = 0
    if state.in_value:
        base = _min_tokens_to_close_value(state.value_stack)
        return base + 1
    if state.mode in {"START", "EXPECT_OP"}:
        return 1
    if state.mode == "EXPECT_OPCODE":
        return 2
    if state.mode == "EXPECT_DEST":
        return 3
    if state.mode == "EXPECT_DEST_VALUE":
        return 2
    if state.mode == "EXPECT_ARG_KEY":
        remaining = max(0, len(state.missing_args) - 1)
        base = 3 + 4 * remaining
        return base + 1
    if state.mode == "EXPECT_VAL":
        remaining = len(state.missing_args)
        if state.arg_key and state.arg_key in state.missing_args:
            remaining -= 1
        remaining = max(0, remaining)
        base = 2 + 4 * remaining
        return base + 1
    if state.mode == "EXPECT_ARG_OR_END":
        if state.missing_args:
            base = 4 * len(state.missing_args)
            return base + 1
        return 1
    return 1


def _resolve_apply_arith_op_ids(tokens: _TokenSets) -> set[int]:
    allowed: set[int] = set()
    # Tier A: STR:<op> with op in PTv1.ARITH_OP_STRINGS
    if _PTV1_ARITH_OP_STRINGS:
        strict: set[int] = set()
        for op in _PTV1_ARITH_OP_STRINGS:
            tok_id = tokens.token_id_map.get(f"STR:{op}")
            if tok_id is not None:
                strict.add(tok_id)
        if strict:
            return strict
    # Tier B fallback: PTv1 op tokens + substring matching
    observed: list[str] | None = None
    if _get_ptv1_apply_arith_op_tokens is not None:
        try:
            observed = list(_get_ptv1_apply_arith_op_tokens())
        except Exception:
            observed = None
    if observed:
        for tok in observed:
            tok_id = tokens.token_id_map.get(tok)
            if tok_id is not None:
                allowed.add(tok_id)
    for tok, tok_id in tokens.token_id_map.items():
        upper = tok.upper()
        if any(sub in upper for sub in _ARITH_OP_SUBSTRINGS):
            allowed.add(tok_id)
    return allowed


def _should_mask_apply_arith_operator(toks: list[str]) -> bool:
    if not toks:
        return False
    last_apply = -1
    for idx in range(len(toks) - 1, -1, -1):
        if toks[idx] == "APPLY_ARITH":
            last_apply = idx
            break
    if last_apply < 0:
        return False
    for j in range(last_apply + 1, len(toks)):
        if toks[j] == "VAL":
            return len(toks) == j + 1
    return False


def _math_apply_arith_mask_enabled() -> bool:
    flag = os.getenv("FC_MATH_APPLY_ARITH_OP_MASK", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def greedy_decode_with_opcode_mask(
    logits: torch.Tensor,
    vocab: TokenVocab,
    *,
    enforce_opcode: bool = False,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    stop_on_emit: bool = True,
    domain_tag: str | None = None,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], DecodeStats]:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [seq_len, vocab]")
    seq_len = logits.shape[0]
    max_len = seq_len if max_tokens is None else max(1, min(seq_len, int(max_tokens)))
    decoded: list[int] = []
    decoded_tokens: list[str] = []
    eos_pos: int | None = None
    eos_token: str | None = None

    if not enforce_opcode:
        pred = torch.argmax(logits[:max_len], dim=-1).detach().cpu().tolist()
        eos_id = vocab.token_to_id.get("<EOS>")
        end_id = vocab.token_to_id.get("END")
        if eos_id is not None and eos_id in pred:
            idx = pred.index(eos_id)
            decoded = [int(i) for i in pred[: idx + 1]]
            eos_pos = idx
            eos_token = "<EOS>"
        elif end_id is not None and end_id in pred:
            idx = pred.index(end_id)
            decoded = [int(i) for i in pred[: idx + 1]]
            if eos_id is not None:
                decoded.append(int(eos_id))
                eos_pos = len(decoded) - 1
                eos_token = "<EOS>"
        else:
            for idx, tok_id in enumerate(pred):
                tok = vocab.decode(int(tok_id))
                if eos_pos is None and tok in ("END", "<EOS>"):
                    eos_pos = idx
                    eos_token = tok
            decoded = [int(i) for i in pred]
        stats = DecodeStats(
            eos_pos=eos_pos,
            eos_token=eos_token,
            truncated=eos_pos is None,
            min_tokens=max(0, int(min_tokens)),
            max_tokens=max_len,
            apply_arith_op_fallback=0,
        )
        return (decoded, stats) if return_stats else decoded

    tokens = _build_token_sets(vocab)
    domain = _normalize_domain(domain_tag)
    emit_id_set = {vocab.token_to_id[op] for op in _EMIT_OPS if op in vocab.token_to_id}
    allowed_emit_ids: set[int] | None = None
    if domain in _EMIT_BY_DOMAIN:
        allowed_emit = _EMIT_BY_DOMAIN[domain]
        if allowed_emit in vocab.token_to_id:
            allowed_emit_ids = {vocab.token_to_id[allowed_emit]}
    state = _Ptv1DecodeState(domain=domain)
    arith_op_ids = _resolve_apply_arith_op_ids(tokens) if domain == "math" else set()
    apply_arith_op_fallback = 0

    min_tokens = max(0, int(min_tokens))
    for idx in range(max_len):
        row = logits[idx].detach()
        allowed = state.allowed_ids(tokens)
        if (
            domain == "math"
            and state.in_value
            and _math_apply_arith_mask_enabled()
            and _should_mask_apply_arith_operator(decoded_tokens)
        ):
            if arith_op_ids:
                filtered = set(allowed) & arith_op_ids
                if filtered:
                    allowed = filtered
                else:
                    apply_arith_op_fallback += 1
        if state.mode == "EXPECT_OPCODE":
            if allowed_emit_ids is not None and emit_id_set:
                allowed = set(allowed) - (emit_id_set - allowed_emit_ids)
            if domain in _OPCODES_BY_DOMAIN:
                allowed_ops = _OPCODES_BY_DOMAIN[domain]
                allowed = {tok_id for tok_id in allowed if vocab.decode(int(tok_id)) in allowed_ops}
        remaining = max_len - idx
        if allowed:
            filtered: set[int] = set()
            for tok_id in allowed:
                probe = state.clone()
                probe.consume(vocab.decode(int(tok_id)))
                needed = _min_tokens_to_complete(probe)
                if needed <= remaining - 1:
                    filtered.add(tok_id)
            if filtered:
                allowed = filtered
            elif not (state.emit_required_arg and state.missing_args):
                fallback = {tok_id for tok_id in (tokens.end_id, tokens.eos_id) if tok_id in allowed}
                if fallback:
                    allowed = fallback
        if idx < min_tokens:
            if tokens.end_id in allowed:
                allowed.discard(tokens.end_id)
            if tokens.eos_id in allowed:
                allowed.discard(tokens.eos_id)
            if not allowed:
                if tokens.end_id is not None:
                    allowed.add(tokens.end_id)
                if tokens.eos_id is not None:
                    allowed.add(tokens.eos_id)
        if not allowed:
            allowed = set(range(row.numel()))
        row = _mask_logits(row, allowed)
        tok_id = int(torch.argmax(row).item())
        decoded.append(tok_id)
        tok = vocab.decode(tok_id)
        decoded_tokens.append(tok)
        if eos_pos is None and tok in ("END", "<EOS>"):
            eos_pos = idx
            eos_token = tok
        state.consume(tok)
        if tok == "<EOS>":
            break
        if tok == "END" and tokens.eos_id is not None and len(decoded) < max_len:
            decoded.append(tokens.eos_id)
            eos_pos = len(decoded) - 1
            eos_token = "<EOS>"
            break
    stats = DecodeStats(
        eos_pos=eos_pos,
        eos_token=eos_token,
        truncated=eos_pos is None,
        min_tokens=min_tokens,
        max_tokens=max_len,
        apply_arith_op_fallback=apply_arith_op_fallback,
    )
    return (decoded, stats) if return_stats else decoded
