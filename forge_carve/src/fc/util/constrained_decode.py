from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from fc.dsl.tokens import ARG_KEYS, OPCODES, TokenVocab

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


@dataclass
class DecodeStats:
    eos_pos: int | None
    eos_token: str | None
    truncated: bool
    min_tokens: int
    max_tokens: int


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
    )


class _Ptv1DecodeState:
    def __init__(self) -> None:
        self.mode = "START"
        self.current_op: str | None = None
        self.missing_args: set[str] = set()
        self.dest_required = False
        self.dest_seen = False
        self.args_started = False
        self.terminate_after_emit = False
        self.done = False
        self.value_stack: list[_ValueContext] = []
        self.arg_key: str | None = None
        self.in_value = False

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
            if not self.args_started and not self.dest_seen and not self.dest_required:
                if tokens.dest_id is not None:
                    allowed.add(tokens.dest_id)
            if self.missing_args:
                if tokens.arg_id is not None:
                    allowed.add(tokens.arg_id)
                return allowed
            if tokens.arg_id is not None:
                allowed.add(tokens.arg_id)
            if not self.terminate_after_emit:
                if tokens.op_id is not None:
                    allowed.add(tokens.op_id)
            for tok_id in (tokens.end_id, tokens.eos_id):
                if tok_id is not None:
                    allowed.add(tok_id)
            return allowed
        return allowed

    def _allowed_value(self, tokens: _TokenSets) -> set[int]:
        allowed: set[int] = set()
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
            self.missing_args = set(_OP_REQUIRED_ARGS.get(tok, set()))
            self.terminate_after_emit = tok in _EMIT_OPS
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
            return

    def _complete_arg_value(self) -> None:
        if self.arg_key and self.arg_key in self.missing_args:
            self.missing_args.discard(self.arg_key)
        self.arg_key = None
        self.in_value = False
        self.mode = "EXPECT_ARG_OR_END"


def _mask_logits(row: torch.Tensor, allowed: Iterable[int]) -> torch.Tensor:
    mask = torch.full_like(row, float("-inf"))
    for tok_id in allowed:
        if 0 <= int(tok_id) < mask.numel():
            mask[int(tok_id)] = row[int(tok_id)]
    return mask


def greedy_decode_with_opcode_mask(
    logits: torch.Tensor,
    vocab: TokenVocab,
    *,
    enforce_opcode: bool = False,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], DecodeStats]:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [seq_len, vocab]")
    seq_len = logits.shape[0]
    max_len = seq_len if max_tokens is None else max(1, min(seq_len, int(max_tokens)))
    decoded: list[int] = []
    eos_pos: int | None = None
    eos_token: str | None = None

    if not enforce_opcode:
        pred = torch.argmax(logits[:max_len], dim=-1).detach().cpu().tolist()
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
        )
        return (decoded, stats) if return_stats else decoded

    tokens = _build_token_sets(vocab)
    state = _Ptv1DecodeState()

    min_tokens = max(0, int(min_tokens))
    for idx in range(max_len):
        row = logits[idx].detach()
        allowed = state.allowed_ids(tokens)
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
        if eos_pos is None and tok in ("END", "<EOS>"):
            eos_pos = idx
            eos_token = tok
        state.consume(tok)
    stats = DecodeStats(
        eos_pos=eos_pos,
        eos_token=eos_token,
        truncated=eos_pos is None,
        min_tokens=min_tokens,
        max_tokens=max_len,
    )
    return (decoded, stats) if return_stats else decoded
