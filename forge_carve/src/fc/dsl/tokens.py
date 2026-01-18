from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

Value = int | float | bool | str | None | dict[str, "Value"] | list["Value"]

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

OPCODES = [
    "EXTRACT_INT",
    "EXTRACT_FLOAT",
    "EXTRACT_STR",
    "BIND",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "APPLY_ARITH",
    "APPLY_TOPO",
    "APPLY_CUMSUM",
    "SOLVE_CSP",
    "EMIT",
    "EMIT_NUM",
    "EMIT_SCHEDULE",
]

ARG_KEYS = [
    "key",
    "index",
    "a",
    "b",
    "schema",
    "fields",
    "op",
    "tasks",
    "constraints",
    "value",
    "stop",
]

NAMES = ["Alice", "Bob", "Cara", "Dion", "Eve", "Fay"]
CITIES = ["Paris", "Berlin", "Oslo", "Lima", "Riga", "Pune"]
TASKS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
OP_SYMBOLS = ["+", "-", "*", "/"]
SYMBOLS = [
    "name",
    "age",
    "city",
    "a",
    "b",
    "c",
    "d",
    "result",
    "t0",
    "t1",
    "t2",
    "schedule",
    "status",
    "tasks",
    "constraints",
    "order",
]

SCHEMAS = ["person", "math", "schedule"]
STATUS = ["ok", "infeasible"]

STRUCT_TOKENS = [
    "OP",
    "DEST",
    "ARG",
    "VAL",
    "BEGIN",
    "END",
    "LIST_START",
    "LIST_END",
    "DICT_START",
    "DICT_END",
    "SEP",
]


@dataclass(frozen=True)
class TokenVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def decode(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "<UNK>")


def _int_tokens(lo: int = -20, hi: int = 50) -> list[str]:
    return [f"INT:{i}" for i in range(lo, hi + 1)]


def _bool_tokens() -> list[str]:
    return ["BOOL:true", "BOOL:false"]


def _str_tokens(values: Sequence[str]) -> list[str]:
    return [f"STR:{v}" for v in values]


def build_default_vocab(extra_tokens: Iterable[str] | None = None) -> TokenVocab:
    base = (
        SPECIAL_TOKENS
        + STRUCT_TOKENS
        + OPCODES
        + ARG_KEYS
        + _int_tokens()
        + _bool_tokens()
        + _str_tokens(NAMES + CITIES + TASKS + SCHEMAS + STATUS + SYMBOLS + OP_SYMBOLS)
    )
    if extra_tokens:
        base.extend(list(extra_tokens))
    # Deterministic ordering: keep original order, then append sorted uniques not already seen.
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in base:
        if tok not in seen:
            ordered.append(tok)
            seen.add(tok)
    token_to_id = {tok: i for i, tok in enumerate(ordered)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return TokenVocab(token_to_id=token_to_id, id_to_token=id_to_token)


def validate_opcode(opcode: str) -> None:
    if opcode not in OPCODES:
        raise ValueError(f"Unknown opcode: {opcode}")
