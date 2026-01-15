from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Any

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def serialize_answer(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_answer(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if cleaned[0] in "{[":
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
    if re.fullmatch(r"-?\d+", cleaned):
        try:
            return int(cleaned)
        except ValueError:
            return cleaned
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    return cleaned


@dataclass(frozen=True)
class AnswerVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @staticmethod
    def build(texts: Iterable[str]) -> "AnswerVocab":
        chars = set()
        for text in texts:
            chars.update(list(text))
        ordered = SPECIAL_TOKENS + sorted(chars)
        token_to_id = {t: i for i, t in enumerate(ordered)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return AnswerVocab(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.token_to_id["<BOS>"]]
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.token_to_id["<UNK>"]))
        ids.append(self.token_to_id["<EOS>"])
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.token_to_id["<PAD>"]] * (max_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        for tok_id in ids:
            tok = self.id_to_token.get(int(tok_id), "<UNK>")
            if tok == "<EOS>":
                break
            if tok in {"<PAD>", "<BOS>"}:
                continue
            tokens.append("?" if tok == "<UNK>" else tok)
        return "".join(tokens)
