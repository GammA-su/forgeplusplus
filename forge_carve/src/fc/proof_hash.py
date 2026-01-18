from __future__ import annotations
import json, hashlib
from typing import Any, Iterable

def hash_tokens(tokens: Iterable[Any]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
