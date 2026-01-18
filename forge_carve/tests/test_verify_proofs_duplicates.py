import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _hash_tokens(tokens: list[str]) -> str:
    payload = json.dumps(tokens, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_verify(path: str) -> str:
    r = subprocess.run([sys.executable, "scripts/verify_proofs.py", path], capture_output=True, text=True)
    return (r.stdout or "").strip()


def test_verify_rejects_duplicate_rows() -> None:
    tokens = [
        "<BOS>", "BEGIN",
        "OP", "EXTRACT_INT", "DEST", "STR:a", "ARG", "index", "VAL", "INT:0",
        "OP", "EXTRACT_INT", "DEST", "STR:b", "ARG", "index", "VAL", "INT:1",
        "OP", "APPLY_ARITH", "DEST", "STR:result", "ARG", "a", "VAL", "STR:a", "ARG", "b", "VAL", "STR:b",
        "ARG", "op", "VAL", "STR:+",
        "OP", "EMIT_NUM", "ARG", "value", "VAL", "STR:result",
        "END", "<EOS>",
    ]
    proof = {"tokens": tokens, "sha256": _hash_tokens(tokens)}
    row = {
        "id": "dup_0",
        "domain_tag": "[MATH]",
        "x": "What is 7 plus 7?",
        "y": 14,
        "constraints": [],
        "proof": proof,
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = tmp / "dups.jsonl"
        path.write_text("\n".join([json.dumps(row), json.dumps(row)]) + "\n", encoding="utf-8")
        out = _run_verify(str(path))
        assert "bad=" in out
        assert "bad=0" not in out
