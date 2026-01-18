import hashlib
import json
import subprocess, sys, textwrap, tempfile
from pathlib import Path

from fc.util.tags import DOMAIN_TAGS

def run_verify(path: str) -> tuple[int,int,str,str]:
    r = subprocess.run([sys.executable, "scripts/verify_proofs.py", path],
                       capture_output=True, text=True)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    checked = bad = None
    for tok in out.split():
        if tok.startswith("checked="):
            checked = int(tok.split("=",1)[1])
        if tok.startswith("bad="):
            bad = int(tok.split("=",1)[1])
    return checked, bad, out, err

def write(tmp: Path, name: str, lines: list[str]) -> str:
    p = tmp / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)

def test_rejects_junk_rows():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = write(tmp, "junk.jsonl", ['{"junk":1}', '{"lol":"nope"}'])
        checked, bad, out, err = run_verify(path)

        # Strict requirement: junk must NOT be accepted as valid.
        assert checked is not None and bad is not None, f"no summary printed. out={out!r} err={err!r}"
        assert bad > 0, f"junk accepted: {out} err={err}"

def _hash_tokens(tokens: list[str]) -> str:
    payload = json.dumps(tokens, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_rejects_wrong_y_when_schema_present():
    # This assumes your schema requires x,y,domain_tag. Adjust if needed.
    tokens = [
        "<BOS>", "BEGIN",
        "OP", "EXTRACT_INT", "DEST", "STR:a", "ARG", "index", "VAL", "INT:0",
        "OP", "EXTRACT_INT", "DEST", "STR:b", "ARG", "index", "VAL", "INT:1",
        "OP", "APPLY_ARITH", "DEST", "STR:result", "ARG", "a", "VAL", "STR:a", "ARG", "b", "VAL", "STR:b",
        "ARG", "op", "VAL", "STR:+",
        "OP", "EMIT_NUM", "ARG", "value", "VAL", "STR:result",
        "END", "<EOS>",
    ]
    row_ok = {
        "id": "t0",
        "domain_tag": DOMAIN_TAGS["math"],
        "x": "What is 7 plus 7?",
        "y": 14,
        "constraints": [],
        "proof": {"tokens": tokens, "sha256": _hash_tokens(tokens)},
        "orbit": [],
        "flips": []
    }
    row_bad = dict(row_ok)
    row_bad["y"] = 999

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = write(tmp, "two.jsonl", [json.dumps(row_ok), json.dumps(row_bad)])
        checked, bad, out, err = run_verify(path)

        assert checked is not None and bad is not None, f"no summary printed. out={out!r} err={err!r}"
        assert bad >= 1, f"wrong y not detected: {out} err={err}"
