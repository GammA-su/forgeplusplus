import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_backfill_idempotent_and_preserves_tokens() -> None:
    row = {
        "id": "row_0",
        "domain_tag": "[MATH]",
        "x": "Compute: 2 + 2.",
        "y": 4,
        "constraints": [],
        "proof": {"tokens": ["<BOS>", "BEGIN", "OP", "EMIT_NUM", "END", "<EOS>"]},
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        inp = tmp / "in.jsonl"
        out1 = tmp / "out1.jsonl"
        out2 = tmp / "out2.jsonl"
        inp.write_text(json.dumps(row) + "\n", encoding="utf-8")

        subprocess.run(
            [sys.executable, "scripts/backfill_proof_sha256_v2.py", str(inp), str(out1)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "scripts/backfill_proof_sha256_v2.py", str(out1), str(out2)],
            check=True,
        )

        lines1 = _read_lines(out1)
        lines2 = _read_lines(out2)
        assert lines1 == lines2
        assert lines1[0]["proof"]["tokens"] == row["proof"]["tokens"]
