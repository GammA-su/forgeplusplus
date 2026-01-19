from __future__ import annotations
import subprocess, sys
from pathlib import Path

def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], text=True, capture_output=True)

def test_datasets_have_proof_sha256():
    files = ["out/data/math.jsonl", "out/data/csp.jsonl"]
    for f in files:
        assert Path(f).exists(), f"missing dataset file: {f}"

    r = run("scripts/check_dataset_sha256_contract.py", *files)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
