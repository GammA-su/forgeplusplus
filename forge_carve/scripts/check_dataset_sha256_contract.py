from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any

def has_sha(proof: Any) -> bool:
    if isinstance(proof, dict):
        return isinstance(proof.get("sha256"), str) and len(proof["sha256"]) == 64
    return False

def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_dataset_sha256_contract.py <file1.jsonl> [file2.jsonl ...]", file=sys.stderr)
        return 2

    rc = 0
    for p in map(Path, sys.argv[1:]):
        rows = bad = 0
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                print(f"[FAIL] {p} line={ln} reason=invalid_json")
                continue
            proof = rec.get("proof")
            if not has_sha(proof):
                bad += 1
                rid = rec.get("id", f"line:{ln}")
                print(f"[FAIL] {p} id={rid} line={ln} reason=missing_or_bad_proof_sha256")
        print(f"{p}: rows={rows} bad={bad}")
        if bad:
            rc = 1
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
