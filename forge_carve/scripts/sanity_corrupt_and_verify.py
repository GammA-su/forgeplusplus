from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_verify(path: str) -> tuple[int, str, str]:
    r = subprocess.run([sys.executable, "scripts/verify_proofs.py", path], capture_output=True, text=True)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    return r.returncode, out, err


def _parse_bad(output: str) -> int | None:
    for tok in output.split():
        if tok.startswith("bad="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _mutate_value(val):
    if isinstance(val, str):
        return val + " CORRUPT"
    if isinstance(val, (int, float)):
        return val + 1
    if isinstance(val, list):
        return (val + ["CORRUPT"]) if val else ["CORRUPT"]
    if isinstance(val, dict):
        out = dict(val)
        out["_CORRUPT_"] = True
        return out
    return "CORRUPT"


def _corrupt_answer(obj: dict) -> dict:
    o = dict(obj)
    if "y" in o:
        o["y"] = _mutate_value(o.get("y"))
        return o
    if "constraints" in o:
        o["constraints"] = _mutate_value(o.get("constraints"))
        return o
    o["x"] = _mutate_value(o.get("x", ""))
    return o


def _corrupt_proof(obj: dict) -> dict:
    o = dict(obj)
    proof = o.get("proof")
    if isinstance(proof, dict) and isinstance(proof.get("tokens"), list):
        tokens = list(proof.get("tokens") or [])
        if tokens:
            idx = random.randrange(len(tokens))
            tokens[idx] = str(tokens[idx]) + "_CORRUPT"
        proof = dict(proof)
        proof["tokens"] = tokens
        o["proof"] = proof
        return o
    o["proof"] = _mutate_value(proof)
    return o


def _write_temp(rows: list[dict]) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
        for rec in rows:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return tmp.name


def main(path: str, n: int, mode: str) -> int:
    src = Path(path)
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    if not rows:
        print("SANITY FAIL: empty file")
        return 1
    base_rc, base_out, base_err = _run_verify(str(src))
    base_bad = _parse_bad(base_out)
    if base_rc != 0 or base_bad:
        print("SANITY FAIL: baseline does not verify; fix dataset first.")
        if base_out:
            print(base_out)
        if base_err:
            print(base_err, file=sys.stderr)
        return 1
    if base_out:
        print(base_out)
    idxs = list(range(len(rows)))
    random.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    corrupted = list(rows)
    if mode == "answer-sanity":
        for i in idxs:
            corrupted[i] = _corrupt_answer(rows[i])
        fail_msg = "SANITY FAIL: verifier accepted corrupted answers (bad=0)."
        ok_msg = "SANITY OK: verifier rejects corrupted answers."
    else:
        for i in idxs:
            corrupted[i] = _corrupt_proof(rows[i])
        fail_msg = "SANITY FAIL: verifier accepted corrupted proofs (bad=0)."
        ok_msg = "SANITY OK: verifier rejects corrupted proofs."

    tmp_path = _write_temp(corrupted)
    rc, out, err = _run_verify(tmp_path)
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    bad = _parse_bad(out)
    if bad == 0:
        print(fail_msg)
        return 1
    if rc == 0 and bad is None:
        print("SANITY FAIL: verifier output missing bad count.")
        return 1
    print(ok_msg)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to jsonl dataset")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--mode", choices=("answer-sanity", "proof-sanity"), default="answer-sanity")
    args = parser.parse_args()
    raise SystemExit(main(args.path, args.n, args.mode))
