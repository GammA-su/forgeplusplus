from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numcanon
from fc.util.runtime_solve import runtime_solve


def _hash_tokens(tokens: Iterable[Any]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_proof_tokens(proof: Any) -> tuple[list[Any], str | None, bool]:
    """
    Returns (tokens, sha, is_dict_proof).
    Accepts proof as dict {"tokens":[...],"sha256":"..."}, list tokens, or JSON string.
    """
    tokens: list[Any] = []
    sha: str | None = None
    is_dict = False

    if isinstance(proof, dict):
        is_dict = True
        tokens = proof.get("tokens") or []
        sha = proof.get("sha256")
    elif isinstance(proof, list):
        tokens = proof
    elif isinstance(proof, str):
        try:
            parsed = json.loads(proof)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            is_dict = True
            tokens = parsed.get("tokens") or []
            sha = parsed.get("sha256")
        elif isinstance(parsed, list):
            tokens = parsed

    return tokens, sha, is_dict


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: fixup_y_to_runtime.py <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    rows = 0
    patched_y = 0
    patched_sha = 0
    bad = 0

    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows += 1

            try:
                rec = json.loads(line)
                proof = rec.get("proof")
                tokens, sha, is_dict = _extract_proof_tokens(proof)

                if not tokens:
                    bad += 1
                    raise ValueError("missing proof tokens")

                # Ensure sha256 exists and matches tokens (backfill if needed)
                want_sha = _hash_tokens(tokens)
                if is_dict and isinstance(proof, dict):
                    if not sha:
                        proof["sha256"] = want_sha
                        patched_sha += 1
                    elif sha != want_sha:
                        # keep as-is; verify_proofs should fail this dataset if corrupted
                        pass

                got = runtime_solve(rec.get("x", ""), rec.get("constraints") or [], tokens)
                got = numcanon.canon_json(got)
                if numcanon.canon_json(rec.get("y")) != got:
                    rec["y"] = got
                    patched_y += 1

                g.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")

            except Exception as exc:
                bad += 1
                # emit the original line so you can inspect, but still fail with rc=1
                g.write(line + "\n")
                print(f"[BAD] line {ln}: {exc}", file=sys.stderr)

    print(f"in={inp} out={outp} rows={rows} patched_y={patched_y} patched_sha={patched_sha} bad={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
