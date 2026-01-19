from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
from typing import Any, Iterable

def _hash_tokens(tokens: Iterable[Any]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _extract_proof_tokens(proof: Any) -> list[Any]:
    if isinstance(proof, dict):
        return proof.get("tokens") or []
    if isinstance(proof, list):
        return proof
    if isinstance(proof, str):
        try:
            parsed = json.loads(proof)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            return parsed.get("tokens") or []
        if isinstance(parsed, list):
            return parsed
    return []

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: backfill_proof_sha256.py <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    n = 0
    patched = 0

    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            rec = json.loads(line)

            proof = rec.get("proof")
            if isinstance(proof, dict):
                tokens = _extract_proof_tokens(proof)
                if tokens and not proof.get("sha256"):
                    proof["sha256"] = _hash_tokens(tokens)
                    patched += 1
            # if proof is list/str, leave it as-is (verifier expects dict w/ sha256 now)

            g.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"rows={n} patched={patched} out={outp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
