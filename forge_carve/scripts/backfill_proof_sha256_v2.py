from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
from typing import Any, Iterable, Tuple

def hash_tokens(tokens: Iterable[Any]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def extract_tokens_and_extra(proof: Any) -> Tuple[list[Any], dict[str, Any]]:
    # returns (tokens, extra_fields_if_dict)
    if isinstance(proof, dict):
        tokens = proof.get("tokens") or []
        extra = {k: v for k, v in proof.items() if k not in ("tokens", "sha256")}
        return tokens, extra
    if isinstance(proof, list):
        return proof, {}
    if isinstance(proof, str):
        try:
            parsed = json.loads(proof)
        except json.JSONDecodeError:
            return [], {}
        if isinstance(parsed, dict):
            tokens = parsed.get("tokens") or []
            extra = {k: v for k, v in parsed.items() if k not in ("tokens", "sha256")}
            return tokens, extra
        if isinstance(parsed, list):
            return parsed, {}
    return [], {}

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: backfill_proof_sha256_v2.py <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    rows = 0
    patched = 0
    normalized = 0
    empty_tokens = 0

    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            rec = json.loads(line)

            proof = rec.get("proof")
            tokens, extra = extract_tokens_and_extra(proof)
            if not tokens:
                empty_tokens += 1
                # keep as-is; verifier will fail these (which is correct)
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            want = hash_tokens(tokens)

            if not isinstance(proof, dict):
                normalized += 1
                rec["proof"] = {"tokens": tokens, "sha256": want, **extra}
                patched += 1
            else:
                have = proof.get("sha256")
                if have != want:
                    proof["sha256"] = want
                    patched += 1

            g.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"in={inp} out={outp} rows={rows} patched={patched} normalized={normalized} empty_tokens={empty_tokens}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
