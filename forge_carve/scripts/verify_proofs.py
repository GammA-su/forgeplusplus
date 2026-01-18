from __future__ import annotations
import numcanon

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from fc.util.tags import DOMAIN_TAGS
from fc.util.runtime_solve import runtime_solve


def _eq(a: Any, b: Any) -> bool:
    # float tolerant compare (only where needed)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9
    return a == b


def _hash_tokens(tokens: Iterable[Any]) -> str:
    payload = json.dumps(list(tokens), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_proof_tokens(proof: Any) -> tuple[list[Any], str | None]:
    tokens: list[Any] = []
    proof_hash: str | None = None
    if isinstance(proof, dict):
        tokens = proof.get("tokens") or []
        proof_hash = proof.get("sha256")
    elif isinstance(proof, list):
        tokens = proof
    elif isinstance(proof, str):
        try:
            parsed = json.loads(proof)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tokens = parsed.get("tokens") or []
            proof_hash = parsed.get("sha256")
        elif isinstance(parsed, list):
            tokens = parsed
    return tokens, proof_hash


def _missing_keys(rec: dict[str, Any], required: Iterable[str]) -> list[str]:
    return [key for key in required if key not in rec or rec.get(key) is None]


def _fail(rid: str, ln: int, reason: str) -> None:
    print(f"[FAIL] {rid} @ line {ln} reason={reason}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_proofs.py <path.jsonl>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    allowed_tags = set(DOMAIN_TAGS.values())
    seen_signatures: set[str] = set()

    bad = 0
    checked = 0

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            checked += 1
            rid = f"line:{ln}"
            try:
                rec: Dict[str, Any] = json.loads(line)
                rid = rec.get("id", rid)

                missing = _missing_keys(rec, ("x", "y", "proof", "constraints", "domain_tag"))
                if missing:
                    bad += 1
                    _fail(rid, ln, f"missing_keys:{missing}")
                    continue

                domain_tag = rec.get("domain_tag")
                if domain_tag not in allowed_tags:
                    bad += 1
                    _fail(rid, ln, f"unknown_domain_tag:{domain_tag}")
                    continue

                x_val = rec.get("x")
                if not isinstance(x_val, str) or not x_val.strip():
                    bad += 1
                    _fail(rid, ln, "degenerate_x")
                    continue

                constraints = rec.get("constraints")
                if not isinstance(constraints, list):
                    bad += 1
                    _fail(rid, ln, "degenerate_constraints")
                    continue

                proof = rec.get("proof")
                tokens, proof_hash = _extract_proof_tokens(proof)
                if not tokens:
                    bad += 1
                    _fail(rid, ln, "missing_proof_tokens")
                    continue
                if not proof_hash:
                    bad += 1
                    _fail(rid, ln, "missing_proof_sha256")
                    continue
                if _hash_tokens(tokens) != proof_hash:
                    bad += 1
                    _fail(rid, ln, "proof_sha256_mismatch")
                    continue

                try:
                    signature = json.dumps(
                        {
                            "domain_tag": domain_tag,
                            "x": x_val,
                            "y": rec.get("y"),
                            "constraints": constraints,
                            "proof": proof,
                        },
                        sort_keys=True,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                except Exception as exc:
                    bad += 1
                    _fail(rid, ln, f"degenerate_signature:{exc}")
                    continue
                if signature in seen_signatures:
                    bad += 1
                    _fail(rid, ln, "duplicate_record")
                    continue
                seen_signatures.add(signature)

                got = runtime_solve(x_val, constraints or [], tokens)
                got = numcanon.canon_json(got)
                exp = rec.get("y")
                exp = numcanon.canon_json(exp)
                if not _eq(got, exp):
                    bad += 1
                    _fail(rid, ln, "answer_mismatch")
                    print("  x:", rec.get("x"))
                    print("  exp:", exp)
                    print("  got:", got)
            except Exception as exc:
                bad += 1
                _fail(rid, ln, f"verify_exception:{exc}")
                continue

    print(f"checked={checked} bad={bad}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
