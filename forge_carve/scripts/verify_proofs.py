from __future__ import annotations
import numcanon

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

from fc.util.tags import DOMAIN_TAGS
from prooftape.ptv1 import PTv1Runtime

_REQUIRED_KEYS = ("x", "y", "domain_tag", "proof")
_ALLOWED_TAGS = set(DOMAIN_TAGS.values())


def _eq(a: Any, b: Any) -> bool:
    return numcanon.json_equal(a, b)


def _hash_tokens(tokens: list[Any]) -> str:
    payload = json.dumps(tokens, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signature(rec: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "domain_tag": rec.get("domain_tag", ""),
            "x": rec.get("x", ""),
            "constraints": rec.get("constraints", []),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _report_fail(
    reason: str,
    rec: dict[str, Any] | None,
    ln: int,
    *,
    got: Any | None = None,
    exp: Any | None = None,
    exc: Exception | None = None,
) -> None:
    rid = f"line:{ln}"
    if isinstance(rec, dict) and rec.get("id"):
        rid = str(rec.get("id"))
    print(f"[FAIL] {rid} @ line {ln} reason={reason}")
    if isinstance(rec, dict) and rec.get("x") is not None:
        print("  x:", rec.get("x"))
    if exp is not None:
        print("  exp:", exp)
    if got is not None:
        print("  got:", got)
    if exc is not None:
        print("  error:", exc)


def _verify_record(rec: dict[str, Any], rt: PTv1Runtime, seen_sigs: set[str]) -> Any:
    missing = [k for k in _REQUIRED_KEYS if k not in rec]
    if missing:
        raise ValueError(f"missing_keys:{missing}")
    domain_tag = rec.get("domain_tag")
    if domain_tag not in _ALLOWED_TAGS:
        raise ValueError(f"unknown_domain_tag:{domain_tag}")
    proof = rec.get("proof")
    if not isinstance(proof, dict):
        raise ValueError("proof_not_object")
    tokens = proof.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("missing_tokens")
    sha = proof.get("sha256")
    if not isinstance(sha, str) or not sha:
        raise ValueError("missing_proof_sha256")
    digest = _hash_tokens(tokens)
    if sha != digest:
        raise ValueError("proof_sha256_mismatch")
    sig = _signature(rec)
    if sig in seen_sigs:
        raise ValueError("duplicate_signature")
    seen_sigs.add(sig)
    return rt.run(rec.get("x", ""), rec.get("constraints") or [], tokens)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_proofs.py <path.jsonl>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    rt = PTv1Runtime()

    bad = 0
    checked = 0
    seen_sigs: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                checked += 1
                rec: Dict[str, Any] | None = None
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise ValueError("record_not_object")
                    got = _verify_record(rec, rt, seen_sigs)
                    got = numcanon.canon_json(got)
                    exp = rec.get("y")
                    exp = numcanon.canon_json(exp)
                    if not _eq(got, exp):
                        bad += 1
                        _report_fail("answer_mismatch", rec, ln, got=got, exp=exp)
                except Exception as exc:
                    bad += 1
                    _report_fail(str(exc), rec, ln, exc=exc)
    finally:
        print(f"checked={checked} bad={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
