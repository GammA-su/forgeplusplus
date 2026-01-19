from __future__ import annotations

import json
from pathlib import Path

from prooftape.ptv1 import PTv1Runtime
from fc.verify.schema import SchemaVerifier


def test_schema_dataset_schema_49_verifies() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "out" / "data" / "schema.jsonl"
    assert path.exists()
    target = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") == "schema_49":
                target = rec
                break
    assert target is not None
    proof = target.get("proof", {})
    assert isinstance(proof, dict) and proof.get("tokens")
    out = PTv1Runtime().run(target.get("x", ""), target.get("constraints") or [], proof.get("tokens"))
    res = SchemaVerifier().verify(target.get("x", ""), program=None, output=out, constraints=target.get("constraints"))
    assert res.valid
