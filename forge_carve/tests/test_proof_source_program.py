from fc.train.data import audit_proof_tokens


def test_audit_program_source_accepts_program_only() -> None:
    row = {
        "id": "x0",
        "domain": "math",
        "x": "Compute 1 + 1.",
        "y": 2,
        "constraints": [],
        "proof": {},
        "program": ["EXTRACT_INT", "EXTRACT_INT", "APPLY_ARITH", "EMIT_NUM"],
    }
    tokens = audit_proof_tokens([row], proof_source="program")
    assert "OP" in tokens
    assert "EMIT_NUM" in tokens


def test_audit_proof_source_ignores_program_field() -> None:
    row = {
        "id": "x1",
        "domain": "math",
        "x": "Compute 1 + 1.",
        "y": 2,
        "constraints": [],
        "proof": {"dsl": "PTv1", "tokens": ["<BOS>", "BEGIN", "OP", "EMIT_NUM", "END", "<EOS>"]},
        "program": ["BOGUS_OPCODE"],
    }
    tokens = audit_proof_tokens([row], proof_source="proof")
    assert "EMIT_NUM" in tokens
