from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("train_phase_script", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load 03_train_phase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_proof_tokens_and_preflight(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "proof": ["OP", "STR:+", "END"]},
        {"id": "b", "proof": "OP STR:+ END"},
        {"id": "c", "proof": {"tokens": ["OP", "STR:+", "END"]}},
    ]
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"prog_vocab": {"OP": 0, "STR:+": 1, "END": 2}}, ckpt_path)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "03_train_phase.py"
    module = _load_script(script_path)

    for row in rows:
        tokens = module._read_proof_tokens(row, "proof")
        assert tokens == ["OP", "STR:+", "END"]

    vocab = module._load_prog_vocab(str(ckpt_path))
    missing_counts, first_bad = module._scan_preflight(
        data_path=str(data_path), proof_source="proof", vocab=vocab, limit=10
    )
    assert missing_counts == {}
    assert first_bad is None
    module._preflight_or_raise(
        data_path=str(data_path), proof_source="proof", vocab=vocab, limit=10, logger=None
    )

    rows[1]["proof"] = ["OP", "MISSING", "END"]
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    missing_counts, first_bad = module._scan_preflight(
        data_path=str(data_path), proof_source="proof", vocab=vocab, limit=10
    )
    assert "MISSING" in missing_counts
    assert first_bad is not None
    with pytest.raises(ValueError):
        module._preflight_or_raise(
            data_path=str(data_path), proof_source="proof", vocab=vocab, limit=10, logger=None
        )
