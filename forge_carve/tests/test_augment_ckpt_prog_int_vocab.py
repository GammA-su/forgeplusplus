from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("augment_ckpt_prog_int_vocab", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load augment_ckpt_prog_int_vocab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_augment_ckpt_prog_int_vocab(tmp_path: Path) -> None:
    in_ckpt = tmp_path / "in.pt"
    out_ckpt = tmp_path / "out.pt"
    prog_vocab = ["INT:0", "OP:APPLY_ARITH"]
    max_prog_len = 2
    vocab_size = len(prog_vocab)
    hidden = 3
    weight = torch.arange(max_prog_len * vocab_size * hidden, dtype=torch.float32).view(
        max_prog_len * vocab_size, hidden
    )
    bias = torch.arange(max_prog_len * vocab_size, dtype=torch.float32)
    ckpt = {
        "prog_vocab": prog_vocab,
        "max_prog_len": max_prog_len,
        "model": {
            "policy.head.weight": weight.clone(),
            "policy.head.bias": bias.clone(),
        },
    }
    torch.save(ckpt, in_ckpt)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "augment_ckpt_prog_int_vocab.py"
    module = _load_script(script_path)
    assert (
        module.main(
            [
                "--in-ckpt",
                str(in_ckpt),
                "--out-ckpt",
                str(out_ckpt),
                "--max-abs",
                "3",
            ]
        )
        == 0
    )
    out = torch.load(out_ckpt, map_location="cpu")
    new_vocab = out["prog_vocab"]
    assert "INT:3" in new_vocab
    assert "INT:-3" in new_vocab
    new_weight = out["model"]["policy.head.weight"]
    new_bias = out["model"]["policy.head.bias"]
    new_vocab_size = len(new_vocab)
    assert new_weight.shape == (max_prog_len * new_vocab_size, hidden)
    assert new_bias.shape == (max_prog_len * new_vocab_size,)

    old_map = {tok: idx for idx, tok in enumerate(prog_vocab)}
    new_w = new_weight.view(max_prog_len, new_vocab_size, hidden)
    new_b = new_bias.view(max_prog_len, new_vocab_size)
    old_w = weight.view(max_prog_len, vocab_size, hidden)
    old_b = bias.view(max_prog_len, vocab_size)
    for tok, old_id in old_map.items():
        new_id = new_vocab[tok]
        assert torch.equal(new_w[:, new_id, :], old_w[:, old_id, :])
        assert torch.equal(new_b[:, new_id], old_b[:, old_id])
