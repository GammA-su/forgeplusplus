from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from fc.util.vocab_identity import vocab_identity


def _load_prog_vocab(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {tok: idx for idx, tok in enumerate(raw)}
    raise ValueError("ckpt.prog_vocab must be a dict or list")


def _resize_policy_head(
    model: dict[str, torch.Tensor],
    old_vocab: dict[str, int],
    new_vocab: dict[str, int],
    max_prog_len: int,
) -> None:
    weight_key = "policy.head.weight"
    bias_key = "policy.head.bias"
    if weight_key not in model or bias_key not in model:
        raise ValueError("ckpt missing policy head weights")
    weight = model[weight_key]
    bias = model[bias_key]
    hidden = weight.shape[1]
    old_vocab_size = len(old_vocab)
    new_vocab_size = len(new_vocab)
    expected_rows = max_prog_len * old_vocab_size
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"policy head shape mismatch expected_rows={expected_rows} "
            f"actual_rows={weight.shape[0]}"
        )
    old_w = weight.view(max_prog_len, old_vocab_size, hidden)
    old_b = bias.view(max_prog_len, old_vocab_size)

    base_id = old_vocab.get("INT:0")
    if base_id is not None:
        base_w = old_w[:, base_id, :].clone()
        base_b = old_b[:, base_id].clone()
        new_w = base_w.unsqueeze(1).expand(max_prog_len, new_vocab_size, hidden).clone()
        new_b = base_b.unsqueeze(1).expand(max_prog_len, new_vocab_size).clone()
    else:
        new_w = torch.zeros((max_prog_len, new_vocab_size, hidden), dtype=weight.dtype, device=weight.device)
        new_b = torch.zeros((max_prog_len, new_vocab_size), dtype=bias.dtype, device=bias.device)

    for tok, old_id in old_vocab.items():
        new_id = new_vocab[tok]
        new_w[:, new_id, :] = old_w[:, old_id, :]
        new_b[:, new_id] = old_b[:, old_id]

    model[weight_key] = new_w.view(max_prog_len * new_vocab_size, hidden)
    model[bias_key] = new_b.view(max_prog_len * new_vocab_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Augment checkpoint prog_vocab with INT literals.")
    parser.add_argument("--in-ckpt", default="out/ckpt.pt", help="Input checkpoint")
    parser.add_argument("--out-ckpt", required=True, help="Output checkpoint path")
    parser.add_argument("--max-abs", type=int, default=20000, help="Max abs INT literal to add")
    args = parser.parse_args(argv)

    ckpt = torch.load(args.in_ckpt, map_location="cpu")
    prog_vocab_raw = ckpt.get("prog_vocab")
    if prog_vocab_raw is None:
        raise ValueError("ckpt missing prog_vocab")
    prog_vocab = _load_prog_vocab(prog_vocab_raw)
    max_id = max(prog_vocab.values(), default=-1)
    next_id = max_id + 1
    for val in range(-args.max_abs, args.max_abs + 1):
        tok = f"INT:{val}"
        if tok not in prog_vocab:
            prog_vocab[tok] = next_id
            next_id += 1

    max_prog_len = ckpt.get("max_prog_len")
    if max_prog_len is None:
        raise ValueError("ckpt missing max_prog_len")
    model = ckpt.get("model") or ckpt.get("state_dict")
    if not isinstance(model, dict):
        raise ValueError("ckpt missing model state dict")
    _resize_policy_head(model, _load_prog_vocab(prog_vocab_raw), prog_vocab, max_prog_len)
    if "model" in ckpt:
        ckpt["model"] = model
    else:
        ckpt["state_dict"] = model

    vocab_id = vocab_identity(prog_vocab)
    ckpt["prog_vocab"] = vocab_id.token_to_id
    ckpt["prog_vocab_tokens"] = vocab_id.tokens_by_id
    ckpt["prog_vocab_sha256"] = vocab_id.sha256

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(
        f"wrote {out_path} vocab_size={len(prog_vocab)} "
        f"max_abs={args.max_abs} added={len(prog_vocab) - len(_load_prog_vocab(prog_vocab_raw))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
