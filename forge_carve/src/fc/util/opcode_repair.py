from __future__ import annotations

from typing import Callable

import torch

from fc.dsl.tokens import OPCODES, TokenVocab


def repair_invalid_opcodes(
    pred_ids: list[int],
    logits: torch.Tensor,
    vocab: TokenVocab,
    *,
    k: int = 30,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[int], bool]:
    if not pred_ids:
        return pred_ids, False
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [seq_len, vocab]")
    valid_ops = set(OPCODES)
    tokens = [vocab.decode(i) for i in pred_ids]
    repaired_ids = list(pred_ids)
    repaired = False
    seq_len = min(len(tokens), logits.shape[0])
    logits_cpu = logits.detach().cpu()

    for idx in range(1, seq_len):
        if tokens[idx - 1] == "<EOS>":
            break
        if tokens[idx - 1] != "OP":
            continue
        if tokens[idx] in valid_ops:
            continue

        if log_fn:
            log_fn(f"  op_repair_pos={idx} invalid_token={tokens[idx]}")

        row = logits_cpu[idx]
        k_eff = max(1, min(int(k), row.numel()))
        probs = torch.softmax(row, dim=-1)
        topk = torch.topk(probs, k_eff)
        candidates: list[tuple[str, float]] = []
        chosen: tuple[str, int, float] | None = None
        for prob, tok_id in zip(topk.values.tolist(), topk.indices.tolist()):
            tok = vocab.decode(int(tok_id))
            if tok in valid_ops:
                candidates.append((tok, float(prob)))
                if chosen is None:
                    chosen = (tok, int(tok_id), float(prob))

        if log_fn:
            if candidates:
                cand_str = ", ".join([f"{tok}:{prob:.4g}" for tok, prob in candidates])
            else:
                cand_str = "<none>"
            log_fn(f"  op_repair_candidates={cand_str}")

        if chosen is not None:
            repaired_ids[idx] = chosen[1]
            tokens[idx] = chosen[0]
            repaired = True
            if log_fn:
                log_fn(f"  op_repair_chosen={chosen[0]} prob={chosen[2]:.4g}")
        else:
            if log_fn:
                log_fn("  op_repair_chosen=<none>")

    return repaired_ids, repaired
