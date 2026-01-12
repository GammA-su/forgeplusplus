from __future__ import annotations

import torch
from torch import nn


class ProgramPolicy(nn.Module):
    def __init__(self, d_slot: int, num_slots: int, vocab_size: int, max_prog_len: int) -> None:
        super().__init__()
        self.max_prog_len = max_prog_len
        self.head = nn.Linear(d_slot * num_slots, max_prog_len * vocab_size)
        self.vocab_size = vocab_size

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        flat = slots.flatten(start_dim=1)
        logits = self.head(flat)
        return logits.view(-1, self.max_prog_len, self.vocab_size)

    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        logits = self.forward(slots)
        return torch.argmax(logits, dim=-1)

    def sample_k(self, slots: torch.Tensor, k: int = 3, topk: int = 5) -> torch.Tensor:
        logits = self.forward(slots)
        return self.sample_k_from_logits(logits, k=k, topk=topk)

    def sample_k_from_logits(self, logits: torch.Tensor, k: int = 3, topk: int = 5) -> torch.Tensor:
        topk = max(1, min(topk, logits.size(-1)))
        _, top_idx = torch.topk(logits, k=topk, dim=-1)
        batch, seq, _ = top_idx.shape
        candidates = []
        for i in range(k):
            choice = top_idx[..., i % topk]
            candidates.append(choice)
        return torch.stack(candidates, dim=1)

    def score_candidates(self, logits: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # candidates: [batch, k, seq]
        log_probs = torch.log_softmax(logits, dim=-1)
        batch, k, seq = candidates.shape
        gather_idx = candidates.unsqueeze(-1)
        gathered = torch.gather(log_probs.unsqueeze(1).expand(batch, k, seq, -1), -1, gather_idx).squeeze(-1)
        return gathered.sum(dim=-1)

    def select_best(self, slots: torch.Tensor, k: int = 3, topk: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(slots)
        candidates = self.sample_k_from_logits(logits, k=k, topk=topk)
        scores = self.score_candidates(logits, candidates)
        best_idx = torch.argmax(scores, dim=1)
        batch = candidates.size(0)
        best = candidates[torch.arange(batch, device=slots.device), best_idx]
        return logits, candidates, scores, best_idx
