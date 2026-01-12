from __future__ import annotations

import torch
from torch import nn


def kkt_loss(c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    relu_c = torch.relu(c)
    return (relu_c**2 + torch.nn.functional.softplus(-mu) ** 2 + (mu * relu_c) ** 2).mean()


def mdl_loss(program_len: torch.Tensor, mu: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return alpha * program_len.float().mean() + beta * mu.abs().mean()


def regret_loss(losses: torch.Tensor, chosen_idx: torch.Tensor, margin: float) -> torch.Tensor:
    # losses: [batch, k]
    min_loss, _ = losses.min(dim=1)
    chosen = losses.gather(1, chosen_idx.view(-1, 1)).squeeze(1)
    return torch.relu(chosen - min_loss + margin).mean()


def _normalize(mu: torch.Tensor) -> torch.Tensor:
    mu = torch.relu(mu)
    return mu / (mu.sum(dim=-1, keepdim=True) + 1e-8)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = _normalize(p)
    q = _normalize(q)
    return (p * (p.add(1e-8).log() - q.add(1e-8).log())).sum(dim=-1).mean()


def orbit_invariance_loss(mu: torch.Tensor, mu_orbit: torch.Tensor, prog_dist: torch.Tensor) -> torch.Tensor:
    return kl_divergence(mu, mu_orbit) + prog_dist.float().mean()


def causal_faithfulness_loss(mu: torch.Tensor, mu_flip: torch.Tensor, delta: float) -> torch.Tensor:
    kl = kl_divergence(mu, mu_flip)
    return torch.relu(delta - kl)


def state_progress_loss(
    state_logits: torch.Tensor,
    state_targets: torch.Tensor,
    adjacency: torch.Tensor,
    next_states: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()(state_logits, state_targets)
    if next_states is None:
        return ce
    # Penalize invalid transitions
    probs = torch.softmax(state_logits, dim=-1)
    # Simple surrogate: expected invalid transition probability
    invalid = 1.0 - adjacency[state_targets, next_states]
    return ce + invalid.float().mean() * probs.mean()
