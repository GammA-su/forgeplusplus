from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PrimalDualConfig:
    num_constraints: int = 7
    steps: int = 2
    d_slot: int = 128


@dataclass
class PrimalDualOutput:
    slots: torch.Tensor
    mu: torch.Tensor
    c_soft: torch.Tensor


class PrimalDualModule(nn.Module):
    def __init__(self, cfg: PrimalDualConfig, d_model: int, num_slots: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_slots = num_slots
        self.constraint_head = nn.Linear(d_model + cfg.d_slot * num_slots, cfg.num_constraints)
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model + cfg.d_slot * num_slots + cfg.num_constraints, cfg.d_slot * num_slots),
            nn.ReLU(),
            nn.Linear(cfg.d_slot * num_slots, cfg.d_slot * num_slots),
        )
        self.gamma = nn.Parameter(torch.ones(cfg.steps, cfg.num_constraints))

    def forward(self, slots: torch.Tensor, hidden: torch.Tensor) -> PrimalDualOutput:
        pooled = hidden.mean(dim=1)
        mu = torch.zeros(slots.shape[0], self.cfg.num_constraints, device=slots.device)
        c_soft = torch.zeros_like(mu)
        for step in range(self.cfg.steps):
            flat_slots = slots.flatten(start_dim=1)
            feat = torch.cat([pooled, flat_slots], dim=-1)
            c_soft = self.constraint_head(feat)
            gamma = self.gamma[step].unsqueeze(0)
            mu = torch.relu(mu + gamma * c_soft)
            upd_in = torch.cat([feat, mu], dim=-1)
            delta = self.update_mlp(upd_in).view(-1, self.num_slots, self.cfg.d_slot)
            slots = slots + delta
        return PrimalDualOutput(slots=slots, mu=mu, c_soft=c_soft)
