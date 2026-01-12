from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SlotConfig:
    num_slots: int = 4
    d_slot: int = 128
    num_states: int = 4


class SlotModule(nn.Module):
    def __init__(self, cfg: SlotConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.to_slots = nn.Linear(d_model, cfg.num_slots * cfg.d_slot)
        self.state_head = nn.Linear(d_model, cfg.num_states)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mean pool
        pooled = hidden.mean(dim=1)
        slots = self.to_slots(pooled).view(-1, self.cfg.num_slots, self.cfg.d_slot)
        state_logits = self.state_head(pooled)
        return slots, state_logits
