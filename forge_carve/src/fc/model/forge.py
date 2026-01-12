from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import torch
from torch import nn

from fc.dsl.program import Program
from fc.model.backbone import BackboneConfig, TransformerBackbone
from fc.model.primal_dual import PrimalDualConfig, PrimalDualModule
from fc.model.slots import SlotConfig, SlotModule
from fc.model.policy import ProgramPolicy


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_prog_len: int
    backbone: BackboneConfig
    slots: SlotConfig
    primal_dual: PrimalDualConfig


class ForgeModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(cfg.backbone)
        self.slots = SlotModule(cfg.slots, d_model=cfg.backbone.d_model)
        self.primal_dual = PrimalDualModule(cfg.primal_dual, d_model=cfg.backbone.d_model, num_slots=cfg.slots.num_slots)
        self.policy = ProgramPolicy(cfg.slots.d_slot, cfg.slots.num_slots, cfg.vocab_size, cfg.max_prog_len)
        self.prog_decoder: Callable[[list[int]], Program] | None = None

    def set_prog_decoder(self, decoder: Callable[[list[int]], Program]) -> None:
        self.prog_decoder = decoder

    def forward(self, input_ids: torch.Tensor) -> dict[str, Any]:
        hidden = self.backbone(input_ids)
        slots, state_logits = self.slots(hidden)
        pd_out = self.primal_dual(slots, hidden)
        logits, candidate_ids, candidate_scores, chosen_idx = self.policy.select_best(pd_out.slots, k=3, topk=5)
        program_ids = candidate_ids[torch.arange(candidate_ids.size(0), device=input_ids.device), chosen_idx]
        programs: list[Program] = []
        if self.prog_decoder is not None:
            for row in program_ids.detach().cpu().tolist():
                try:
                    programs.append(self.prog_decoder(row))
                except Exception:
                    programs.append(Program(instructions=[]))
        return {
            "logits": logits,
            "mu": pd_out.mu,
            "c_soft": pd_out.c_soft,
            "c_hard": None,
            "state_logits": state_logits,
            "program_ids": program_ids,
            "candidate_ids": candidate_ids,
            "candidate_scores": candidate_scores,
            "chosen_index": chosen_idx,
            "programs": programs,
        }

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(input_ids)
        return torch.argmax(outputs["logits"], dim=-1)
