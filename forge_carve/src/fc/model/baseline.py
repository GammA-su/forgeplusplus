from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fc.model.backbone import BackboneConfig, TransformerBackbone


@dataclass(frozen=True)
class BaselineConfig:
    vocab_size: int
    answer_vocab_size: int
    max_answer_len: int
    backbone: BackboneConfig


class BaselineModel(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = TransformerBackbone(cfg.backbone)
        self.head = nn.Linear(cfg.backbone.d_model, cfg.max_answer_len * cfg.answer_vocab_size)

    def forward(self, input_ids: torch.Tensor) -> dict[str, Any]:
        hidden = self.backbone(input_ids)
        pooled = hidden.mean(dim=1)
        logits = self.head(pooled).view(-1, self.cfg.max_answer_len, self.cfg.answer_vocab_size)
        answer_ids = torch.argmax(logits, dim=-1)
        return {"logits": logits, "answer_ids": answer_ids}

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(input_ids)
        return outputs["answer_ids"]
