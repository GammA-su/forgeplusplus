from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from fc.model.backbone import BackboneConfig
from fc.model.baseline import BaselineConfig, BaselineModel
from fc.train.answer import AnswerVocab, serialize_answer
from fc.train.data import Example, TextVocab, load_dataset, load_dataset_with_variants
from fc.util.jsonl import write_jsonl
from fc.util.logging import configure_logging, get_logger
from fc.util.seed import set_seed


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    steps: int
    batch_size: int
    lr: float
    max_text_len: int
    max_answer_len: int | None = None


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_examples(
    schema_path: str,
    math_path: str,
    csp_path: str,
    include_variants: bool = True,
) -> list[Example]:
    loader = load_dataset_with_variants if include_variants else load_dataset
    return loader(schema_path) + loader(math_path) + loader(csp_path)


def _collate_batch(
    batch: list[Example],
    text_vocab: TextVocab,
    answer_vocab: AnswerVocab,
    max_text_len: int,
    max_answer_len: int,
) -> dict[str, Any]:
    input_ids = torch.tensor([text_vocab.encode(ex.x, max_text_len) for ex in batch], dtype=torch.long)
    answer_ids = torch.tensor(
        [answer_vocab.encode(serialize_answer(ex.y), max_answer_len) for ex in batch],
        dtype=torch.long,
    )
    return {"input_ids": input_ids, "answer_ids": answer_ids}


def train(
    examples: list[Example],
    config_path: str,
    out_dir: str,
    device: str | torch.device | None = None,
) -> Path:
    cfg = load_config(config_path)
    train_params = {k: v for k, v in cfg["train"].items() if k in TrainConfig.__dataclass_fields__}
    train_cfg = TrainConfig(**train_params)
    set_seed(train_cfg.seed)
    configure_logging()
    logger = get_logger(__name__)

    texts = [ex.x for ex in examples]
    text_vocab = TextVocab.build(texts)
    answers = [serialize_answer(ex.y) for ex in examples]
    answer_vocab = AnswerVocab.build(answers)

    cfg["text_vocab_size"] = len(text_vocab.token_to_id)
    max_answer_len = train_cfg.max_answer_len or cfg.get("max_answer_len") or cfg.get("max_prog_len", 64)
    cfg["max_answer_len"] = max_answer_len
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    mcfg = BaselineConfig(
        vocab_size=cfg["text_vocab_size"],
        answer_vocab_size=len(answer_vocab.token_to_id),
        max_answer_len=max_answer_len,
        backbone=bcfg,
    )
    model = BaselineModel(mcfg)
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.train()
    logger.info("baseline trainer device=%s", torch_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    pad_id = answer_vocab.token_to_id["<PAD>"]
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    train_logs: list[dict[str, Any]] = []

    for step in range(train_cfg.steps):
        batch = [examples[(step + i) % len(examples)] for i in range(train_cfg.batch_size)]
        batch_data = _collate_batch(batch, text_vocab, answer_vocab, train_cfg.max_text_len, max_answer_len)
        input_ids = batch_data["input_ids"].to(torch_device)
        answer_ids = batch_data["answer_ids"].to(torch_device)
        outputs = model(input_ids)
        logits = outputs["logits"]
        ce = ce_loss_fn(logits.view(-1, logits.size(-1)), answer_ids.view(-1))

        optimizer.zero_grad(set_to_none=True)
        ce.backward()
        optimizer.step()

        log_row = {"step": step, "loss": float(ce.item())}
        train_logs.append(log_row)
        if step % 10 == 0:
            logger.info("baseline step=%d loss=%.4f", step, ce.item())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "mode": "baseline",
        "model": model.state_dict(),
        "text_vocab": text_vocab.token_to_id,
        "answer_vocab": answer_vocab.token_to_id,
        "config": cfg,
    }
    ckpt_path = out_path / "baseline_ckpt.pt"
    torch.save(ckpt, ckpt_path)
    (out_path / "baseline_text_vocab.json").write_text(json.dumps(text_vocab.token_to_id, indent=2))
    (out_path / "baseline_answer_vocab.json").write_text(json.dumps(answer_vocab.token_to_id, indent=2))
    write_jsonl(out_path / "baseline_train_log.jsonl", train_logs)
    return ckpt_path


def train_from_paths(
    config_path: str,
    out_dir: str,
    schema_path: str = "out/data/schema.jsonl",
    math_path: str = "out/data/math.jsonl",
    csp_path: str = "out/data/csp.jsonl",
    device: str | torch.device | None = None,
) -> Path:
    examples = load_examples(schema_path, math_path, csp_path, include_variants=True)
    return train(examples, config_path=config_path, out_dir=out_dir, device=device)
