import json
from pathlib import Path

import torch
import yaml

from fc.dsl.tokens import build_default_vocab
from fc.model.backbone import BackboneConfig
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig
from fc.train.data import TextVocab, generate_dataset, save_dataset
from fc.train.trainer import train_from_dataset
from fc.util.vocab_identity import vocab_identity


def _write_init_ckpt(tmp_path: Path, text_vocab: TextVocab, max_prog_len: int) -> Path:
    prog_vocab = build_default_vocab()
    cfg = {
        "text_vocab_size": len(text_vocab.token_to_id),
        "max_prog_len": max_prog_len,
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.0, "max_len": 16},
        "slots": {"num_slots": 2, "d_slot": 16, "num_states": 2},
        "primal_dual": {"num_constraints": 7, "steps": 1, "d_slot": 16},
        "train": {"max_text_len": 16, "max_prog_len": max_prog_len},
    }
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(
        vocab_size=len(prog_vocab.token_to_id),
        max_prog_len=max_prog_len,
        backbone=bcfg,
        slots=scfg,
        primal_dual=pcfg,
    )
    model = ForgeModel(mcfg)
    prog_id = vocab_identity(prog_vocab.token_to_id)
    ckpt = {
        "mode": "forge",
        "model": model.state_dict(),
        "text_vocab": text_vocab.token_to_id,
        "prog_vocab": prog_vocab.token_to_id,
        "prog_vocab_sha256": prog_id.sha256,
        "max_prog_len": max_prog_len,
        "config": cfg,
    }
    path = tmp_path / "init_ckpt.pt"
    torch.save(ckpt, path)
    return path


def test_train_reuses_init_text_vocab(tmp_path: Path) -> None:
    dataset = generate_dataset("math", n=2, seed=7)
    for ex in dataset:
        ex.x = f"{ex.x} newword"
    data_path = tmp_path / "math.jsonl"
    save_dataset(str(data_path), dataset)

    init_text_vocab = TextVocab.build(["known"])
    init_ckpt = _write_init_ckpt(tmp_path, init_text_vocab, max_prog_len=16)

    cfg = {
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.0, "max_len": 32},
        "slots": {"num_slots": 2, "d_slot": 16, "num_states": 2},
        "primal_dual": {"num_constraints": 7, "steps": 1, "d_slot": 16},
        "max_prog_len": 16,
        "train": {
            "seed": 1,
            "steps": 1,
            "batch_size": 2,
            "lr": 0.001,
            "max_text_len": 16,
            "max_prog_len": 16,
            "mdl_alpha": 0.01,
            "mdl_beta": 0.01,
            "regret_margin": 0.1,
            "causal_delta": 0.1,
            "proof_supervision": True,
        },
        "weights": {
            "ce": 1.0,
            "kkt": 0.0,
            "mdl": 0.0,
            "regret": 0.0,
            "orbit": 0.0,
            "causal": 0.0,
            "state": 0.0,
        },
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "out"
    train_from_dataset(
        config_path=str(config_path),
        data_path=str(data_path),
        out_dir=str(out_dir),
        device="cpu",
        init_ckpt_path=str(init_ckpt),
        steps=1,
        include_variants=False,
    )

    ckpt_path = out_dir / "ckpt.pt"
    saved = torch.load(ckpt_path, map_location="cpu")
    assert len(saved["text_vocab"]) == len(init_text_vocab.token_to_id)
    assert saved["text_vocab"] == init_text_vocab.token_to_id
