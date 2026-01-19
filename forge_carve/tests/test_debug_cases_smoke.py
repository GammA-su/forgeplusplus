import subprocess
import sys
from pathlib import Path

import torch

from fc.dsl.tokens import build_default_vocab
from fc.model.backbone import BackboneConfig
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig
from fc.train.data import TextVocab, generate_dataset, save_dataset


def _write_forge_ckpt(tmp_path: Path, texts: list[str]) -> Path:
    text_vocab = TextVocab.build(texts)
    prog_vocab = build_default_vocab()
    cfg = {
        "text_vocab_size": len(text_vocab.token_to_id),
        "max_prog_len": 32,
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.1, "max_len": 32},
        "slots": {"num_slots": 2, "d_slot": 16, "num_states": 2},
        "primal_dual": {"num_constraints": 7, "steps": 1, "d_slot": 16},
        "train": {"max_text_len": 32},
    }
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(
        vocab_size=len(prog_vocab.token_to_id),
        max_prog_len=cfg["max_prog_len"],
        backbone=bcfg,
        slots=scfg,
        primal_dual=pcfg,
    )
    model = ForgeModel(mcfg)
    ckpt = {
        "mode": "forge",
        "model": model.state_dict(),
        "text_vocab": text_vocab.token_to_id,
        "prog_vocab": prog_vocab.token_to_id,
        "config": cfg,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    return path


def test_debug_cases_script_smoke(tmp_path: Path) -> None:
    dataset = generate_dataset("math", n=2, seed=9)
    data_path = tmp_path / "math.jsonl"
    save_dataset(str(data_path), dataset)
    texts = []
    for ex in dataset:
        texts.append(ex.x)
        texts.extend([o.x for o in ex.orbit])
        texts.extend([f.x for f in ex.flips])
    ckpt_path = _write_forge_ckpt(tmp_path, texts)
    script = Path(__file__).resolve().parents[1] / "scripts" / "debug_cases.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--domain",
            "math",
            "--data",
            str(data_path),
            "--ckpt",
            str(ckpt_path),
            "--n",
            "2",
            "--device",
            "cpu",
            "--constrained-op",
        ],
        check=True,
        cwd=str(script.parent.parent),
        capture_output=True,
        text=True,
    )
    assert "INVALID_OPCODE" not in (result.stdout or "")
