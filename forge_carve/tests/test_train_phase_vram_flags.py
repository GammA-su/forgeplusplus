from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from typer.testing import CliRunner


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("train_phase_script_vram", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load 03_train_phase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_phase_vram_flags(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("train:\n  proof_supervision_source: proof\n")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps({"id": "a", "proof": ["OP", "END"]}) + "\n")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"prog_vocab": {"OP": 0, "END": 1}}, ckpt_path)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "03_train_phase.py"
    module = _load_script(script_path)

    cfg = {"train": {}}
    updated = module._apply_train_overrides(
        cfg,
        precision="fp16",
        microbatch=2,
        grad_accum=4,
        grad_checkpoint=False,
        optimizer="adamw",
    )
    assert updated["train"]["precision"] == "fp16"
    assert updated["train"]["microbatch"] == 2
    assert updated["train"]["grad_accum"] == 4
    assert updated["train"]["grad_checkpoint"] is False
    assert updated["train"]["optimizer"] == "adamw"

    runner = CliRunner()
    result = runner.invoke(
        module.app,
        [
            "--config",
            str(cfg_path),
            "--data",
            str(data_path),
            "--init",
            str(ckpt_path),
            "--proof-preflight-only",
            "--precision",
            "fp16",
            "--microbatch",
            "2",
            "--grad-accum",
            "4",
            "--no-grad-checkpoint",
            "--optimizer",
            "adamw",
        ],
    )
    assert result.exit_code == 0
