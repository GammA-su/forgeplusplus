import json
from pathlib import Path

import yaml

from fc.train.data import generate_dataset
from fc.train.trainer import train


def test_trainer_runs_all_steps(tmp_path: Path) -> None:
    examples = generate_dataset("schema", n=3, seed=1)
    cfg = {
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.0, "max_len": 32},
        "slots": {"num_slots": 2, "d_slot": 16, "num_states": 2},
        "primal_dual": {"num_constraints": 7, "steps": 1, "d_slot": 16},
        "max_prog_len": 32,
        "train": {
            "seed": 1,
            "steps": 50,
            "batch_size": 2,
            "lr": 0.001,
            "max_text_len": 32,
            "max_prog_len": 32,
            "mdl_alpha": 0.01,
            "mdl_beta": 0.01,
            "regret_margin": 0.1,
            "causal_delta": 0.1,
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
    train(examples, config_path=str(config_path), out_dir=str(out_dir), device="cpu")
    log_path = out_dir / "train_log.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert rows
    assert rows[-1]["step"] == 49
