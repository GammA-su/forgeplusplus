import torch

from fc.train.trainer import TrainConfig, _build_model, _sync_max_prog_len


def test_train_max_prog_len_sets_model_logits() -> None:
    cfg = {
        "text_vocab_size": 16,
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.1, "max_len": 16},
        "slots": {"num_slots": 2, "d_slot": 8, "num_states": 2},
        "primal_dual": {"num_constraints": 3, "steps": 1, "d_slot": 8},
        "train": {
            "seed": 1,
            "steps": 1,
            "batch_size": 2,
            "lr": 0.001,
            "max_text_len": 8,
            "max_prog_len": 128,
            "mdl_alpha": 0.1,
            "mdl_beta": 0.1,
            "regret_margin": 0.1,
            "causal_delta": 0.1,
        },
    }
    train_cfg = TrainConfig(**cfg["train"])
    _sync_max_prog_len(cfg, train_cfg)
    model = _build_model(cfg, vocab_size=32)
    input_ids = torch.randint(0, cfg["text_vocab_size"], (2, 8))
    outputs = model(input_ids)
    assert outputs["logits"].shape[1] == 128
