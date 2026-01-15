import json
from pathlib import Path

import torch

from fc.dsl.tokens import build_default_vocab
from fc.eval.compare import run_compare
from fc.model.backbone import BackboneConfig
from fc.model.baseline import BaselineConfig, BaselineModel
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig
from fc.train.answer import AnswerVocab, serialize_answer
from fc.train.data import TextVocab, generate_dataset, save_dataset


def _write_baseline_ckpt(tmp_path: Path, texts: list[str], answers: list[str]) -> Path:
    text_vocab = TextVocab.build(texts)
    answer_vocab = AnswerVocab.build(answers)
    cfg = {
        "text_vocab_size": len(text_vocab.token_to_id),
        "max_answer_len": 32,
        "backbone": {"d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32, "dropout": 0.1, "max_len": 32},
        "train": {"max_text_len": 32},
    }
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    mcfg = BaselineConfig(
        vocab_size=cfg["text_vocab_size"],
        answer_vocab_size=len(answer_vocab.token_to_id),
        max_answer_len=cfg["max_answer_len"],
        backbone=bcfg,
    )
    model = BaselineModel(mcfg)
    ckpt = {
        "mode": "baseline",
        "model": model.state_dict(),
        "text_vocab": text_vocab.token_to_id,
        "answer_vocab": answer_vocab.token_to_id,
        "config": cfg,
    }
    path = tmp_path / "baseline_ckpt.pt"
    torch.save(ckpt, path)
    return path


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
    path = tmp_path / "forge_ckpt.pt"
    torch.save(ckpt, path)
    return path


def test_eval_compare_runs(tmp_path: Path) -> None:
    schema = generate_dataset("schema", n=2, seed=1)
    math = generate_dataset("math", n=2, seed=2)
    csp = generate_dataset("csp", n=2, seed=3)
    schema_path = tmp_path / "schema.jsonl"
    math_path = tmp_path / "math.jsonl"
    csp_path = tmp_path / "csp.jsonl"
    save_dataset(str(schema_path), schema)
    save_dataset(str(math_path), math)
    save_dataset(str(csp_path), csp)
    examples = schema + math + csp
    texts = []
    for ex in examples:
        texts.append(ex.x)
        texts.extend([o.x for o in ex.orbit])
        texts.extend([f.x for f in ex.flips])
    answers = [serialize_answer(ex.y) for ex in examples]

    baseline_ckpt = _write_baseline_ckpt(tmp_path, texts, answers)
    forge_ckpt = _write_forge_ckpt(tmp_path, texts)
    out_path = tmp_path / "compare.json"
    report = run_compare(
        schema_path=str(schema_path),
        math_path=str(math_path),
        csp_path=str(csp_path),
        out_path=str(out_path),
        baseline_ckpt=str(baseline_ckpt),
        forge_ckpt=str(forge_ckpt),
    )
    assert "baseline" in report
    assert "forge" in report
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert set(loaded["baseline"].keys()) == {
        "verified_accuracy",
        "orbit_invariance",
        "flip_sensitivity",
        "confident_error_rate",
    }
