from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("make_math_teacher_cegis_vocab", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load make_math_teacher_cegis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_math_teacher_cegis_vocabfilter(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"id": "bad", "x": "[MATH] bad 1 + 1", "y": 2},
        {"id": "ok", "x": "[MATH] 2 + 2", "y": 4},
        {"id": "conv", "x": "[MATH] 9 + 9", "y": 18},
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    with in_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "prog_vocab": {
                "OP": 0,
                "STR:+": 1,
                "END": 2,
                "INT:1": 3,
                "STR:999": 4,
            }
        },
        ckpt_path,
    )

    def fake_repair(text, target, **_kwargs):
        if "bad" in text:
            return ["OP", "INT:999999", "END"], {}
        if target == 18:
            return ["OP", "INT:999", "END"], {}
        return ["OP", "INT:1", "END"], {}

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_math_teacher_cegis.py"
    module = _load_script(script_path)
    monkeypatch.setattr(module, "repair_math_cegis_target", fake_repair)
    monkeypatch.setattr(module, "runtime_solve", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(module, "outputs_equivalent", lambda *_args, **_kwargs: True)

    assert (
        module.main(
            [
                "--in-jsonl",
                str(in_path),
                "--out-jsonl",
                str(out_path),
                "--limit",
                "3",
                "--workers",
                "1",
                "--ckpt",
                str(ckpt_path),
                "--no-prefilter-int-range",
            ]
        )
        == 0
    )
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    for obj in objs:
        assert obj["proof"]["sha256"]
        assert obj["proof_tokens_gold"] == obj["proof"]["tokens"]
    conv = next(obj for obj in objs if obj["id"] == "conv")
    assert "STR:999" in conv["proof"]["tokens"]
