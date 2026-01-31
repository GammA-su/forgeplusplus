from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("make_math_teacher_cegis", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load make_math_teacher_cegis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_math_teacher_cegis_smoke(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"id": "a", "x": "[MATH] What is 2 + 3?", "y": 5},
        {"id": "b", "x": "[MATH] What is 7 - 4?", "y": 3},
        {"id": "c", "x": "[MATH] What is 6 / 2?", "y": 3},
    ]
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    with in_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "make_math_teacher_cegis.py"
    module = _load_script(script_path)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"prog_vocab": {"OP": 0, "STR:+": 1, "END": 2}}, ckpt_path)

    def fake_repair(text, target, **_kwargs):
        return ["OP", "STR:+", "END"], {"repair_cegis_mode": "brute", "repair_cegis_kind": "frac_depth4"}

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
            ]
        )
        == 0
    )
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert obj["proof"]["dsl"] == "PTv1"
        assert obj["teacher_cegis"]["hit"] is True
