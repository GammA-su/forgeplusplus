from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner


def _load_eval_suite_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("eval_suite_script_math_override", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load eval_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_math_decode_override_cli(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval_suite.py"
    cfg_path = tmp_path / "eval_suite.yaml"
    cfg_path.write_text("")

    module = _load_eval_suite_module(script_path)
    captured = {}

    def fake_run_eval_suite(config_path: str, **kwargs):
        captured["decode_overrides"] = kwargs.get("decode_overrides")
        return {"ok": True}

    monkeypatch.setattr(module, "run_eval_suite", fake_run_eval_suite)

    runner = CliRunner()
    result = runner.invoke(module.app, ["--config", str(cfg_path), "--math-no-constrained-op"])
    assert result.exit_code == 0, result.output
    assert captured.get("decode_overrides") == {"math": {"constrained_op": False}}
