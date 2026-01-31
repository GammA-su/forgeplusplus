from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from fc.eval.suite import merge_eval_suite_config


def _load_eval_suite_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("eval_suite_script_cegis", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load eval_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cegis_mode_default_merge() -> None:
    merged = merge_eval_suite_config({}, None)
    assert merged.get("cegis_mode") == "brute"


def test_cegis_mode_default_cli(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval_suite.py"
    cfg_path = tmp_path / "eval_suite.yaml"
    cfg_path.write_text("")

    module = _load_eval_suite_module(script_path)
    captured = {}

    def fake_run_eval_suite(config_path: str, **kwargs):
        captured["cegis_mode"] = kwargs.get("cegis_mode")
        return {"ok": True}

    monkeypatch.setattr(module, "run_eval_suite", fake_run_eval_suite)

    runner = CliRunner()
    result = runner.invoke(module.app, ["--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert captured.get("cegis_mode") == "brute"
