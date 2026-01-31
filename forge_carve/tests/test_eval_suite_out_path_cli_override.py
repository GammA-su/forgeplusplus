from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner


def _load_eval_suite_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("eval_suite_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load eval_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_suite_out_path_cli_override(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "eval_suite.py"
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "eval_suite.yaml"
    cfg_path.write_text("out_path: out/report.json\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    custom_path = tmp_path / "custom.json"

    module = _load_eval_suite_module(script_path)

    def fake_run_eval_suite(config_path: str, **kwargs):
        out_path = kwargs.get("out_path")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text("{}")
        return {"ok": True}

    monkeypatch.setattr(module, "run_eval_suite", fake_run_eval_suite)

    runner = CliRunner()
    result = runner.invoke(
        module.app,
        ["--config", str(cfg_path), "--out-path", str(custom_path)],
    )
    assert result.exit_code == 0, result.output
    assert custom_path.exists()
    assert not (tmp_path / "out" / "report.json").exists()
