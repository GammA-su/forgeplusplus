from __future__ import annotations

from fc.eval.suite import merge_eval_suite_config


def test_eval_suite_cli_overrides() -> None:
    cfg = {"schema_path": "A", "csp_path": "B"}
    merged = merge_eval_suite_config(cfg, {"schema_path": "X"})
    assert merged["schema_path"] == "X"
    assert merged["csp_path"] == "B"
