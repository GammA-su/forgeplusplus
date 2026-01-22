from __future__ import annotations

from fc.util.decode_limits import resolve_max_prog_len


def test_default_max_prog_len_is_256_without_config() -> None:
    assert resolve_max_prog_len(None, {}) == 256


def test_cli_overrides_config_max_prog_len() -> None:
    cfg = {"max_prog_len": 64}
    assert resolve_max_prog_len(128, cfg) == 128


def test_eval_config_overrides_train_config() -> None:
    cfg = {"train": {"max_prog_len": 64}, "eval": {"max_prog_len": 192}}
    assert resolve_max_prog_len(None, cfg) == 192
