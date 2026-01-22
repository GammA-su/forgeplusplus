from __future__ import annotations

from typing import Any


def resolve_max_prog_len(
    cli_value: int | None,
    cfg: dict[str, Any] | None,
    *,
    ckpt: dict[str, Any] | None = None,
    default: int = 256,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    if cfg:
        eval_cfg = cfg.get("eval")
        if isinstance(eval_cfg, dict) and "max_prog_len" in eval_cfg:
            return int(eval_cfg["max_prog_len"])
        train_cfg = cfg.get("train")
        if isinstance(train_cfg, dict) and "max_prog_len" in train_cfg:
            return int(train_cfg["max_prog_len"])
        if "max_prog_len" in cfg:
            return int(cfg["max_prog_len"])
    if ckpt and "max_prog_len" in ckpt:
        return int(ckpt["max_prog_len"])
    return int(default)
