from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
