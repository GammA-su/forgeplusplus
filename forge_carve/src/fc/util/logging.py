from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO, stderr: bool = True) -> None:
    console = Console(stderr=stderr)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
