from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class DomainSpec:
    name: str
    parser: Callable[[str], dict[str, Any]] | None = None


def registry() -> dict[str, DomainSpec]:
    return {
        "schema": DomainSpec(name="schema"),
        "math": DomainSpec(name="math"),
        "csp": DomainSpec(name="csp"),
    }
