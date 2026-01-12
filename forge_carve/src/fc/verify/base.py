from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class VerifierResult(BaseModel):
    valid: bool
    violations: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class Verifier(Protocol):
    name: str

    def verify(
        self,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> VerifierResult:
        ...


class MeshReport(BaseModel):
    constraint_names: list[str]
    c: list[float]
    meta: dict[str, Any] = Field(default_factory=dict)


def normalize_constraints(constraints: list[Any] | None) -> list[dict[str, Any]]:
    if not constraints:
        return []
    normalized: list[dict[str, Any]] = []
    for constraint in constraints:
        if isinstance(constraint, dict):
            normalized.append(constraint)
        elif hasattr(constraint, "model_dump"):
            normalized.append(constraint.model_dump())
        else:
            normalized.append({"type": str(constraint)})
    return normalized
