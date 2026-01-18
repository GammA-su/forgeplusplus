from __future__ import annotations

from typing import Any, Iterable

from prooftape.ptv1 import PTv1Runtime

_RUNTIME: PTv1Runtime | None = None


def _get_runtime() -> PTv1Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = PTv1Runtime()
    return _RUNTIME


def runtime_solve(x: str, constraints: Iterable[dict[str, Any]], proof_tokens: Iterable[Any]) -> Any:
    rt = _get_runtime()
    return rt.run(x, list(constraints), list(proof_tokens))
