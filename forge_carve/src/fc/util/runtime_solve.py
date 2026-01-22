from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from prooftape.ptv1 import PTv1Runtime
from fc.util.proof_repair import scan_tokens

_RUNTIME: PTv1Runtime | None = None


def _get_runtime() -> PTv1Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = PTv1Runtime()
    return _RUNTIME


@dataclass(frozen=True)
class RuntimeSolveFailure:
    code: str
    detail: str


def _classify_error(exc: Exception) -> RuntimeSolveFailure:
    msg = str(exc)
    if isinstance(exc, IndexError):
        return RuntimeSolveFailure(code="PARSE_FAIL", detail=msg)
    if isinstance(exc, KeyError):
        return RuntimeSolveFailure(code="MISSING_KEY", detail=msg)
    if isinstance(exc, ValueError):
        lowered = msg.lower()
        if "unsupported op" in lowered:
            return RuntimeSolveFailure(code="UNSUPPORTED_OP", detail=msg)
        if "missing dest" in lowered:
            return RuntimeSolveFailure(code="MISSING_DEST", detail=msg)
        if "missing value" in lowered:
            return RuntimeSolveFailure(code="MISSING_VALUE", detail=msg)
        if "bad decimal" in lowered:
            return RuntimeSolveFailure(code="PARSE_FAIL", detail=msg)
    return RuntimeSolveFailure(code="RUNTIME_ERROR", detail=msg)


def _parse_failure_detail(proof_tokens: list[Any]) -> str | None:
    tokens = [tok if isinstance(tok, str) else str(tok) for tok in proof_tokens]
    ok, failure, _ = scan_tokens(tokens)
    if ok or failure is None:
        return None
    got = failure.got_token if failure.got_token is not None else "EOF"
    return f"ParseError pos={failure.pos} expected={failure.expected_class} got={got} reason={failure.reason}"


def runtime_solve(
    x: str,
    constraints: Iterable[dict[str, Any]],
    proof_tokens: Iterable[Any],
    *,
    return_error: bool = False,
) -> Any:
    rt = _get_runtime()
    tokens = list(proof_tokens)
    try:
        out = rt.run(x, list(constraints), tokens)
    except Exception as exc:
        parse_detail = _parse_failure_detail(tokens)
        if parse_detail is not None:
            failure = RuntimeSolveFailure(code="PARSE_FAIL", detail=parse_detail)
        else:
            failure = _classify_error(exc)
        if return_error:
            return None, failure
        return None
    if out is None:
        failure = RuntimeSolveFailure(code="NO_OUTPUT", detail="runtime returned None")
        if return_error:
            return None, failure
    return (out, None) if return_error else out
