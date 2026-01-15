from __future__ import annotations

import subprocess
import sys
from typing import Any

from fc.verify.base import VerifierResult


class CodeVerifier:
    name = "code"

    def verify(self, text: str, program: Any, output: Any) -> VerifierResult:
        violations: dict[str, float] = {}
        meta: dict[str, Any] = {}
        if not isinstance(output, dict):
            return VerifierResult(valid=False, violations={"code_format": 1.0}, meta=meta)
        code = output.get("code")
        tests = output.get("tests")
        if not code or not tests:
            return VerifierResult(valid=False, violations={"code_missing": 1.0}, meta=meta)
        payload = f"{code}\n\n{tests}\n"
        try:
            result = subprocess.run(
                [sys.executable, "-c", payload],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(valid=False, violations={"code_timeout": 1.0}, meta=meta)
        if result.returncode != 0:
            violations["code_fail"] = 1.0
            meta["stderr"] = result.stderr[-200:]
        return VerifierResult(valid=not violations, violations=violations, meta=meta)

    def verify_batch(
        self,
        text: str,
        programs: list[Any],
        outputs: list[Any],
        constraints: list[dict[str, Any]] | None = None,
    ) -> list[VerifierResult]:
        return [self.verify(text, program, output) for program, output in zip(programs, outputs)]
