from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Sequence

from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.verify.arithmetic import ArithmeticVerifier
from fc.verify.base import MeshReport
from fc.dsl.repair import repair_program
from fc.verify.code import CodeVerifier
from fc.verify.csp import CSPVerifier
from fc.util.tags import domain_from_tag
from fc.verify.schema import SchemaVerifier

CONSTRAINT_NAMES = ["schema", "arith", "csp", "code", "orbit", "flip", "adv_caught"]
ORBIT_PARALLELISM = False
ORBIT_MAX_WORKERS = 4


def set_orbit_parallelism(enabled: bool, max_workers: int | None = None) -> None:
    global ORBIT_PARALLELISM, ORBIT_MAX_WORKERS
    ORBIT_PARALLELISM = bool(enabled)
    if max_workers is not None:
        ORBIT_MAX_WORKERS = max(1, int(max_workers))


class VerifierMesh:
    def __init__(self) -> None:
        self.schema = SchemaVerifier()
        self.arith = ArithmeticVerifier()
        self.csp = CSPVerifier()
        self.code = CodeVerifier()
        self.interp = Interpreter()
        self.constraint_names = CONSTRAINT_NAMES
        self._formal_idx = {
            "schema": CONSTRAINT_NAMES.index("schema"),
            "math": CONSTRAINT_NAMES.index("arith"),
            "csp": CONSTRAINT_NAMES.index("csp"),
            "code": CONSTRAINT_NAMES.index("code"),
        }

    def _select_formal_verifier(self, domain: str) -> tuple[Any, int]:
        if domain == "schema":
            return self.schema, self._formal_idx["schema"]
        if domain == "math":
            return self.arith, self._formal_idx["math"]
        if domain == "csp":
            return self.csp, self._formal_idx["csp"]
        return self.code, self._formal_idx["code"]

    def _select_adv_verifier(self, domain: str) -> Any:
        if domain == "schema":
            return self.schema
        if domain == "math":
            return self.arith
        return self.csp

    def _verify_formal(
        self,
        verifier: Any,
        text: str,
        program: Any,
        output: Any,
        constraints: list[dict[str, Any]] | None = None,
    ) -> Any:
        return verifier.verify(text, program, output, constraints=constraints)

    def _verify_formal_batch(
        self,
        verifier: Any,
        text: str,
        programs: Sequence[Any],
        outputs: Sequence[Any],
        constraints: list[dict[str, Any]] | None = None,
    ) -> list[Any]:
        batch = getattr(verifier, "verify_batch", None)
        if callable(batch):
            return list(batch(text, programs, outputs, constraints=constraints))
        results = []
        for program, output in zip(programs, outputs):
            results.append(verifier.verify(text, program, output, constraints=constraints))
        return results

    def _orbit_outputs(self, program: Any, orbits: Sequence[str]) -> list[Any]:
        if not orbits:
            return []
        def _exec_orbit(text: str) -> Any:
            return self.interp.execute(program, text)[0]
        if ORBIT_PARALLELISM and len(orbits) > 1 and ORBIT_MAX_WORKERS > 1:
            max_workers = min(ORBIT_MAX_WORKERS, len(orbits))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(_exec_orbit, orbits))
        return [_exec_orbit(t) for t in orbits]

    def run(
        self,
        text: str,
        program: Any,
        output: Any,
        domain: str,
        expected: Any | None = None,
        orbits: Iterable[str] | None = None,
        flips: Iterable[str] | None = None,
        mutator: Any | None = None,
        constraints: list[dict[str, Any]] | None = None,
        repair: bool = False,
        max_repairs: int = 2,
    ) -> MeshReport:
        c = [0.0 for _ in CONSTRAINT_NAMES]
        meta: dict[str, Any] = {"formal": {}, "orbit": {}, "flip": {}, "adv": {}}
        resolved_domain = domain_from_tag(text) or domain
        meta["domain_used"] = resolved_domain
        verifier, formal_idx = self._select_formal_verifier(resolved_domain)
        res = self._verify_formal(verifier, text, program, output, constraints=constraints)
        c[formal_idx] = 1.0 if not res.valid else 0.0
        meta["formal"] = res.model_dump()

        if expected is not None:
            meta["correct"] = outputs_equivalent(output, expected)

        # Orbit invariance: program should yield same output on orbits.
        if orbits:
            orbit_texts = list(orbits)
            orbit_outs = self._orbit_outputs(program, orbit_texts)
            orbit_fail = sum(1 for o in orbit_outs if not outputs_equivalent(o, output))
            if orbit_fail > 0:
                c[CONSTRAINT_NAMES.index("orbit")] = 1.0
            meta["orbit"]["fails"] = orbit_fail

        # Flip sensitivity: output should change on flips.
        if flips:
            flip_fail = 0
            for ftext in flips:
                out, _, _ = self.interp.execute(program, ftext)
                if outputs_equivalent(out, output):
                    flip_fail += 1
            if flip_fail > 0:
                c[CONSTRAINT_NAMES.index("flip")] = 1.0
            meta["flip"]["fails"] = flip_fail

        # Adversarial: mutated programs should be rejected by formal verifiers.
        if mutator is not None:
            mutants = mutator.mutate(program)
            adv_fail = 0
            if mutants:
                adv_verifier = self._select_adv_verifier(resolved_domain)
                adv_outs = [self.interp.execute(mprog, text)[0] for mprog in mutants]
                results = self._verify_formal_batch(adv_verifier, text, mutants, adv_outs, constraints=constraints)
                adv_fail = sum(1 for res in results if res.valid)
            if adv_fail > 0:
                c[CONSTRAINT_NAMES.index("adv_caught")] = 1.0
            meta["adv"]["fails"] = adv_fail
            meta["adv"]["caught"] = max(0, len(mutants) - adv_fail)

        if repair:
            def _score(prog: Any) -> tuple[float, dict[str, Any]]:
                out, _, _ = self.interp.execute(prog, text)
                res = self._verify_formal(verifier, text, prog, out, constraints=constraints)
                score = float(sum(res.violations.values())) if res.violations else 0.0
                meta = dict(res.meta)
                meta["violations"] = res.violations
                meta["c_hard"] = score
                return score, meta

            def _score_batch(programs: Sequence[Any]) -> list[tuple[float, dict[str, Any]]]:
                outs = [self.interp.execute(prog, text)[0] for prog in programs]
                results = self._verify_formal_batch(verifier, text, list(programs), outs, constraints=constraints)
                scored: list[tuple[float, dict[str, Any]]] = []
                for res in results:
                    score = float(sum(res.violations.values())) if res.violations else 0.0
                    meta = dict(res.meta)
                    meta["violations"] = res.violations
                    meta["c_hard"] = score
                    scored.append((score, meta))
                return scored

            _score.batch = _score_batch  # type: ignore[attr-defined]

            base_score, base_meta = _score(program)
            best_prog, best_score, steps, best_meta = repair_program(
                program,
                evaluator=_score,
                max_steps=max_repairs,
            )
            meta["repair"] = {
                "success": best_score <= 0.0 and base_score > 0.0,
                "steps": steps,
                "base_score": base_score,
                "best_score": best_score,
                "meta": best_meta,
            }

        return MeshReport(constraint_names=CONSTRAINT_NAMES, c=c, meta=meta)
