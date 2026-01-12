from __future__ import annotations

from typing import Any, Iterable

from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.verify.arithmetic import ArithmeticVerifier
from fc.verify.base import MeshReport
from fc.verify.code import CodeVerifier
from fc.verify.csp import CSPVerifier
from fc.verify.schema import SchemaVerifier

CONSTRAINT_NAMES = ["schema", "arith", "csp", "code", "orbit", "flip", "adv"]


class VerifierMesh:
    def __init__(self) -> None:
        self.schema = SchemaVerifier()
        self.arith = ArithmeticVerifier()
        self.csp = CSPVerifier()
        self.code = CodeVerifier()
        self.interp = Interpreter()
        self.constraint_names = CONSTRAINT_NAMES

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
    ) -> MeshReport:
        c = [0.0 for _ in CONSTRAINT_NAMES]
        meta: dict[str, Any] = {"formal": {}, "orbit": {}, "flip": {}, "adv": {}}
        if domain == "schema":
            res = self.schema.verify(text, program, output, constraints=constraints)
            c[CONSTRAINT_NAMES.index("schema")] = 1.0 if not res.valid else 0.0
            meta["formal"] = res.model_dump()
        elif domain == "math":
            res = self.arith.verify(text, program, output, constraints=constraints)
            c[CONSTRAINT_NAMES.index("arith")] = 1.0 if not res.valid else 0.0
            meta["formal"] = res.model_dump()
        elif domain == "csp":
            res = self.csp.verify(text, program, output, constraints=constraints)
            c[CONSTRAINT_NAMES.index("csp")] = 1.0 if not res.valid else 0.0
            meta["formal"] = res.model_dump()
        else:
            res = self.code.verify(text, program, output)
            c[CONSTRAINT_NAMES.index("code")] = 1.0 if not res.valid else 0.0
            meta["formal"] = res.model_dump()

        if expected is not None:
            meta["correct"] = outputs_equivalent(output, expected)

        # Orbit invariance: program should yield same output on orbits.
        if orbits:
            orbit_fail = 0
            for otext in orbits:
                out, _, _ = self.interp.execute(program, otext)
                if not outputs_equivalent(out, output):
                    orbit_fail += 1
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
            adv_fail = 0
            mutants = mutator.mutate(program)
            for mprog in mutants:
                mout, _, _ = self.interp.execute(mprog, text)
                if domain == "schema":
                    res = self.schema.verify(text, mprog, mout, constraints=constraints)
                elif domain == "math":
                    res = self.arith.verify(text, mprog, mout, constraints=constraints)
                else:
                    res = self.csp.verify(text, mprog, mout, constraints=constraints)
                if res.valid:
                    adv_fail += 1
            if adv_fail > 0:
                c[CONSTRAINT_NAMES.index("adv")] = 1.0
            meta["adv"]["fails"] = adv_fail

        return MeshReport(constraint_names=CONSTRAINT_NAMES, c=c, meta=meta)
