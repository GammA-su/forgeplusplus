from __future__ import annotations

from typing import Iterable

from fc.adv.mutator import ProgramMutator
from fc.interp.core import Interpreter
from fc.verify.mesh import VerifierMesh


def attack_success_rate(
    texts: Iterable[str],
    programs: Iterable[object],
    domain: str,
    mesh: VerifierMesh,
) -> float:
    mutator = ProgramMutator()
    interp = Interpreter()
    total = 0
    success = 0
    for text, prog in zip(texts, programs):
        mutants = mutator.mutate(prog)
        for mprog in mutants:
            total += 1
            mout, _, _ = interp.execute(mprog, text)
            report = mesh.run(text, mprog, mout, domain=domain)
            formal_violation = report.c[mesh.constraint_names.index("schema")]
            formal_violation += report.c[mesh.constraint_names.index("arith")]
            formal_violation += report.c[mesh.constraint_names.index("csp")]
            if formal_violation == 0:
                success += 1
    if total == 0:
        return 0.0
    return success / total
