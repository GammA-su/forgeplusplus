from __future__ import annotations

from typing import Iterable, Any

from fc.adv.mutator import ProgramMutator
from fc.interp.core import Interpreter
from fc.verify.mesh import VerifierMesh


def attack_success_rate(
    texts: Iterable[str],
    programs: Iterable[object],
    domain: str,
    mesh: VerifierMesh,
    constraints: Iterable[list[dict[str, Any]]] | None = None,
) -> float:
    mutator = ProgramMutator()
    interp = Interpreter()
    total = 0
    success = 0
    if constraints is None:
        triplets = ((t, p, None) for t, p in zip(texts, programs))
    else:
        triplets = ((t, p, c) for t, p, c in zip(texts, programs, constraints))
    for text, prog, c_list in triplets:
        mutants = mutator.mutate(prog)
        for mprog in mutants:
            total += 1
            mout, _, _ = interp.execute(mprog, text)
            report = mesh.run(text, mprog, mout, domain=domain, constraints=c_list)
            formal_violation = report.c[mesh.constraint_names.index("schema")]
            formal_violation += report.c[mesh.constraint_names.index("arith")]
            formal_violation += report.c[mesh.constraint_names.index("csp")]
            if formal_violation == 0:
                success += 1
    if total == 0:
        return 0.0
    return success / total


def attack_report(
    texts: Iterable[str],
    programs: Iterable[object],
    domain: str,
    mesh: VerifierMesh,
    constraints: Iterable[list[dict[str, Any]]] | None = None,
) -> dict[str, float]:
    rate = attack_success_rate(texts, programs, domain, mesh, constraints=constraints)
    return {"attack_success_rate": rate, "attack_caught_rate": float(1.0 - rate)}
