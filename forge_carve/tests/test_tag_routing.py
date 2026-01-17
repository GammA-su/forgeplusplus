from fc.dsl.tokens import build_default_vocab
from fc.interp.core import Interpreter
from fc.train.data import generate_dataset, proof_to_program
from fc.util.tags import domain_from_tag
from fc.verify.mesh import VerifierMesh


def test_tag_routing() -> None:
    mesh = VerifierMesh()
    for domain in ["schema", "math", "csp"]:
        ex = generate_dataset(domain, n=1, seed=21)[0]
        program = proof_to_program(ex.proof, build_default_vocab())
        out, _, errors = Interpreter().execute(program, ex.x)
        assert not errors
        wrong_domain = "schema" if domain != "schema" else "math"
        report = mesh.run(ex.x, program, out, domain=wrong_domain, constraints=ex.constraints)
        assert report.meta["domain_used"] == domain
        assert domain_from_tag(ex.x) == domain
