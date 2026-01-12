from fc.dsl.program import Program
from fc.interp.core import Interpreter
from fc.train.data import (
    generate_dataset,
    load_dataset,
    load_dataset_with_variants,
    save_dataset,
)
from fc.verify.arithmetic import ArithmeticVerifier
from fc.verify.csp import CSPVerifier
from fc.verify.schema import SchemaVerifier


def test_jsonl_rows_validate_and_expand(tmp_path) -> None:
    for domain in ["schema", "math", "csp"]:
        examples = generate_dataset(domain, n=2, seed=3)
        path = tmp_path / f"{domain}.jsonl"
        save_dataset(str(path), examples)
        loaded = load_dataset(str(path))
        assert len(loaded) == 2
        expanded = load_dataset_with_variants(str(path))
        assert len(expanded) >= len(loaded)
        assert expanded[0].domain == domain
        if loaded[0].orbit:
            assert loaded[0].orbit[0].x
        if loaded[0].flips:
            assert loaded[0].flips[0].x


def test_verifiers_accept_generated_rows() -> None:
    for domain in ["schema", "math", "csp"]:
        ex = generate_dataset(domain, n=1, seed=11)[0]
        program = Program.from_dict(ex.proof)
        out, _, errors = Interpreter().execute(program, ex.x)
        assert not errors
        constraints = [c.model_dump() for c in ex.constraints]
        if domain == "schema":
            res = SchemaVerifier().verify(ex.x, program, out, constraints=constraints)
        elif domain == "math":
            res = ArithmeticVerifier().verify(ex.x, program, out, constraints=constraints)
        else:
            res = CSPVerifier().verify(ex.x, program, out, constraints=constraints)
        assert isinstance(res.valid, bool)
