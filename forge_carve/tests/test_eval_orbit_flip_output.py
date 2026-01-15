from fc.eval.metrics import flip_output_pass, orbit_output_pass
from fc.morph.equiv import outputs_equivalent


def test_orbit_output_pass_strict_schema() -> None:
    base = {"name": "Ada", "age": 30}
    orbit_same = {"name": "Ada", "age": 30}
    orbit_json = '{"age": 30, "name": "Ada"}'
    orbit_str_age = {"name": "Ada", "age": "30"}
    assert orbit_output_pass(base, [orbit_same])
    assert orbit_output_pass(base, [orbit_json])
    assert not orbit_output_pass(base, [orbit_str_age])


def test_flip_output_pass_conditions() -> None:
    base_y = {"name": "Ada", "age": 30}
    base_output = {"name": "Ada", "age": 30}
    flip_y = {"name": "Ada", "age": 31}
    flip_output = {"name": "Ada", "age": 31}
    assert flip_output_pass(base_output, base_y, [flip_output], [flip_y])
    assert flip_output_pass(base_output, None, [flip_output], [flip_y], base_correct=True)
    assert flip_output_pass(base_output, base_y, [flip_output], [flip_y], base_correct=False)
    bad_base = {"name": "Ada", "age": 29}
    assert not flip_output_pass(bad_base, base_y, [flip_output], [flip_y])
    assert not flip_output_pass(base_output, base_y, [base_output], [flip_y])


def test_outputs_equivalent_math_and_csp() -> None:
    assert outputs_equivalent({"result": "4"}, {"result": 4})
    assert outputs_equivalent("4", 4)
    a = {"status": "ok", "schedule": {"A": 0, "B": 1}}
    b = {"status": "ok", "schedule": {"A": "0", "B": "1"}}
    assert outputs_equivalent(a, b)
