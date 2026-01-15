import pytest

from fc.morph.equiv import outputs_equivalent
from fc.train.data import generate_dataset


@pytest.mark.parametrize("domain", ["schema", "math", "csp"])
def test_data_counts_and_flips(domain: str) -> None:
    n = 20
    orbits = 3
    flips = 2
    examples = generate_dataset(domain, n=n, seed=7, orbits=orbits, flips=flips)
    assert len(examples) == n
    for ex in examples:
        assert ex.constraints
        assert len(ex.orbit) == orbits
        assert len(ex.flips) == flips
        for orb in ex.orbit:
            assert orb.x
            assert outputs_equivalent(orb.y, ex.y)
        assert ex.flips
        for flip in ex.flips:
            assert flip.x
            assert flip.y is not None
            assert not outputs_equivalent(flip.y, ex.y)
