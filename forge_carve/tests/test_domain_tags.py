from fc.train.data import generate_dataset
from fc.util.tags import DOMAIN_TAGS


def test_domain_tags_present() -> None:
    for domain, tag in DOMAIN_TAGS.items():
        ex = generate_dataset(domain, n=2, seed=13)[0]
        assert ex.x.split(maxsplit=1)[0] == tag
        assert ex.domain_tag == tag
        for orb in ex.orbit:
            assert orb.x.split(maxsplit=1)[0] == tag
        for flip in ex.flips:
            assert flip.x.split(maxsplit=1)[0] == tag
