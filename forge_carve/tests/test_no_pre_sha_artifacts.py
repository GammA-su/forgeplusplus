from pathlib import Path

def test_no_pre_sha_artifacts():
    bad = []
    for p in Path("out/data").glob("*.pre_sha.jsonl"):
        bad.append(str(p))
    assert not bad, f"remove pre_sha artifacts: {bad}"
