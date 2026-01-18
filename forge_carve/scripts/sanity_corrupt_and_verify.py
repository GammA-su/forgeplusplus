from __future__ import annotations
import json, random, subprocess, sys, tempfile
from pathlib import Path

def run_verify(path: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "scripts/verify_proofs.py", path],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out

def corrupt_tokens_inplace(rec: dict) -> bool:
    p = rec.get("proof")
    if not isinstance(p, dict): 
        return False
    toks = p.get("tokens")
    if not isinstance(toks, list) or not toks:
        return False
    # mutate one token without updating sha256
    i = random.randrange(len(toks))
    old = toks[i]
    if isinstance(old, str):
        toks[i] = old + "_CORRUPT"
    elif isinstance(old, (int, float)):
        toks[i] = old + 1
    else:
        toks[i] = "CORRUPT"
    return True

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: sanity_corrupt_and_verify.py <path.jsonl> <n>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    n = int(sys.argv[2])

    # 1) baseline must PASS
    rc0, out0 = run_verify(str(src))
    if rc0 != 0:
        print("SANITY FAIL: baseline does not verify; fix dataset first.")
        print(out0.strip())
        return 1

    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    if not rows:
        print("SANITY FAIL: empty file")
        return 1

    # sample n rows
    idxs = list(range(len(rows)))
    random.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    changed = 0
    for i in idxs:
        if corrupt_tokens_inplace(rows[i]):
            changed += 1

    if changed == 0:
        print("SANITY FAIL: could not corrupt any proof tokens (schema unexpected).")
        return 1

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as tmp:
        for rec in rows:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    # 2) corrupted must FAIL and ideally due to sha mismatch
    rc1, out1 = run_verify(tmp_path)
    if rc1 == 0:
        print("SANITY FAIL: verifier accepted corrupted proofs.")
        return 1

    if "proof_sha256_mismatch" not in out1 and "answer_mismatch" not in out1:
        print("SANITY FAIL: corrupted file failed, but not for expected reasons.")
        print(out1.strip())
        return 1

    print("SANITY OK: verifier rejects corrupted proofs (sha mismatch / answer mismatch).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
