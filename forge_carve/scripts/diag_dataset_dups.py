import json, hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict

def h(obj) -> str:
    s = json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main(path: str, topk: int = 10):
    p = Path(path)
    n = 0

    cx = Counter()
    cc = Counter()
    cp = Counter()
    cx_c = Counter()
    cx_c_p = Counter()

    empty_x = 0
    nonlist_constraints = 0
    missing_sha = 0
    ids = Counter()

    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1

        rid = rec.get("id")
        if rid is not None:
            ids[str(rid)] += 1

        x = rec.get("x")
        if not isinstance(x, str) or not x.strip():
            empty_x += 1
            x = "" if x is None else str(x)

        constraints = rec.get("constraints")
        if not isinstance(constraints, list):
            nonlist_constraints += 1
            constraints = [] if constraints is None else [constraints]

        proof = rec.get("proof", {})
        if isinstance(proof, dict):
            tokens = proof.get("tokens")
            sha = proof.get("sha256")
            if not sha:
                missing_sha += 1
        else:
            tokens = None
            sha = None
            missing_sha += 1

        hx = h(x)
        hc = h(constraints)
        hp = h(tokens if tokens is not None else [])
        cx[hx] += 1
        cc[hc] += 1
        cp[hp] += 1
        cx_c[hx + ":" + hc] += 1
        cx_c_p[hx + ":" + hc + ":" + hp] += 1

    def uniq(c: Counter) -> int:
        return sum(1 for _k, v in c.items() if v == 1)

    print(f"file={path} rows={n}")
    print(f"empty/whitespace x={empty_x}")
    print(f"non-list constraints={nonlist_constraints}")
    print(f"missing proof.sha256={missing_sha}")
    print(f"uniq_x={len(cx)}  uniq_constraints={len(cc)}  uniq_proof_tokens={len(cp)}")
    print(f"uniq(x,constraints)={len(cx_c)}  uniq(x,constraints,proof)={len(cx_c_p)}")
    print()

    print(f"TOP repeated x hashes:")
    for k,v in cx.most_common(topk):
        if v <= 1: break
        print(f"  {v:6d}  {k}")

    print(f"\nTOP repeated (x,constraints) hashes:")
    for k,v in cx_c.most_common(topk):
        if v <= 1: break
        print(f"  {v:6d}  {k}")

    print(f"\nTOP repeated (x,constraints,proof) hashes:")
    for k,v in cx_c_p.most_common(topk):
        if v <= 1: break
        print(f"  {v:6d}  {k}")

    dup_ids = sum(1 for _k,v in ids.items() if v > 1)
    if ids:
        print(f"\nid_count={len(ids)} dup_id_count={dup_ids} max_id_freq={max(ids.values())}")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10))
