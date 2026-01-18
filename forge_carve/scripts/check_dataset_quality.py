import json, hashlib, sys
from pathlib import Path

def h(x): 
    return hashlib.sha256(x.encode("utf-8")).hexdigest()

def main(p: str, min_uniq_pairs: float = 0.2):
    rows = 0
    uniq_pairs = set()
    uniq_x = set()
    uniq_proof = set()

    for line in Path(p).read_text().splitlines():
        if not line.strip(): 
            continue
        rec = json.loads(line)
        x = rec.get("x","")
        proof = rec.get("proof", {})
        tokens = proof.get("tokens") if isinstance(proof, dict) else None
        rows += 1

        xs = h(x)
        ps = h(json.dumps(tokens, ensure_ascii=True, separators=(",", ":")) if tokens is not None else "")
        uniq_x.add(xs)
        uniq_proof.add(ps)
        uniq_pairs.add(xs + ":" + ps)

    frac = len(uniq_pairs)/max(rows,1)
    print(f"{p}: rows={rows} uniq_x={len(uniq_x)} uniq_proof={len(uniq_proof)} uniq_pairs={len(uniq_pairs)} frac_pairs={frac:.4f}")
    if frac < min_uniq_pairs:
        print(f"[FAIL] uniq_pairs fraction {frac:.4f} < {min_uniq_pairs}")
        return 1
    return 0

if __name__ == "__main__":
    p = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    raise SystemExit(main(p, thr))
