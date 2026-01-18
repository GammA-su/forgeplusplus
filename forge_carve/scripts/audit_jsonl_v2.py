import json, hashlib
from pathlib import Path
from collections import Counter

def size(x):
    if isinstance(x, str): return len(x)
    if isinstance(x, list): return len(x)
    if isinstance(x, dict): return len(x)
    return 0

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def main(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    n = len(rows)

    # Heuristic: choose "problem/statement" as the largest string among top-level fields
    # and choose "proof/witness/trace" as the largest non-trivial structured field among top-level.
    def pick_fields(r):
        top = list(r.items())
        str_fields = [(k,v) for k,v in top if isinstance(v,str)]
        obj_fields = [(k,v) for k,v in top if isinstance(v,(list,dict,str))]

        stmt_k = None
        if str_fields:
            stmt_k = max(str_fields, key=lambda kv: len(kv[1]))[0]

        proof_k = None
        if obj_fields:
            proof_k = max(obj_fields, key=lambda kv: size(kv[1]))[0]

        return stmt_k, proof_k

    stmt_k, proof_k = pick_fields(rows[0]) if rows else (None, None)
    print(f"{path}: n={n} auto_stmt={stmt_k} auto_proof={proof_k}")

    stmt_sigs = []
    proof_sigs = []
    pair_sigs = []
    stmt_sizes = []
    proof_sizes = []
    emptystmt = 0
    emptyproof = 0

    for r in rows:
        stmt = r.get(stmt_k) if stmt_k else None
        prf  = r.get(proof_k) if proof_k else None

        if stmt in (None,""): emptystmt += 1
        if prf  in (None,"",[],{}): emptyproof += 1

        stmt_sizes.append(size(stmt))
        proof_sizes.append(size(prf))

        stmt_sigs.append(digest(stmt))
        proof_sigs.append(digest(prf))
        pair_sigs.append(digest({"stmt": stmt, "proof": prf}))

    def uniq(xs): return len(set(xs))

    print(f"emptystmt={emptystmt} emptyproof={emptyproof}")
    print(f"stmt_size: min={min(stmt_sizes) if stmt_sizes else 0} max={max(stmt_sizes) if stmt_sizes else 0}")
    print(f"proof_size: min={min(proof_sizes) if proof_sizes else 0} max={max(proof_sizes) if proof_sizes else 0}")
    print(f"uniq stmt={uniq(stmt_sigs)} / {n}")
    print(f"uniq proof={uniq(proof_sigs)} / {n}")
    print(f"uniq pair={uniq(pair_sigs)} / {n}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
