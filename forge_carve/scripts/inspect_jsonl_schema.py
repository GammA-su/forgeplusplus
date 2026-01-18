import json
from pathlib import Path
from collections import Counter

def walk(x, path="$", out=None, depth=0, max_depth=4):
    if out is None: out = []
    if depth > max_depth:
        out.append((path, type(x).__name__, "DEPTH_CAP"))
        return out
    if isinstance(x, dict):
        out.append((path, "dict", f"keys={len(x)}"))
        for k, v in x.items():
            walk(v, f"{path}.{k}", out, depth+1, max_depth)
    elif isinstance(x, list):
        out.append((path, "list", f"len={len(x)}"))
        for i, v in enumerate(x[:5]):
            walk(v, f"{path}[{i}]", out, depth+1, max_depth)
    elif isinstance(x, str):
        out.append((path, "str", f"len={len(x)}"))
    else:
        out.append((path, type(x).__name__, str(x)[:80]))
    return out

def main(path, n=3):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    print(f"file={path} rows={len(rows)}")
    for i, r in enumerate(rows[:n]):
        print(f"\n--- sample[{i}] top-level keys ({len(r)}): {sorted(r.keys())}")
        flat = walk(r, max_depth=4)
        types = Counter(t for _, t, _ in flat)
        print("types:", dict(types))

        # show biggest string fields
        big = [(p, meta) for (p,t,meta) in flat if t=="str"]
        big_sorted = sorted(big, key=lambda x: int(x[1].split("len=")[-1]), reverse=True)[:10]
        if big_sorted:
            print("top string fields:")
            for p, meta in big_sorted:
                print(" ", p, meta)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else 3)
