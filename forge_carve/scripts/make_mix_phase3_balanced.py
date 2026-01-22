import json, random
from collections import defaultdict
from pathlib import Path

random.seed(123)

SRC = {
  "schema": Path("out/train/schema_train.jsonl"),
  "math":   Path("out/train/math_train.jsonl"),
  "csp":    Path("out/train/csp_train.jsonl"),
}
OUT = Path("out/train/forge_mix_phase3.jsonl")

def sig(rec):
    # stable-ish signature to dedup: x + proof sha if present
    pr = rec.get("proof") or {}
    sha = pr.get("sha256") if isinstance(pr, dict) else None
    return json.dumps({"x": rec.get("x"), "sha": sha}, sort_keys=True, ensure_ascii=False)

buckets = defaultdict(list)
for name, path in SRC.items():
    assert path.exists(), f"missing {path}"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            r = json.loads(line)
            buckets[name].append(r)

for k in buckets:
    random.shuffle(buckets[k])

# balance to min bucket size (or cap if you want)
m = min(len(v) for v in buckets.values())
target = m
print({k: len(v) for k,v in buckets.items()}, "target_per_domain=", target)

seen = set()
rows = []
for k, lst in buckets.items():
    take = 0
    for r in lst:
        s = sig(r)
        if s in seen:
            continue
        seen.add(s)
        rows.append(r)
        take += 1
        if take >= target:
            break

random.shuffle(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote", OUT, "rows=", len(rows))
