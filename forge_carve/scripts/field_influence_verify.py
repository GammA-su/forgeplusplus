import json, tempfile, subprocess, sys
from pathlib import Path

def run_verify(tmp_path: str) -> str:
    r = subprocess.run([sys.executable, "scripts/verify_proofs.py", tmp_path],
                       capture_output=True, text=True)
    return (r.stdout or "").strip()

def parse_bad(s: str) -> int:
    # expects "... bad=NUM"
    for tok in s.split():
        if tok.startswith("bad="):
            try: return int(tok.split("=",1)[1])
            except: pass
    return -1

def mutate(v):
    if isinstance(v, str):
        return v + " CORRUPT"
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, list):
        return (v + ["CORRUPT"]) if v else ["CORRUPT"]
    if isinstance(v, dict):
        w = dict(v)
        w["_CORRUPT_"] = True
        return w
    return "CORRUPT"

def main(path: str):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    item = rows[0]
    base_out = None

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.jsonl"
        base.write_text(json.dumps(item, ensure_ascii=False) + "\n")
        base_out = run_verify(str(base))
        base_bad = parse_bad(base_out)
        print("BASE:", base_out)

        keys = list(item.keys())
        for k in keys:
            # variant A: delete key
            a = dict(item); a.pop(k, None)
            pa = Path(td) / f"del_{k}.jsonl"
            pa.write_text(json.dumps(a, ensure_ascii=False) + "\n")
            out_a = run_verify(str(pa))
            bad_a = parse_bad(out_a)

            # variant B: corrupt key
            b = dict(item); b[k] = mutate(b.get(k))
            pb = Path(td) / f"mut_{k}.jsonl"
            pb.write_text(json.dumps(b, ensure_ascii=False) + "\n")
            out_b = run_verify(str(pb))
            bad_b = parse_bad(out_b)

            changed = ("!" if (bad_a != base_bad or bad_b != base_bad) else " ")
            print(f"{changed} key={k:20s} del_bad={bad_a:3d} mut_bad={bad_b:3d}")

if __name__ == "__main__":
    main(sys.argv[1])
