import json, hashlib, sys
from pathlib import Path

def sig(rec):
    # Stable signature of *task identity* (not id, not proof hash).
    # Keep this aligned with verify_proofs.py duplicate signature.
    x = rec.get("x","")
    constraints = rec.get("constraints", [])
    domain_tag = rec.get("domain_tag","")
    payload = json.dumps(
        {"domain_tag": domain_tag, "x": x, "constraints": constraints},
        ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def main(inp, outp):
    inp = Path(inp); outp = Path(outp)
    seen = set()
    kept = 0
    dropped = 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = sig(rec)
            if s in seen:
                dropped += 1
                continue
            seen.add(s)
            w.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")
            kept += 1
    print(f"in={inp} out={outp} kept={kept} dropped={dropped} uniq_frac={kept/max(kept+dropped,1):.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
