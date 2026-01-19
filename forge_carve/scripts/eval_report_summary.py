import json, sys
from pathlib import Path

KEYS = [
    "verified_accuracy",
    "orbit_invariance_pass_rate",
    "flip_sensitivity_score",
    "proof_validity_correlation",
    "repair_success_rate",
    "attack_success_rate",
]

def f(x):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        # compact float formatting
        return f"{x:.6g}"
    return str(x)

def main():
    if len(sys.argv) != 2:
        print("usage: eval_report_summary.py <report.json>")
        raise SystemExit(2)
    p = Path(sys.argv[1])
    rep = json.loads(p.read_text())
    print(f"report: {p}")
    for dom, d in rep.items():
        print(f"\n[{dom}]")
        for k in KEYS:
            print(f"  {k:<28} {f(d.get(k))}")
        sa = d.get("selective_accuracy") or []
        if sa:
            print("  selective_accuracy:")
            for it in sa:
                print(f"    thr={it.get('threshold')}  acc={f(it.get('accuracy'))}  cov={f(it.get('coverage'))}")

if __name__ == "__main__":
    main()
