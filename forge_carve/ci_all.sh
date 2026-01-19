#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-123}"
N="${N:-8000}"
QUAL_THR="${QUAL_THR:-0.20}"
SANITY_N="${SANITY_N:-300}"

math_out="${MATH_OUT:-out/data/math.jsonl}"
csp_out="${CSP_OUT:-out/data/csp.jsonl}"

train_cfg="${TRAIN_CFG:-train_mvp.yaml}"
eval_cfg="${EVAL_CFG:-eval_suite.yaml}"

# Resolve config paths (root or configs/)
if [[ ! -f "$train_cfg" && -f "configs/$train_cfg" ]]; then train_cfg="configs/$train_cfg"; fi
if [[ ! -f "$eval_cfg"  && -f "configs/$eval_cfg"  ]]; then eval_cfg="configs/$eval_cfg";  fi

echo "[1/8] pytest"
uv run pytest -q

echo "[2/8] make_data_schema (seed=$SEED)"
uv run python scripts/make_data_schema.py --seed "$SEED"

echo "[3/8] make_data_math + make_data_csp (n=$N seed=$SEED)"
uv run python scripts/make_data_math.py --n "$N" --seed "$SEED" --out "$math_out"
uv run python scripts/make_data_csp.py  --n "$N" --seed "$SEED" --out "$csp_out"

echo "[4/8] verify_proofs"
uv run python scripts/verify_proofs.py "$math_out"
uv run python scripts/verify_proofs.py "$csp_out"

echo "[5/8] dataset_quality (thr=$QUAL_THR)"
uv run python scripts/check_dataset_quality.py "$math_out" "$QUAL_THR"
uv run python scripts/check_dataset_quality.py "$csp_out"  "$QUAL_THR"

echo "[6/8] sanity_corrupt_and_verify (n=$SANITY_N)"
uv run python scripts/sanity_corrupt_and_verify.py "$math_out" "$SANITY_N"
uv run python scripts/sanity_corrupt_and_verify.py "$csp_out"  "$SANITY_N"

echo "[7/8] train + eval (cfgs: $train_cfg / $eval_cfg)"
uv run python scripts/train_mvp.py --config configs/train_mvp.yaml 
uv run python scripts/eval_suite.py --config configs/train_mvp.yaml 

echo "[8/8] suite summary"
uv run python scripts/suite_report_summary.py "runs/suites/*/report.json" || true

echo "CI OK"
