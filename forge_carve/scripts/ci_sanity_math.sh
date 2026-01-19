#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/sanity_corrupt_and_verify.py --mode answer-sanity --n 300 out/data/math.jsonl | tee /tmp/sanity_math.txt
rg -n "baseline line: checked=8000 bad=0" /tmp/sanity_math.txt >/dev/null
rg -n "SANITY OK: verifier rejects corrupted answers" /tmp/sanity_math.txt >/dev/null
echo "CI_SANITY_MATH: OK"
