from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _load_tokens(row: dict[str, Any]) -> list[str]:
    proof = row.get("proof")
    if isinstance(proof, dict):
        tokens = proof.get("tokens")
        if isinstance(tokens, list):
            return [str(tok) for tok in tokens]
    tokens = row.get("proof_tokens_gold")
    if isinstance(tokens, list):
        return [str(tok) for tok in tokens]
    return []


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose APPLY_ARITH token windows in PTv1 proofs.")
    parser.add_argument(
        "--data",
        default="out/train/forge_mix_phase3_mathdom.jsonl",
        help="Path to JSONL dataset with PTv1 proofs.",
    )
    parser.add_argument("--max-rows", type=int, default=5000, help="Number of rows to scan.")
    parser.add_argument("--examples", type=int, default=5, help="Number of example windows to print.")
    parser.add_argument("--ckpt", default="", help="Ignored compatibility flag.")
    parser.add_argument("--max-prog-len", type=int, default=0, help="Ignored compatibility flag.")
    args = parser.parse_args()

    path = Path(args.data)
    max_rows = args.max_rows
    max_examples = args.examples
    if not path.exists():
        print(f"missing jsonl: {path}")
        return 1

    counters = {offset: Counter() for offset in range(1, 6)}
    examples: list[list[str]] = []
    seen = 0
    arith_hits = 0
    for row in _iter_jsonl(path):
        if seen >= max_rows:
            break
        seen += 1
        tokens = _load_tokens(row)
        if not tokens:
            continue
        for idx, tok in enumerate(tokens):
            if tok != "APPLY_ARITH":
                continue
            arith_hits += 1
            for offset in range(1, 6):
                if idx + offset < len(tokens):
                    counters[offset][tokens[idx + offset]] += 1
            if len(examples) < max_examples:
                start = max(0, idx - 4)
                end = min(len(tokens), idx + 9)
                examples.append(tokens[start:end])

    print(f"rows_scanned={seen} apply_arith_hits={arith_hits}")
    for offset in range(1, 6):
        most = counters[offset].most_common(10)
        parts = [f"{tok}:{count}" for tok, count in most]
        print(f"offset_t+{offset} top10=" + ", ".join(parts))
    if examples:
        print("examples:")
        for ex in examples:
            print("  " + " ".join(ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
