from __future__ import annotations

import argparse
import json
import random
import time
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

import torch

from fc.proof_hash import hash_tokens
from fc.util.runtime_solve import runtime_solve
from pathlib import Path

from fc.util.repair_math_cegis import repair_math_cegis_target
from fc.morph.equiv import outputs_equivalent


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _process_row(
    row: dict[str, Any],
    max_seconds: float,
    vocab: set[str],
) -> tuple[dict[str, Any] | None, float, str, int, int]:
    target = row.get("y")
    text = row.get("x") or row.get("prompt") or ""
    if not text:
        return None, 0.0, "", 0, 0
    t0 = time.perf_counter()
    result = repair_math_cegis_target(
        text,
        target,
        max_nums=8,
        depth=4,
        limit=20000,
        max_seconds=max_seconds,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if result is None:
        return None, elapsed_ms, "", 0, 0
    tokens, meta = result
    canon_tokens: list[Any] = []
    int_to_str_n = 0
    int_to_str_fail_n = 0
    for tok in tokens:
        if isinstance(tok, str) and tok.startswith("INT:"):
            suffix = tok.split(":", 1)[1]
            if suffix.startswith("-"):
                candidate = f"STR:{suffix}"
            else:
                candidate = f"STR:{suffix}"
            if candidate in vocab:
                canon_tokens.append(candidate)
                int_to_str_n += 1
            else:
                canon_tokens.append(tok)
                int_to_str_fail_n += 1
        else:
            canon_tokens.append(tok)
    tokens = canon_tokens
    constraints = row.get("constraints") or []
    expected = row.get("y")
    if expected is not None:
        out = runtime_solve(text, constraints, tokens)
        if not outputs_equivalent(out, expected):
            return None, elapsed_ms, "verify", int_to_str_n, int_to_str_fail_n
    for tok in tokens:
        if tok not in vocab:
            if isinstance(tok, str) and tok.startswith("INT:"):
                return None, elapsed_ms, "int_oov", int_to_str_n, int_to_str_fail_n
            return None, elapsed_ms, "oov", int_to_str_n, int_to_str_fail_n
    row = dict(row)
    if "proof" in row:
        row.setdefault("proof_orig", row.get("proof"))
    row["proof"] = {
        "dsl": "PTv1",
        "tokens": tokens,
        "sha256": hash_tokens(tokens),
        "aux": {"source": "teacher_cegis"},
    }
    row["proof_tokens_gold"] = tokens
    row["teacher_cegis"] = {
        "hit": True,
        "mode": meta.get("repair_cegis_mode", "brute"),
        "kind": meta.get("repair_cegis_kind", "frac_depth4"),
        "depth": meta.get("repair_cegis_depth", 4),
        "states": meta.get("repair_cegis_states", 0),
        "time_ms": meta.get("repair_cegis_time_ms", int(elapsed_ms)),
    }
    return row, elapsed_ms, "", int_to_str_n, int_to_str_fail_n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create teacher-labeled math proofs via CEGIS brute.")
    parser.add_argument("--in-jsonl", required=True, help="Input JSONL dataset")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL with teacher proofs")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--max-seconds", type=float, default=0.08, help="Max seconds per sample")
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="Checkpoint with prog_vocab")
    parser.add_argument(
        "--prefilter-int-range",
        dest="prefilter_int_range",
        action="store_true",
        help="Skip samples with integers outside the prog_vocab INT range before CEGIS.",
    )
    parser.add_argument(
        "--no-prefilter-int-range",
        dest="prefilter_int_range",
        action="store_false",
        help="Disable integer-range prefilter.",
    )
    parser.set_defaults(prefilter_int_range=True)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    total = 0
    hits = 0
    written = 0
    oov_skip = 0
    int_oov_skip = 0
    int_prefilter_skip = 0
    verify_skip = 0
    time_sum = 0.0
    ckpt = torch.load(args.ckpt, map_location="cpu")
    prog_vocab = ckpt.get("prog_vocab", {})
    vocab = set(prog_vocab.keys())
    max_abs_int = None
    if args.prefilter_int_range:
        max_abs = 0
        for tok in vocab:
            if not isinstance(tok, str) or not tok.startswith("INT:"):
                continue
            try:
                val = int(tok.split(":", 1)[1])
            except ValueError:
                continue
            max_abs = max(max_abs, abs(val))
        if max_abs > 0:
            max_abs_int = max_abs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in _iter_jsonl(in_path):
        total += 1
        if args.limit and total > args.limit:
            break
        rows.append(row)

    int_to_str_n = 0
    int_to_str_fail_n = 0
    with out_path.open("w", encoding="utf-8") as out_handle:
        if args.workers <= 1:
            for row in rows:
                if max_abs_int is not None:
                    text = row.get("x") or row.get("prompt") or ""
                    ints = [int(tok) for tok in re.findall(r"-?\\d+", text)]
                    if any(abs(v) > max_abs_int for v in ints):
                        int_prefilter_skip += 1
                        continue
                updated, elapsed_ms, skip_reason, n_str, n_fail = _process_row(row, args.max_seconds, vocab)
                int_to_str_n += n_str
                int_to_str_fail_n += n_fail
                if updated is None:
                    if skip_reason == "oov":
                        oov_skip += 1
                    elif skip_reason == "int_oov":
                        int_oov_skip += 1
                    elif skip_reason == "verify":
                        verify_skip += 1
                    continue
                hits += 1
                time_sum += elapsed_ms
                out_handle.write(json.dumps(updated, ensure_ascii=True) + "\n")
                written += 1
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                if max_abs_int is not None:
                    filtered = []
                    for row in rows:
                        text = row.get("x") or row.get("prompt") or ""
                        ints = [int(tok) for tok in re.findall(r"-?\\d+", text)]
                        if any(abs(v) > max_abs_int for v in ints):
                            int_prefilter_skip += 1
                            continue
                        filtered.append(row)
                    rows_iter = filtered
                else:
                    rows_iter = rows
                for updated, elapsed_ms, skip_reason, n_str, n_fail in pool.map(
                    _process_row,
                    rows_iter,
                    [args.max_seconds] * len(rows_iter),
                    [vocab] * len(rows_iter),
                    chunksize=64,
                ):
                    int_to_str_n += n_str
                    int_to_str_fail_n += n_fail
                    if updated is None:
                        if skip_reason == "oov":
                            oov_skip += 1
                        elif skip_reason == "int_oov":
                            int_oov_skip += 1
                        elif skip_reason == "verify":
                            verify_skip += 1
                        continue
                    hits += 1
                    time_sum += elapsed_ms
                    out_handle.write(json.dumps(updated, ensure_ascii=True) + "\n")
                    written += 1

    hit_rate = (hits / total) if total else 0.0
    avg_time = (time_sum / hits) if hits else 0.0
    print(
        "total=%d hit_n=%d written_n=%d oov_skip_n=%d int_oov_skip_n=%d "
        "int_prefilter_skip_n=%d verify_skip_n=%d int_to_str_n=%d int_to_str_fail_n=%d "
        "hit_rate=%.4f avg_time_ms=%.2f"
        % (
            total,
            hits,
            written,
            oov_skip,
            int_oov_skip,
            int_prefilter_skip,
            verify_skip,
            int_to_str_n,
            int_to_str_fail_n,
            hit_rate,
            avg_time,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
