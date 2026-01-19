from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _count_failures(report: dict[str, Any]) -> tuple[int, int]:
    """
    Tries a few common report shapes.
    Returns (tasks_total, failing_total).
    """
    # Shape A: {"tasks":[{"status":"ok"/"fail"/...}, ...]}
    tasks = report.get("tasks")
    if isinstance(tasks, list):
        failing = 0
        for t in tasks:
            if isinstance(t, dict):
                st = (t.get("status") or t.get("result") or "").lower()
                if st in ("fail", "failed", "error"):
                    failing += 1
        return (len(tasks), failing)

    # Shape B: {"results":[...]} or {"cases":[...]}
    for k in ("results", "cases"):
        arr = report.get(k)
        if isinstance(arr, list):
            failing = 0
            for t in arr:
                if isinstance(t, dict):
                    st = (t.get("status") or t.get("result") or "").lower()
                    if st in ("fail", "failed", "error"):
                        failing += 1
            return (len(arr), failing)

    # Shape C: {"summary":{"tasks":N,"failing":M}} or similar
    summary = report.get("summary")
    if isinstance(summary, dict):
        t = summary.get("tasks") or summary.get("total") or summary.get("n") or 0
        f = summary.get("failing") or summary.get("failed") or summary.get("errors") or 0
        if isinstance(t, int) and isinstance(f, int):
            return (t, f)

    return (0, 0)


def _find_fail_examples(report: dict[str, Any], limit: int = 5) -> list[str]:
    out: list[str] = []
    for k in ("tasks", "results", "cases"):
        arr = report.get(k)
        if not isinstance(arr, list):
            continue
        for t in arr:
            if not isinstance(t, dict):
                continue
            st = (t.get("status") or t.get("result") or "").lower()
            if st not in ("fail", "failed", "error"):
                continue
            name = t.get("id") or t.get("name") or t.get("case") or "<?>"
            reason = t.get("reason") or t.get("error") or t.get("message") or ""
            out.append(f"- {name}: {reason}".rstrip())
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="Path or glob (e.g. runs/suites/*/report.json)")
    args = ap.parse_args()

    pat = args.report or "runs/suites/*/report.json"
    paths = [Path(p) for p in glob.glob(pat)]
    if not paths:
        raise SystemExit(f"no reports matched: {pat!r}")

    # newest first
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in paths[:10]:
        j = json.loads(p.read_text())
        t, f = _count_failures(j)
        print(f"report: {p}  tasks: {t}  failing: {f}")
        if f:
            for line in _find_fail_examples(j, limit=5):
                print(line)
        break


if __name__ == "__main__":
    main()
