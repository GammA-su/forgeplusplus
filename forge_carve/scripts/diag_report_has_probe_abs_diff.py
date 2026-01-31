from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _collect_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    google_re = report.get("google_re")
    if isinstance(google_re, dict):
        google_samples = google_re.get("samples")
        if isinstance(google_samples, list):
            samples.extend([s for s in google_samples if isinstance(s, dict)])
    top_samples = report.get("samples")
    if isinstance(top_samples, list):
        samples.extend([s for s in top_samples if isinstance(s, dict)])
    return samples


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: diag_report_has_probe_abs_diff.py <report.json>")
        return 2
    report_path = Path(sys.argv[1])
    if not report_path.exists():
        print(f"missing report: {report_path}")
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        print("report is not a JSON object")
        return 1
    samples = _collect_samples(report)
    has_probe = any("probe_abs_diff" in sample for sample in samples)
    keys: set[str] = set()
    for sample in samples:
        keys.update(str(k) for k in sample.keys())
    print(f"has_probe_abs_diff={str(has_probe).lower()}")
    print("sample_keys=" + ",".join(sorted(keys)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
