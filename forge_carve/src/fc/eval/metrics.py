from __future__ import annotations

from typing import Iterable

import numpy as np

from fc.morph.equiv import outputs_equivalent

def verified_accuracy(correct: Iterable[bool]) -> float:
    vals = list(correct)
    return float(np.mean(vals)) if vals else 0.0


def orbit_invariance_pass_rate(passes: Iterable[bool]) -> float:
    vals = list(passes)
    return float(np.mean(vals)) if vals else 0.0


def flip_sensitivity_score(changed: Iterable[bool]) -> float:
    vals = list(changed)
    return float(np.mean(vals)) if vals else 0.0


def orbit_output_pass(base_output: object, orbit_outputs: Iterable[object]) -> bool:
    return all(outputs_equivalent(base_output, out) for out in orbit_outputs)


def flip_output_pass(
    base_output: object,
    base_y: object | None,
    flip_outputs: Iterable[object],
    flip_ys: Iterable[object | None],
    base_correct: bool | None = None,
) -> bool:
    flip_outputs_list = list(flip_outputs)
    flip_ys_list = list(flip_ys)
    if len(flip_outputs_list) != len(flip_ys_list):
        return False
    base_ok = False
    if base_correct:
        base_ok = True
    if base_y is not None and outputs_equivalent(base_output, base_y):
        base_ok = True
    if not base_ok:
        return False
    for out, expected in zip(flip_outputs_list, flip_ys_list):
        if expected is None:
            return False
        if not outputs_equivalent(out, expected):
            return False
        if outputs_equivalent(out, base_output):
            return False
    return True


def proof_validity_correlation(valid: Iterable[bool], correct: Iterable[bool]) -> float:
    v = np.array(list(valid), dtype=float)
    c = np.array(list(correct), dtype=float)
    if v.size == 0:
        return 0.0
    if np.std(v) == 0 or np.std(c) == 0:
        return 0.0
    return float(np.corrcoef(v, c)[0, 1])


def repair_success_rate(repaired: Iterable[bool]) -> float:
    vals = list(repaired)
    return float(np.mean(vals)) if vals else 0.0


def attack_success_rate(successes: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(successes / total)


def selective_accuracy(conf: Iterable[float], correct: Iterable[bool], thresholds: Iterable[float]) -> list[dict[str, float]]:
    confs = np.array(list(conf), dtype=float)
    corr = np.array(list(correct), dtype=bool)
    out = []
    for thr in thresholds:
        mask = confs >= thr
        if mask.sum() == 0:
            out.append({"threshold": float(thr), "accuracy": 0.0, "coverage": 0.0})
        else:
            out.append(
                {
                    "threshold": float(thr),
                    "accuracy": float(corr[mask].mean()),
                    "coverage": float(mask.mean()),
                }
            )
    return out
