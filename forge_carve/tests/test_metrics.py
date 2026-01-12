from fc.eval.metrics import (
    attack_success_rate,
    flip_sensitivity_score,
    orbit_invariance_pass_rate,
    proof_validity_correlation,
    selective_accuracy,
    verified_accuracy,
)


def test_metrics_basic() -> None:
    assert verified_accuracy([True, False, True]) == 2 / 3
    assert orbit_invariance_pass_rate([True, True]) == 1.0
    assert flip_sensitivity_score([True, False]) == 0.5
    corr = proof_validity_correlation([True, False], [True, False])
    assert corr > 0.9
    sel = selective_accuracy([0.9, 0.1], [True, False], thresholds=[0.5])
    assert sel[0]["accuracy"] == 1.0
    assert attack_success_rate(1, 4) == 0.25


def test_metrics_edge_cases() -> None:
    corr = proof_validity_correlation([True, True], [False, False])
    assert corr == 0.0
    sel = selective_accuracy([0.2, 0.1], [True, False], thresholds=[0.9])
    assert sel[0]["coverage"] == 0.0
