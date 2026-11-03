"""Unit tests for statistical intervals and hypothesis testing routines."""

import numpy as np

from driving_wm.evaluation.metrics import compute_steering_jerk
from driving_wm.evaluation.stats import (
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_test,
    wilson_score_interval,
)


def test_wilson_score_interval_known_values() -> None:
    # 0/10 successes: lower must be 0.0, upper > 0
    p, low, high = wilson_score_interval(0, 10)
    assert p == 0.0
    assert low == 0.0
    assert 0.25 < high < 0.35

    # 10/10 successes: upper must be 1.0, lower < 1.0
    p, low, high = wilson_score_interval(10, 10)
    assert p == 1.0
    assert 0.65 < low < 0.75
    assert high == 1.0

    # 50/100 successes: symmetric around 0.5
    p, low, high = wilson_score_interval(50, 100)
    assert p == 0.5
    assert abs(0.5 - low - (high - 0.5)) < 1e-4
    assert 0.39 < low < 0.41
    assert 0.59 < high < 0.61


def test_bootstrap_ci_coverage() -> None:
    # Normal distribution with mean 10.0
    rng = np.random.default_rng(42)
    data = rng.normal(10.0, 1.0, size=100)

    mean_val, low, high = bootstrap_ci(data, num_bootstrap=1000, seed=42)
    assert 9.5 < mean_val < 10.5
    assert low < mean_val < high
    assert (high - low) < 0.6


def test_paired_bootstrap_test() -> None:
    # A is clearly better than B
    a = np.array([0.9, 0.95, 0.88, 0.92, 0.89])
    b = np.array([0.4, 0.45, 0.38, 0.42, 0.39])

    res = paired_bootstrap_test(a, b, num_bootstrap=1000, seed=42)
    assert res["mean_diff"] > 0.45
    assert res["ci_lower"] > 0.40
    assert res["p_value"] < 0.01


def test_mcnemar_test() -> None:
    # 10 identical outcomes -> discordant is 0 -> stat=0, p=1.0
    out_a = [True] * 10
    out_b = [True] * 10
    res = mcnemar_test(out_a, out_b)
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0


def test_compute_steering_jerk() -> None:
    # Constant steering -> jerk is 0.0
    const_steer = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert compute_steering_jerk(const_steer) == 0.0

    # Linear steering -> 2nd derivative is 0.0 -> jerk is 0.0
    linear_steer = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert compute_steering_jerk(linear_steer) < 1e-6
