"""Statistical estimation: Wilson score intervals, bootstrap CIs, and paired tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute Wilson score interval for a binomial proportion.

    Returns:
        (center_rate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = successes / total
    # z = 1.96 for 95% confidence
    z = 1.95996 if abs(confidence - 0.95) < 1e-3 else 2.576
    denominator = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denominator
    half_width = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denominator

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return p, lower, upper


def bootstrap_ci(
    values: list[float] | np.ndarray,
    num_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute empirical bootstrap confidence interval for mean of continuous values.

    Returns:
        (sample_mean, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(arr)), lower, upper


def paired_bootstrap_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap hypothesis test for difference in performance on identical seeds."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    assert len(a) == len(b), "Arrays must have identical length for paired test"

    diffs = a - b
    observed_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_diffs = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        sample_diffs = rng.choice(diffs, size=n, replace=True)
        boot_diffs[i] = np.mean(sample_diffs)

    p_value = float(np.mean(boot_diffs <= 0.0) if observed_diff > 0 else np.mean(boot_diffs >= 0.0))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def mcnemar_test(outcomes_a: list[bool], outcomes_b: list[bool]) -> dict[str, Any]:
    """McNemar test for paired nominal success/failure outcomes on identical seeds."""
    assert len(outcomes_a) == len(outcomes_b), "Outcome lists must match in length"
    n01 = 0  # a failed, b succeeded
    n10 = 0  # a succeeded, b failed

    for a, b in zip(outcomes_a, outcomes_b):
        if not a and b:
            n01 += 1
        elif a and not b:
            n10 += 1

    total_discordant = n01 + n10
    diff = abs(n01 - n10)
    if diff <= 1 or total_discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        # Continuity-corrected chi-square
        stat = ((diff - 1.0) ** 2) / total_discordant
        # 1-df chi-square survival function approx
        p_val = math.erfc(math.sqrt(stat) / math.sqrt(2.0))

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_a_better": n10,
        "n_b_better": n01,
    }
