"""Evaluation metrics computation for closed-loop controller benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EpisodeResult:
    controller_name: str
    suite_name: str
    map_layout: str
    seed: int
    num_steps: int
    arrive_dest: bool
    crash: bool
    out_of_road: bool
    success: bool
    route_completion: float
    mean_speed_mps: float
    rms_lateral_error_m: float
    steering_jerk: float
    min_obstacle_clearance_m: float
    termination_reason: str
    mean_planning_time_ms: float = 0.0


def compute_steering_jerk(steer_history: list[float], dt: float = 0.1) -> float:
    """Compute mean squared second difference of steering actions."""
    if len(steer_history) < 3:
        return 0.0
    arr = np.asarray(steer_history, dtype=np.float64)
    # 2nd derivative: (s[t+2] - 2*s[t+1] + s[t]) / dt^2
    d2 = (arr[2:] - 2.0 * arr[1:-1] + arr[:-2]) / (dt**2)
    return float(np.mean(d2**2))


def aggregate_episode_results(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """Aggregate a list of episode results into summary rates and means with confidence intervals."""
    from driving_wm.evaluation.stats import bootstrap_ci, wilson_score_interval

    n = len(episodes)
    if n == 0:
        return {}

    num_success = sum(1 for e in episodes if e.success)
    num_crashes = sum(1 for e in episodes if e.crash)
    num_out_of_road = sum(1 for e in episodes if e.out_of_road)

    succ_rate, succ_low, succ_high = wilson_score_interval(num_success, n)
    crash_rate, crash_low, crash_high = wilson_score_interval(num_crashes, n)
    oor_rate, oor_low, oor_high = wilson_score_interval(num_out_of_road, n)

    completions = [e.route_completion for e in episodes]
    speeds = [e.mean_speed_mps for e in episodes]
    lat_errs = [e.rms_lateral_error_m for e in episodes]
    jerks = [e.steering_jerk for e in episodes]
    plan_times = [e.mean_planning_time_ms for e in episodes if e.mean_planning_time_ms > 0]

    comp_mean, comp_low, comp_high = bootstrap_ci(completions)
    speed_mean, speed_low, speed_high = bootstrap_ci(speeds)
    lat_mean, lat_low, lat_high = bootstrap_ci(lat_errs)
    jerk_mean, jerk_low, jerk_high = bootstrap_ci(jerks)
    plan_mean = float(np.mean(plan_times)) if plan_times else 0.0

    # Reason breakdown
    reasons: dict[str, int] = {}
    for e in episodes:
        reasons[e.termination_reason] = reasons.get(e.termination_reason, 0) + 1

    return {
        "num_episodes": n,
        "success_rate": {"mean": succ_rate, "ci_lower": succ_low, "ci_upper": succ_high},
        "collision_rate": {"mean": crash_rate, "ci_lower": crash_low, "ci_upper": crash_high},
        "out_of_road_rate": {"mean": oor_rate, "ci_lower": oor_low, "ci_upper": oor_high},
        "route_completion": {"mean": comp_mean, "ci_lower": comp_low, "ci_upper": comp_high},
        "mean_speed_mps": {"mean": speed_mean, "ci_lower": speed_low, "ci_upper": speed_high},
        "rms_lateral_error_m": {"mean": lat_mean, "ci_lower": lat_low, "ci_upper": lat_high},
        "steering_jerk": {"mean": jerk_mean, "ci_lower": jerk_low, "ci_upper": jerk_high},
        "mean_planning_time_ms": plan_mean,
        "termination_reasons": reasons,
    }
