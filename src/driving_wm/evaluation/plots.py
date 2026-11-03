"""Plotting routines for benchmark metrics, ablations, and trajectory overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_bar_charts(
    summary_data: dict[str, Any],
    save_path: str | Path,
) -> None:
    """Generate side-by-side bar plots comparing controllers across benchmark suites."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    controllers = list(summary_data.keys())
    # Metrics to display: Success Rate, Route Completion, Collision Rate
    suites = ["test_unseen_seeds", "test_unseen_layouts", "test_traffic"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, suite in enumerate(suites):
        ax = axes[idx]
        names = []
        succ_means = []
        succ_errs_low = []
        succ_errs_high = []

        comp_means = []

        for c_name in controllers:
            if suite in summary_data[c_name]:
                data = summary_data[c_name][suite]
                names.append(c_name)
                # Success rate & Wilson CI
                s_m = data["success_rate"]["mean"]
                s_l = data["success_rate"]["ci_lower"]
                s_h = data["success_rate"]["ci_upper"]
                succ_means.append(s_m)
                succ_errs_low.append(s_m - s_l)
                succ_errs_high.append(s_h - s_m)

                # Route completion
                c_m = data["route_completion"]["mean"]
                comp_means.append(c_m)

        x = np.arange(len(names))
        width = 0.35

        ax.bar(
            x - width / 2,
            succ_means,
            width,
            yerr=[succ_errs_low, succ_errs_high],
            capsize=4,
            label="Success Rate",
            color="#2ca02c",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            comp_means,
            width,
            label="Route Completion",
            color="#1f77b4",
            alpha=0.85,
        )

        suite_title = suite.replace("test_", "").replace("_", " ").title()
        ax.set_title(suite_title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ablation_safety(
    mpc_with_safety: dict[str, Any],
    mpc_without_safety: dict[str, Any],
    save_path: str | Path,
) -> None:
    """Plot Safety Constraints ON vs OFF ablation comparison."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Collision Rate", "Out-of-Road Rate", "Success Rate"]
    with_s = [
        mpc_with_safety["collision_rate"]["mean"],
        mpc_with_safety["out_of_road_rate"]["mean"],
        mpc_with_safety["success_rate"]["mean"],
    ]
    without_s = [
        mpc_without_safety["collision_rate"]["mean"],
        mpc_without_safety["out_of_road_rate"]["mean"],
        mpc_without_safety["success_rate"]["mean"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, with_s, width, label="MPC + Safety Constraints", color="#1f77b4")
    plt.bar(x + width / 2, without_s, width, label="MPC (Unconstrained Cost Only)", color="#d62728")

    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title("Ablation A: Impact of Explicit Safety Constraints")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_ablation_horizon_samples(
    grid_results: list[dict[str, Any]],
    save_path: str | Path,
) -> None:
    """Plot Ablation B: Horizon x Candidate-Count trade-off vs Planning Time and Completion."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # grid_results is list of dicts: {'horizon': H, 'samples': N, 'time_ms': T, 'completion': C}
    horizons = sorted(list({r["horizon"] for r in grid_results}))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for h in horizons:
        times = [r["time_ms"] for r in grid_results if r["horizon"] == h]
        ns = [r["samples"] for r in grid_results if r["horizon"] == h]
        ax1.plot(ns, times, "o-", label=f"Horizon H={h}", lw=1.8)

        comps = [r["completion"] for r in grid_results if r["horizon"] == h]
        ax2.plot(ns, comps, "s--", label=f"Horizon H={h}", lw=1.8)

    ax1.set_title("CPU Planning Latency vs Sample Count")
    ax1.set_xlabel("Number of Candidate Trajectories N")
    ax1.set_ylabel("Mean Planning Time (ms/step)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Route Completion vs Sample Count")
    ax2.set_xlabel("Number of Candidate Trajectories N")
    ax2.set_ylabel("Mean Route Completion")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trajectory_overlay(
    trajectories: dict[str, np.ndarray],  # name -> (T, 2) [x, y]
    save_path: str | Path,
) -> None:
    """Plot 2D trajectory spatial traces for controllers on a shared map."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    colors = {
        "MPC (Learned)": "#1f77b4",
        "Stanley PID": "#2ca02c",
        "Constant Throttle": "#ff7f0e",
        "Random": "#7f7f7f",
    }

    for name, traj in trajectories.items():
        color = colors.get(name, None)
        plt.plot(traj[:, 0], traj[:, 1], label=name, lw=2.0, color=color)

    plt.title("Closed-Loop Trajectory Traces on Held-Out Scenario")
    plt.xlabel("World X Position (m)")
    plt.ylabel("World Y Position (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
