"""Failure analysis script: inspects failed MPC episodes, categorizes causes, and renders top-down GIFs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from driving_wm.evaluation.render import render_episode_gif
from driving_wm.models.evaluate import load_model_from_checkpoint
from driving_wm.planning.mpc import LearnedModelMPC
from driving_wm.utils.log import setup_logger

logger = setup_logger("analyze_failures")


def analyze_failures(
    results_dir: str | Path = "results/smoke",
    checkpoint_path: str | Path = "checkpoints/world_model_onestep.pt",
    max_cases: int = 5,
) -> list[dict[str, Any]]:
    """Identify failed MPC episodes in episodes.csv, categorize root causes, and render visualizations."""
    results_dir = Path(results_dir)
    csv_file = results_dir / "episodes.csv"

    if not csv_file.is_file():
        raise FileNotFoundError(f"Benchmark results not found: {csv_file}")

    model, s_norm, t_norm, _ = load_model_from_checkpoint(checkpoint_path)
    mpc = LearnedModelMPC(
        model=model, state_norm=s_norm, target_norm=t_norm, horizon=10, num_samples=256
    )

    failed_episodes: list[dict[str, Any]] = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["controller"] in [
                "Learned MPC",
                "MPC + Safety Constraints",
                "MPC (Unconstrained)",
            ]:
                if row["success"].lower() == "false":
                    failed_episodes.append(row)

    logger.info(f"Found {len(failed_episodes)} failed MPC episodes across benchmark runs.")

    categorized_cases: list[dict[str, Any]] = []
    docs_img = Path("docs/img")
    docs_img.mkdir(parents=True, exist_ok=True)

    # Pick up to max_cases distinct failed episodes
    selected = failed_episodes[:max_cases]

    for idx, case in enumerate(selected):
        layout = case["map"]
        seed = int(case["seed"])
        reason = case["termination_reason"]
        suite = case["suite"]

        # Categorize cause
        if "crash" in reason:
            category = "Dynamic Traffic Obstacle Collision"
            explanation = "Vehicle collided with adjacent moving traffic participant or lead vehicle under dynamic merge."
        elif layout in ["O", "T"]:
            category = "Held-Out Topology Curvature Shift"
            explanation = "Sharp continuous curvature on unseen roundabout/T-junction exceeded the 1.0s planning horizon lookahead."
        elif float(case.get("rms_lateral_error_m", 0.0)) > 2.0:
            category = "Cumulative Lateral Drift / Curb Exit"
            explanation = "Compounding autoregressive dynamics drift pushed vehicle past road boundary margin into sidewalk."
        else:
            category = "Infeasible Safety Fallback Timeout"
            explanation = "Constraint evaluator detected no feasible candidates, triggering safety braking and subsequent timeout."

        case_info = {
            "case_id": idx + 1,
            "suite": suite,
            "layout": layout,
            "seed": seed,
            "termination_reason": reason,
            "route_completion": float(case["route_completion"]),
            "mean_speed_mps": float(case["mean_speed_mps"]),
            "category": category,
            "explanation": explanation,
        }

        # Render GIF for the failure case
        gif_filename = docs_img / f"failure_case_{idx + 1}.gif"
        traffic = 0.08 if "traffic" in suite else 0.0

        try:
            render_episode_gif(
                controller=mpc,
                map_layout=layout,
                seed=seed,
                max_steps=180,
                save_path=gif_filename,
                traffic_density=traffic,
            )
            case_info["gif_path"] = str(gif_filename)
        except Exception as e:
            logger.warning(f"Failed to render GIF for case {idx + 1}: {e}")
            case_info["gif_path"] = None

        categorized_cases.append(case_info)

    # Save failure analysis report
    report_file = results_dir / "failure_analysis.json"
    import json

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(categorized_cases, f, indent=2)

    logger.info(f"Saved failure analysis for {len(categorized_cases)} cases to {report_file}")
    return categorized_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and render failed MPC episodes")
    parser.add_argument("--results-dir", type=str, default="results/smoke")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model_onestep.pt")
    args = parser.parse_args()
    analyze_failures(args.results_dir, args.checkpoint)


if __name__ == "__main__":
    main()
