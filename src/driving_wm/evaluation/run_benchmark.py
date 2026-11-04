"""Closed-loop benchmarking runner across controllers, map suites, and seeds."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from driving_wm.controllers.pid import (
    BaseController,
)
from driving_wm.evaluation.metrics import (
    EpisodeResult,
    aggregate_episode_results,
    compute_steering_jerk,
)
from driving_wm.evaluation.stats import paired_bootstrap_test
from driving_wm.utils.config import load_yaml
from driving_wm.utils.log import setup_logger

logger = setup_logger("evaluation.benchmark")


def run_single_episode(
    env: Any,
    controller: BaseController,
    controller_name: str,
    suite_name: str,
    layout: str,
    seed: int,
    max_steps: int = 300,
) -> tuple[EpisodeResult, np.ndarray]:
    """Execute a single episode and return EpisodeResult and (T, 2) trajectory."""
    obs, info = env.reset(seed=seed)
    controller.reset()

    agent = env.agent
    step_count = 0
    done = False

    trajectory_pts: list[tuple[float, float]] = []
    steer_history: list[float] = []
    speeds: list[float] = []
    lateral_errors: list[float] = []
    min_clearance = float("inf")

    term_reason = "max_step"

    while not done and step_count < max_steps:
        pos = np.array(agent.position)
        trajectory_pts.append((float(pos[0]), float(pos[1])))
        speeds.append(float(agent.speed))

        # Lane tracking error
        lane = agent.lane or (
            agent.navigation.current_ref_lanes[0] if agent.navigation.current_ref_lanes else None
        )
        if lane is not None:
            _, lat_offset = lane.local_coordinates(pos)
            lateral_errors.append(float(lat_offset))

        # Obstacle clearance
        if hasattr(env, "engine") and env.engine is not None:
            surrounding = env.engine.get_objects(
                lambda obj: obj is not agent and hasattr(obj, "position")
            )
            for other_obj in surrounding.values():
                d = float(np.hypot(other_obj.position[0] - pos[0], other_obj.position[1] - pos[1]))
                if d < min_clearance:
                    min_clearance = d

        action = controller.compute_action(env, info)
        steer_history.append(float(action[0]))

        obs, reward, terminated, truncated, step_info = env.step(action)
        step_count += 1
        done = terminated or truncated

        if step_info.get("arrive_dest"):
            term_reason = "arrive_dest"
        elif step_info.get("crash"):
            term_reason = "crash"
        elif step_info.get("out_of_road"):
            term_reason = "out_of_road"
        elif truncated:
            term_reason = "truncated"

    pos = np.array(agent.position)
    trajectory_pts.append((float(pos[0]), float(pos[1])))

    arrive_dest = bool(step_info.get("arrive_dest", False))
    crash = bool(step_info.get("crash", False))
    out_of_road = bool(step_info.get("out_of_road", False))
    success = arrive_dest and not crash and not out_of_road
    route_comp = float(step_info.get("route_completion", 0.0))

    rms_lat = math.sqrt(float(np.mean(np.array(lateral_errors) ** 2))) if lateral_errors else 0.0
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    jerk = compute_steering_jerk(steer_history)

    # Planning time if controller has telemetry
    plan_time = 0.0
    if hasattr(controller, "planning_times_ms") and controller.planning_times_ms and step_count > 0:
        plan_time = float(np.mean(controller.planning_times_ms[-step_count:]))

    res = EpisodeResult(
        controller_name=controller_name,
        suite_name=suite_name,
        map_layout=layout,
        seed=seed,
        num_steps=step_count,
        arrive_dest=arrive_dest,
        crash=crash,
        out_of_road=out_of_road,
        success=success,
        route_completion=route_comp,
        mean_speed_mps=mean_speed,
        rms_lateral_error_m=rms_lat,
        steering_jerk=jerk,
        min_obstacle_clearance_m=min_clearance if min_clearance != float("inf") else -1.0,
        termination_reason=term_reason,
        mean_planning_time_ms=plan_time,
    )

    return res, np.array(trajectory_pts, dtype=np.float32)


def run_benchmark_suite(
    controllers: dict[str, BaseController],
    map_suites_path: str | Path,
    output_dir: str | Path,
    episodes_per_suite: int = 15,
    max_steps: int = 300,
) -> tuple[Path, Path]:
    """Run cross-map benchmark comparing all controllers on identical maps and seeds."""
    from metadrive.envs import MetaDriveEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maps_cfg = load_yaml(map_suites_path)
    suites = ["test_unseen_seeds", "test_unseen_layouts", "test_traffic"]

    all_results: list[EpisodeResult] = []
    trajectories_sample: dict[str, np.ndarray] = {}

    csv_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.json"

    # Seed list for each suite to guarantee shared scenarios across controllers
    suite_scenarios: dict[str, list[tuple[str, int, float]]] = {}
    for s_name in suites:
        s_cfg = maps_cfg[s_name]
        layouts = s_cfg["layouts"]
        start_seed = s_cfg["start_seed"]
        traffic = s_cfg.get("traffic_density", 0.0)
        scenario_list = []
        for i in range(episodes_per_suite):
            layout = layouts[i % len(layouts)]
            seed = start_seed + i
            scenario_list.append((layout, seed, traffic))
        suite_scenarios[s_name] = scenario_list

    logger.info(
        f"Starting benchmark across {len(controllers)} controllers on {len(suites)} suites "
        f"({episodes_per_suite} episodes per suite = {episodes_per_suite * len(suites)} total per controller)."
    )

    for c_name, ctrl in controllers.items():
        logger.info(f"--- Evaluating Controller: {c_name} ---")
        for s_name, scenarios in suite_scenarios.items():
            # Group scenarios by (layout, traffic) so we can run them on the same MetaDriveEnv instance
            grouped_scenarios: dict[tuple[str, float], list[int]] = {}
            for layout, seed, traffic in scenarios:
                key = (layout, traffic)
                grouped_scenarios.setdefault(key, []).append(seed)

            for (layout, traffic), seeds in grouped_scenarios.items():
                min_s = min(seeds)
                max_s = max(seeds)
                env_config = {
                    "use_render": False,
                    "manual_control": False,
                    "traffic_density": traffic,
                    "map": layout,
                    "start_seed": min_s,
                    "num_scenarios": max_s - min_s + 1,
                    "random_agent_model": False,
                }

                env = MetaDriveEnv(env_config)

                for seed in seeds:
                    ep_res, traj = run_single_episode(
                        env, ctrl, c_name, s_name, layout, seed, max_steps=max_steps
                    )
                    all_results.append(ep_res)

                    # Save first trajectory of test_unseen_seeds for trajectory overlay plot
                    if s_name == "test_unseen_seeds" and seed == scenarios[0][1]:
                        trajectories_sample[c_name] = traj

                env.close()

            # Log interim status for this suite
            suite_eps = [
                e for e in all_results if e.controller_name == c_name and e.suite_name == s_name
            ]
            succ = sum(1 for e in suite_eps if e.success)
            comp = np.mean([e.route_completion for e in suite_eps])
            logger.info(
                f"[{c_name}] Suite '{s_name}': Success {succ}/{len(suite_eps)} ({succ / len(suite_eps) * 100:.1f}%), "
                f"Route Completion: {comp * 100:.1f}%"
            )

    # Save episodes.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "controller",
                "suite",
                "map",
                "seed",
                "steps",
                "success",
                "arrive_dest",
                "crash",
                "out_of_road",
                "route_completion",
                "mean_speed_mps",
                "rms_lateral_error_m",
                "steering_jerk",
                "min_obstacle_clearance_m",
                "mean_planning_time_ms",
                "termination_reason",
            ]
        )
        for r in all_results:
            writer.writerow(
                [
                    r.controller_name,
                    r.suite_name,
                    r.map_layout,
                    r.seed,
                    r.num_steps,
                    r.success,
                    r.arrive_dest,
                    r.crash,
                    r.out_of_road,
                    f"{r.route_completion:.4f}",
                    f"{r.mean_speed_mps:.3f}",
                    f"{r.rms_lateral_error_m:.3f}",
                    f"{r.steering_jerk:.4f}",
                    f"{r.min_obstacle_clearance_m:.3f}",
                    f"{r.mean_planning_time_ms:.2f}",
                    r.termination_reason,
                ]
            )

    logger.info(f"Saved {len(all_results)} episode rows to {csv_path}")

    # Build summary.json
    summary: dict[str, Any] = {}
    for c_name in controllers.keys():
        summary[c_name] = {}
        for s_name in suites:
            subset = [
                e for e in all_results if e.controller_name == c_name and e.suite_name == s_name
            ]
            summary[c_name][s_name] = aggregate_episode_results(subset)

    # Paired comparisons against Stanley PID baseline
    if "Stanley PID" in controllers and "Learned MPC" in controllers:
        summary["paired_comparisons"] = {}
        for s_name in suites:
            mpc_eps = [
                e
                for e in all_results
                if e.controller_name == "Learned MPC" and e.suite_name == s_name
            ]
            pid_eps = [
                e
                for e in all_results
                if e.controller_name == "Stanley PID" and e.suite_name == s_name
            ]
            # Ensure matched seeds
            mpc_by_seed = {e.seed: e for e in mpc_eps}
            pid_by_seed = {e.seed: e for e in pid_eps}
            common_seeds = sorted(list(set(mpc_by_seed.keys()) & set(pid_by_seed.keys())))

            mpc_comp = [mpc_by_seed[s].route_completion for s in common_seeds]
            pid_comp = [pid_by_seed[s].route_completion for s in common_seeds]
            test_res = paired_bootstrap_test(mpc_comp, pid_comp)
            summary["paired_comparisons"][s_name] = test_res

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary metrics to {summary_path}")

    # Plot benchmark bar chart
    from driving_wm.evaluation.plots import plot_benchmark_bar_charts, plot_trajectory_overlay

    chart_path = output_dir / "benchmark_comparison.png"
    plot_benchmark_bar_charts(summary, chart_path)

    if trajectories_sample:
        traj_path = output_dir / "trajectories_overlay.png"
        plot_trajectory_overlay(trajectories_sample, traj_path)

    return csv_path, summary_path
