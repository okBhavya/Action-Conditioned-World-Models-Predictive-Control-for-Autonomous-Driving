"""End-to-end autonomous driving world model research pipeline.

Executes data collection, dynamics model training, prediction evaluation,
safety-constrained MPC planning, ablation experiments, and full benchmarking.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from driving_wm.controllers.pid import (
    ConstantThrottleController,
    RandomController,
    StanleyPIDController,
    ZeroController,
)
from driving_wm.data.collect import collect_dataset
from driving_wm.data.dataset import validate_dataset
from driving_wm.evaluation.plots import (
    plot_ablation_horizon_samples,
    plot_ablation_safety,
)
from driving_wm.evaluation.render import render_episode_gif
from driving_wm.evaluation.run_benchmark import run_benchmark_suite, run_single_episode
from driving_wm.models.evaluate import (
    evaluate_one_step_metrics,
    evaluate_rollout_horizons,
    load_model_from_checkpoint,
    plot_prediction_error_growth,
)
from driving_wm.models.train import check_overfit_one_batch, train_model
from driving_wm.planning.mpc import LearnedModelMPC
from driving_wm.utils.config import compute_config_hash, load_yaml
from driving_wm.utils.log import setup_logger
from driving_wm.utils.seeding import seed_everything

logger = setup_logger("run_pipeline")


def get_environment_info() -> dict[str, Any]:
    """Capture runtime system information, hardware, and dependency versions."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for pkg in ["metadrive-simulator", "gymnasium", "numpy", "matplotlib", "pyyaml"]:
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except Exception:
            info[pkg] = "unknown"

    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        info["git_commit"] = "unknown"

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end driving world model pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/smoke.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--skip-collection", action="store_true", help="Skip data collection if npz exists"
    )
    args = parser.parse_args()

    pipeline_start_t = time.time()
    cfg_path = Path(args.config)
    config = load_yaml(cfg_path)
    maps_cfg = load_yaml("configs/maps.yaml")

    seed = config.get("seed", 42)
    seed_everything(seed)

    # Output directories
    b_cfg = config.get("benchmark", {})
    results_dir = Path(b_cfg.get("results_dir", "results/smoke"))
    results_dir.mkdir(parents=True, exist_ok=True)

    d_cfg = config.get("data", {})
    data_dir = Path(d_cfg.get("output_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = config.get("model", {})
    ckpt_dir = Path(m_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage_timings: dict[str, float] = {}

    logger.info("=" * 70)
    logger.info(f"STARTING PIPELINE RUN WITH CONFIG: {cfg_path}")
    logger.info(f"Results destination: {results_dir.resolve()}")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # STAGE 1: Data Collection
    # ---------------------------------------------------------
    s1_t = time.time()
    logger.info("\n>>> STAGE 1: Data Collection across Map Suites...")

    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"

    if args.skip_collection and train_npz.is_file() and val_npz.is_file():
        logger.info("Found existing datasets, skipping collection (--skip-collection enabled).")
    else:
        collect_dataset(
            map_suite_config=maps_cfg["train"],
            split_name="train",
            target_episodes=d_cfg.get("train_episodes", 15),
            max_steps_per_episode=d_cfg.get("max_steps_per_episode", 200),
            output_dir=data_dir,
            seed=1000,
        )
        collect_dataset(
            map_suite_config=maps_cfg["val"],
            split_name="val",
            target_episodes=d_cfg.get("val_episodes", 5),
            max_steps_per_episode=d_cfg.get("max_steps_per_episode", 200),
            output_dir=data_dir,
            seed=2000,
        )

    # Validate dataset consistency
    train_meta = validate_dataset(train_npz)
    val_meta = validate_dataset(val_npz)
    logger.info(f"Train Dataset Validated: {train_meta}")
    logger.info(f"Val Dataset Validated: {val_meta}")
    stage_timings["data_collection_seconds"] = time.time() - s1_t

    # ---------------------------------------------------------
    # STAGE 2: World Model Training (One-Step Loss)
    # ---------------------------------------------------------
    s2_t = time.time()
    logger.info("\n>>> STAGE 2: Dynamics Model Training (One-Step)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sanity check: overfit one batch
    check_overfit_one_batch(train_npz, device=device)

    loss_plot_path = results_dir / "loss_curve_onestep.png"
    ckpt_onestep = train_model(
        train_npz=train_npz,
        val_npz=val_npz,
        config=config,
        checkpoint_dir=ckpt_dir,
        multistep_ablation=False,
        plot_path=loss_plot_path,
    )
    stage_timings["training_onestep_seconds"] = time.time() - s2_t

    # ---------------------------------------------------------
    # STAGE 3: Multi-Step Unrolled Model Training (Ablation C)
    # ---------------------------------------------------------
    s3_t = time.time()
    logger.info("\n>>> STAGE 3: Multi-Step Model Training (Ablation C)...")
    loss_plot_multistep = results_dir / "loss_curve_multistep.png"
    ckpt_multistep = train_model(
        train_npz=train_npz,
        val_npz=val_npz,
        config=config,
        checkpoint_dir=ckpt_dir,
        multistep_ablation=True,
        plot_path=loss_plot_multistep,
    )
    logger.info(f"Multi-step checkpoint saved: {ckpt_multistep}")
    stage_timings["training_multistep_seconds"] = time.time() - s3_t

    # ---------------------------------------------------------
    # STAGE 4: Model Prediction Error Evaluation
    # ---------------------------------------------------------
    s4_t = time.time()
    logger.info("\n>>> STAGE 4: Open-Loop Prediction Error Evaluation...")
    model_one, s_norm, t_norm, _ = load_model_from_checkpoint(ckpt_onestep, device=device)

    one_step_metrics = evaluate_one_step_metrics(model_one, s_norm, t_norm, val_npz, device=device)
    logger.info(
        f"One-Step Prediction MAE (Physical units): {json.dumps(one_step_metrics['per_dimension_mae'], indent=2)}"
    )

    with open(results_dir / "prediction_one_step.json", "w", encoding="utf-8") as f:
        json.dump(one_step_metrics, f, indent=2)

    rollout_summary = evaluate_rollout_horizons(
        model_one,
        s_norm,
        t_norm,
        val_npz,
        horizons=[1, 2, 5, 10, 20],
        max_episodes=10,
        device=device,
    )
    logger.info(f"Rollout Position Error Growth (m): {rollout_summary['mean_position_error']}")

    with open(results_dir / "prediction_rollout_horizons.json", "w", encoding="utf-8") as f:
        json.dump(rollout_summary, f, indent=2)

    plot_prediction_error_growth(rollout_summary, results_dir / "prediction_error_growth.png")
    stage_timings["model_eval_seconds"] = time.time() - s4_t

    # ---------------------------------------------------------
    # STAGE 5: Ablation A: Safety Constraints ON vs OFF
    # ---------------------------------------------------------
    s5_t = time.time()
    logger.info("\n>>> STAGE 5: Ablation A (Safety Constraints ON vs OFF)...")
    mpc_cfg = config.get("mpc", {})
    h = mpc_cfg.get("horizon", 10)
    n_samples = mpc_cfg.get("num_samples", 256)

    mpc_safe = LearnedModelMPC(
        model=model_one,
        state_norm=s_norm,
        target_norm=t_norm,
        horizon=h,
        num_samples=n_samples,
        safety_constraints=True,
        target_speed=mpc_cfg.get("target_speed", 10.0),
        cost_weights=mpc_cfg.get("cost_weights", {}),
    )
    mpc_unconstrained = LearnedModelMPC(
        model=model_one,
        state_norm=s_norm,
        target_norm=t_norm,
        horizon=h,
        num_samples=n_samples,
        safety_constraints=False,
        target_speed=mpc_cfg.get("target_speed", 10.0),
        cost_weights=mpc_cfg.get("cost_weights", {}),
    )

    ablation_a_controllers = {
        "MPC + Safety Constraints": mpc_safe,
        "MPC (Unconstrained)": mpc_unconstrained,
    }
    ab_a_csv, ab_a_summary = run_benchmark_suite(
        controllers=ablation_a_controllers,
        map_suites_path="configs/maps.yaml",
        output_dir=results_dir / "ablation_a",
        episodes_per_suite=min(5, b_cfg.get("episodes_per_suite", 5)),
        max_steps=200,
    )
    ab_a_data = load_yaml(ab_a_summary)
    plot_ablation_safety(
        ab_a_data["MPC + Safety Constraints"]["test_unseen_seeds"],
        ab_a_data["MPC (Unconstrained)"]["test_unseen_seeds"],
        results_dir / "ablation_a_safety.png",
    )
    stage_timings["ablation_a_seconds"] = time.time() - s5_t

    # ---------------------------------------------------------
    # STAGE 6: Ablation B: Horizon x Candidate Count Grid
    # ---------------------------------------------------------
    s6_t = time.time()
    logger.info("\n>>> STAGE 6: Ablation B (Horizon x Candidate Count Grid)...")
    grid_params = [
        (5, 128),
        (5, 256),
        (10, 128),
        (10, 256),
    ]
    grid_records: list[dict[str, Any]] = []

    from metadrive.envs import MetaDriveEnv

    grid_env = MetaDriveEnv(
        {
            "use_render": False,
            "manual_control": False,
            "traffic_density": 0.0,
            "map": "SC",
            "start_seed": 2000,
            "num_scenarios": 1,
        }
    )

    for grid_h, grid_n in grid_params:
        test_mpc = LearnedModelMPC(
            model=model_one,
            state_norm=s_norm,
            target_norm=t_norm,
            horizon=grid_h,
            num_samples=grid_n,
            safety_constraints=True,
            target_speed=mpc_cfg.get("target_speed", 10.0),
        )
        res, _ = run_single_episode(
            grid_env, test_mpc, f"MPC_H{grid_h}_N{grid_n}", "ablation_b", "SC", 2000, max_steps=100
        )
        grid_records.append(
            {
                "horizon": grid_h,
                "samples": grid_n,
                "time_ms": res.mean_planning_time_ms,
                "completion": res.route_completion,
            }
        )
    grid_env.close()

    with open(results_dir / "ablation_b_grid.json", "w", encoding="utf-8") as f:
        json.dump(grid_records, f, indent=2)

    plot_ablation_horizon_samples(grid_records, results_dir / "ablation_b_tradeoff.png")
    stage_timings["ablation_b_seconds"] = time.time() - s6_t

    # ---------------------------------------------------------
    # STAGE 7: Headline Closed-Loop Benchmark
    # ---------------------------------------------------------
    s7_t = time.time()
    logger.info("\n>>> STAGE 7: Closed-Loop Controller Benchmark across Suites...")

    headline_controllers = {
        "Learned MPC": mpc_safe,
        "Stanley PID": StanleyPIDController(target_speed=10.0),
        "Constant Throttle": ConstantThrottleController(throttle=0.4),
        "Zero Action": ZeroController(),
        "Random": RandomController(seed=seed),
    }

    episodes_per_suite = b_cfg.get("episodes_per_suite", 5)
    max_steps = b_cfg.get("max_steps", 250)

    csv_path, summary_path = run_benchmark_suite(
        controllers=headline_controllers,
        map_suites_path="configs/maps.yaml",
        output_dir=results_dir,
        episodes_per_suite=episodes_per_suite,
        max_steps=max_steps,
    )
    stage_timings["benchmark_seconds"] = time.time() - s7_t

    # ---------------------------------------------------------
    # STAGE 8: Failure Analysis & Visual Demonstrations
    # ---------------------------------------------------------
    s8_t = time.time()
    logger.info("\n>>> STAGE 8: Failure Analysis and Top-Down Video Rendering...")
    docs_img_dir = Path("docs/img")
    docs_img_dir.mkdir(parents=True, exist_ok=True)

    # Render headline MPC demonstration
    demo_gif = docs_img_dir / "demo.gif"
    render_episode_gif(
        controller=mpc_safe,
        map_layout="SC",
        seed=3000,
        max_steps=200,
        save_path=demo_gif,
    )

    # Search for an episode where MPC crashed or went out of road, or render a tight curve scenario
    failure_gif = docs_img_dir / "failure_case_traffic.gif"
    render_episode_gif(
        controller=mpc_safe,
        map_layout="X",
        seed=5000,
        max_steps=150,
        save_path=failure_gif,
        traffic_density=0.12,
    )
    stage_timings["rendering_seconds"] = time.time() - s8_t

    # ---------------------------------------------------------
    # STAGE 9: Environment Metadata & Final Reports
    # ---------------------------------------------------------
    total_elapsed = time.time() - pipeline_start_t
    stage_timings["total_pipeline_seconds"] = total_elapsed

    env_metadata = {
        "system": get_environment_info(),
        "stage_timings_seconds": stage_timings,
        "config_hash": compute_config_hash(config),
    }
    with open(results_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(env_metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"PIPELINE RUN COMPLETED IN {total_elapsed:.1f} s ({total_elapsed / 60.0:.1f} min)")
    logger.info(f"Saved artifacts in: {results_dir.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
