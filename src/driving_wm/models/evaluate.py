"""Evaluation of world models against kinematic and persistence baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from driving_wm.data.dataset import Normalizer
from driving_wm.data.state import TARGET_LABELS, ego_to_world, wrap_angle
from driving_wm.models.dynamics import KinematicBicycleModel, PersistenceBaseline, WorldModel
from driving_wm.utils.log import setup_logger

logger = setup_logger("models.evaluate")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[WorldModel, Normalizer, Normalizer, dict[str, Any]]:
    """Load trained WorldModel and its normalization statistics from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    m_cfg = ckpt["model_config"]

    model = WorldModel(
        state_dim=m_cfg["state_dim"],
        action_dim=m_cfg["action_dim"],
        target_dim=m_cfg["target_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        activation=m_cfg["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state_norm = Normalizer.from_dict(ckpt["state_norm"])
    target_norm = Normalizer.from_dict(ckpt["target_norm"])

    return model, state_norm, target_norm, ckpt


def evaluate_one_step_metrics(
    model: WorldModel,
    state_norm: Normalizer,
    target_norm: Normalizer,
    npz_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Compute per-dimension MAE and RMSE in physical units across all transitions."""
    data = np.load(npz_path)
    states = torch.tensor(data["states"], dtype=torch.float32).to(device)
    actions = torch.tensor(data["actions"], dtype=torch.float32).to(device)
    targets = data["targets"]  # (N, 7) in physical units

    kinematic = KinematicBicycleModel()
    persistence = PersistenceBaseline()

    with torch.no_grad():
        norm_s = state_norm.normalize(states)
        pred_norm_targets = model(norm_s, actions)
        pred_targets = target_norm.denormalize(pred_norm_targets).cpu().numpy()

    n = len(targets)
    s_np = states.cpu().numpy()
    a_np = actions.cpu().numpy()
    model_mae = np.mean(np.abs(pred_targets - targets), axis=0)
    model_rmse = np.sqrt(np.mean((pred_targets - targets) ** 2, axis=0))

    # Persistence baseline
    pers_preds = np.zeros_like(targets)
    for i in range(min(n, 5000)):
        _, p_inc = persistence.step(s_np[i], a_np[i])
        pers_preds[i, 4:] = p_inc
    pers_mae = np.mean(np.abs(pers_preds[: min(n, 5000)] - targets[: min(n, 5000)]), axis=0)

    # Kinematic baseline
    kin_preds = np.zeros_like(targets)
    for i in range(min(n, 5000)):
        _, p_inc = kinematic.step(s_np[i], a_np[i])
        kin_preds[i, 4:] = p_inc
    kin_mae = np.mean(np.abs(kin_preds[: min(n, 5000)] - targets[: min(n, 5000)]), axis=0)

    results = {
        "num_samples": n,
        "per_dimension_mae": {
            TARGET_LABELS[i]: float(model_mae[i]) for i in range(len(TARGET_LABELS))
        },
        "per_dimension_rmse": {
            TARGET_LABELS[i]: float(model_rmse[i]) for i in range(len(TARGET_LABELS))
        },
        "persistence_mae": {
            TARGET_LABELS[i]: float(pers_mae[i]) for i in range(len(TARGET_LABELS))
        },
        "kinematic_mae": {TARGET_LABELS[i]: float(kin_mae[i]) for i in range(len(TARGET_LABELS))},
    }
    return results


def evaluate_rollout_horizons(
    model: WorldModel,
    state_norm: Normalizer,
    target_norm: Normalizer,
    npz_path: str | Path,
    horizons: list[int] = [1, 2, 5, 10, 20, 30],
    max_episodes: int = 15,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate open-loop autoregressive rollout error compounding across horizons."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(npz_path)
    states = data["states"]
    actions = data["actions"]
    pose_increments = data["pose_increments"]
    episode_ids = data["episode_ids"]

    unique_eps = np.unique(episode_ids)[:max_episodes]
    max_h = max(horizons)

    # Horizon -> list of position errors, yaw errors, speed errors
    pos_errors: dict[int, list[float]] = {h: [] for h in horizons}
    yaw_errors: dict[int, list[float]] = {h: [] for h in horizons}
    speed_errors: dict[int, list[float]] = {h: [] for h in horizons}

    kinematic = KinematicBicycleModel()
    kin_pos_errors: dict[int, list[float]] = {h: [] for h in horizons}

    for ep in unique_eps:
        mask = episode_ids == ep
        ep_states = states[mask]
        ep_actions = actions[mask]
        ep_poses_inc = pose_increments[mask]

        ep_len = len(ep_states)
        if ep_len <= max_h:
            continue

        # Evaluate rollouts originating at different starts within episode
        for start_t in range(0, ep_len - max_h, max(1, max_h // 2)):
            # Ground truth world pose accumulation starting at (0, 0, 0)
            gt_x, gt_y, gt_yaw = 0.0, 0.0, 0.0
            gt_poses = [(gt_x, gt_y, gt_yaw)]
            for step_i in range(start_t, start_t + max_h):
                dx_e, dy_e, dyaw = ep_poses_inc[step_i]
                dx_w, dy_w = ego_to_world(dx_e, dy_e, gt_yaw)
                gt_x += float(dx_w)
                gt_y += float(dy_w)
                gt_yaw = float(wrap_angle(gt_yaw + dyaw))
                gt_poses.append((gt_x, gt_y, gt_yaw))

            # Model rollout
            init_s = torch.tensor(ep_states[start_t : start_t + 1], dtype=torch.float32).to(device)
            init_p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(device)
            act_seq = torch.tensor(
                ep_actions[start_t : start_t + max_h][None, ...], dtype=torch.float32
            ).to(device)

            with torch.inference_mode():
                pred_states, pred_poses = model.rollout(
                    init_s, init_p, act_seq, state_norm, target_norm
                )

            pred_p_np = pred_poses[0].cpu().numpy()  # (max_h + 1, 3)
            pred_s_np = pred_states[0].cpu().numpy()

            # Kinematic rollout
            kin_s = ep_states[start_t].copy()
            kin_x, kin_y, kin_yaw = 0.0, 0.0, 0.0
            kin_poses = [(kin_x, kin_y, kin_yaw)]
            for step_i in range(start_t, start_t + max_h):
                kin_s, k_inc = kinematic.step(kin_s, ep_actions[step_i])
                k_dx, k_dy = ego_to_world(k_inc[0], k_inc[1], kin_yaw)
                kin_x += float(k_dx)
                kin_y += float(k_dy)
                kin_yaw = float(wrap_angle(kin_yaw + k_inc[2]))
                kin_poses.append((kin_x, kin_y, kin_yaw))

            for h in horizons:
                gt_p = gt_poses[h]
                pr_p = pred_p_np[h]
                kn_p = kin_poses[h]

                dist_err = float(np.hypot(pr_p[0] - gt_p[0], pr_p[1] - gt_p[1]))
                y_err = float(abs(wrap_angle(pr_p[2] - gt_p[2])))
                v_err = float(abs(pred_s_np[h, 0] - ep_states[start_t + h, 0]))

                kin_dist = float(np.hypot(kn_p[0] - gt_p[0], kn_p[1] - gt_p[1]))

                pos_errors[h].append(dist_err)
                yaw_errors[h].append(y_err)
                speed_errors[h].append(v_err)
                kin_pos_errors[h].append(kin_dist)

    summary = {
        "horizons": horizons,
        "mean_position_error": {h: float(np.mean(pos_errors[h])) for h in horizons},
        "median_position_error": {h: float(np.median(pos_errors[h])) for h in horizons},
        "mean_yaw_error_rad": {h: float(np.mean(yaw_errors[h])) for h in horizons},
        "mean_speed_error_mps": {h: float(np.mean(speed_errors[h])) for h in horizons},
        "kinematic_position_error": {h: float(np.mean(kin_pos_errors[h])) for h in horizons},
    }
    return summary


def plot_prediction_error_growth(
    eval_summary: dict[str, Any],
    save_path: str | Path,
) -> None:
    """Plot position and heading error growth across prediction horizons."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = eval_summary["horizons"]
    model_pos = [eval_summary["mean_position_error"][h] for h in horizons]
    kin_pos = [eval_summary["kinematic_position_error"][h] for h in horizons]
    yaw_err = [eval_summary["mean_yaw_error_rad"][h] for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(horizons, model_pos, "o-", label="Learned World Model", lw=2.2, color="#1f77b4")
    ax1.plot(horizons, kin_pos, "s--", label="Kinematic Bicycle", lw=1.8, color="#ff7f0e")
    ax1.set_title("Position Error Compound vs Rollout Horizon")
    ax1.set_xlabel("Horizon Steps (0.1s dt)")
    ax1.set_ylabel("Mean Euclidean Position Error (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(horizons, np.rad2deg(yaw_err), "o-", lw=2.2, color="#2ca02c")
    ax2.set_title("Heading Error Compound vs Rollout Horizon")
    ax2.set_xlabel("Horizon Steps (0.1s dt)")
    ax2.set_ylabel("Mean Heading Error (deg)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved prediction error plot to {save_path}")
