"""Trajectory collection module across MetaDrive map suites with mixed behavior policies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from driving_wm.controllers.pid import StanleyPIDController
from driving_wm.data.state import (
    ACTION_DIM,
    STATE_DIM,
    TARGET_DIM,
    extract_vehicle_dynamic_state,
    world_to_ego,
    wrap_angle,
)
from driving_wm.utils.config import compute_config_hash
from driving_wm.utils.log import setup_logger

logger = setup_logger("data.collect")


class BehaviorPolicyMixture:
    """Generates actions by mixing noisy baseline, smoothed random, and disturbance-recovery."""

    def __init__(
        self,
        baseline_controller: StanleyPIDController,
        noisy_baseline_ratio: float = 0.6,
        random_smooth_ratio: float = 0.2,
        disturbance_recovery_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.baseline = baseline_controller
        self.ratios = [noisy_baseline_ratio, random_smooth_ratio, disturbance_recovery_ratio]
        self.rng = np.random.default_rng(seed)

        # Policy state
        self.current_mode = "noisy_baseline"
        self.ou_noise = np.zeros(2, dtype=np.float32)
        self.smooth_action = np.zeros(2, dtype=np.float32)
        self.disturbance_remaining = 0
        self.disturbance_action = np.zeros(2, dtype=np.float32)

    def reset_episode(self) -> None:
        """Select a behavior mode for the upcoming episode and reset internal state."""
        self.baseline.reset()
        mode_idx = self.rng.choice(3, p=self.ratios)
        self.current_mode = ["noisy_baseline", "smooth_random", "disturbance_recovery"][mode_idx]
        self.ou_noise = np.zeros(2, dtype=np.float32)
        self.smooth_action = np.array([0.0, self.rng.uniform(0.2, 0.6)], dtype=np.float32)
        self.disturbance_remaining = 0

    def compute_action(
        self, env: Any, step_idx: int, info: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Compute the next action based on the active episode mode."""
        base_action = self.baseline.compute_action(env, info)

        if self.current_mode == "noisy_baseline":
            # Ornstein-Uhlenbeck noise added to baseline
            noise_sample = self.rng.normal(0.0, [0.12, 0.15], size=2).astype(np.float32)
            self.ou_noise = 0.75 * self.ou_noise + 0.25 * noise_sample
            action = base_action + self.ou_noise

        elif self.current_mode == "smooth_random":
            # Smoothed random action walk
            rand_delta = self.rng.normal(0.0, [0.10, 0.12], size=2).astype(np.float32)
            self.smooth_action = 0.85 * self.smooth_action + 0.15 * rand_delta
            # Bias throttle positive to keep vehicle moving
            self.smooth_action[1] = np.clip(self.smooth_action[1] + 0.05, -0.3, 0.8)
            action = self.smooth_action

        else:  # disturbance_recovery
            # Normal baseline interspersed with disturbance bursts
            if self.disturbance_remaining > 0:
                action = self.disturbance_action
                self.disturbance_remaining -= 1
            else:
                # Trigger a new disturbance with 5% chance
                if self.rng.random() < 0.06 and step_idx > 15:
                    self.disturbance_remaining = self.rng.integers(4, 9)
                    steer_dist = self.rng.choice([-0.45, 0.45])
                    self.disturbance_action = np.array([steer_dist, 0.3], dtype=np.float32)
                    action = self.disturbance_action
                else:
                    action = base_action

        return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_dataset(
    map_suite_config: dict[str, Any],
    split_name: str,
    target_episodes: int,
    max_steps_per_episode: int,
    output_dir: str | Path,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Collect driving episodes using MetaDrive and save compressed npz + metadata.json."""
    from metadrive.envs import MetaDriveEnv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layouts = map_suite_config.get("layouts", ["S", "C", "SC"])
    start_seed = map_suite_config.get("start_seed", 1000)
    traffic_density = map_suite_config.get("traffic_density", 0.0)

    logger.info(
        f"Collecting {target_episodes} episodes for split '{split_name}' across {len(layouts)} layouts..."
    )

    baseline_ctrl = StanleyPIDController(target_speed=10.0)
    policy_mix = BehaviorPolicyMixture(baseline_ctrl, seed=seed)

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_next_states: list[np.ndarray] = []
    all_pose_increments: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_episode_ids: list[int] = []
    all_steps: list[int] = []
    all_map_ids: list[str] = []
    all_seeds: list[int] = []
    all_terminated: list[bool] = []
    all_truncated: list[bool] = []
    all_reasons: list[str] = []

    episodes_collected = 0
    total_transitions = 0

    # Distribute episodes across layouts
    episodes_per_layout = math.ceil(target_episodes / len(layouts))

    for layout in layouts:
        if episodes_collected >= target_episodes:
            break

        env_config = {
            "use_render": False,
            "manual_control": False,
            "traffic_density": traffic_density,
            "map": layout,
            "start_seed": start_seed,
            "num_scenarios": target_episodes * 2,
            "random_agent_model": False,
        }

        env = MetaDriveEnv(env_config)

        for _ in range(episodes_per_layout):
            if episodes_collected >= target_episodes:
                break

            current_seed = start_seed + episodes_collected
            obs, info = env.reset(seed=current_seed)
            policy_mix.reset_episode()

            agent = env.agent
            dt = env.config["decision_repeat"] * env.config["physics_world_step_size"]

            curr_state, curr_heading = extract_vehicle_dynamic_state(
                agent, prev_heading=None, dt=dt
            )
            curr_pos = np.array(agent.position, dtype=np.float64)

            ep_transitions = 0
            done = False
            term_reason = "max_step"

            while not done and ep_transitions < max_steps_per_episode:
                action = policy_mix.compute_action(env, ep_transitions, info)

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_transitions += 1

                agent = env.agent
                next_pos = np.array(agent.position, dtype=np.float64)
                next_heading = float(agent.heading_theta)
                next_state, _ = extract_vehicle_dynamic_state(
                    agent, prev_heading=curr_heading, dt=dt
                )

                # Pose increment in ego body frame at time t
                d_pos_world = next_pos - curr_pos
                dx_ego, dy_ego = world_to_ego(d_pos_world[0], d_pos_world[1], curr_heading)
                dyaw = wrap_angle(next_heading - curr_heading)
                pose_inc = np.array([dx_ego, dy_ego, dyaw], dtype=np.float32)

                # State delta
                d_state = next_state - curr_state
                # Target: [dv_long, dv_lat, dyaw_rate, dsteering, dx_ego, dy_ego, dyaw]
                target = np.concatenate([d_state, pose_inc]).astype(np.float32)

                done = terminated or truncated
                if step_info.get("arrive_dest"):
                    term_reason = "arrive_dest"
                elif step_info.get("crash"):
                    term_reason = "crash"
                elif step_info.get("out_of_road"):
                    term_reason = "out_of_road"
                elif truncated:
                    term_reason = "truncated"

                all_states.append(curr_state)
                all_actions.append(action)
                all_next_states.append(next_state)
                all_pose_increments.append(pose_inc)
                all_targets.append(target)
                all_episode_ids.append(episodes_collected)
                all_steps.append(ep_transitions - 1)
                all_map_ids.append(layout)
                all_seeds.append(current_seed)
                all_terminated.append(terminated)
                all_truncated.append(truncated)
                all_reasons.append(term_reason if done else "in_progress")

                curr_state = next_state
                curr_heading = next_heading
                curr_pos = next_pos

            episodes_collected += 1
            total_transitions += ep_transitions

        env.close()

    logger.info(
        f"Split '{split_name}' complete: {episodes_collected} episodes, {total_transitions} transitions."
    )

    # Save npz
    npz_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        npz_path,
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        pose_increments=np.array(all_pose_increments, dtype=np.float32),
        targets=np.array(all_targets, dtype=np.float32),
        episode_ids=np.array(all_episode_ids, dtype=np.int32),
        steps=np.array(all_steps, dtype=np.int32),
        seeds=np.array(all_seeds, dtype=np.int32),
        terminated=np.array(all_terminated, dtype=bool),
        truncated=np.array(all_truncated, dtype=bool),
    )

    # Save metadata
    meta = {
        "split": split_name,
        "num_episodes": episodes_collected,
        "num_transitions": total_transitions,
        "layouts": layouts,
        "start_seed": start_seed,
        "seed_range": [int(min(all_seeds)), int(max(all_seeds))] if all_seeds else [],
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "target_dim": TARGET_DIM,
        "config_hash": compute_config_hash(map_suite_config),
    }
    meta_path = output_dir / f"{split_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return npz_path, meta_path
