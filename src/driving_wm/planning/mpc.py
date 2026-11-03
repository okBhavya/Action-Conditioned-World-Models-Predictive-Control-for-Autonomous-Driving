"""Receding-horizon Model Predictive Control (MPC) using learned world model rollouts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from driving_wm.controllers.pid import BaseController
from driving_wm.data.dataset import Normalizer
from driving_wm.data.state import extract_vehicle_dynamic_state
from driving_wm.models.dynamics import WorldModel
from driving_wm.planning.constraints import evaluate_all_constraints
from driving_wm.planning.cost import MPCCost
from driving_wm.utils.log import setup_logger

logger = setup_logger("planning.mpc")


class LearnedModelMPC(BaseController):
    """Action-conditioned model predictive controller with explicit safety constraint filtering."""

    def __init__(
        self,
        model: WorldModel,
        state_norm: Normalizer,
        target_norm: Normalizer,
        horizon: int = 10,
        num_samples: int = 512,
        dt: float = 0.1,
        target_speed: float = 10.0,
        safety_constraints: bool = True,
        use_cem: bool = False,
        cem_iterations: int = 3,
        cem_elite_frac: float = 0.1,
        cost_weights: dict[str, float] | None = None,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.state_norm = state_norm
        self.target_norm = target_norm
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.target_speed = target_speed
        self.safety_constraints = safety_constraints
        self.use_cem = use_cem
        self.cem_iterations = cem_iterations
        self.num_elites = max(4, int(num_samples * cem_elite_frac))

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.rng = np.random.default_rng(seed)

        c_w = cost_weights or {}
        self.cost_evaluator = MPCCost(
            w_lat=c_w.get("w_lat", 3.0),
            w_heading=c_w.get("w_heading", 6.0),
            w_speed=c_w.get("w_speed", 1.0),
            w_steer=c_w.get("w_steer", 0.3),
            w_acc=c_w.get("w_acc", 0.1),
            w_steer_rate=c_w.get("w_steer_rate", 0.5),
            w_acc_rate=c_w.get("w_acc_rate", 0.2),
            target_speed=target_speed,
        )

        self.prev_solution: np.ndarray | None = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_heading: float | None = None

        # Telemetry
        self.planning_times_ms: list[float] = []
        self.fallback_count: int = 0
        self.total_steps: int = 0
        self.total_constraint_violations = 0

    def reset(self) -> None:
        """Reset MPC state between episodes."""
        self.prev_solution = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_heading = None
        self.planning_times_ms.clear()
        self.fallback_count = 0
        self.total_steps = 0
        self.total_constraint_violations = 0

    def _sample_action_sequences(
        self,
        mean_seq: np.ndarray | None = None,
        std_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate N candidate action sequences with temporal smoothing and warm-starting."""
        N = self.num_samples
        H = self.horizon

        if mean_seq is None:
            # Nominal base action sequence: zero steering, moderate acceleration
            mean = np.zeros((H, 2), dtype=np.float32)
            mean[:, 1] = 0.35  # forward throttle bias
        else:
            mean = mean_seq

        # Correlated random walk noise across horizon
        std_steer = 0.18 * std_scale
        std_acc = 0.22 * std_scale

        white_noise = self.rng.normal(0.0, [std_steer, std_acc], size=(N, H, 2)).astype(np.float32)

        # Apply exponential moving average filter for temporal smoothness
        smooth_noise = np.zeros_like(white_noise)
        alpha = 0.65
        curr = np.zeros((N, 2), dtype=np.float32)
        for t in range(H):
            curr = alpha * curr + (1.0 - alpha) * white_noise[:, t, :]
            smooth_noise[:, t, :] = curr

        candidates = mean[None, :, :] + smooth_noise

        # Warm start slot: candidate 0 is exact shifted prior solution
        if self.prev_solution is not None:
            warm_start = np.zeros((H, 2), dtype=np.float32)
            warm_start[:-1] = self.prev_solution[1:]
            warm_start[-1] = self.prev_solution[-1]
            candidates[0] = warm_start

        candidates = np.clip(candidates, -1.0, 1.0)
        return torch.tensor(candidates, dtype=torch.float32, device=self.device)

    def _extract_surrounding_obstacles(self, env: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        """Identify nearby dynamic and static objects for clearance constraints."""
        obstacles: list[tuple[np.ndarray, np.ndarray]] = []
        agent = env.agent
        if not hasattr(env, "engine") or env.engine is None:
            return obstacles

        surrounding = env.engine.get_objects(
            lambda obj: obj is not agent and hasattr(obj, "position") and hasattr(obj, "velocity")
        )
        agent_pos = np.array(agent.position)
        for obj in surrounding.values():
            pos = np.array(obj.position, dtype=np.float32)
            # Only consider objects within 30m
            if np.hypot(pos[0] - agent_pos[0], pos[1] - agent_pos[1]) < 30.0:
                vel = np.array(getattr(obj, "velocity", [0.0, 0.0]), dtype=np.float32)
                obstacles.append((pos, vel))

        return obstacles

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        """Plan trajectory and return first control action [steering, throttle_brake]."""
        start_t = time.perf_counter()
        agent = env.agent

        # Extract current physical dynamic state and world pose
        curr_state_np, curr_heading = extract_vehicle_dynamic_state(
            agent, prev_heading=self.prev_heading, dt=self.dt
        )
        self.prev_heading = curr_heading
        curr_pos_np = np.array(agent.position, dtype=np.float32)
        curr_pose_np = np.array([curr_pos_np[0], curr_pos_np[1], curr_heading], dtype=np.float32)

        # Target lane reference
        target_lane = agent.lane
        if (
            target_lane is None
            and hasattr(agent, "navigation")
            and agent.navigation.current_ref_lanes
        ):
            target_lane = agent.navigation.current_ref_lanes[0]

        if target_lane is None:
            # Fallback if no reference lane is found
            return np.array([0.0, -0.5], dtype=np.float32)

        obstacles = self._extract_surrounding_obstacles(env)

        # Prepare PyTorch tensors
        init_s = torch.tensor(
            curr_state_np[None, :], dtype=torch.float32, device=self.device
        ).expand(self.num_samples, -1)
        init_p = torch.tensor(
            curr_pose_np[None, :], dtype=torch.float32, device=self.device
        ).expand(self.num_samples, -1)

        mean_solution = self.prev_solution
        best_candidate_idx = 0
        is_fallback = False

        num_iters = self.cem_iterations if self.use_cem else 1

        for it in range(num_iters):
            std_scale = 1.0 / (1.0 + it * 0.5)
            candidates = self._sample_action_sequences(mean_solution, std_scale=std_scale)

            # Fast batch rollout through learned neural world model under inference_mode
            with torch.inference_mode():
                states, poses = self.model.rollout(
                    init_s, init_p, candidates, self.state_norm, self.target_norm
                )

            # Evaluate trajectory costs
            costs_t = self.cost_evaluator.evaluate(
                states, poses, candidates, self.prev_action, target_lane
            )
            costs = costs_t.cpu().numpy()

            if self.safety_constraints:
                # Evaluate explicit safety constraints
                is_feasible, violations, v_counts = evaluate_all_constraints(
                    states,
                    poses,
                    candidates,
                    self.prev_action,
                    target_lane,
                    obstacles,
                    dt=self.dt,
                )
                self.total_constraint_violations += sum(v_counts.values())

                feasible_indices = np.where(is_feasible)[0]
                if len(feasible_indices) > 0:
                    best_candidate_idx = int(feasible_indices[np.argmin(costs[feasible_indices])])
                else:
                    # Infeasible: fallback to minimum violation candidate
                    is_fallback = True
                    self.fallback_count += 1
                    best_candidate_idx = int(np.argmin(violations))
            else:
                best_candidate_idx = int(np.argmin(costs))

            if self.use_cem and it < num_iters - 1:
                # Refit mean around top elite candidates
                sorted_idx = np.argsort(costs)
                elites = candidates[sorted_idx[: self.num_elites]].cpu().numpy()
                mean_solution = np.mean(elites, axis=0)

        # Selected best sequence
        best_seq = candidates[best_candidate_idx].cpu().numpy()  # (H, 2)
        chosen_action = best_seq[0].copy()

        if is_fallback:
            # Active safety fallback: steer toward lane centerline and apply braking
            long_p, lat_p = target_lane.local_coordinates(agent.position)
            center_steer = float(np.clip(lat_p * 0.5, -0.6, 0.6))
            chosen_action = np.array([center_steer, -0.5], dtype=np.float32)

        self.prev_solution = best_seq
        self.prev_action = chosen_action
        self.total_steps += 1

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.planning_times_ms.append(elapsed_ms)

        return chosen_action

    def get_telemetry_summary(self) -> dict[str, float]:
        """Return planning time distribution and safety violation metrics."""
        times = self.planning_times_ms or [0.0]
        return {
            "mean_planning_time_ms": float(np.mean(times)),
            "p95_planning_time_ms": float(np.percentile(times, 95)),
            "fallback_rate": float(self.fallback_count / max(1, self.total_steps)),
            "total_steps": self.total_steps,
            "constraint_violations": self.total_constraint_violations,
        }
