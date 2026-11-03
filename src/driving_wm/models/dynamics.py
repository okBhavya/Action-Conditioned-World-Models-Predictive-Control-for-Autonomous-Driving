"""Action-conditioned neural world model and kinematic baselines."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from driving_wm.data.dataset import Normalizer
from driving_wm.data.state import (
    ACTION_DIM,
    STATE_DIM,
    TARGET_DIM,
    ego_to_world,
    wrap_angle,
)


class WorldModel(nn.Module):
    """Multi-layer perceptron world model predicting normalized state deltas and pose increments."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        target_dim: int = TARGET_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_dim = target_dim

        act_cls = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, target_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, norm_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict normalized target [dv_long, dv_lat, dyaw_rate, dsteering, dx_ego, dy_ego, dyaw]."""
        x = torch.cat([norm_state, action], dim=-1)
        return self.net(x)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_norm: Normalizer,
        target_norm: Normalizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one forward prediction step in physical units.

        Returns:
            next_state: [v_long, v_lat, yaw_rate, steering] clamped to valid physics ranges.
            pose_inc: [dx_ego, dy_ego, dyaw] ego-frame displacement.
        """
        norm_s = state_norm.normalize(state)
        norm_target_pred = self.forward(norm_s, action)
        target_pred = target_norm.denormalize(norm_target_pred)

        d_state = target_pred[..., : self.state_dim]
        pose_inc = target_pred[..., self.state_dim :]

        next_state = state + d_state

        # Physically bounded outputs: non-negative longitudinal speed, bounded steering
        v_long = torch.clamp(next_state[..., 0:1], min=0.0)
        v_lat = next_state[..., 1:2]
        yaw_rate = next_state[..., 2:3]
        steering = torch.clamp(next_state[..., 3:4], min=-1.0, max=1.0)
        clamped_state = torch.cat([v_long, v_lat, yaw_rate, steering], dim=-1)

        return clamped_state, pose_inc

    def rollout(
        self,
        init_state: torch.Tensor,
        init_pose: torch.Tensor,
        actions: torch.Tensor,
        state_norm: Normalizer,
        target_norm: Normalizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch rollout across horizon H for MPC planning.

        Args:
            init_state: (B, 4) initial states [v_long, v_lat, yaw_rate, steering]
            init_pose: (B, 3) initial poses [x, y, heading_theta] in world frame
            actions: (B, H, 2) planned control actions
            state_norm: normalizer for states
            target_norm: normalizer for targets

        Returns:
            states: (B, H+1, 4) state trajectory
            poses: (B, H+1, 3) world-frame pose trajectory
        """
        B, H, _ = actions.shape
        states = [init_state]
        poses = [init_pose]

        curr_state = init_state
        curr_x = init_pose[:, 0]
        curr_y = init_pose[:, 1]
        curr_yaw = init_pose[:, 2]

        for t in range(H):
            action_t = actions[:, t, :]
            next_state, pose_inc = self.step(curr_state, action_t, state_norm, target_norm)

            dx_ego = pose_inc[:, 0]
            dy_ego = pose_inc[:, 1]
            dyaw = pose_inc[:, 2]

            # Rotate ego delta into world frame
            dx_world, dy_world = ego_to_world(dx_ego, dy_ego, curr_yaw)
            curr_x = curr_x + dx_world
            curr_y = curr_y + dy_world
            curr_yaw = wrap_angle(curr_yaw + dyaw)

            curr_state = next_state
            states.append(curr_state)
            poses.append(torch.stack([curr_x, curr_y, curr_yaw], dim=-1))

        return torch.stack(states, dim=1), torch.stack(poses, dim=1)


class KinematicBicycleModel:
    """Kinematic bicycle model baseline for physical comparative benchmarking."""

    def __init__(self, wheelbase: float = 2.8, dt: float = 0.1, max_steer_rad: float = 0.5) -> None:
        self.wheelbase = wheelbase
        self.dt = dt
        self.max_steer_rad = max_steer_rad
        self.lr = wheelbase / 2.0

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One kinematic integration step.

        State: [v_long, v_lat, yaw_rate, steering]
        Action: [steer_cmd, throttle_cmd]
        """
        v_long = max(0.0, float(state[0]))
        steer = float(np.clip(action[0], -1.0, 1.0))
        steer_rad = steer * self.max_steer_rad
        accel = float(action[1]) * 4.0  # max ~4 m/s^2

        beta = math.atan2(self.lr * math.tan(steer_rad), self.wheelbase)
        yaw_rate = (v_long / self.wheelbase) * math.sin(steer_rad) * math.cos(beta)

        v_next = max(0.0, v_long + accel * self.dt)
        dx_ego = v_long * math.cos(beta) * self.dt
        dy_ego = v_long * math.sin(beta) * self.dt
        dyaw = yaw_rate * self.dt

        next_state = np.array([v_next, 0.0, yaw_rate, steer], dtype=np.float32)
        pose_inc = np.array([dx_ego, dy_ego, dyaw], dtype=np.float32)
        return next_state, pose_inc


class PersistenceBaseline:
    """Persistence baseline: assumes state and pose increments continue unchanged."""

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt

    def step(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_long = max(0.0, float(state[0]))
        dx_ego = v_long * self.dt
        dy_ego = 0.0
        dyaw = 0.0
        next_state = state.copy()
        pose_inc = np.array([dx_ego, dy_ego, dyaw], dtype=np.float32)
        return next_state, pose_inc
