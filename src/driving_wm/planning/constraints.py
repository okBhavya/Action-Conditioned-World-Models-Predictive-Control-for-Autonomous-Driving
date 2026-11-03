"""Explicit safety constraints for predictive trajectory filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def check_road_bounds(
    poses_np: np.ndarray,  # (B, H+1, 3)
    target_lane: Any,
    road_half_width: float = 5.25,
    safety_margin: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Constraint 1: Keep lateral offset within physical road bounds plus margin."""
    B, H_plus_1, _ = poses_np.shape
    max_allowed = max(0.5, road_half_width - safety_margin)

    feasible = np.ones(B, dtype=bool)
    violations = np.zeros(B, dtype=np.float32)

    for b in range(B):
        max_lat = 0.0
        for t in range(1, H_plus_1):
            px, py, _ = poses_np[b, t]
            _, lat_p = target_lane.local_coordinates([px, py])
            abs_lat = abs(lat_p)
            if abs_lat > max_lat:
                max_lat = abs_lat

        if max_lat > max_allowed:
            feasible[b] = False
            violations[b] = max_lat - max_allowed

    return feasible, violations


def check_speed_limit(
    states: torch.Tensor,  # (B, H+1, 4)
    speed_limit: float = 14.0,  # m/s (~50 km/h)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constraint 2: Longitudinal speed below maximum speed limit."""
    v_long = states[:, 1:, 0]  # (B, H)
    max_v, _ = torch.max(v_long, dim=1)
    viol = torch.clamp(max_v - speed_limit, min=0.0)
    feasible = viol <= 1e-4
    return feasible, viol


def check_obstacle_clearance(
    poses_np: np.ndarray,  # (B, H+1, 3)
    obstacles: list[tuple[np.ndarray, np.ndarray]],  # list of (pos_0, vel)
    min_clearance: float = 3.5,  # meters
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Constraint 3: Clearance to nearby obstacles using constant-velocity extrapolation."""
    B, H_plus_1, _ = poses_np.shape
    feasible = np.ones(B, dtype=bool)
    violations = np.zeros(B, dtype=np.float32)

    if not obstacles:
        return feasible, violations

    for b in range(B):
        min_dist = float("inf")
        for t in range(1, H_plus_1):
            ego_pos = poses_np[b, t, :2]
            for obs_pos0, obs_vel in obstacles:
                # Constant velocity extrapolation
                pred_obs_pos = obs_pos0 + obs_vel * (t * dt)
                dist = float(np.hypot(ego_pos[0] - pred_obs_pos[0], ego_pos[1] - pred_obs_pos[1]))
                if dist < min_dist:
                    min_dist = dist

        if min_dist < min_clearance:
            feasible[b] = False
            violations[b] = min_clearance - min_dist

    return feasible, violations


def check_steering_rate(
    actions: torch.Tensor,  # (B, H, 2)
    prev_action: np.ndarray | torch.Tensor,  # (2,)
    max_steer_rate: float = 6.0,  # normalized units per second
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constraint 4: Bounded steering rate of change."""
    B, H, _ = actions.shape
    device = actions.device

    if isinstance(prev_action, np.ndarray):
        prev_act_t = torch.tensor(prev_action, dtype=torch.float32, device=device).expand(B, 1, 2)
    else:
        prev_act_t = prev_action.to(device).expand(B, 1, 2)

    ext = torch.cat([prev_act_t, actions], dim=1)
    d_steer = torch.abs(ext[:, 1:, 0] - ext[:, :-1, 0]) / max(dt, 1e-4)
    max_rate, _ = torch.max(d_steer, dim=1)

    viol = torch.clamp(max_rate - max_steer_rate, min=0.0)
    feasible = viol <= 1e-4
    return feasible, viol


def evaluate_all_constraints(
    states: torch.Tensor,
    poses: torch.Tensor,
    actions: torch.Tensor,
    prev_action: np.ndarray | torch.Tensor,
    target_lane: Any,
    obstacles: list[tuple[np.ndarray, np.ndarray]],
    road_half_width: float = 5.25,
    speed_limit: float = 14.0,
    min_clearance: float = 3.5,
    max_steer_rate: float = 6.0,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Evaluate all four safety constraints across B candidates."""
    poses_np = poses.detach().cpu().numpy()

    c1_feas, c1_viol = check_road_bounds(poses_np, target_lane, road_half_width)
    c2_feas_t, c2_viol_t = check_speed_limit(states, speed_limit)
    c3_feas, c3_viol = check_obstacle_clearance(poses_np, obstacles, min_clearance, dt)
    c4_feas_t, c4_viol_t = check_steering_rate(actions, prev_action, max_steer_rate, dt)

    c2_feas = c2_feas_t.cpu().numpy()
    c2_viol = c2_viol_t.cpu().numpy()
    c4_feas = c4_feas_t.cpu().numpy()
    c4_viol = c4_viol_t.cpu().numpy()

    all_feasible = c1_feas & c2_feas & c3_feas & c4_feas
    total_violation = c1_viol * 10.0 + c2_viol * 2.0 + c3_viol * 10.0 + c4_viol * 1.0

    violation_counts = {
        "road_bound_violations": int(np.sum(~c1_feas)),
        "speed_limit_violations": int(np.sum(~c2_feas)),
        "obstacle_clearance_violations": int(np.sum(~c3_feas)),
        "steering_rate_violations": int(np.sum(~c4_feas)),
    }

    return all_feasible, total_violation, violation_counts
