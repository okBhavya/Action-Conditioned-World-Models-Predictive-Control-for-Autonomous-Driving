"""Trajectory cost formulation for receding-horizon predictive control."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from driving_wm.data.state import wrap_angle


class MPCCost:
    """Computes tracking, effort, and rate-of-change costs across candidate rollouts."""

    def __init__(
        self,
        w_lat: float = 3.0,
        w_heading: float = 6.0,
        w_speed: float = 1.0,
        w_steer: float = 0.3,
        w_acc: float = 0.1,
        w_steer_rate: float = 0.5,
        w_acc_rate: float = 0.2,
        w_terminal: float = 2.0,
        target_speed: float = 10.0,
    ) -> None:
        self.w_lat = w_lat
        self.w_heading = w_heading
        self.w_speed = w_speed
        self.w_steer = w_steer
        self.w_acc = w_acc
        self.w_steer_rate = w_steer_rate
        self.w_acc_rate = w_acc_rate
        self.w_terminal = w_terminal
        self.target_speed = target_speed

    def evaluate(
        self,
        states: torch.Tensor,  # (B, H+1, 4)
        poses: torch.Tensor,  # (B, H+1, 3): [x, y, yaw]
        actions: torch.Tensor,  # (B, H, 2): [steer, throttle_brake]
        prev_action: np.ndarray | torch.Tensor,  # (2,)
        target_lane: Any,
    ) -> torch.Tensor:
        """Compute scalar cost for each of the B candidate trajectories."""
        B, H, _ = actions.shape
        device = states.device

        # 1. Action effort cost
        steer = actions[..., 0]
        acc = actions[..., 1]
        effort_cost = self.w_steer * torch.sum(steer**2, dim=1) + self.w_acc * torch.sum(
            acc**2, dim=1
        )

        # 2. Action rate of change (smoothness)
        if isinstance(prev_action, np.ndarray):
            prev_act_t = torch.tensor(prev_action, dtype=torch.float32, device=device).expand(
                B, 1, 2
            )
        else:
            prev_act_t = prev_action.to(device).expand(B, 1, 2)

        extended_actions = torch.cat([prev_act_t, actions], dim=1)  # (B, H+1, 2)
        d_actions = extended_actions[:, 1:, :] - extended_actions[:, :-1, :]
        rate_cost = self.w_steer_rate * torch.sum(
            d_actions[..., 0] ** 2, dim=1
        ) + self.w_acc_rate * torch.sum(d_actions[..., 1] ** 2, dim=1)

        # 3. Speed tracking cost
        v_long = states[:, 1:, 0]  # (B, H)
        speed_cost = self.w_speed * torch.sum((v_long - self.target_speed) ** 2, dim=1)

        # 4. Route tracking costs (projected poses along target lane)
        poses_np = poses.detach().cpu().numpy()
        lat_costs = np.zeros(B, dtype=np.float32)
        head_costs = np.zeros(B, dtype=np.float32)

        # Fast projection of horizon endpoints and intermediate waypoints
        # For efficiency, compute on steps [1, 2, ..., H]
        for b in range(B):
            b_lat_sq = 0.0
            b_head_sq = 0.0
            for t in range(1, H + 1):
                px, py, pyaw = poses_np[b, t]
                long_p, lat_p = target_lane.local_coordinates([px, py])
                lane_yaw = float(
                    target_lane.heading_theta_at(max(0.0, min(target_lane.length, long_p)))
                )
                h_err = wrap_angle(pyaw - lane_yaw)

                weight = self.w_terminal if t == H else 1.0
                b_lat_sq += weight * (lat_p**2)
                b_head_sq += weight * (h_err**2)

            lat_costs[b] = b_lat_sq
            head_costs[b] = b_head_sq

        tracking_cost = self.w_lat * torch.tensor(
            lat_costs, device=device
        ) + self.w_heading * torch.tensor(head_costs, device=device)

        total_cost = effort_cost + rate_cost + speed_cost + tracking_cost
        return total_cost
