"""State representation, coordinate transformations, and angle wrapping utilities.

Coordinates and units:
- Positions: meters (m), world frame [x, y]
- Velocities: meters per second (m/s), ego frame [v_long, v_lat]
- Heading: radians (rad), counter-clockwise from +x axis, wrapped to [-pi, pi]
- Yaw rate: radians per second (rad/s)
- Steering: normalized to [-1.0, 1.0]
- Actions: [steering, throttle_brake], each bounded in [-1.0, 1.0]
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

STATE_DIM = 4  # [v_long, v_lat, yaw_rate, steering]
ACTION_DIM = 2  # [steering, throttle_brake]
TARGET_DIM = 7  # [dv_long, dv_lat, dyaw_rate, dsteering, dx_ego, dy_ego, dyaw]

STATE_LABELS = ["v_long", "v_lat", "yaw_rate", "steering"]
TARGET_LABELS = ["dv_long", "dv_lat", "dyaw_rate", "dsteering", "dx_ego", "dy_ego", "dyaw"]


def wrap_angle(theta: float | np.ndarray | torch.Tensor) -> float | np.ndarray | torch.Tensor:
    """Wrap angle or angle difference to [-pi, pi]."""
    if isinstance(theta, torch.Tensor):
        return (theta + math.pi) % (2.0 * math.pi) - math.pi
    elif isinstance(theta, np.ndarray):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
    else:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi


def world_to_ego(
    dx_world: float | np.ndarray,
    dy_world: float | np.ndarray,
    heading_theta: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Transform world displacement vector to ego body frame at given heading theta.

    Ego frame:
    - +x: forward along vehicle heading
    - +y: lateral left
    """
    cos_th = np.cos(heading_theta)
    sin_th = np.sin(heading_theta)
    dx_ego = dx_world * cos_th + dy_world * sin_th
    dy_ego = -dx_world * sin_th + dy_world * cos_th
    return dx_ego, dy_ego


def ego_to_world(
    dx_ego: float | np.ndarray | torch.Tensor,
    dy_ego: float | np.ndarray | torch.Tensor,
    heading_theta: float | np.ndarray | torch.Tensor,
) -> tuple[float | np.ndarray | torch.Tensor, float | np.ndarray | torch.Tensor]:
    """Transform ego frame displacement vector back to world coordinates."""
    if isinstance(dx_ego, torch.Tensor):
        cos_th = torch.cos(heading_theta)
        sin_th = torch.sin(heading_theta)
        dx_world = dx_ego * cos_th - dy_ego * sin_th
        dy_world = dx_ego * sin_th + dy_ego * cos_th
    else:
        cos_th = np.cos(heading_theta)
        sin_th = np.sin(heading_theta)
        dx_world = dx_ego * cos_th - dy_ego * sin_th
        dy_world = dx_ego * sin_th + dy_ego * cos_th
    return dx_world, dy_world


def extract_vehicle_dynamic_state(
    vehicle: Any,
    prev_heading: float | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Extract ego body-frame dynamic state [v_long, v_lat, yaw_rate, steering] and current heading.

    Yaw rate is derived via finite difference of wrapped heading over dt.
    """
    heading = float(vehicle.heading_theta)
    vx, vy = float(vehicle.velocity[0]), float(vehicle.velocity[1])

    cos_th = math.cos(heading)
    sin_th = math.sin(heading)
    v_long = vx * cos_th + vy * sin_th
    v_lat = -vx * sin_th + vy * cos_th

    if prev_heading is not None:
        dyaw = wrap_angle(heading - prev_heading)
        yaw_rate = dyaw / max(dt, 1e-4)
    else:
        yaw_rate = 0.0

    steering = float(vehicle.steering)

    state = np.array([v_long, v_lat, yaw_rate, steering], dtype=np.float32)
    return state, heading
