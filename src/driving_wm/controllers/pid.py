"""Baseline controllers: Stanley path tracker with PID speed control, plus floor policies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from driving_wm.data.state import wrap_angle


class BaseController:
    """Abstract base class for driving controllers."""

    def reset(self) -> None:
        """Reset internal controller state between episodes."""
        pass

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        """Compute [steering, throttle_brake] in [-1.0, 1.0]."""
        raise NotImplementedError


class StanleyPIDController(BaseController):
    """Stanley steering controller + PID speed controller with curvature & gap awareness.

    Steering:
        delta = wrap(heading_err) + arctan(ke * lat_err / (v + eps)) - kd * yaw_rate
    Speed:
        PID tracking of curvature-aware target speed, with leader vehicle gap keeping.
    """

    def __init__(
        self,
        kp_steer: float = 0.85,
        kd_steer: float = 0.15,
        kp_speed: float = 0.6,
        ki_speed: float = 0.05,
        kd_speed: float = 0.02,
        target_speed: float = 10.0,
        min_curve_speed: float = 5.0,
        max_lat_accel: float = 3.0,
        safe_time_gap: float = 1.5,
        min_safe_dist: float = 6.0,
    ) -> None:
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.target_speed = target_speed
        self.min_curve_speed = min_curve_speed
        self.max_lat_accel = max_lat_accel
        self.safe_time_gap = safe_time_gap
        self.min_safe_dist = min_safe_dist

        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0

    def reset(self) -> None:
        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        agent = env.agent
        v_speed = float(agent.speed)  # m/s
        heading = float(agent.heading_theta)
        pos = agent.position

        # 1. Path tracking reference via agent's lane or navigation ref lanes
        lane = agent.lane
        if lane is None and hasattr(agent, "navigation") and agent.navigation.current_ref_lanes:
            lane = agent.navigation.current_ref_lanes[0]

        if lane is None:
            # Fallback if outside mapped lanes
            return np.array([0.0, 0.0], dtype=np.float32)

        long_pos, lat_offset = lane.local_coordinates(pos)
        # Note on MetaDrive sign convention:
        # positive lat_offset means vehicle is right of centerline; negative means left.
        # To steer toward center when to the right (lat > 0), we steer left (positive steering).
        # When to the left (lat < 0), we steer right (negative steering).
        # Lane tangent heading:
        lane_heading = float(lane.heading_theta_at(max(0.0, min(lane.length, long_pos))))
        heading_error = wrap_angle(lane_heading - heading)

        # Stanley steering law
        cross_track_term = math.atan2(self.kp_steer * lat_offset, max(v_speed, 1.0))
        steer_cmd = heading_error + cross_track_term

        # Damping using yaw rate if available
        yaw_rate = float(
            agent.velocity[0] * math.sin(heading) - agent.velocity[1] * math.cos(heading)
        )
        steer_cmd -= self.kd_steer * yaw_rate
        steer = float(np.clip(steer_cmd, -1.0, 1.0))

        # 2. Curvature-aware target speed
        curv_lookahead = min(lane.length, long_pos + max(5.0, v_speed * 1.0))
        ahead_heading = float(lane.heading_theta_at(curv_lookahead))
        curvature = abs(wrap_angle(ahead_heading - lane_heading)) / max(
            1.0, curv_lookahead - long_pos
        )
        if curvature > 0.01:
            curve_speed = math.sqrt(self.max_lat_accel / curvature)
            v_ref = max(self.min_curve_speed, min(self.target_speed, curve_speed))
        else:
            v_ref = self.target_speed

        # 3. Traffic gap-keeping and emergency braking
        if hasattr(agent, "lidar") and agent.lidar is not None:
            # Use front lidar cloud / side detector to identify leader distance
            surrounding = env.engine.get_objects(
                lambda obj: obj is not agent and hasattr(obj, "position")
            )
            min_front_dist = float("inf")
            for other_id, other_obj in surrounding.items():
                d_pos = np.array(other_obj.position) - np.array(pos)
                dx_body = d_pos[0] * math.cos(heading) + d_pos[1] * math.sin(heading)
                dy_body = -d_pos[0] * math.sin(heading) + d_pos[1] * math.cos(heading)
                if dx_body > 0 and abs(dy_body) < 2.0:  # In same lane corridor ahead
                    if dx_body < min_front_dist:
                        min_front_dist = dx_body

            safe_dist = self.min_safe_dist + v_speed * self.safe_time_gap
            if min_front_dist < safe_dist:
                # Proportional speed reduction or hard brake
                gap_factor = max(
                    0.0,
                    (min_front_dist - self.min_safe_dist)
                    / max(1.0, safe_dist - self.min_safe_dist),
                )
                v_ref = min(v_ref, v_ref * gap_factor)
                if min_front_dist < self.min_safe_dist * 0.8:
                    return np.array([steer, -1.0], dtype=np.float32)

        # 4. PID longitudinal control
        speed_err = v_ref - v_speed
        self.speed_error_integral = np.clip(
            self.speed_error_integral + speed_err * 0.1, -10.0, 10.0
        )
        speed_deriv = (speed_err - self.prev_speed_error) / 0.1
        self.prev_speed_error = speed_err

        acc_cmd = (
            self.kp_speed * speed_err
            + self.ki_speed * self.speed_error_integral
            + self.kd_speed * speed_deriv
        )
        throttle_brake = float(np.clip(acc_cmd, -1.0, 1.0))

        return np.array([steer, throttle_brake], dtype=np.float32)


class ZeroController(BaseController):
    """Floor policy 1: Zero inputs (coasts to a stop)."""

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class ConstantThrottleController(BaseController):
    """Floor policy 2: Constant throttle, zero steering."""

    def __init__(self, throttle: float = 0.5) -> None:
        self.throttle = throttle

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        return np.array([0.0, self.throttle], dtype=np.float32)


class RandomController(BaseController):
    """Floor policy 3: Uniformly random actions."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def compute_action(self, env: Any, info: dict[str, Any] | None = None) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
