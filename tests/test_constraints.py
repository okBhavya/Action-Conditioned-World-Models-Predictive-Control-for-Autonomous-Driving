"""Unit tests for explicit safety constraints on analytical boundary cases."""

import numpy as np
import torch

from driving_wm.planning.constraints import (
    check_obstacle_clearance,
    check_road_bounds,
    check_speed_limit,
    check_steering_rate,
)


class MockLane:
    """Mock lane with simple straight local coordinate projection for testing."""

    def local_coordinates(self, pos: list[float] | np.ndarray) -> tuple[float, float]:
        # X is longitudinal, Y is lateral
        return float(pos[0]), float(pos[1])


def test_check_speed_limit_boundary() -> None:
    limit = 12.0
    # Candidate 0: strictly below limit (max 10.0) -> feasible
    # Candidate 1: at limit (max 12.0) -> feasible
    # Candidate 2: exceeds limit (max 14.5) -> infeasible
    states = torch.zeros(3, 6, 4)
    states[0, :, 0] = torch.tensor([0.0, 5.0, 8.0, 10.0, 9.0, 7.0])
    states[1, :, 0] = torch.tensor([0.0, 5.0, 10.0, 12.0, 11.0, 8.0])
    states[2, :, 0] = torch.tensor([0.0, 5.0, 10.0, 14.5, 12.0, 9.0])

    feas, viol = check_speed_limit(states, speed_limit=limit)
    assert feas[0].item() is True
    assert feas[1].item() is True
    assert feas[2].item() is False
    assert abs(viol[2].item() - 2.5) < 1e-4


def test_check_road_bounds_boundary() -> None:
    lane = MockLane()
    half_width = 5.0
    margin = 0.5
    # max allowed lat is 4.5m

    poses = np.zeros((3, 5, 3), dtype=np.float32)
    # Candidate 0: within bounds (lat=2.0) -> feasible
    poses[0, :, 1] = 2.0
    # Candidate 1: exact margin (lat=4.5) -> feasible
    poses[1, :, 1] = 4.5
    # Candidate 2: outside bounds (lat=5.2) -> infeasible
    poses[2, :, 1] = 5.2

    feas, viol = check_road_bounds(poses, lane, road_half_width=half_width, safety_margin=margin)
    assert feas[0] is np.True_ or feas[0] == 1
    assert feas[1] is np.True_ or feas[1] == 1
    assert feas[2] is np.False_ or feas[2] == 0
    assert abs(viol[2] - 0.7) < 1e-4


def test_check_obstacle_clearance_extrapolation() -> None:
    # Ego vehicle moving at 10 m/s along x from x=0
    poses = np.zeros((2, 5, 3), dtype=np.float32)
    dt = 0.1
    for t in range(5):
        poses[0, t, 0] = t * 10.0 * dt  # candidate 0 x: 0, 1, 2, 3, 4 m
        poses[1, t, 0] = t * 2.0 * dt  # candidate 1 x: 0, 0.2, 0.4, 0.6, 0.8 m

    # Static obstacle at x=3.0, y=0.0
    obstacles = [(np.array([3.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))]
    min_clearance = 2.0

    # Candidate 0 passes within 0.0m of obstacle at t=3 -> infeasible
    # Candidate 1 max x is 0.8m, distance to obstacle is 3.0 - 0.8 = 2.2m > 2.0m -> feasible
    feas, viol = check_obstacle_clearance(poses, obstacles, min_clearance=min_clearance, dt=dt)
    assert feas[0] is np.False_ or feas[0] == 0
    assert feas[1] is np.True_ or feas[1] == 1


def test_check_steering_rate_boundary() -> None:
    prev_a = np.array([0.0, 0.5], dtype=np.float32)
    dt = 0.1
    max_rate = 4.0  # max change per step is 4.0 * 0.1 = 0.4

    # Candidate 0: gentle steering changes (0.1 per step) -> feasible
    actions = torch.zeros(2, 4, 2)
    actions[0, :, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    # Candidate 1: sudden jerk from 0.0 to 0.7 at step 0 -> rate = 7.0 > 4.0 -> infeasible
    actions[1, :, 0] = torch.tensor([0.7, 0.7, 0.7, 0.7])

    feas, viol = check_steering_rate(actions, prev_a, max_steer_rate=max_rate, dt=dt)
    assert feas[0].item() is True
    assert feas[1].item() is False
    assert abs(viol[1].item() - 3.0) < 1e-4
