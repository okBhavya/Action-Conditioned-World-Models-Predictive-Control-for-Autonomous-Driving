"""Unit tests for state representation, coordinate transformations, and angle wrapping."""

import math

import numpy as np
import torch

from driving_wm.data.state import (
    ACTION_DIM,
    STATE_DIM,
    TARGET_DIM,
    ego_to_world,
    world_to_ego,
    wrap_angle,
)


def test_wrap_angle_scalars() -> None:
    assert abs(wrap_angle(0.0)) < 1e-6
    assert abs(wrap_angle(math.pi) - math.pi) < 1e-6 or abs(wrap_angle(math.pi) + math.pi) < 1e-6
    assert (
        abs(wrap_angle(3.0 * math.pi) - math.pi) < 1e-6
        or abs(wrap_angle(3.0 * math.pi) + math.pi) < 1e-6
    )
    assert (
        abs(wrap_angle(-3.0 * math.pi) - math.pi) < 1e-6
        or abs(wrap_angle(-3.0 * math.pi) + math.pi) < 1e-6
    )
    assert abs(wrap_angle(2.5 * math.pi) - 0.5 * math.pi) < 1e-6


def test_wrap_angle_arrays_and_tensors() -> None:
    arr = np.array([-4.0 * np.pi, -np.pi, 0.0, np.pi, 3.5 * np.pi])
    wrapped = wrap_angle(arr)
    assert np.all(wrapped >= -np.pi - 1e-6)
    assert np.all(wrapped <= np.pi + 1e-6)

    t = torch.tensor([-4.0 * math.pi, -math.pi, 0.0, math.pi, 3.5 * math.pi])
    wrapped_t = wrap_angle(t)
    assert torch.all(wrapped_t >= -math.pi - 1e-6)
    assert torch.all(wrapped_t <= math.pi + 1e-6)


def test_ego_world_coordinate_roundtrip() -> None:
    headings = [-math.pi / 3, 0.0, math.pi / 4, math.pi / 2, 2.3]
    for th in headings:
        dx_w, dy_w = 4.25, -2.15
        dx_e, dy_e = world_to_ego(dx_w, dy_w, th)
        dx_w_rec, dy_w_rec = ego_to_world(dx_e, dy_e, th)
        assert abs(dx_w - dx_w_rec) < 1e-6
        assert abs(dy_w - dy_w_rec) < 1e-6


def test_ego_to_world_torch_batch() -> None:
    B = 16
    dx_e = torch.randn(B)
    dy_e = torch.randn(B)
    yaw = torch.rand(B) * 2 * math.pi - math.pi

    dx_w, dy_w = ego_to_world(dx_e, dy_e, yaw)
    assert dx_w.shape == (B,)
    assert dy_w.shape == (B,)
    assert torch.all(torch.isfinite(dx_w))
    assert torch.all(torch.isfinite(dy_w))


def test_dimension_constants() -> None:
    assert STATE_DIM == 4
    assert ACTION_DIM == 2
    assert TARGET_DIM == 7
