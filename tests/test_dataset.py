"""Unit tests for dataset loading, normalization, and validation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from driving_wm.data.dataset import (
    DrivingDataset,
    Normalizer,
    validate_dataset,
)
from driving_wm.data.state import ACTION_DIM, STATE_DIM, TARGET_DIM


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    """Generate a clean synthetic dataset for testing without simulator dependencies."""
    n_per_ep = 20
    n_eps = 3
    total_n = n_per_ep * n_eps

    states = np.zeros((total_n, STATE_DIM), dtype=np.float32)
    actions = np.zeros((total_n, ACTION_DIM), dtype=np.float32)
    next_states = np.zeros((total_n, STATE_DIM), dtype=np.float32)
    pose_increments = np.zeros((total_n, 3), dtype=np.float32)
    targets = np.zeros((total_n, TARGET_DIM), dtype=np.float32)
    episode_ids = np.zeros(total_n, dtype=np.int32)
    steps = np.zeros(total_n, dtype=np.int32)

    for ep in range(n_eps):
        start = ep * n_per_ep
        end = start + n_per_ep
        episode_ids[start:end] = ep
        steps[start:end] = np.arange(n_per_ep)

        # Synthetic continuous trajectory
        s = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for t in range(n_per_ep):
            idx = start + t
            a = np.array([0.1 * math_sin(t), 0.2], dtype=np.float32)
            d_s = np.array([0.05, 0.0, 0.01, 0.0], dtype=np.float32)
            s_next = s + d_s
            p_inc = np.array([0.5, 0.0, 0.01], dtype=np.float32)

            states[idx] = s
            actions[idx] = a
            next_states[idx] = s_next
            pose_increments[idx] = p_inc
            targets[idx] = np.concatenate([d_s, p_inc])

            s = s_next

    npz_path = tmp_path / "synthetic.npz"
    np.savez_compressed(
        npz_path,
        states=states,
        actions=actions,
        next_states=next_states,
        pose_increments=pose_increments,
        targets=targets,
        episode_ids=episode_ids,
        steps=steps,
        seeds=np.full(total_n, 42, dtype=np.int32),
        terminated=np.zeros(total_n, dtype=bool),
        truncated=np.zeros(total_n, dtype=bool),
    )
    return npz_path


def math_sin(x: float) -> float:
    import math

    return math.sin(x)


def test_normalizer_floor_tiny_std() -> None:
    mean = np.array([10.0, 0.0], dtype=np.float32)
    std = np.array([2.0, 0.0], dtype=np.float32)  # Zero variance dimension

    norm = Normalizer(mean, std, eps=1e-3)
    x = torch.tensor([[12.0, 0.0], [8.0, 0.0]], dtype=torch.float32)

    x_norm = norm.normalize(x)
    assert torch.all(torch.isfinite(x_norm))
    assert abs(x_norm[0, 0].item() - 1.0) < 1e-4

    # Round trip
    x_rec = norm.denormalize(x_norm)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_validate_dataset_pass(synthetic_npz: Path) -> None:
    res = validate_dataset(synthetic_npz)
    assert res["alignment_verified"] is True
    assert res["finite_verified"] is True
    assert res["action_bounds_verified"] is True
    assert res["num_transitions"] == 60
    assert res["num_episodes"] == 3


def test_driving_dataset_no_boundary_crossing(synthetic_npz: Path) -> None:
    unroll = 5
    ds = DrivingDataset(synthetic_npz, unroll_steps=unroll)

    # Each episode has 20 transitions. With unroll=5, valid indices per episode are 20 - 5 + 1 = 16.
    # Total valid: 3 * 16 = 48
    assert len(ds) == 48

    for i in range(len(ds)):
        item = ds[i]
        assert item["state"].shape == (unroll, STATE_DIM)
        assert item["action"].shape == (unroll, ACTION_DIM)
        assert item["target"].shape == (unroll, TARGET_DIM)
