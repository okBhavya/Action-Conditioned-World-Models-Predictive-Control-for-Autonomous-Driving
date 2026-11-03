"""Dataset and normalization utilities for world model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from driving_wm.data.state import ACTION_DIM, STATE_DIM, TARGET_DIM


class Normalizer:
    """Mean and standard deviation normalizer with tiny-variance flooring."""

    def __init__(
        self,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        eps: float = 1e-4,
    ) -> None:
        if isinstance(mean, torch.Tensor):
            self.mean = mean.float()
            self.std = torch.clamp(std.float(), min=eps)
        else:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(np.maximum(std, eps), dtype=torch.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std.to(x_norm.device) + self.mean.to(x_norm.device)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Normalizer:
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )


class DrivingDataset(Dataset):
    """PyTorch Dataset loading single-step or multi-step driving transitions."""

    def __init__(
        self,
        npz_path: str | Path,
        unroll_steps: int = 1,
        state_normalizer: Normalizer | None = None,
        target_normalizer: Normalizer | None = None,
    ) -> None:
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = np.load(path)
        self.states = torch.tensor(data["states"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)
        self.next_states = torch.tensor(data["next_states"], dtype=torch.float32)
        self.pose_increments = torch.tensor(data["pose_increments"], dtype=torch.float32)
        self.targets = torch.tensor(data["targets"], dtype=torch.float32)
        self.episode_ids = data["episode_ids"]
        self.steps = data["steps"]

        self.unroll_steps = unroll_steps
        self.state_normalizer = state_normalizer
        self.target_normalizer = target_normalizer

        # Build valid indices for multi-step sequences ensuring no episode boundary crossings
        self.valid_indices: list[int] = []
        n = len(self.states)
        for i in range(n - unroll_steps + 1):
            if self.episode_ids[i] == self.episode_ids[i + unroll_steps - 1]:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start_i = self.valid_indices[idx]
        end_i = start_i + self.unroll_steps

        # Sequence of states, actions, targets
        states = self.states[start_i:end_i]
        actions = self.actions[start_i:end_i]
        next_states = self.next_states[start_i:end_i]
        targets = self.targets[start_i:end_i]

        norm_states = (
            self.state_normalizer.normalize(states) if self.state_normalizer is not None else states
        )
        norm_targets = (
            self.target_normalizer.normalize(targets)
            if self.target_normalizer is not None
            else targets
        )

        return {
            "state": states[0] if self.unroll_steps == 1 else states,
            "action": actions[0] if self.unroll_steps == 1 else actions,
            "next_state": next_states[0] if self.unroll_steps == 1 else next_states,
            "target": targets[0] if self.unroll_steps == 1 else targets,
            "norm_state": norm_states[0] if self.unroll_steps == 1 else norm_states,
            "norm_target": norm_targets[0] if self.unroll_steps == 1 else norm_targets,
        }


def compute_normalization_stats(npz_path: str | Path) -> tuple[Normalizer, Normalizer]:
    """Compute mean and std statistics for states and targets on the training dataset."""
    data = np.load(npz_path)
    states = data["states"]
    targets = data["targets"]

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)

    target_mean = np.mean(targets, axis=0)
    target_std = np.std(targets, axis=0)

    state_norm = Normalizer(state_mean, state_std)
    target_norm = Normalizer(target_mean, target_std)

    return state_norm, target_norm


def validate_dataset(npz_path: str | Path) -> dict[str, Any]:
    """Run full consistency audit on dataset: shapes, alignment, finite checks, bounds."""
    data = np.load(npz_path)
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    targets = data["targets"]
    episode_ids = data["episode_ids"]
    steps = data["steps"]

    n = len(states)
    assert len(actions) == n
    assert len(next_states) == n
    assert len(targets) == n
    assert states.shape[1] == STATE_DIM
    assert actions.shape[1] == ACTION_DIM
    assert targets.shape[1] == TARGET_DIM

    # Finite values
    assert np.all(np.isfinite(states)), "NaN or Inf detected in states"
    assert np.all(np.isfinite(actions)), "NaN or Inf detected in actions"
    assert np.all(np.isfinite(next_states)), "NaN or Inf detected in next_states"
    assert np.all(np.isfinite(targets)), "NaN or Inf detected in targets"

    # Action bounds in [-1.0, 1.0]
    assert np.all(actions >= -1.0 - 1e-4) and np.all(actions <= 1.0 + 1e-4), (
        "Action bounds violated"
    )

    # Alignment check within episodes: state[t+1] == next_state[t]
    alignment_violations = 0
    for i in range(n - 1):
        if episode_ids[i] == episode_ids[i + 1]:
            # Step must increment by 1
            assert steps[i + 1] == steps[i] + 1, f"Step discontinuity at transition {i}"
            diff = np.max(np.abs(next_states[i] - states[i + 1]))
            if diff > 1e-4:
                alignment_violations += 1

    assert alignment_violations == 0, f"Found {alignment_violations} alignment violations"

    return {
        "num_transitions": int(n),
        "num_episodes": int(len(np.unique(episode_ids))),
        "alignment_verified": True,
        "finite_verified": True,
        "action_bounds_verified": True,
    }
