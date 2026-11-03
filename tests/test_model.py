"""Unit tests for world model architecture, forward shapes, and checkpointing."""

from pathlib import Path

import torch

from driving_wm.data.dataset import Normalizer
from driving_wm.data.state import ACTION_DIM, STATE_DIM, TARGET_DIM
from driving_wm.models.dynamics import WorldModel
from driving_wm.models.evaluate import load_model_from_checkpoint


def test_world_model_forward_shape() -> None:
    B = 32
    model = WorldModel(hidden_dim=64, num_layers=3)
    norm_s = torch.randn(B, STATE_DIM)
    a = torch.randn(B, ACTION_DIM)

    pred = model(norm_s, a)
    assert pred.shape == (B, TARGET_DIM)
    assert torch.all(torch.isfinite(pred))


def test_world_model_step_physical_clamps() -> None:
    model = WorldModel(hidden_dim=64, num_layers=3)
    s_norm = Normalizer(torch.zeros(STATE_DIM), torch.ones(STATE_DIM))
    t_norm = Normalizer(torch.zeros(TARGET_DIM), torch.ones(TARGET_DIM))

    state = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # Huge negative acceleration
    action = torch.tensor([[0.0, -1.0]], dtype=torch.float32)

    next_state, pose_inc = model.step(state, action, s_norm, t_norm)
    assert next_state.shape == (1, STATE_DIM)
    assert pose_inc.shape == (1, 3)
    # Non-negative longitudinal speed constraint
    assert next_state[0, 0].item() >= 0.0
    # Steering in [-1, 1]
    assert abs(next_state[0, 3].item()) <= 1.0


def test_world_model_rollout_shape_and_consistency() -> None:
    B, H = 16, 10
    model = WorldModel(hidden_dim=64, num_layers=3)
    s_norm = Normalizer(torch.zeros(STATE_DIM), torch.ones(STATE_DIM))
    t_norm = Normalizer(torch.zeros(TARGET_DIM), torch.ones(TARGET_DIM))

    init_s = torch.zeros(B, STATE_DIM)
    init_p = torch.zeros(B, 3)
    actions = torch.zeros(B, H, ACTION_DIM)

    states, poses = model.rollout(init_s, init_p, actions, s_norm, t_norm)
    assert states.shape == (B, H + 1, STATE_DIM)
    assert poses.shape == (B, H + 1, 3)
    assert torch.all(torch.isfinite(states))
    assert torch.all(torch.isfinite(poses))


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    model = WorldModel(hidden_dim=64, num_layers=3)
    s_norm = Normalizer(torch.ones(STATE_DIM), torch.ones(STATE_DIM) * 2.0)
    t_norm = Normalizer(torch.zeros(TARGET_DIM), torch.ones(TARGET_DIM))

    ckpt_file = tmp_path / "test_ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_norm": s_norm.to_dict(),
            "target_norm": t_norm.to_dict(),
            "model_config": {
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "target_dim": TARGET_DIM,
                "hidden_dim": 64,
                "num_layers": 3,
                "activation": "silu",
            },
        },
        ckpt_file,
    )

    loaded_m, loaded_sn, loaded_tn, _ = load_model_from_checkpoint(ckpt_file)
    assert isinstance(loaded_m, WorldModel)
    assert torch.allclose(loaded_sn.mean, s_norm.mean)
    assert torch.allclose(loaded_sn.std, s_norm.std)
