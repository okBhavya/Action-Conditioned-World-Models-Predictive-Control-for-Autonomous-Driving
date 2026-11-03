"""Integration tests with MetaDrive simulator (marked with @pytest.mark.sim)."""

import pytest


@pytest.mark.sim
def test_metadrive_headless_step() -> None:
    """Verify headless execution and 5-tuple step return."""
    from metadrive.envs import MetaDriveEnv

    env = MetaDriveEnv(
        {
            "use_render": False,
            "manual_control": False,
            "traffic_density": 0.0,
            "map": "S",
            "start_seed": 100,
            "num_scenarios": 1,
        }
    )
    obs, info = env.reset(seed=100)
    assert obs is not None
    assert isinstance(info, dict)

    for _ in range(10):
        step_res = env.step([0.0, 0.5])
        assert len(step_res) == 5
        obs, reward, terminated, truncated, info = step_res
        if terminated or truncated:
            break

    env.close()
