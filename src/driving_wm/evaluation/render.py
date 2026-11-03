"""Top-down episode rendering and GIF generation."""

from __future__ import annotations

from pathlib import Path

from driving_wm.controllers.pid import BaseController
from driving_wm.utils.log import setup_logger

logger = setup_logger("evaluation.render")


def render_episode_gif(
    controller: BaseController,
    map_layout: str = "SC",
    seed: int = 42,
    max_steps: int = 250,
    save_path: str | Path = "docs/img/demo.gif",
    traffic_density: float = 0.0,
) -> Path:
    """Execute an episode with off-screen top-down recording and save as GIF under 5MB."""
    from metadrive.envs import MetaDriveEnv

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env_config = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": traffic_density,
        "map": map_layout,
        "start_seed": seed,
        "num_scenarios": 1,
        "random_agent_model": False,
    }

    env = MetaDriveEnv(env_config)
    obs, info = env.reset(seed=seed)
    controller.reset()

    steps = 0
    done = False

    while not done and steps < max_steps:
        action = controller.compute_action(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        # Record top-down frame
        env.render(mode="topdown", window=False, screen_record=True)
        steps += 1
        done = terminated or truncated

    # Generate animated GIF
    temp_gif = str(save_path)
    if hasattr(env, "top_down_renderer") and env.top_down_renderer is not None:
        env.top_down_renderer.generate_gif(temp_gif)
    env.close()

    size_mb = save_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Generated GIF at {save_path} ({size_mb:.2f} MB, {steps} steps, arrive_dest={info.get('arrive_dest')})"
    )
    assert size_mb < 5.0, f"GIF exceeds 5 MB limit ({size_mb:.2f} MB)"

    return save_path
