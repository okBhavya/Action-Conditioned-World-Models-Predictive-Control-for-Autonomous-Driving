"""Environment and simulator setup verification script.

Checks OS, Python, CPU/GPU hardware, installed package versions,
and executes a 100-step headless smoke test with MetaDrive via the Gymnasium API.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
import time
from typing import Any


def get_git_info() -> dict[str, str]:
    """Retrieve git commit hash."""
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def print_environment_info() -> None:
    """Print system architecture and package versions."""
    print("=" * 70)
    print("ENVIRONMENT & HARDWARE INSPECTION")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.machine()} ({platform.processor()})")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")

    import os

    print(f"CPU count (logical): {os.cpu_count()}")

    import torch

    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    packages = [
        "metadrive-simulator",
        "gymnasium",
        "numpy",
        "torch",
        "pyyaml",
        "matplotlib",
        "imageio",
        "pytest",
        "ruff",
    ]
    print("\nPackage Versions:")
    for pkg in packages:
        try:
            ver = importlib.metadata.version(pkg)
            print(f"  - {pkg}: {ver}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  - {pkg}: NOT INSTALLED")

    git_info = get_git_info()
    print("\nGit Config:")
    print(f"  - commit: {git_info.get('commit', 'N/A')}")
    print("=" * 70)


def run_smoke_test(num_steps: int = 100) -> dict[str, Any]:
    """Execute a headless MetaDrive smoke test with random actions."""
    print("\nRUNNING HEADLESS METADRIVE SMOKE TEST...")
    start_time = time.time()

    from metadrive.envs import MetaDriveEnv

    env_config = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.0,
        "num_scenarios": 1,
        "start_seed": 42,
    }

    env = MetaDriveEnv(env_config)
    obs, info = env.reset(seed=42)

    obs_shape = obs.shape if hasattr(obs, "shape") else None
    action_space = env.action_space
    dt = env.config["decision_repeat"] * env.config["physics_world_step_size"]

    print(f"Reset initial observation shape: {obs_shape}")
    print(f"Action space: {action_space}")
    print(f"Control timestep dt: {dt:.3f} s ({1.0 / dt:.1f} Hz)")

    # Test vehicle and navigation properties
    agent = env.agent
    print(f"Initial agent position: {agent.position}")
    print(f"Initial agent heading (theta): {agent.heading_theta:.4f} rad")
    print(f"Initial agent speed: {agent.speed:.4f} m/s ({agent.speed_km_h:.2f} km/h)")
    print(f"Initial agent velocity: {agent.velocity}")
    print(f"Initial agent steering: {agent.steering}")
    if hasattr(agent, "navigation"):
        print(f"Navigation current ref lanes count: {len(agent.navigation.current_ref_lanes)}")

    steps_taken = 0
    all_info_keys = set(info.keys())

    for i in range(num_steps):
        action = env.action_space.sample()
        step_result = env.step(action)
        assert len(step_result) == 5, f"Expected 5-tuple, got {len(step_result)}"
        obs, reward, terminated, truncated, step_info = step_result
        all_info_keys.update(step_info.keys())
        steps_taken += 1
        if terminated or truncated:
            obs, reset_info = env.reset()
            all_info_keys.update(reset_info.keys())

    env.close()
    elapsed = time.time() - start_time
    fps = steps_taken / elapsed

    print(f"\nSmoke test completed {steps_taken} steps in {elapsed:.2f} s ({fps:.1f} steps/s).")
    print(f"Recorded {len(all_info_keys)} info keys:")
    for k in sorted(all_info_keys):
        print(f"  - {k}")

    print("\nHeadless smoke test PASSED successfully.")
    print("=" * 70)

    return {
        "obs_shape": list(obs_shape) if obs_shape else None,
        "action_space": str(action_space),
        "control_timestep": dt,
        "num_steps": steps_taken,
        "elapsed_seconds": elapsed,
        "fps": fps,
        "info_keys": sorted(list(all_info_keys)),
    }


def main() -> None:
    print_environment_info()
    run_smoke_test(num_steps=100)


if __name__ == "__main__":
    main()
