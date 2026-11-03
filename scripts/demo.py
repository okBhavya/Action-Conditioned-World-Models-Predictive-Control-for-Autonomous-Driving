"""Demo script: runs closed-loop control on MetaDrive and renders a top-down GIF under 5MB."""

from __future__ import annotations

import argparse
from pathlib import Path

from driving_wm.controllers.pid import (
    ConstantThrottleController,
    RandomController,
    StanleyPIDController,
    ZeroController,
)
from driving_wm.evaluation.render import render_episode_gif
from driving_wm.models.evaluate import load_model_from_checkpoint
from driving_wm.planning.mpc import LearnedModelMPC
from driving_wm.utils.log import setup_logger

logger = setup_logger("scripts.demo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render driving demo with selected controller to GIF"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="mpc",
        choices=["mpc", "pid", "throttle", "zero", "random"],
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/world_model_onestep.pt")
    parser.add_argument("--map", type=str, default="SC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--output", type=str, default="docs/img/demo.gif")
    parser.add_argument("--traffic", type=float, default=0.0)
    args = parser.parse_args()

    out_path = Path(args.output)

    if args.controller == "mpc":
        ckpt_p = Path(args.checkpoint)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"World model checkpoint not found: {ckpt_p}")
        model, state_norm, target_norm, _ = load_model_from_checkpoint(ckpt_p)
        controller = LearnedModelMPC(
            model=model,
            state_norm=state_norm,
            target_norm=target_norm,
            horizon=10,
            num_samples=256,
            safety_constraints=True,
        )
    elif args.controller == "pid":
        controller = StanleyPIDController(target_speed=10.0)
    elif args.controller == "throttle":
        controller = ConstantThrottleController(throttle=0.5)
    elif args.controller == "zero":
        controller = ZeroController()
    else:
        controller = RandomController(seed=args.seed)

    logger.info(
        f"Rendering demo with controller '{args.controller}' on map '{args.map}', seed {args.seed}..."
    )
    gif_file = render_episode_gif(
        controller=controller,
        map_layout=args.map,
        seed=args.seed,
        max_steps=args.max_steps,
        save_path=out_path,
        traffic_density=args.traffic,
    )
    logger.info(f"Demo GIF created: {gif_file} ({gif_file.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
