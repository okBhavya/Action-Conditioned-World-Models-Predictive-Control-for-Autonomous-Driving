"""Unit tests for baseline controllers and floor policies."""

import numpy as np

from driving_wm.controllers.pid import (
    ConstantThrottleController,
    RandomController,
    StanleyPIDController,
    ZeroController,
)


def test_floor_policies_action_bounds() -> None:
    zero_ctrl = ZeroController()
    a_zero = zero_ctrl.compute_action(None)
    assert np.all(a_zero == 0.0)

    const_ctrl = ConstantThrottleController(throttle=0.45)
    a_const = const_ctrl.compute_action(None)
    assert a_const[0] == 0.0
    assert a_const[1] == 0.45

    rand_ctrl = RandomController(seed=123)
    for _ in range(50):
        a_rand = rand_ctrl.compute_action(None)
        assert np.all(a_rand >= -1.0)
        assert np.all(a_rand <= 1.0)


def test_stanley_pid_reset() -> None:
    pid = StanleyPIDController()
    pid.speed_error_integral = 5.0
    pid.prev_speed_error = 2.0
    pid.prev_heading = 0.5
    pid.reset()
    assert pid.speed_error_integral == 0.0
    assert pid.prev_speed_error == 0.0
    assert pid.prev_heading is None


class MockLane:
    def __init__(self, length: float = 100.0, heading: float = 0.0) -> None:
        self.length = length
        self._heading = heading

    def local_coordinates(self, pos: list[float] | np.ndarray) -> tuple[float, float]:
        return float(pos[0]), float(pos[1])

    def heading_theta_at(self, s: float) -> float:
        return self._heading


class MockAgent:
    def __init__(
        self,
        pos: tuple[float, float] = (0.0, 0.5),
        heading: float = 0.1,
        speed: float = 8.0,
    ) -> None:
        self.position = list(pos)
        self.heading_theta = heading
        self.speed = speed
        self.velocity = [speed * np.cos(heading), speed * np.sin(heading)]
        self.steering = 0.0
        self.lane = MockLane(100.0, 0.0)
        self.navigation = None


class MockEnv:
    def __init__(self, agent: MockAgent) -> None:
        self.agent = agent
        self.engine = None


def test_stanley_pid_yaw_rate_damping() -> None:
    pid = StanleyPIDController(kd_steer=0.5)
    agent = MockAgent(pos=(0.0, 0.0), heading=0.0, speed=10.0)
    env = MockEnv(agent)

    # Step 1: initial step, prev_heading is None
    a1 = pid.compute_action(env)
    assert pid.prev_heading == 0.0

    # Step 2: heading rotated CCW to 0.2 rad -> yaw_rate = (0.2 - 0.0)/0.1 = 2.0 rad/s
    agent.heading_theta = 0.2
    agent.velocity = [10.0 * np.cos(0.2), 10.0 * np.sin(0.2)]
    a2 = pid.compute_action(env)
    assert pid.prev_heading == 0.2
    # Steer command should include damping of -kd * 2.0 = -1.0
    # Heading error is -0.2, cross track is 0.0 -> steer_cmd = -0.2 - 0.5 * 2.0 = -1.2 -> clipped to -1.0
    assert a2[0] < a1[0]


def test_learned_model_mpc_fallback_and_reset() -> None:
    import torch

    from driving_wm.data.dataset import Normalizer
    from driving_wm.data.state import STATE_DIM, TARGET_DIM
    from driving_wm.models.dynamics import WorldModel
    from driving_wm.planning.mpc import LearnedModelMPC

    model = WorldModel(hidden_dim=32, num_layers=2)
    s_norm = Normalizer(torch.zeros(STATE_DIM), torch.ones(STATE_DIM))
    t_norm = Normalizer(torch.zeros(TARGET_DIM), torch.ones(TARGET_DIM))

    mpc = LearnedModelMPC(
        model=model,
        state_norm=s_norm,
        target_norm=t_norm,
        horizon=5,
        num_samples=16,
        safety_constraints=True,
    )

    agent = MockAgent(pos=(0.0, 8.0), heading=0.2, speed=10.0)  # lat=8.0 violates 4.75m road bound
    env = MockEnv(agent)

    action = mpc.compute_action(env)
    # Because agent is at lat=8.0 (way outside road bounds), all rollouts are infeasible -> fallback triggered
    assert mpc.fallback_count == 1
    # Fallback applies braking: action[1] == -0.5
    assert action[1] == -0.5
    # prev_solution is cleared on fallback so infeasible sequence is not reused as warm start
    assert mpc.prev_solution is None

    # Reset clears telemetry and state
    mpc.reset()
    assert mpc.fallback_count == 0
    assert mpc.total_steps == 0
    assert mpc.prev_solution is None
