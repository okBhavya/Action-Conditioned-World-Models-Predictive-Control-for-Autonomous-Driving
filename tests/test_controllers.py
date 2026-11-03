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
    pid.reset()
    assert pid.speed_error_integral == 0.0
    assert pid.prev_speed_error == 0.0
