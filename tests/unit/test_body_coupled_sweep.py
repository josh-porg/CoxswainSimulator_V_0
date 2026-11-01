"""The body-coupled drive: the hands carried by the shoulders while the arms are straight (SOURCES §208)."""
from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.drive_law import BodyCoupledDriveSweep, PopulationDriveSweep
from coxswain.crew.stroke import OnWaterTiming

CATCH, FINISH = np.radians(60.0), np.radians(-44.0)
TIMING = OnWaterTiming(31.0)


def _sweeps(straight):
    law = PopulationDriveSweep(catch_angle=CATCH, finish_angle=FINISH)
    body = BodyCoupledDriveSweep(catch_angle=CATCH, finish_angle=FINISH, straight=straight)
    return law, body


def _times(n=400):
    return np.linspace(0.0, float(TIMING.period), n, endpoint=False)


def test_an_inactive_constraint_leaves_the_population_law():
    x = tuple(np.linspace(0.0, 1.0, 51))
    law, body = _sweeps((x, tuple(np.full(51, np.nan))))
    t = _times()
    np.testing.assert_allclose(body(t, TIMING), law(t, TIMING), atol=1e-12)
    np.testing.assert_allclose(body.rate(t, TIMING), law.rate(t, TIMING), atol=1e-9)
    assert body.arm_draw_onset() == 0.0


def test_the_hands_never_pass_the_straight_arm_angle_and_follow_it_where_it_binds():
    x = np.linspace(0.0, 1.0, 201)
    # shoulders that run ahead of the population's hands through the first half of the drive, then fall away
    straight = CATCH + (FINISH - CATCH) * np.clip(1.6 * x, 0.0, 1.0) ** 1.2 + np.where(x > 0.55, 2.0, 0.0)
    law, body = _sweeps((tuple(x), tuple(straight)))
    D = float(TIMING.drive_fraction) * float(TIMING.period)
    t = np.linspace(0.0, 0.999 * D, 300)
    phi, s = body(t, TIMING), body._straight()[0](t / D)        # the sweep's own (shape-preserving) table
    assert np.all(phi <= s + 1e-9)                               # never beyond a straight arm's reach
    assert np.all(phi <= law(t, TIMING) + 1e-12)
    bound = (law(t, TIMING) - s) > 2 * body.blend
    assert bound.any()
    np.testing.assert_allclose(phi[bound], s[bound], atol=1e-6)
    onset = body.arm_draw_onset()
    assert 0.3 < onset <= 0.6                                    # the draw begins where the law overtakes


def test_rate_is_the_derivative_of_the_angle():
    x = np.linspace(0.0, 1.0, 201)
    straight = CATCH + (FINISH - CATCH) * np.clip(1.4 * x, 0.0, 1.0)
    _law, body = _sweeps((tuple(x), tuple(straight)))
    t = np.linspace(0.02, 0.95, 200) * float(TIMING.drive_fraction) * float(TIMING.period)
    h = 1e-5
    num = (body(t + h, TIMING) - body(t - h, TIMING)) / (2 * h)
    np.testing.assert_allclose(body.rate(t, TIMING), num, rtol=2e-3, atol=2e-3)


def test_the_recovery_is_unchanged():
    x = np.linspace(0.0, 1.0, 51)
    law, body = _sweeps((tuple(x), tuple(np.full(51, CATCH - 0.5))))
    D = float(TIMING.drive_fraction) * float(TIMING.period)
    t = np.linspace(D * 1.001, float(TIMING.period) * 0.999, 100)
    np.testing.assert_allclose(body(t, TIMING), law(t, TIMING), atol=1e-12)
