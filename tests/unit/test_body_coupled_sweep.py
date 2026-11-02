"""The drive the body generates: a smooth table of the oar angle, the recovery unchanged (SOURCES §208, §209)."""
from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.drive_law import BodyCoupledDriveSweep, PopulationDriveSweep
from coxswain.crew.stroke import OnWaterTiming

CATCH, FINISH = np.radians(60.0), np.radians(-44.0)
TIMING = OnWaterTiming(31.0)
X = np.linspace(0.0, 1.0, 241)
# a body-like drive: slow off the catch, fastest mid-drive, at rest into the finish
TABLE = CATCH + (FINISH - CATCH) * (10 * X**3 - 15 * X**4 + 6 * X**5)


def _sweep(table=TABLE, **kw):
    return BodyCoupledDriveSweep(catch_angle=CATCH, finish_angle=FINISH, drive=(tuple(X), tuple(table)), **kw)


def _drive_times(n=300):
    D = float(TIMING.drive_fraction) * float(TIMING.period)
    return np.linspace(0.0, 0.999 * D, n), D


def test_the_drive_is_the_table():
    sweep = _sweep(onset=0.47)
    t, D = _drive_times()
    np.testing.assert_allclose(sweep(t, TIMING), np.interp(t / D, X, TABLE), atol=np.radians(0.01))
    assert sweep.arm_draw_onset() == 0.47


def test_rate_is_the_derivative_of_the_angle():
    sweep = _sweep()
    t, _D = _drive_times()
    t = t[5:-5]
    h = 1e-5
    num = (sweep(t + h, TIMING) - sweep(t - h, TIMING)) / (2 * h)
    np.testing.assert_allclose(sweep.rate(t, TIMING), num, rtol=1e-4, atol=1e-6)


def test_the_recovery_is_unchanged():
    law = PopulationDriveSweep(catch_angle=CATCH, finish_angle=FINISH)
    _t, D = _drive_times()
    t = np.linspace(D * 1.001, float(TIMING.period) * 0.999, 100)
    np.testing.assert_allclose(_sweep()(t, TIMING), law(t, TIMING), atol=1e-12)


def test_a_table_that_misses_the_catch_or_finish_or_cannot_be_reached_is_refused():
    with pytest.raises(ValueError, match="catch"):
        _sweep(TABLE + np.radians(1.0))(np.array([0.1]), TIMING)
    bad = TABLE.copy()
    bad[100] = np.nan
    with pytest.raises(ValueError, match="cannot reach"):
        _sweep(bad)(np.array([0.1]), TIMING)
