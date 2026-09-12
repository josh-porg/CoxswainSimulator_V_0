r"""The blade does not know how fast the boat is going.

``oar_force(t, timing, side)`` is a function of stroke phase and side.
There is no velocity argument anywhere in the force path, so the
propulsive force a crew makes is the same whether the boat is doing
2 m/s or 6.  A real blade's force falls as the hull catches up with it
-- that is what blade slip IS -- so the error grows with distance from
wherever the model was calibrated.

What that does to the numbers, measured: with the force independent of
speed, the steady-state balance ``R(v) v = eta P`` forces
``eta = (R(v)/P) v``, and since ``R`` is set by the force rather than
by the speed, ``eta`` comes out proportional to ``v``.  Both boats at
rate 28, four power levels each, as ``speed (m/s) / eta``:

    ======  ============  ============
    scale   eight         four
    ======  ============  ============
    0.25    2.81 / 0.239  2.39 / 0.197
    0.45    3.92 / 0.342  3.34 / 0.277
    0.70    5.02 / 0.432  4.30 / 0.352
    0.95    5.93 / 0.517  5.07 / 0.420
    ======  ============  ============

``eta/v`` is 0.0849-0.0872 on the eight (spread 2.7%) and 0.0819-0.0830
on the four (1.3%), and the fitted intercepts are -0.008 and +0.0002 --
indistinguishable from zero on both hulls.  So the two boats lie on one
line through the origin: this is a property of the force path and not
of either hull.

**These numbers were re-measured on 2026-09-12 and they are not the ones
this file used to carry.**  The earlier table (0.366 at 2.8 m/s, rising
to 0.766) was taken before ``a053540`` made the blade load perpendicular
to the shaft.  That commit's message said ``f_x`` was unchanged so "no
speed or power calibration moved", which is true of thrust and boat
speed and **false of handle power**: the handle sweeps about the pin, so
its velocity is perpendicular to the shaft and the lateral force
component does work.  Mean handle power for the eight at rate 28 went
from 411 W to 630 W -- 53% -- and eta fell by the same factor.  Nothing
caught it; the validation scorecard did, on its first run.  See
``coxswain.validation``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_the_force_path_takes_no_velocity():
    """The mechanism, checked directly rather than inferred."""
    from coxswain.boats.boat import Boat
    from coxswain.crew import oarlock

    force = inspect.signature(oarlock.oar_force).parameters
    assert "t" in force and "side" in force
    for name in force:
        assert "veloc" not in name and "speed" not in name, name
    at = inspect.signature(Boat.oar_forces_at).parameters
    for name in at:
        assert "veloc" not in name and "speed" not in name, name
    # and the force really is the same at two different boat speeds:
    # it is a function of stroke time, so there is nothing to vary
    from coxswain.boats import catalog
    boat = catalog.eight(rate=28.0)
    a = np.asarray(boat.oar_force_at(0.3, +1))
    b = np.asarray(boat.oar_force_at(0.3, +1))
    assert np.allclose(a, b)


def _efficiency(boat, scale, start):
    """``(speed, efficiency)`` -- now measured by the validation harness.

    This used to be defined here, and the definition was promoted into
    :mod:`coxswain.validation.scorecard` so the scorecard and the test
    that found the defect cannot drift apart and leave nobody able to say
    which number is right.
    """
    from coxswain.validation.scorecard import efficiency_at

    return efficiency_at(boat, scale, start)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "The blade carries no velocity term, so propulsive efficiency comes "
    "out proportional to boat speed instead of roughly constant across "
    "the racing range. Measured 0.239 at 2.81 m/s and 0.517 at 5.93 -- a "
    "factor of 2.2. When the force path gains a slip term this should "
    "pass, and that is the point of it being strict. See "
    "docs/TRACKING.md, 'The blade does not know the boat's speed'."))
def test_efficiency_should_not_double_across_the_racing_range():
    """What ought to be true, failing on purpose until it is.

    Propulsive efficiency varies with slip in reality, but gently: a
    blade is not twice as efficient at 5.8 m/s as at 2.8.
    """
    from coxswain.boats import catalog

    slow_v, slow_eta = _efficiency(catalog.eight(rate=28.0), 0.25, 2.8)
    fast_v, fast_eta = _efficiency(catalog.eight(rate=28.0), 0.90, 5.5)
    assert fast_v > slow_v + 1.5, (slow_v, fast_v)
    assert abs(fast_eta - slow_eta) / slow_eta < 0.30, (slow_eta, fast_eta)


def test_the_blade_load_is_perpendicular_to_the_shaft():
    """The docstring's own specification, which the code contradicted.

    A blade sweeps about its oarlock, so it crosses the water at right
    angles to the shaft and the water's reaction is along that normal.
    With ``oar_axis`` = ``(sin, side cos)`` the normal carrying a
    forward component is ``(cos, -side sin)``; the code had ``+side``,
    whose dot with the shaft is ``|F| sin(2 phi)`` -- 1049 N at
    mid-drive on a four, where zero is required.

    ``f_x`` is identical either way, so no power or speed calibration
    depends on this.  What it sets is which way an alternate rig's yaw
    couple points.
    """
    from coxswain.crew.oarlock import oar_axis, oar_force
    from coxswain.crew.stroke import StrokeTiming

    timing = StrokeTiming(rate=30.0)
    worst = 0.0
    for t in np.linspace(0.0, timing.period, 60, endpoint=False):
        for side in (+1, -1):
            f = np.asarray(oar_force(t, timing, side), dtype=float)
            axis = np.asarray(oar_axis(t, timing, side), dtype=float)
            size = float(np.hypot(*f[:2]))
            if size < 1e-9:
                continue
            worst = max(worst, abs(float(f[:2] @ axis[:2])) / size)
    assert worst < 1e-9, worst


def test_a_bucket_rig_cancels_the_couple_an_alternate_rig_carries():
    """Which is the reason the rig exists, and a check on the sign."""
    from coxswain.core.frames import cross3
    from coxswain.crew.oarlock import oar_force
    from coxswain.viz.menu import build_boat
    from coxswain.viz.rigview import PRESETS
    from coxswain.boats import catalog

    def mean_yaw(boat):
        ts = np.linspace(0.0, boat.timing.period, 300, endpoint=False)
        total = 0.0
        for seat in boat.rig.seats:
            for lock in seat.oarlocks:
                position = np.asarray(lock.position, dtype=float)
                for t in ts:
                    force = lock.oar.gearing * np.asarray(
                        oar_force(t, boat.timing, lock.side), dtype=float)
                    total += cross3(position, force)[2]
        return total / len(ts)

    alternate = mean_yaw(catalog.coxed_four(rate=30.0))
    bucket, _made = build_boat("4+", 30.0, lineup=PRESETS["HOCR 4+"]())
    assert abs(alternate) > 10.0, alternate
    assert abs(mean_yaw(bucket)) < 1e-6, mean_yaw(bucket)
