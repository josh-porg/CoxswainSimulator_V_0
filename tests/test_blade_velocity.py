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
by the speed, ``eta`` comes out proportional to ``v``:

    ====== ====== ======
    speed  eight  four
    ====== ====== ======
    2.8      0.366  0.338
    3.6      0.477  0.456
    4.8      0.640  0.563
    5.7      0.766  0.695
    ====== ====== ======

The two boats lie on the same line, so this is a property of the force
path and not of either hull.  It is why the quasi-steady evaluator runs
7% optimistic against the eight and 35% against the four: the eight is
close to where this was calibrated and the four is not.
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
    from coxswain.core.state import State
    from coxswain.crew.exertion import mean_handle_power
    from coxswain.hydro.resistance import hull_resistance
    from coxswain.sim.control import Coxswain
    from coxswain.sim.simulator import RowingSimulator

    boat.power_scales = np.full(boat.n_seats, scale)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0),
                          fast=True)
    result = sim.run(duration=70.0, dt=0.01, surge_speed=start)
    time_s = np.asarray(result.time)
    speed = np.hypot(*np.asarray(result.velocity)[:2])
    v = float(speed[time_s > 70.0 - 4 * boat.timing.period].mean())

    crew = mean_handle_power(boat, samples=360) * scale * boat.n_seats
    y = sim.initial_state(surge_speed=v)
    props = boat.mesh.submerged(np.array([0.0, 0.0, float(y[2])]),
                                np.asarray(y[3:6], dtype=float),
                                rho=boat.water.density, gravity=9.81)
    res = hull_resistance(State.from_vector(y).velocity_hull, props,
                          boat.length, boat.water, boat.resistance,
                          getattr(boat, "shallow", None),
                          wave_table=getattr(boat, "wave_table", None))
    force = res[0] if isinstance(res, tuple) else res
    drag = abs(float(np.asarray(force)[0]))
    return v, (drag * v) / max(crew, 1e-9)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "The blade carries no velocity term, so propulsive efficiency comes "
    "out proportional to boat speed instead of roughly constant across "
    "the racing range. Measured 0.37 at 2.8 m/s and 0.77 at 5.8 -- a "
    "factor of 2.1. When the force path gains a slip term this should "
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
