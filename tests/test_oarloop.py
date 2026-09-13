r"""The reduced surge-plus-oar model, and the gate it answers.

Phase 2's question is whether making the oar angle a state removes the
defect: with the blade force independent of speed, the steady balance
``R(v) v = eta P`` forces ``eta`` proportional to ``v`` -- a straight line
through the origin.  Scored as *where the fitted line reaches zero
efficiency, as a multiple of mean speed*, the baseline reads **0.020** on
the eight; a blade with real physics should not pass anywhere near the
origin.

Measured here: **6.4**.  The gate passes by a factor of about forty.
(It was 8.9 with the blade force at the oar's tip; the force now acts at the
blade centre, [CR06]'s ``l`` -- 2.30 m on this eight, not 2.56 -- which slows
the blade and moves every row below.)

    ========  =======  ===========  ==========
    peak tau  speed    W per rower  eta
    ========  =======  ===========  ==========
    200 N m   2.92     79           0.528
    300       3.41     119          0.540
    450       3.95     178          0.557
    650       4.54     257          0.563
    900       5.13     356          0.569
    ========  =======  ===========  ==========

``eta`` is now nearly flat -- and at a level, about 0.55, that a blade
efficiency can actually be.  It also does something the prescribed model
could not: at **one realistic handle power** the eight and the four reach
their published race paces, where the prescribed model needed 720 W and
795 W per rower to get there and overshot every band anyway.

What this cannot show
---------------------
The crew's mass does not move in this model, so there is no intracycle
surge swing -- and the swing is what destroyed the efficiency-only wiring,
because the hull's minimum coincides with peak oar force.  A pass here is
**necessary and not sufficient**.  It settles the shape of ``eta(v)``; it
cannot settle whether the coupled model stays stable once the crew is
moving again.  That is what wiring it into the simulator is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.sim.oarloop import (drag_curve, settle_at_power,
                                  settle_coupled, torque_shape)


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=28.0)


@pytest.fixture(scope="module")
def parts(eight):
    from coxswain.crew.oardynamics import OarDynamics

    return OarDynamics.from_boat(eight), drag_curve(eight)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def test_the_drag_curve_agrees_with_the_simulators_own(eight):
    """One definition of hull drag, not two."""
    from coxswain.core.state import State
    from coxswain.hydro.resistance import hull_resistance
    from coxswain.sim.simulator import RowingSimulator

    curve = drag_curve(eight)
    sim = RowingSimulator(eight, fast=True)
    for speed in (3.0, 4.5, 5.5):
        y = sim.initial_state(surge_speed=speed)
        props = eight.mesh.submerged(
            np.array([0.0, 0.0, float(y[2])]),
            np.asarray(y[3:6], dtype=float),
            rho=eight.water.density, gravity=9.81)
        result = hull_resistance(State.from_vector(y).velocity_hull, props,
                                 eight.length, eight.water, eight.resistance,
                                 getattr(eight, "shallow", None),
                                 wave_table=getattr(eight, "wave_table", None))
        force = result[0] if isinstance(result, tuple) else result
        expected = abs(float(np.asarray(force)[0]))
        assert curve(speed) == pytest.approx(expected, rel=0.02), speed


def test_the_torque_shape_is_the_measured_one(eight):
    """Front-loaded, peaking near 40% of the drive, and by ANGLE.

    Parameterised by angle progress rather than time, because the drive
    duration is now an output -- a curve in time would need the answer
    before it could be evaluated.
    """
    shape = torque_shape(eight)
    catch = float(eight.oar_sweep.catch_angle)
    finish = float(eight.oar_sweep.finish_angle)
    progress = np.linspace(0.0, 1.0, 51)
    values = np.array([shape(catch - u * (catch - finish)) for u in progress])

    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values[-1] == pytest.approx(0.0, abs=1e-9)
    peak = progress[int(np.argmax(values))]
    assert 0.30 < peak < 0.50, peak
    assert values.max() == pytest.approx(1.0, rel=0.02)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_efficiency_is_no_longer_proportional_to_speed(eight, parts):
    """The whole point of phase 2, scored the way the baseline was.

    Baseline: the fitted line reaches zero efficiency at 0.020 of mean
    speed -- through the origin. Here it should be far outside the
    racing range, which is what "not proportional to v" means.
    """
    oar, resistance = parts
    runs = [settle_coupled(eight, tau, oar=oar, resistance=resistance,
                           cycles=30)
            for tau in (200.0, 300.0, 450.0, 650.0, 900.0)]
    assert all(run.settled for run in runs)

    speeds = np.array([run.speed for run in runs])
    etas = np.array([run.efficiency for run in runs])
    assert speeds[-1] > 1.5 * speeds[0], speeds

    slope, intercept = np.polyfit(speeds, etas, 1)
    crossing = abs(-intercept / slope) / speeds.mean()
    assert crossing > 1.0, (slope, intercept, crossing)

    # And eta/v is no longer constant, which was the robust statement of
    # the defect: it was flat to 2.7% on the baseline.
    ratio = etas / speeds
    assert np.ptp(ratio) / ratio.mean() > 0.3, ratio


@pytest.mark.slow
def test_the_efficiency_level_is_one_a_blade_could_have(eight, parts):
    """A guard on the level, not just the shape -- and it caught a bug.

    The first version applied the rig's gearing to the BLADE force as
    well as to the handle force, by analogy with ``hull_load``. For the
    hull-plus-crew system the only external horizontal forces are the
    blade force and the drag: the pull on the handle, the handle's push
    back and the stretcher reaction are all internal. The lever decides
    how hard the rower must pull for a given blade force, not how much
    of it reaches the boat. Charging it twice cost a factor of 3.2 and
    put efficiency at 0.18, which no blade has.
    """
    oar, resistance = parts
    run = settle_coupled(eight, 650.0, oar=oar, resistance=resistance,
                         cycles=30)
    assert 0.40 < run.efficiency < 0.85, run.efficiency
    # And the power that goes with it is one a rower could produce.
    assert 150.0 < run.handle_power < 500.0, run.handle_power


# ---------------------------------------------------------------------------
# against published pace
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_racing_boats_reach_published_pace_at_a_real_power():
    """What ``test_steady_speed_matches_published_race_pace`` wanted.

    That test compares settled speed against published race pace while
    driving each boat at ``power_scales = 1.0`` -- 720 W per rower for
    the eight at rate 32, 795 for the four -- so every boat beat its
    band, which was the only thing that could have happened. It is a
    strict xfail for exactly that reason.

    Here the boats are driven at **one stated handle power**, the same
    for both, and both land in band. One consistent, physiologically
    plausible number reproducing two different published bands is a
    great deal more than a fit.
    """
    from coxswain.boats import catalog

    for name, rate, low, high in (("8+", 32.0, 5.0, 5.6),
                                  ("4+", 32.0, 4.5, 5.1)):
        boat = catalog.build(name, rate=rate)
        run = settle_at_power(boat, 380.0, cycles=25)
        assert run.handle_power == pytest.approx(380.0, rel=0.02)
        assert low <= run.speed <= high, (name, rate, run.speed)


@pytest.mark.slow
def test_the_boat_classes_come_out_in_the_right_order():
    """At equal power per rower: eight, then double, then single.

    More crew per unit of drag. It is the most basic ordering in the
    sport and it is not imposed anywhere -- it has to fall out of the
    hulls, the rigs and the blade model.
    """
    from coxswain.boats import catalog

    speeds = {}
    for name in ("8+", "4+", "2x", "1x"):
        boat = catalog.build(name, rate=30.0)
        speeds[name] = settle_at_power(boat, 380.0, cycles=25).speed
    assert speeds["8+"] > speeds["4+"] > speeds["1x"], speeds
    assert speeds["2x"] > speeds["1x"], speeds


@pytest.mark.slow
def test_a_sculler_is_charged_for_both_oars():
    """Per rower, not per oar.

    A single has one seat and two oarlocks. Collecting two oars' thrust
    while counting one oar's work flattered it by a factor of two, and
    put it 28% above its published race pace.
    """
    from coxswain.boats import catalog

    single = catalog.build("1x", rate=30.0)
    run = settle_at_power(single, 380.0, cycles=25)
    assert run.handle_power == pytest.approx(380.0, rel=0.02)
    # With only one oar counted this landed at 5.35 m/s against a
    # published 4.1-4.7.
    assert run.speed < 4.8, run.speed


@pytest.mark.slow
def test_the_drive_fraction_rises_with_rate(eight):
    """Because the drive takes a physical time and the period shrinks.

    Measured on the water it rises too, 0.296 at 20.6 spm to 0.395 at
    31.5. The model's rises about twice as fast at a FIXED power, which
    is expected and is itself a finding: real crews pull harder at higher
    rates, and pulling harder shortens the drive. Comparing across rates
    properly needs a published power-against-rate relationship, which is
    recorded as blocked in docs/PHYSICS_PROGRAMME.md.
    """
    from coxswain.boats import catalog

    fractions = [settle_at_power(catalog.eight(rate=rate), 380.0,
                                 cycles=25).drive_fraction
                 for rate in (24.0, 32.0, 38.0)]
    assert fractions == sorted(fractions), fractions
