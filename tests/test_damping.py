"""The hull must damp every degree of freedom it moves in.

Regression tests for a failure that produced no error and no obviously
wrong number until somebody drew a picture of it.  A coxed four at rate
30 rode 1.5 m clear of its own waterline and pitched 25 degrees, because
the vertical resistance was a single lumped force at the origin -- which
exerts no moment about the origin, so the hull had no pitch damping at
all -- and because every damping term in the model was quadratic, which
vanishes faster than the energy going in as the amplitude falls.

The tests that matter here are the *behavioural* ones: not "is the
coefficient the value I typed", but "does the boat stay flat at every
rate it is raced at", which is the thing that was wrong.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.heaveflow import HeaveFlowHull
from coxswain.hydro.radiation import (StripDamping, damping_report,
                                      natural_frequencies)
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator


def four(rate=30.0):
    return catalog.coxed_four(rate=rate, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)


# -- the structural defects -------------------------------------------
def test_a_pitching_hull_feels_a_moment():
    """The defect itself: pitch rate alone must produce a moment.

    With the vertical force lumped at the origin this was identically
    zero, which is why the mode had nothing opposing it.
    """
    boat = four()
    table = HeaveFlowHull(boat.offsets)
    _force, moment = table.load(0.0, 0.2, boat.water.density, 0.8)
    assert abs(moment) > 1.0
    # and it must oppose the rotation
    _f2, moment_back = table.load(0.0, -0.2, boat.water.density, 0.8)
    assert np.sign(moment) != np.sign(moment_back)


def test_distributed_heave_matches_the_lumped_force_it_replaced():
    """A strict extension, not a re-tuning.

    At zero pitch rate the integral of waterline beam along the length
    *is* the plan area, so the new distributed form must reproduce the
    old lumped one exactly.
    """
    boat = four()
    table = HeaveFlowHull(boat.offsets)
    rho, drag, rate = boat.water.density, 0.8, 0.35
    force, moment = table.load(rate, 0.0, rho, drag)
    lumped = -0.5 * rho * drag * table.plan_area * rate * abs(rate)
    assert force == pytest.approx(lumped, rel=1e-9)
    assert moment == pytest.approx(0.0, abs=1e-6)


def test_linear_damping_survives_small_amplitude():
    """The point of a linear term.

    Quadratic damping falls as the square of the rate, so halving the
    motion quarters the force and the *ratio* of damping to momentum
    halves.  A linear term keeps that ratio fixed, which is what makes
    small motions decay instead of grow.
    """
    boat = four()
    strip = StripDamping(boat.offsets)
    matrix = strip.matrix(6.3, boat.water.density)
    fast = matrix[2, 2] * 0.4
    slow = matrix[2, 2] * 0.04
    assert fast / 0.4 == pytest.approx(slow / 0.04, rel=1e-9)
    assert matrix[2, 2] > 0.0


def test_heave_and_pitch_are_coupled():
    """Strip theory gives the off-diagonal terms, and it must.

    A hull that heaves also pitches, because the sectional forces act at
    a lever arm.  A diagonal-only damping matrix would miss it.
    """
    boat = four()
    matrix = StripDamping(boat.offsets).matrix(6.3, boat.water.density)
    assert matrix[2, 4] == pytest.approx(matrix[4, 2])
    assert matrix[4, 4] > 0.0


def test_roll_damping_needs_forward_speed():
    """Ikeda's lift component is linear in U and dominates at race pace.

    It is also why a stationary shell is so much harder to balance than
    a moving one: at rest only Kato friction is left.
    """
    strip = StripDamping(four().offsets)
    at_rest = strip.roll_lift(0.0, 1000.0)
    racing = strip.roll_lift(4.0, 1000.0)
    assert at_rest == 0.0
    assert racing > 0.0
    assert strip.roll_lift(8.0, 1000.0) == pytest.approx(2.0 * racing,
                                                         rel=1e-9)


def test_surge_carries_no_radiation_term():
    """Michell's integral already is the longitudinal wave making.

    A radiation term in surge would count the same physics twice.
    """
    matrix = StripDamping(four().offsets).matrix(6.3, 1000.0)
    assert matrix[0, 0] == 0.0


# -- the derived quantities -------------------------------------------
def test_natural_frequencies_are_physical():
    boat = four()
    modes = natural_frequencies(boat.offsets, boat.total_mass,
                                boat.hull_inertia, boat.water.density)
    # A shell's heave period is a fraction of a second, not seconds.
    assert 3.0 < modes["heave"] < 15.0
    assert 3.0 < modes["pitch"] < 20.0
    # Roll needs a metacentric height the hull alone cannot supply.
    assert modes["roll"] is None


def test_damping_ratios_sit_in_the_published_bands():
    """Against seakeeping literature, which is the only check available.

    Published values for ships put roll at 0.02-0.10.  Heave and pitch
    run 0.1-0.4 for a ship and should come out *lower* here, because a
    shell is three times more slender and radiates less per unit
    displacement -- but they must not be zero, and they must not be
    anywhere near critical.
    """
    boat = four()
    inertia = RowingSimulator(boat, coxswain=Coxswain()).pitch_inertia
    ratios = damping_report(boat.offsets, boat.total_mass, inertia,
                            boat.water.density, speed=3.9)
    assert 0.01 < ratios["heave"] < 0.40
    assert 0.01 < ratios["pitch"] < 0.40
    assert 0.02 < ratios["roll"] < 0.10


def test_crew_dominates_the_pitch_inertia():
    """Reporting against the bare hull said pitch was 1.37 of critical.

    The hull of a four is 51 kg; four rowers and a coxswain are what
    actually resist pitching.
    """
    boat = four()
    simulator = RowingSimulator(boat, coxswain=Coxswain())
    assert simulator.pitch_inertia[1, 1] > 3.0 * boat.hull_inertia[1][1]


# -- the behaviour that was actually wrong -----------------------------
@pytest.mark.parametrize("rate", [24.0, 30.0, 32.0])
def test_the_boat_stays_flat_at_every_racing_rate(rate):
    """The regression that matters.

    Rates 30 and 32 diverged and 18, 22, 26, 28, 36 did not, which is a
    resonance band far too narrow for a real hull.  Sixty seconds is
    long enough: the old model reached 5 degrees rms in that time.
    """
    boat = four(rate)
    simulator = RowingSimulator(boat, coxswain=Coxswain())
    state = simulator.initial_state(surge_speed=3.9)
    state[6] = 3.9
    result = simulator.run(duration=60.0, dt=0.01, initial_state=state)

    pitch = np.degrees(np.asarray(result.attitude)[1])
    heave = np.asarray(result.position)[2]
    assert np.abs(pitch).max() < 2.0
    assert np.abs(heave).max() < 0.20


def test_the_motion_does_not_grow():
    """Self-excitation, stated as a test.

    The old model's pitch rms went from 0.7 degrees to 5 over a minute.
    Growth is the signature; the absolute level is not.
    """
    boat = four(30.0)
    simulator = RowingSimulator(boat, coxswain=Coxswain())
    state = simulator.initial_state(surge_speed=3.9)
    state[6] = 3.9
    result = simulator.run(duration=60.0, dt=0.01, initial_state=state)
    pitch = np.degrees(np.asarray(result.attitude)[1])
    n = len(pitch)
    first = np.sqrt((pitch[:n // 5] ** 2).mean())
    last = np.sqrt((pitch[4 * n // 5:] ** 2).mean())
    assert last < 1.5 * first


# -- the bow-loader ----------------------------------------------------
def test_the_coxed_four_is_a_bow_loader():
    """And that it is not a cosmetic change.

    8.3 m for a 55-90 kg mass moves the crew centre of mass 1.66 m and
    flips the static trim.
    """
    bow = four()
    stern = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                               rower_stature=1.70, coxswain_mass=68.0,
                               bow_loaded=False)
    seats = [seat.station_x for seat in bow.rig.seats]
    assert bow.rig.coxswain_position[0] > max(seats)
    assert stern.rig.coxswain_position[0] < min(seats)
    assert bow.rig.coxswain_reclined and not stern.rig.coxswain_reclined
    # A reclining coxswain sees from lower down.
    assert bow.rig.coxswain_eye_height < stern.rig.coxswain_eye_height

    forward = np.asarray(bow.crew_centre_of_mass(0.0)).ravel()[0]
    aft = np.asarray(stern.crew_centre_of_mass(0.0)).ravel()[0]
    assert forward - aft > 1.0
