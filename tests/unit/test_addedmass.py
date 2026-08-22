"""Hull added mass, and the steering behaviour it governs.

A hull accelerating through water accelerates water with it.  For a
rowing shell the entrained water is not a correction: in sway and yaw it
is comparable to the boat or larger, and omitting it leaves the boat
several times too easy to turn.  See SOURCES sec. 35.
"""
import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.addedmass import (AddedMass, sectional_heave,
                                      sectional_sway, surge_coefficient)
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator


def test_lamb_surge_coefficient_matches_the_published_table():
    """Lamb (1932) art. 71, prolate spheroid inertia coefficient k1."""
    # a sphere entrains half its displaced mass
    assert surge_coefficient(1.0, 1.0) == pytest.approx(0.5, abs=0.02)
    # published values for elongated spheroids
    assert surge_coefficient(2.0, 1.0) == pytest.approx(0.209, abs=0.01)
    assert surge_coefficient(5.0, 1.0) == pytest.approx(0.059, abs=0.01)
    # and it vanishes for a very slender body
    assert surge_coefficient(40.0, 1.0) < 0.01


def test_sectional_values_use_the_right_dimension():
    """Sway presents the draft to the flow; heave presents the beam."""
    rho = 1000.0
    # a section twice as deep entrains four times as much in sway
    assert sectional_sway(0.2, rho) == pytest.approx(
        4.0 * sectional_sway(0.1, rho))
    # heave depends on beam, not draft
    assert sectional_heave(0.4, rho) == pytest.approx(
        4.0 * sectional_heave(0.2, rho))
    # half of the full-ellipse result: rho pi T^2 / 2
    assert sectional_sway(0.135, rho) == pytest.approx(
        0.5 * rho * np.pi * 0.135 ** 2)


def test_added_mass_matrix_is_symmetric():
    boat = catalog.double_scull()
    m = AddedMass.from_offsets(boat.offsets, rho=boat.water.density).matrix
    assert np.allclose(m, m.T)
    assert np.all(np.diag(m) >= 0.0)


@pytest.mark.parametrize("factory", [catalog.single_scull,
                                     catalog.double_scull,
                                     catalog.eight])
def test_surge_added_mass_is_small_but_sway_is_not(factory):
    """The asymmetry is the whole point.

    A slender hull slips along its own axis entraining almost nothing,
    but shoved sideways it drags a boat's worth of water with it.
    """
    boat = factory()
    a = np.diag(AddedMass.from_offsets(boat.offsets,
                                       rho=boat.water.density).matrix)
    assert a[0] < 0.02 * boat.total_mass          # surge: negligible
    assert a[1] > 0.5 * boat.total_mass           # sway: comparable or more


def test_added_yaw_inertia_dominates_a_small_boat():
    """The number that makes a single scull feel unsteerable in a model.

    Physical yaw inertia of a 1x is tens of kg m^2; the water it swings
    is hundreds.
    """
    boat = catalog.single_scull()
    added = AddedMass.from_offsets(boat.offsets,
                                   rho=boat.water.density).matrix[5, 5]
    mass, position, _, _ = boat.crew_field(0.0)
    crew_yaw = float(np.sum(mass * (position[:, 0] ** 2
                                    + position[:, 1] ** 2)))
    physical = float(boat.hull_inertia[2, 2]) + crew_yaw
    assert added > 4.0 * physical


def test_added_mass_slows_the_yaw_response_to_rudder():
    """The steering-relevant consequence, end to end.

    Steady turn rate is set by rudder moment against yaw damping and
    barely moves.  What added mass changes is the *transient* -- and a
    coxswain's corrections live in the transient.
    """
    boat = catalog.eight()
    period = boat.timing.period

    def amplitude(added_mass, osc=4.0):
        cox = Coxswain(rudder_override=lambda t, s:
                       np.radians(5.0) * np.sin(2 * np.pi * t / osc))
        sim = RowingSimulator(boat, coxswain=cox, added_mass=added_mass)
        res = sim.run(duration=max(4 * osc, 8 * period), dt=0.006,
                      surge_speed=4.5)
        t = res.time
        yaw = np.unwrap(np.asarray(res.attitude)[2])
        w = t > osc
        detrended = yaw[w] - np.polyval(np.polyfit(t[w], yaw[w], 1), t[w])
        return float(np.ptp(detrended))

    assert amplitude(True) < 0.85 * amplitude(False)


def test_added_mass_can_be_switched_off():
    """The old behaviour stays reachable, for comparison runs."""
    boat = catalog.double_scull()
    assert RowingSimulator(boat, added_mass=False)._added_mass is None
    assert RowingSimulator(boat)._added_mass is not None
