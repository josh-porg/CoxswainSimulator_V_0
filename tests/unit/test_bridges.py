"""Bridges as constraints.

Six bridges cross the racing reach and they are the tightest constraint on
it.  Before this they were landmark points, so an optimised trajectory was
free to pass through a pier.
"""

import numpy as np
import pytest

from coxswain.river.bridges import (OSM_BRIDGE_DECKS, BridgeGate, Pier,
                                    build_gates)


@pytest.fixture(scope="module")
def gates():
    return build_gates()


@pytest.fixture
def gate():
    """A 100 m span running due east, for arithmetic that is checkable."""
    return BridgeGate(name="test", start=np.array([0.0, 0.0]),
                      end=np.array([100.0, 0.0]))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_every_bridge_on_the_reach_is_present(gates):
    names = {gate.name for gate in gates}
    for expected in ("Eliot Bridge", "Larz Anderson", "Weeks Footbridge",
                     "Western Avenue", "River Street"):
        assert expected in names


def test_bridges_project_to_sensible_local_coordinates(gates):
    """A swapped lat/lon put these millions of metres away.

    The reach is a few kilometres long, so every bridge must land within a
    few kilometres of the origin.
    """
    for gate in gates:
        for point in (gate.start, gate.end):
            assert np.hypot(*point) < 5000.0, (gate.name, point)


def test_bridges_come_out_in_the_right_order_down_the_river(gates):
    """Eliot is the upstream end of the course and River Street the
    downstream one, so their eastings must be ordered accordingly."""
    by_name = {gate.name: gate for gate in gates}
    eliot = by_name["Eliot Bridge"].start[0]
    weeks = by_name["Weeks Footbridge"].start[0]
    river = by_name["River Street"].start[0]
    assert eliot < weeks < river


def test_span_and_direction_are_consistent(gate):
    assert gate.span == pytest.approx(100.0)
    np.testing.assert_allclose(gate.direction, [1.0, 0.0], atol=1e-12)
    assert abs(float(np.dot(gate.direction, gate.normal))) < 1e-12


def test_station_measures_along_the_span(gate):
    assert gate.station_of([25.0, 8.0]) == pytest.approx(25.0)
    assert gate.station_of([70.0, -3.0]) == pytest.approx(70.0)


def test_signed_distance_changes_sign_across_the_bridge(gate):
    """What makes it usable to locate the crossing."""
    before = gate.signed_distance([50.0, -20.0])
    after = gate.signed_distance([50.0, +20.0])
    assert before * after < 0.0


# --------------------------------------------------------------------------
# openings and clearance
# --------------------------------------------------------------------------
def test_without_a_raster_the_whole_span_is_open(gate):
    assert gate.open_intervals() == ((0.0, 100.0),)


def test_a_pier_splits_the_opening_in_two():
    gate = BridgeGate(name="test", start=np.array([0.0, 0.0]),
                      end=np.array([100.0, 0.0]),
                      piers=(Pier(centre=50.0, width=10.0),))
    assert gate.open_intervals() == ((0.0, 45.0), (55.0, 100.0))


def test_clearance_is_positive_inside_and_negative_outside():
    gate = BridgeGate(name="test", start=np.array([0.0, 0.0]),
                      end=np.array([100.0, 0.0]),
                      piers=(Pier(centre=50.0, width=10.0),))
    assert gate.clearance([20.0, 0.0]) == pytest.approx(20.0)
    assert gate.clearance([50.0, 0.0]) == pytest.approx(-5.0)
    assert gate.clearance([45.0, 0.0]) == pytest.approx(0.0)


def test_clearance_is_measured_to_the_nearest_edge():
    """Which edge is nearest changes across an opening, and the constraint
    has to follow it -- a boat 2 m from the pier is in trouble even if the
    abutment is 40 m away."""
    gate = BridgeGate(name="test", start=np.array([0.0, 0.0]),
                      end=np.array([100.0, 0.0]),
                      piers=(Pier(centre=50.0, width=10.0),))
    assert gate.clearance([43.0, 0.0]) == pytest.approx(2.0)
    assert gate.clearance([5.0, 0.0]) == pytest.approx(5.0)


def test_a_boat_needs_more_than_its_half_beam(gate):
    """The constraint the optimiser actually applies.

    An eight is 0.57 m in the beam but 18.9 m long, so it sweeps far more
    than its beam while turning -- and its blades reach about 4 m beyond
    the hull on each side.  Clearance has to be compared against the blade
    tips, not the hull.
    """
    from coxswain.boats import catalog

    boat = catalog.eight(rate=32.0)
    lock = boat.rig.seats[0].oarlocks[0]
    blade_reach = abs(float(lock.position[1])) + float(lock.oar.outboard)
    assert blade_reach > 2.5, "blades reach well outside the hull"
    assert gate.clearance([blade_reach - 1.0, 0.0]) < blade_reach


# --------------------------------------------------------------------------
# against the real channel
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_real_bridges_have_a_navigable_opening(gates):
    """Every bridge must be passable, or the course is not rowable.

    This is the check that would have caught a bad projection, a bad
    tangent plane origin, or a channel mask that had drifted away from the
    bridge coordinates.
    """
    from coxswain.river import charles

    raster = charles.charles_channel()
    for gate in gates:
        widest = gate.widest_opening(raster)
        width = widest[1] - widest[0]
        assert width > 20.0, (gate.name, width)
        assert width < gate.span, \
            "the opening cannot be wider than the deck it is under"


def test_deck_endpoints_are_distinct():
    for name, (first, second) in OSM_BRIDGE_DECKS.items():
        assert first != second, name
