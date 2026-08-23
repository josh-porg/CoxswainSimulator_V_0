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


# --------------------------------------------------------------------------
# arches: the spans, the piers, and which of them the rules allow
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def channel():
    from coxswain.river.charles import charles_channel
    return charles_channel()


@pytest.fixture(scope="module")
def rigged(channel):
    return build_gates(channel=channel)


def test_every_gate_runs_from_the_boston_bank_to_the_cambridge_bank(rigged):
    """Arch numbering only means something if every gate is oriented the
    same way, so ``start`` must be Boston and ``end`` Cambridge at all
    seven bridges.  Six of the seven cross the river roughly south to
    north; Western Avenue crosses west to east.  Either way the Cambridge
    end must be the further one along north-plus-east."""
    for gate in rigged:
        delta = gate.end - gate.start
        assert delta[0] + delta[1] > 0.0, gate.name


def test_the_boston_arch_is_never_legal(rigged, channel):
    """The regatta puts the left arch of every bridge out of bounds."""
    from coxswain.river.bridges import bridge_arches
    for gate in rigged:
        arches = bridge_arches(gate, channel)
        if arches:
            assert not arches[0].legal, gate.name


def test_every_bridge_leaves_a_legal_way_through(rigged, channel):
    """A course a crew cannot legally complete would be a modelling bug."""
    from coxswain.river.bridges import EIGHT_ROWED_WIDTH, candidate_arches
    for gate in rigged:
        legal = candidate_arches(gate, channel)
        assert legal, gate.name
        assert max(a.width for a in legal) > EIGHT_ROWED_WIDTH, gate.name


def test_the_cambridge_arch_is_open_on_the_powerhouse_stretch(rigged, channel):
    """River Street, Western Avenue and Weeks carry no penalty for the
    Cambridge arch, so both it and the centre must survive as candidates.

    This is the test that stops a legal line being quietly discarded.  At
    two of the three the Cambridge arch is the *wider* opening, so which
    of them is quicker is a question for the trajectory solver, and it can
    only answer it if both are still on the table.
    """
    from coxswain.river.bridges import candidate_arches
    by_name = {gate.name: gate for gate in rigged}
    for name in ("River Street", "Western Avenue", "Weeks Footbridge"):
        gate = by_name[name]
        legal = candidate_arches(gate, channel)
        assert len(legal) == 2, name
        assert legal[-1].label == "Cambridge shore", name


def test_anderson_and_eliot_are_centre_arch_only(rigged, channel):
    """Both bar the Cambridge arch as well as the Boston one."""
    from coxswain.river.bridges import candidate_arches
    by_name = {gate.name: gate for gate in rigged}
    for name in ("Larz Anderson", "Eliot Bridge"):
        legal = candidate_arches(by_name[name], channel)
        assert len(legal) == 1, name
        assert legal[0].label == "centre", name


def test_eliot_opening_matches_its_published_navigation_clearance(rigged,
                                                                  channel):
    """An independent check on the whole pier construction.

    The centre opening here is derived: span length from one National
    Bridge Inventory column, pier thickness measured off the trestle a
    mile downstream.  The navigation clearance is a different column
    entirely.  They agree to within a metre, which they need not have.
    """
    from coxswain.river.bridges import BRIDGE_STRUCTURE, racing_arch
    gate = {g.name: g for g in rigged}["Eliot Bridge"]
    arch = racing_arch(gate, channel)
    published = BRIDGE_STRUCTURE["Eliot Bridge"].permitted_width
    assert abs(arch.width - published) < 1.0


def test_an_opening_never_exceeds_the_bridge_that_spans_it(rigged, channel):
    """The depth raster will report water where an abutment stands, which
    at River Street invented a 26 m shore arch out of a wing wall."""
    from coxswain.river.bridges import BRIDGE_STRUCTURE, waterway
    for gate in rigged:
        length = BRIDGE_STRUCTURE[gate.name].structure_length
        if length is None:
            continue
        low, high = waterway(gate, channel)
        assert high - low <= length + 1e-6, gate.name


def test_the_trestle_piers_are_the_measured_ones(rigged, channel):
    """Grand Junction is the one bridge with surveyed piers, five of them,
    at the roughly 25 m spacing the survey shows."""
    gate = {g.name: g for g in rigged}["Grand Junction RR"]
    assert len(gate.piers) == 5
    spacing = np.diff([pier.centre for pier in gate.piers])
    assert spacing.min() > 15.0
    assert spacing.max() < 30.0


def test_piers_sit_inside_the_opening_they_divide(rigged, channel):
    from coxswain.river.bridges import waterway
    for gate in rigged:
        low, high = waterway(gate, channel)
        for pier in gate.piers:
            assert low <= pier.centre <= high, gate.name


def test_racing_arch_is_a_default_not_a_restriction(rigged, channel):
    """``racing_arch`` may only ever name an arch the rules allow."""
    from coxswain.river.bridges import candidate_arches, racing_arch
    for gate in rigged:
        arch = racing_arch(gate, channel)
        assert arch is not None, gate.name
        assert arch.index in {a.index for a in candidate_arches(gate, channel)}


def test_a_square_crossing_loses_nothing(rigged, channel):
    """Five of the seven bridges cross within about 6 degrees of square,
    so their structural and effective widths must be near enough equal."""
    from coxswain.river.bridges import bridge_arches
    by_name = {gate.name: gate for gate in rigged}
    for name in ("BU Bridge", "Weeks Footbridge", "Eliot Bridge"):
        for arch in bridge_arches(by_name[name], channel):
            assert arch.effective_width == pytest.approx(arch.width, rel=0.02)


def test_the_skewed_trestle_is_narrower_to_a_boat_than_to_the_bridge(rigged,
                                                                     channel):
    """The Grand Junction carries the railway diagonally across the river.

    Its openings are measured along the deck, but a boat runs along the
    river, and between two pier faces `w` apart on a deck meeting the
    river at angle phi the corridor is only ``w sin(phi)`` wide.  At 41
    degrees off square that is a quarter of every opening, and reporting
    the deck width would tell the coxswain the gap is bigger than it is.
    """
    from coxswain.river.bridges import crossing_angle, racing_arch
    gate = {g.name: g for g in rigged}["Grand Junction RR"]
    assert crossing_angle(gate, channel) == pytest.approx(0.75, abs=0.06)
    arch = racing_arch(gate, channel)
    assert arch.effective_width < 0.8 * arch.width
    assert arch.effective_width == pytest.approx(16.3, abs=1.0)


def test_effective_width_is_what_counts_boats(rigged, channel):
    from coxswain.river.bridges import EIGHT_ROWED_WIDTH, bridge_arches
    for gate in rigged:
        for arch in bridge_arches(gate, channel):
            assert arch.fits() == pytest.approx(
                arch.effective_width / EIGHT_ROWED_WIDTH)


def test_the_trestle_deck_follows_the_railway_it_carries(rigged):
    """The recorded deck is the Grand Junction Running Track's own
    alignment across the river, so its bearing must match the rails."""
    import math
    gate = {g.name: g for g in rigged}["Grand Junction RR"]
    bearing = math.degrees(math.atan2(gate.direction[0],
                                      gate.direction[1])) % 180.0
    assert bearing == pytest.approx(47.4, abs=1.5)
