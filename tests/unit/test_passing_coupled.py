"""The passing rules coupled to a speed field.

The state machine alone makes every yield free: a crew moves over, changes
a number, and rows on at the same speed.  On the Charles that is exactly
wrong, because the line *is* the deep water.  These tests pin the coupling
that makes a yield cost something.
"""

import numpy as np
import pytest

from coxswain.river.passing import Entry, HeadRace, PassingRules


def ridge(station, lateral):
    """A river fastest on the centreline, falling away to either side.

    Deliberately synthetic: the point is to test that the coupling
    *transmits* a speed field, not to re-test the bathymetry.
    """
    return 4.0 - 0.02 * lateral ** 2


def flat(station, lateral):
    return 4.0


def build(speed_fn, interval=12.0, gain=0.04, length=2000.0,
          compliance=1.0):
    leader = Entry(bow=1, start=0.0, speed=4.0, name="ahead",
                   speed_fn=speed_fn)
    chaser = Entry(bow=2, start=interval, speed=4.0, name="you",
                   speed_fn=lambda s, y: speed_fn(s, y) * (1.0 + gain))
    return HeadRace([leader, chaser], length=length,
                    rules=PassingRules(), compliance=compliance)


# --------------------------------------------------------------------------
# the constant-speed path is untouched
# --------------------------------------------------------------------------
def test_without_a_speed_field_position_is_still_closed_form():
    """The rule tests depend on an exactly reproducible geometry."""
    entry = Entry(bow=1, start=10.0, speed=4.0)
    assert entry.position(0.0) == 0.0
    assert entry.position(10.0) == 0.0
    assert entry.position(20.0) == pytest.approx(40.0)


def test_a_speed_field_switches_position_to_the_integrated_station():
    entry = Entry(bow=1, start=0.0, speed=4.0, speed_fn=flat)
    entry.station = 123.0
    assert entry.position(999.0) == 123.0


# --------------------------------------------------------------------------
# the coupling
# --------------------------------------------------------------------------
def test_a_crew_moves_at_the_field_speed():
    event = build(flat, length=400.0)
    event.run(dt=0.5)
    leader = event.entries[1]
    assert leader.finished is not None
    assert leader.finished == pytest.approx(400.0 / 4.0, rel=0.02)


def test_a_faster_chaser_closes_and_the_rules_fire():
    """Gaps must open and close on their own, not on a script."""
    event = build(flat, interval=10.0, gain=0.08, length=3000.0)
    log = event.run(dt=0.5)
    assert log.of_kind("declare"), "the chaser never caught up"


def test_yielding_costs_time_on_a_ridged_river():
    """The headline: being pushed off the fast line is a real penalty."""
    event = build(ridge, interval=10.0, gain=0.08, length=3000.0)
    log = event.run(dt=0.5)
    assert log.of_kind("yield"), "no yield happened"
    leader = event.entries[1]
    assert leader.lost_to_yield > 0.0


def test_yielding_costs_nothing_on_a_flat_river():
    """The control, and it is what makes the previous test mean anything.

    If a yield cost time even where the water is uniform, the cost would
    be an artefact of the bookkeeping rather than the bathymetry.
    """
    event = build(flat, interval=10.0, gain=0.08, length=3000.0)
    log = event.run(dt=0.5)
    assert log.of_kind("yield")
    assert event.entries[1].lost_to_yield == pytest.approx(0.0, abs=1e-9)


def test_a_steeper_ridge_costs_more():
    def steep(station, lateral):
        return 4.0 - 0.06 * lateral ** 2

    gentle = build(ridge, interval=10.0, gain=0.08, length=3000.0)
    gentle.run(dt=0.5)
    sharp = build(steep, interval=10.0, gain=0.08, length=3000.0)
    sharp.run(dt=0.5)
    assert sharp.entries[1].lost_to_yield > gentle.entries[1].lost_to_yield


def test_the_yield_cost_is_bounded_by_the_manoeuvre():
    """A crew cannot lose more than the detour is worth.

    Guards the accounting: ``lost_to_yield`` integrates a speed
    difference, and a sign slip there would let it grow without limit.
    """
    event = build(ridge, interval=10.0, gain=0.08, length=3000.0)
    event.run(dt=0.5)
    leader = event.entries[1]
    width = PassingRules().yield_width
    worst = 0.02 * width ** 2          # the ridge's deficit at full offset
    assert leader.lost_to_yield < worst * (leader.finished or 0.0)


def test_the_passer_is_not_the_one_paying():
    """The obligation is one-directional, and so is its cost."""
    event = build(ridge, interval=10.0, gain=0.08, length=3000.0)
    event.run(dt=0.5)
    assert event.entries[2].lost_to_yield == pytest.approx(0.0, abs=1e-9)
    assert event.entries[1].lost_to_yield > 0.0


def test_both_crews_still_finish():
    event = build(ridge, interval=10.0, gain=0.08, length=3000.0)
    event.run(dt=0.5)
    for entry in event.entries.values():
        assert entry.finished is not None


def test_a_non_complying_crew_pays_the_penalty_instead_of_the_detour():
    """It is a real trade, and the model should show both sides of it."""
    event = build(ridge, interval=10.0, gain=0.08, length=3000.0,
                  compliance=0.0)
    log = event.run(dt=0.5)
    assert log.of_kind("penalty")
    assert event.entries[1].lost_to_yield == pytest.approx(0.0, abs=1e-9)
