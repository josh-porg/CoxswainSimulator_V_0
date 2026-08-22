"""Blades touching the water on the recovery.

Three separate effects, and they are not the same size:

* a **feathered** skim is nearly free -- 22 N against 330 N of hull drag,
  and it saturates;
* a **squared** blade at the same heel is a crab -- 1250 N, four times the
  whole hull resistance;
* the **truncated drive** costs length, and unlike the skim it does not go
  away with good blade work.

[D96] "Balance of Racing Rowing Boats", Furnivall Sculling Club, 1996.
"""

import dataclasses

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.blade_contact import BladeContact


@pytest.fixture
def eight():
    return catalog.eight(rate=32.0)


@pytest.fixture
def contact(eight):
    return BladeContact.from_boat(eight)


# --------------------------------------------------------------------------
# geometry: why a small heel matters
# --------------------------------------------------------------------------
def test_the_blade_sits_on_a_long_lever(contact):
    """3.4 m from the centreline, so roll moves it a lot."""
    assert contact.reach == pytest.approx(3.41, abs=0.05)


def test_a_degree_or_two_of_heel_puts_a_blade_down(contact):
    """The operating point, not an edge case.

    Sections 15-16 put an eight's roll swing at about a degree.  A crew
    carrying a normal 5-10 cm of clearance is roughly that far from
    touching.
    """
    assert np.degrees(contact.roll_to_touch(0.05)) < 1.0
    assert np.degrees(contact.roll_to_touch(0.10)) < 2.0


def test_more_clearance_buys_more_heel(contact):
    assert contact.roll_to_touch(0.15) > contact.roll_to_touch(0.05)


def test_the_blades_are_clear_when_the_boat_is_level(contact):
    assert contact.immersion(0.0, 1.0) == pytest.approx(0.0)
    assert contact.immersion(0.0, -1.0) == pytest.approx(0.0)


def test_only_the_low_side_touches(contact):
    roll = np.radians(2.5)
    low = contact.immersion(roll, 1.0)
    high = contact.immersion(roll, -1.0)
    assert low > 0.0
    assert high == pytest.approx(0.0)


def test_no_loads_at_all_when_level(contact):
    drag, moment = contact.loads(0.0, 4.6)
    assert drag == pytest.approx(0.0)
    assert moment == pytest.approx(0.0)


# --------------------------------------------------------------------------
# feathered versus squared -- the distinction that matters most
# --------------------------------------------------------------------------
def test_a_feathered_blade_drags_far_less_than_a_squared_one(contact):
    """Feathering is not a stylistic matter.

    Feathered, the blade presents its edge and the frontal area saturates
    at thickness by span.  Squared, it presents its face and the area keeps
    growing with depth.
    """
    roll = np.radians(3.0)
    squared = dataclasses.replace(contact, feathered=False)
    feathered_drag = abs(contact.loads(roll, 4.6)[0])
    squared_drag = abs(squared.loads(roll, 4.6)[0])
    assert squared_drag > 20.0 * feathered_drag


def test_feathered_drag_saturates_with_depth(contact):
    """Once the blade is under by its own thickness there is no more edge
    to wet.  A model that keeps growing here is treating it as face-on."""
    shallow = abs(contact.loads(np.radians(2.0), 4.6)[0])
    deep = abs(contact.loads(np.radians(4.0), 4.6)[0])
    assert deep == pytest.approx(shallow, rel=0.05)


def test_squared_drag_does_not_saturate(contact):
    """The control: a crab gets worse the deeper it goes."""
    squared = dataclasses.replace(contact, feathered=False)
    shallow = abs(squared.loads(np.radians(2.0), 4.6)[0])
    deep = abs(squared.loads(np.radians(4.0), 4.6)[0])
    assert deep > 2.0 * shallow


def test_a_feathered_skim_is_cheap_against_hull_drag(contact):
    """About 7% of the hull's resistance, and it stops there."""
    drag = abs(contact.loads(np.radians(3.0), 4.6)[0])
    assert drag < 0.15 * 330.0


def test_drag_opposes_motion(contact):
    assert contact.loads(np.radians(2.5), 4.6)[0] < 0.0


def test_drag_grows_with_speed(contact):
    roll = np.radians(2.5)
    slow = abs(contact.loads(roll, 3.0)[0])
    fast = abs(contact.loads(roll, 6.0)[0])
    assert fast > 3.0 * slow


# --------------------------------------------------------------------------
# the stabilising half
# --------------------------------------------------------------------------
def test_contact_produces_a_restoring_roll_moment(contact):
    """[D96]: the spoon takes some of the weight off that rigger."""
    roll = np.radians(2.5)
    _, moment = contact.loads(roll, 4.6)
    assert moment < 0.0, "a heel to port must be pushed back"


def test_the_stabilising_moment_dwarfs_the_crew(eight, contact):
    """Which is why skimming is a technique and not merely a fault.

    Feathering sheds the drag but not the lift -- a feathered blade is
    exactly the right shape to plane -- so this is close to free
    stabilisation.
    """
    from coxswain.crew.balance import PhaseAuthority

    authority = PhaseAuthority.from_boat(eight)
    _, moment = contact.loads(np.radians(2.0), 4.6)
    assert abs(moment) > 10.0 * authority.recovery


def test_lift_is_capped(contact):
    """A blade pushed too hard dives rather than carrying more."""
    gentle = abs(contact.loads(np.radians(2.0), 4.6)[1])
    hard = abs(contact.loads(np.radians(6.0), 4.6)[1])
    assert hard < 4.0 * gentle


# --------------------------------------------------------------------------
# the truncated drive
# --------------------------------------------------------------------------
def test_an_early_blade_costs_length(eight, contact):
    """The cost the coxswain described: once the blade is in you have to go
    with it, so the drive starts early and short."""
    rate = abs(float(eight.oar_sweep.rate(0.02 * eight.timing.period,
                                          eight.timing)))
    sweep = eight.oar_sweep.total_sweep
    level = contact.length_fraction(0.0, rate, sweep)
    heeled = contact.length_fraction(np.radians(3.0), rate, sweep)
    assert level == pytest.approx(1.0)
    assert heeled < 0.95


def test_length_lost_grows_with_heel(eight, contact):
    rate = abs(float(eight.oar_sweep.rate(0.02 * eight.timing.period,
                                          eight.timing)))
    sweep = eight.oar_sweep.total_sweep
    fractions = [contact.length_fraction(np.radians(d), rate, sweep)
                 for d in (1.5, 2.0, 2.5, 3.0)]
    assert fractions == sorted(fractions, reverse=True)


def test_the_length_cost_does_not_go_away_with_feathering(eight, contact):
    """Unlike the skim drag.

    Feathering is what makes contact survivable for *drag*; it does nothing
    for the length, because the blade is in the water either way.  This is
    why an unset boat is slow even when nobody catches a crab.
    """
    rate = abs(float(eight.oar_sweep.rate(0.02 * eight.timing.period,
                                          eight.timing)))
    sweep = eight.oar_sweep.total_sweep
    squared = dataclasses.replace(contact, feathered=False)
    assert contact.length_fraction(np.radians(3.0), rate, sweep) == \
        pytest.approx(squared.length_fraction(np.radians(3.0), rate, sweep))


def test_length_fraction_is_bounded(eight, contact):
    rate = abs(float(eight.oar_sweep.rate(0.02 * eight.timing.period,
                                          eight.timing)))
    sweep = eight.oar_sweep.total_sweep
    for degrees in (0.0, 5.0, 20.0):
        value = contact.length_fraction(np.radians(degrees), rate, sweep)
        assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------
# the smoothing
# --------------------------------------------------------------------------
def test_the_contact_switch_is_smoothed_by_a_physical_width(contact):
    """Surface roughness and small waves, not numerical taste."""
    pytest.importorskip("casadi")
    error = contact.switch_error()
    assert error == pytest.approx(contact.softness * np.log(2.0), rel=0.2)
    assert error < 0.01


def test_the_smoothed_immersion_is_differentiable_through_contact(contact):
    pytest.importorskip("casadi")
    import casadi as ca

    roll = ca.MX.sym("roll")
    depth = contact.immersion(roll, 1.0, ca)
    jacobian = ca.Function("j", [roll], [ca.jacobian(depth, roll)])
    for degrees in (0.0, 1.34, 3.0):
        value = float(ca.DM(jacobian(np.radians(degrees))))
        assert np.isfinite(value)
