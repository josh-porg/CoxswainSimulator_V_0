"""Unit tests for the coxswain's control loops."""

import numpy as np
import pytest

from coxswain.crew.anthropometry import PORT, STARBOARD
from coxswain.core.state import State
from coxswain.sim.control import BalanceController, Coxswain, HeadingController


# --------------------------------------------------------------------------
# rudder
# --------------------------------------------------------------------------
def test_rudder_override_takes_precedence():
    cox = Coxswain(rudder_override=lambda t, s: 0.05)
    assert cox.rudder(0.0, State.zeros()) == pytest.approx(0.05)


def test_rudder_falls_back_to_the_heading_controller():
    cox = Coxswain()
    assert cox.rudder(0.0, State.zeros()) == pytest.approx(
        cox.heading.deflection(0.0, State.zeros()))


def test_heading_controller_steers_towards_the_target():
    """Off to one side, the rudder must be deflected to come back."""
    controller = HeadingController()
    off_course = State.create(attitude=(0.0, 0.0, np.radians(10.0)))
    on_course = State.zeros()
    assert abs(controller.deflection(0.0, off_course)) > abs(
        controller.deflection(0.0, on_course))


def test_balance_controller_opposes_roll():
    controller = BalanceController()
    heeled = State.create(attitude=(np.radians(3.0), 0.0, 0.0))
    moment = controller.moment(heeled.roll, 0.0)
    assert moment * heeled.roll < 0.0, "must act against the heel"


def test_balance_moment_is_zero_when_level():
    assert BalanceController().moment(0.0, 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# pressure split -- the coxswain's second control
# --------------------------------------------------------------------------
def test_split_defaults_to_zero():
    assert Coxswain().split(0.0, State.zeros()) == 0.0


def test_split_accepts_a_constant_or_a_callable():
    state = State.zeros()
    assert Coxswain(pressure_split=0.2).split(1.0, state) == pytest.approx(0.2)
    varying = Coxswain(pressure_split=lambda t, s: 0.1 * t)
    assert varying.split(3.0, state) == pytest.approx(0.3)


def test_split_is_clamped_to_unit_magnitude():
    state = State.zeros()
    assert Coxswain(pressure_split=5.0).split(0.0, state) == pytest.approx(1.0)
    assert Coxswain(pressure_split=-5.0).split(0.0, state) == pytest.approx(
        -1.0)


def test_side_gain_is_symmetric_about_one():
    """The split must be a pure couple: net thrust unchanged.

    Otherwise the optimiser could accelerate by steering.
    """
    port = Coxswain.side_gain(0.3, PORT)
    starboard = Coxswain.side_gain(0.3, STARBOARD)
    assert port + starboard == pytest.approx(2.0)
    assert port > 1.0 > starboard


def test_zero_split_leaves_both_sides_untouched():
    assert Coxswain.side_gain(0.0, PORT) == pytest.approx(1.0)
    assert Coxswain.side_gain(0.0, STARBOARD) == pytest.approx(1.0)


def test_split_sign_convention():
    """Positive split means the port side pulls harder."""
    assert Coxswain.side_gain(0.4, PORT) > Coxswain.side_gain(0.4, STARBOARD)


# --------------------------------------------------------------------------
# effect on the full model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("split", [0.15, 0.30])
def test_split_turns_the_boat_in_the_full_model(split):
    """The headline result: a pressure split is real steering authority.

    Rudder alone holds a 259 m turn radius; the Charles demands 103-146 m
    at its tightest bends.  This is what closes that gap.
    """
    from coxswain.boats import catalog
    from coxswain.sim.simulator import RowingSimulator

    boat = catalog.eight(rate=32.0)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=split)
    result = RowingSimulator(boat, coxswain=cox).run(
        duration=8.0, dt=0.01, surge_speed=5.2)
    rate = float(np.mean(result.omega[2][result.last_cycles(2)]))
    assert abs(rate) > np.radians(0.1), "a split must produce a real turn"


def test_split_does_not_change_net_thrust():
    """A symmetric split is a pure couple, so speed should barely move."""
    from coxswain.boats import catalog
    from coxswain.sim.simulator import RowingSimulator

    boat = catalog.eight(rate=32.0)
    straight = RowingSimulator(
        boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0)).run(
            duration=8.0, dt=0.01, surge_speed=5.2)
    split = RowingSimulator(
        boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0,
                                pressure_split=0.2)).run(
            duration=8.0, dt=0.01, surge_speed=5.2)
    # some loss is expected from the induced turn, but not a step change
    assert split.mean_speed() == pytest.approx(straight.mean_speed(), rel=0.05)
