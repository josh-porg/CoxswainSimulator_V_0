"""The physics must not move when the code around it does.

Two promises, both of which the speed work could have broken silently.

**The golden trajectory.**  ``tests/data/golden_trajectory.npz`` is 12 s
of a coxed four at rate 30, integrated at ``dt = 0.02``, recorded before
any of the acceleration work.  Every optimisation so far -- the 3-vector
cross product, the scalar clamp, the scalar path through the Fourier
series -- computes the same expressions in the same order, so the
assertion is **exact equality**, not a tolerance.  A tolerance here would
let a real model change through disguised as rounding.

**Stepping equals running.**  A game loop calls
:meth:`~coxswain.sim.simulator.RowingSimulator.step` once per tick; the
studies call :meth:`~coxswain.sim.simulator.RowingSimulator.run`.  If
those two ever disagree, the trainer teaches a boat the analysis does not
describe, which is the one failure this project cannot tolerate.  They
share :func:`coxswain.core.integrators.rk4_step`, and this holds them to
it.
"""

import os

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.sim.control import Coxswain
from coxswain.sim.realtime import (ControlInput, FixedStepLoop, LiveControl,
                                   Recording, interpolate)
from coxswain.sim.simulator import RowingSimulator

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "golden_trajectory.npz")


def make_boat():
    return catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)


@pytest.fixture(scope="module")
def simulator():
    return RowingSimulator(make_boat(), coxswain=Coxswain())


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden trajectory")
def test_the_trajectory_is_bit_for_bit_what_it_was(simulator):
    golden = np.load(GOLDEN)
    result = simulator.run(duration=12.0, dt=0.02,
                           initial_state=golden["initial"])
    assert result.states.shape == golden["states"].shape
    assert np.array_equal(result.states, golden["states"]), (
        "the physics moved: max |difference| %.3e"
        % np.abs(result.states - golden["states"]).max())


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden trajectory")
def test_stepping_by_hand_matches_running(simulator):
    """The seam the real-time loop is built on."""
    golden = np.load(GOLDEN)
    state = np.asarray(golden["initial"], dtype=float)
    t, dt = 0.0, 0.02
    for index in range(1, 120):
        state = simulator.step(state, t, dt)
        t += dt
        assert np.array_equal(state, golden["states"][:, index]), (
            "step and run diverged at step %d" % index)


def test_step_does_not_mutate_the_state_it_is_given(simulator):
    """The loop keeps the previous state to interpolate from; if ``step``
    wrote through its argument, the drawn pose would be the new state
    blended with itself and the motion would judder."""
    state = simulator.initial_state(surge_speed=4.0)
    keep = state.copy()
    simulator.step(state, 0.0, 0.02)
    assert np.array_equal(state, keep)


# -- the loop -------------------------------------------------------------

def test_the_loop_takes_the_right_number_of_steps(simulator):
    """Steps follow the accumulator, and the remainder carries over."""
    loop = FixedStepLoop(simulator, rate=100.0)      # dt = 0.01
    loop.start(simulator.initial_state(surge_speed=4.0))
    assert loop.advance(0.025) == 2          # 0.005 s left over
    assert loop.advance(0.005) == 1          # the carried 0.005 completes it
    assert loop.advance(0.010) == 1
    assert loop.steps == 4
    # No backlog: whatever is left is less than one step.
    assert loop.alpha < 1.0


def test_a_long_stall_does_not_spiral(simulator):
    """One frame may never ask for more than ``max_frame`` of catch-up.

    Thirty seconds of stall must not become three thousand steps, or the
    catch-up takes longer than the stall and the loop never recovers.
    The count is ``max_frame / dt`` give or take one -- 0.01 s is not
    representable in binary, so 0.25 s of accumulator is 24 steps and a
    remainder, not 25.
    """
    loop = FixedStepLoop(simulator, rate=100.0, max_frame=0.25)
    loop.start(simulator.initial_state(surge_speed=4.0))
    taken = loop.advance(30.0)
    assert taken * loop.dt <= loop.max_frame + 1e-12
    assert taken >= int(loop.max_frame / loop.dt) - 1
    assert loop.alpha < 1.0                  # nothing queued for next frame


def test_the_drawn_pose_is_between_the_two_states(simulator):
    loop = FixedStepLoop(simulator, rate=50.0)
    loop.start(simulator.initial_state(surge_speed=4.0))
    loop.advance(0.03)                        # one step, 0.01 left over
    pose = loop.pose()
    low = np.minimum(loop.previous, loop.state)
    high = np.maximum(loop.previous, loop.state)
    assert (pose >= low - 1e-12).all() and (pose <= high + 1e-12).all()
    assert np.array_equal(interpolate(loop.previous, loop.state, 0.0),
                          loop.previous)


# -- live control ---------------------------------------------------------

def test_live_control_reaches_the_force_path():
    """A hand on the tiller has to move the boat, through the same
    override the path follower and the MPC use."""
    live = LiveControl()
    cox = Coxswain(rudder_override=live.rudder)
    sim = RowingSimulator(make_boat(), coxswain=cox)
    state = sim.initial_state(surge_speed=4.0)

    straight = state.copy()
    for _ in range(60):
        straight = sim.step(straight, 0.0, 0.02)

    live.set(ControlInput(rudder=0.12))
    turned = state.copy()
    for _ in range(60):
        turned = sim.step(turned, 0.0, 0.02)

    assert abs(turned[5] - straight[5]) > 1e-4, "rudder did nothing to yaw"


def test_a_recording_replays_exactly():
    """Store the commands, not the trajectory, and get it back."""
    live = LiveControl()
    cox = Coxswain(rudder_override=live.rudder)
    sim = RowingSimulator(make_boat(), coxswain=cox)
    state = sim.initial_state(surge_speed=4.0)

    session = Recording(initial=state, dt=0.02)
    session.record(0.0, ControlInput(rudder=0.0))
    session.record(0.2, ControlInput(rudder=0.08))
    session.record(0.6, ControlInput(rudder=-0.05))

    first = session.replay(sim, live, duration=1.0)[1]
    second = session.replay(sim, live, duration=1.0)[1]
    assert np.array_equal(first, second)
    # and the commands actually took effect
    assert not np.array_equal(first[:, -1], first[:, 0])
