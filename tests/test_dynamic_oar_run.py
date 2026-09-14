r"""A dynamic-oar simulator, driven the way every consumer drives a simulator.

The report cannot use research physics yet, and the reason is not physics.
Every consumer -- ``settled_speed``, ``steer``, ``fit_reduced_model`` --
constructs ``RowingSimulator(...)`` and calls ``run(duration, ...)`` on a
12-element hull state.  A dynamic-oar simulator carries more state than that,
takes a torque rather than a power scale, and until now had only
``run_strokes``.

Two pieces close the gap, and these tests were written before either:

* :func:`coxswain.sim.dynamic_oar.simulator_for` builds the simulator a
  boat's physics needs.  A dynamic-oar boat must state its handle power per
  rower as ``boat.handle_watts`` -- the analogue of ``power_scales`` -- and is
  refused without it rather than handed a default nobody chose.
* :meth:`DynamicOarSimulator.run` takes the base class's arguments, resets
  the oars at every catch, and returns an ordinary ``SimulationResult`` of
  the twelve hull states, so nothing downstream has to know.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.core.state import STATE_SIZE, State
from coxswain.sim.dynamic_oar import (DynamicOarSimulator, _match_key,
                                      simulator_for)
from coxswain.sim.simulator import RowingSimulator


def _research(name="1x", rate=30.0, watts=380.0):
    boat = physics.resolve("research").apply(catalog.build(name, rate=rate))
    if watts is not None:
        boat.handle_watts = float(watts)
    return boat


@pytest.fixture(autouse=True)
def _torque_without_the_settle(request, monkeypatch):
    """Plant a torque for the research boats this module builds.

    Since 2026-09-13 the research profile catches by the sweep, and
    ``simulator_for`` matches its torque on a settle -- 55 s for the single,
    paid by whichever test here builds a simulator first.  These tests are
    about how the simulator is built and run, not about the torque, so the
    closed form is planted for each boat, which is the torque this module ran
    at before the switch.  The planted entries are removed after each test.
    The settle itself is tested in ``tests/test_torque_for_power.py``.
    """
    if request.node.get_closest_marker("slow"):
        return
    for name, rate, watts in (("1x", 30.0, 380.0), ("4+", 32.0, 380.0)):
        boat = _research(name=name, rate=rate, watts=watts)
        monkeypatch.setitem(
            DynamicOarSimulator._MATCHED,
            _match_key(boat, watts, "sweep", "slip"),
            DynamicOarSimulator.peak_torque_for_power(boat, watts))


# ---------------------------------------------------------------------------
# the factory
# ---------------------------------------------------------------------------
def test_the_factory_builds_what_the_boats_physics_needs():
    shipped = physics.resolve(physics.SHIPPED).apply(
        catalog.build("1x", rate=30.0))
    assert type(simulator_for(shipped)) is RowingSimulator

    research = _research()
    sim = simulator_for(research)
    assert isinstance(sim, DynamicOarSimulator)
    # Since 2026-09-13 the research profile catches by the sweep, whose rower
    # work includes the energy carried into the water, so its torque is
    # matched on a settle rather than the closed form (tests/
    # test_torque_for_power.py).
    assert sim.catch == "sweep"
    assert sim.peak_torque == pytest.approx(
        DynamicOarSimulator.torque_for_power(research, 380.0, catch="sweep"))


def test_a_dynamic_boat_without_stated_watts_is_refused():
    """No default wattage. A default here would be a number nobody chose,
    and every speed downstream of it would inherit that."""
    with pytest.raises(ValueError, match="handle_watts"):
        simulator_for(_research(watts=None))


def test_the_factory_passes_the_simulator_options_through():
    from coxswain.sim.control import Coxswain

    cox = Coxswain(rudder_override=lambda t, s: 0.0)
    sim = simulator_for(_research(), coxswain=cox, fast=True)
    assert sim.coxswain is cox


# ---------------------------------------------------------------------------
# run(), the way consumers call it
# ---------------------------------------------------------------------------
def test_run_returns_an_ordinary_result():
    boat = _research()
    sim = simulator_for(boat)
    period = float(boat.timing.period)
    result = sim.run(duration=2.0 * period, surge_speed=4.2)

    assert result.states.shape[0] == STATE_SIZE
    assert np.all(np.diff(result.time) > 0.0), "no duplicated stroke seams"
    assert result.time[0] == pytest.approx(0.0)
    assert result.time[-1] == pytest.approx(2.0 * period)
    assert result.mean_speed(1) > 3.0


def test_run_agrees_with_run_strokes_to_the_bit():
    """Same arithmetic, same step, same resets -- so the same boat."""
    boat = _research()
    torque = DynamicOarSimulator.peak_torque_for_power(boat, 380.0)
    period = float(boat.timing.period)

    by_run = DynamicOarSimulator(boat, peak_torque=torque).run(
        duration=2.0 * period, surge_speed=4.2)
    by_strokes = DynamicOarSimulator(boat, peak_torque=torque).run_strokes(
        2, surge_speed=4.2)

    final_run = float(np.hypot(*np.asarray(by_run.velocity)[:2, -1]))
    final_strokes = float(by_strokes.last_speed[-1])
    assert final_run == pytest.approx(final_strokes, rel=0.0, abs=1e-12)


def test_run_starts_from_a_positioned_and_headed_hull_state():
    """``steer`` places the boat on the path and points it down it."""
    boat = _research()
    sim = simulator_for(boat)
    state = sim.initial_state(surge_speed=4.0)
    state[0], state[1], state[5] = 120.0, -30.0, 0.3
    state[6], state[7] = 4.0 * np.cos(0.3), 4.0 * np.sin(0.3)

    result = sim.run(duration=float(boat.timing.period), initial_state=state)
    assert result.position[0, 0] == pytest.approx(120.0)
    assert result.position[1, 0] == pytest.approx(-30.0)
    # And it went roughly the way it was pointed.
    travel = np.array([result.position[0, -1] - 120.0,
                       result.position[1, -1] + 30.0])
    assert np.degrees(abs(np.arctan2(travel[1], travel[0]) - 0.3)) < 5.0


def test_run_calls_on_stroke_once_per_stroke():
    boat = _research()
    sim = simulator_for(boat)
    seen = []
    sim.run(duration=2.0 * float(boat.timing.period), surge_speed=4.2,
            on_stroke=lambda index, b: seen.append(index))
    assert seen == [0, 1]


def test_run_refuses_adaptive_stepping():
    """The oar resets at every catch and the blade switches out at the
    finish angle; both need the fixed-step path."""
    sim = simulator_for(_research())
    with pytest.raises(ValueError, match="fixed"):
        sim.run(duration=1.0, method="adaptive")


def test_run_refuses_a_state_it_cannot_read():
    sim = simulator_for(_research())
    with pytest.raises(ValueError):
        sim.run(duration=1.0, initial_state=np.zeros(5))


# ---------------------------------------------------------------------------
# the latent bug the port turned up
# ---------------------------------------------------------------------------
def test_the_pressure_split_is_read_at_the_real_time():
    """``_torque`` used to ask the coxswain for the split at t = 0.

    Harmless while every split in the project was a constant, and wrong the
    moment a cox changes the call mid-piece -- which is exactly what a
    steering controller does.
    """
    boat = _research(name="4+", rate=32.0)
    sim = DynamicOarSimulator(
        boat, peak_torque=DynamicOarSimulator.peak_torque_for_power(boat,
                                                                    380.0))

    class _Cox:
        def split(self, t, state):
            return 0.0 if t < 1.0 else 0.5

        def side_gain(self, split, side):
            return 1.0 + split * side

    sim.coxswain = _Cox()
    state = State.from_vector(np.zeros(STATE_SIZE))
    angle = np.radians(10.0)
    early = sim._torque(0, angle, state, 0.5)
    late = sim._torque(0, angle, state, 1.5)
    assert early != pytest.approx(late)
