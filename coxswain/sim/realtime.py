r"""Driving the boat in real time, from a person instead of a controller.

The batch world and the interactive one differ in exactly two places:
**who owns the clock**, and **where the steering comes from**.  Everything
else -- the hull, the crew, the water, the wind -- is the same code, and
that is the point.  A training tool that disagreed with the analysis
would be teaching the wrong boat.

The clock
---------
:class:`FixedStepLoop` runs the physics at a fixed rate and lets the
frame rate be whatever the machine manages.  A game loop that integrates
by the frame time makes the boat behave differently on a fast machine
than a slow one, and on this project it would also make the results
irreproducible, which would cost the thing the analysis is for.  Fixed
steps, an accumulator, and an interpolated pose for the renderer
(Gafferon-Games' "fix your timestep", and the reason your replay is exact).

The steering
------------
:class:`~coxswain.sim.control.Coxswain` already takes a
``rudder_override``, a callable ``(t, state) -> float``, because that is
how :class:`~coxswain.sim.guidance.PathFollower` and the MPC drive the
boat.  A person is just another one of those.  :class:`LiveControl` is a
mailbox: the input thread writes a :class:`ControlInput` into it, the
force path reads it, and neither knows about the other.

So the same simulator is driven by a path follower for a study, by the
MPC for the optimised line, and by a hand on a tiller for training --
and a recorded human run replays through the analysis unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = ["ControlInput", "LiveControl", "FixedStepLoop", "interpolate"]


@dataclass(frozen=True)
class ControlInput:
    """What a coxswain asks of the boat at one instant.

    Deliberately small and plain: this is the thing a keyboard, a
    gamepad axis, a replay file and an autopilot all produce, and making
    it a frozen record rather than a bundle of setters is what lets a
    session be recorded as a stream of these and replayed exactly.
    """

    #: Rudder angle, radians.  Positive turns the bow to port, matching
    #: :meth:`coxswain.sim.control.Coxswain.rudder`.
    rudder: float = 0.0
    #: Port/starboard pressure split in ``[-1, 1]``; positive means the
    #: port side pulls harder, which yaws the bow to starboard.
    pressure_split: float = 0.0
    #: Stroke rate the crew is asked for, in strokes per minute.  ``None``
    #: leaves the boat's own rate alone.
    rate: Optional[float] = None


class LiveControl:
    """A mailbox between whoever is steering and the force path.

    Plugs into :class:`~coxswain.sim.control.Coxswain` as its
    ``rudder_override`` and ``pressure_split``, so nothing in the
    dynamics changes or even notices::

        live = LiveControl()
        cox = Coxswain(rudder_override=live.rudder,
                       pressure_split=live.split)
        sim = RowingSimulator(boat, coxswain=cox, fast=True)
        ...
        live.set(ControlInput(rudder=0.05))   # from the input device

    ``fast=True`` puts the wetted-surface sweep on the compiled kernel,
    which is what a real-time loop wants and what the studies do not.

    Reads are cheap and allocation-free because the force path makes
    them four times per step.
    """

    def __init__(self, initial: ControlInput = None):
        self._input = initial or ControlInput()

    @property
    def current(self) -> ControlInput:
        return self._input

    def set(self, control: ControlInput) -> None:
        self._input = control

    # -- the two callables Coxswain expects -------------------------------
    def rudder(self, t: float, state) -> float:
        return self._input.rudder

    def split(self, t: float, state) -> float:
        return self._input.pressure_split


def interpolate(previous: np.ndarray, current: np.ndarray,
                alpha: float) -> np.ndarray:
    """Blend two states for drawing, ``alpha`` in ``[0, 1]``.

    Linear on every component including the Euler angles.  That is right
    for a boat: roll, pitch and yaw stay far from the wrap at
    :math:`\\pm\\pi` over one physics step, and a slerp here would cost
    more than it buys.  If a course is ever added that turns through
    north, this is the function to revisit.
    """
    return previous + (current - previous) * float(alpha)


@dataclass
class FixedStepLoop:
    """Physics at a fixed rate, rendering at whatever rate you get.

    ``rate`` is the physics rate in hertz.  ``max_frame`` clamps how much
    time a single frame may hand the accumulator: without it, one long
    stall (a garbage collection, a chunk load, the laptop throttling)
    asks for a hundred catch-up steps, which takes longer still, which
    asks for more -- the spiral of death.  Dropping simulated time is the
    right answer; falling further behind for ever is not.
    """

    simulator: object
    #: Called once per fixed step as ``on_step(t, state, dt)``, for
    #: state that advances with the physics but cannot live inside the
    #: derivative.  See the note in :meth:`advance`.
    on_step: object = None
    #: Physics rate, hertz.  100 Hz costs about a third of one core with
    #: the compiled kernel on; drop it to 60 on a weaker machine before
    #: dropping fidelity anywhere else.
    rate: float = 100.0
    max_frame: float = 0.25
    #: How many physics steps one frame may take before the rest of the
    #: accumulated time is DROPPED.  Clamping ``max_frame`` alone is not
    #: a guard: at 60 Hz a quarter of a second is fifteen steps, each
    #: as slow as the step that put the machine behind in the first
    #: place, so a laptop that fell one frame behind would spend the
    #: next frame catching up and fall further behind doing it -- the
    #: spiral, merely bounded.  Four steps is 67 ms of simulated time
    #: at 60 Hz; beyond that the boat runs briefly slow rather than the
    #: program stopping, and :attr:`dropped` counts what was let go so
    #: the telemetry can say so.
    max_steps: int = 4
    #: Simulated seconds discarded by the cap, in total.
    dropped: float = 0.0
    #: Wall-clock source; swapped in tests for a deterministic one.
    clock: Callable[[], float] = time.perf_counter

    t: float = 0.0
    state: np.ndarray = None
    previous: np.ndarray = None
    steps: int = 0
    _accumulator: float = 0.0
    _last: float = None

    @property
    def dt(self) -> float:
        return 1.0 / float(self.rate)

    def start(self, state: np.ndarray, t: float = 0.0) -> None:
        self.state = np.asarray(state, dtype=float)
        self.previous = self.state.copy()
        self.t = float(t)
        self._accumulator = 0.0
        self._last = self.clock()

    def advance(self, elapsed: float = None) -> int:
        """Consume one frame's worth of time.  Returns steps taken.

        Pass ``elapsed`` to drive it deterministically; leave it out and
        it reads the wall clock.
        """
        if self.state is None:
            raise RuntimeError("call start() before advance()")
        if elapsed is None:
            now = self.clock()
            elapsed = now - self._last
            self._last = now
        self._accumulator += min(float(elapsed), self.max_frame)

        dt = self.dt
        taken = 0
        while self._accumulator >= dt:
            if taken >= int(self.max_steps):
                # Let the rest of the frame's time go.  See max_steps.
                self.dropped += self._accumulator
                self._accumulator = 0.0
                break
            self.previous = self.state
            self.state = self.simulator.step(self.state, self.t, dt)
            self.t += dt
            self._accumulator -= dt
            self.steps += 1
            taken += 1
            if self.on_step is not None:
                # Anything stateful that has to advance with the physics
                # and cannot live inside derivative().
                #
                # The derivative is called four times per step at three
                # different times by RK4, and must be a pure function of
                # (t, state) or the integration is not what it claims to
                # be.  Crew synchronisation is the case in point: the
                # phases are their own dynamical system, driven by the
                # hull, and they advance ONCE per step -- here.
                self.on_step(self.t, self.state, dt)
        return taken

    @property
    def alpha(self) -> float:
        """How far between :attr:`previous` and :attr:`state` we are."""
        return self._accumulator / self.dt

    def pose(self) -> np.ndarray:
        """The state to draw: interpolated, so motion is smooth even when
        the frame rate and the physics rate do not divide."""
        return interpolate(self.previous, self.state, self.alpha)


@dataclass
class Recording:
    """A session as its inputs, not its trajectory.

    The state is a deterministic function of the initial condition and
    the command stream, so storing the commands stores everything --
    a few kilobytes for a full race, exactly replayable, and replayable
    *through the analysis*: a human run becomes a `SimulationResult`
    like any other.  It is also how the physics gets a regression test
    with a person in the loop.
    """

    initial: np.ndarray
    dt: float
    inputs: List[Tuple[float, ControlInput]] = field(default_factory=list)

    def record(self, t: float, control: ControlInput) -> None:
        if not self.inputs or self.inputs[-1][1] != control:
            self.inputs.append((float(t), control))

    def at(self, t: float) -> ControlInput:
        """The command in force at time ``t`` (zero-order hold)."""
        held = ControlInput()
        for when, control in self.inputs:
            if when > t:
                break
            held = control
        return held

    def replay(self, simulator, live: LiveControl, duration: float):
        """Re-run the session, returning ``(times, states)``."""
        state = np.asarray(self.initial, dtype=float)
        times, states = [0.0], [state]
        t, steps = 0.0, int(round(duration / self.dt))
        for _ in range(steps):
            live.set(self.at(t))
            state = simulator.step(state, t, self.dt)
            t += self.dt
            times.append(t)
            states.append(state)
        return np.asarray(times), np.asarray(states).T
