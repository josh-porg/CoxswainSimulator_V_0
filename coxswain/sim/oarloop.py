r"""Surge and the oar angle, coupled -- the smallest model that closes the loop.

Why a reduced model at all
--------------------------
The question phase 2 has to answer is whether making the oar angle a state
removes the defect: with the blade force independent of speed, the steady
balance ``R(v) v = eta P`` forces ``eta`` proportional to ``v``, a straight
line through the origin.  Answering it does not need the 6-DOF hull, the
river, or the crew's joint chain.  It needs thrust that depends on boat
speed, and a hull that slows down.

So this integrates exactly two things::

    m dv/dt      = n_oars * F_x(phi, phi_dot, v) - R(v)
    I(phi) phi_ddot + (1/2)(dI/dphi) phi_dot^2
                 = tau_handle(t) + l F_n(phi, phi_dot, v)

over repeated stroke cycles, until the speed settles.  Building it before
wiring :class:`~coxswain.crew.oardynamics.OarDynamics` into the simulator
is the same discipline that caught the efficiency-only wiring: measure the
thing you are about to commit to, on the smallest model that can show it.

What it deliberately leaves out, and why that matters
-----------------------------------------------------
**The crew's mass does not move.**  In the full model the crew surging
fore and aft swings the hull's speed by 56% peak to peak, and it was
exactly that swing which destroyed the efficiency-only wiring: the hull's
minimum coincides with peak oar force, so the blade was at its worst
precisely when it was loaded hardest.

This model has no such swing.  It is therefore a **best case**, and a
result here is necessary but not sufficient: it can show that the shape of
``eta(v)`` is fixed, and it cannot show that the coupled model is stable
once the surge variation is back.  That is what the full wiring is for.
The limitation is stated here rather than discovered later.

The recovery
------------
The blade is out of the water, so there is no blade force and no thrust:
the hull simply decelerates under drag.  The oar returns to the catch on a
raised cosine, because out of the water the rower *is* in control of it and
prescribing the return is not the approximation that prescribing the drive
was.

The cycle period is set by the stroke rate, and the drive takes as long as
the balance says.  So the drive FRACTION is an output, and a pull hard
enough to need longer than the whole cycle is reported rather than
silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["CoupledRun", "drag_curve", "settle_coupled", "torque_shape",
           "settle_at_power"]


def torque_shape(boat, samples: int = 201):
    """The rower's pull through the drive, as a function of oar ANGLE.

    Reuses the fitted Kleshnev curve already in
    :class:`~coxswain.crew.oarlock.OarForceProfile` -- front-loaded, peak
    at 38% of the drive, down to 10% of peak by 88% -- rather than the
    constant pull a first pass used.  The difference is not cosmetic: a
    constant torque accelerates the oar all the way to the finish, so the
    sweep rate peaks where the blade should be easing off, and the blade
    is driven hard through the water exactly where it has least grip.

    Parameterised by **angle progress**, ``u = (catch - phi)/(catch -
    finish)``, and not by time.  Two reasons, one practical and one
    physical.  Practically, the drive duration is now an output, so a
    curve in time would need the answer before it could be evaluated.
    Physically, force curves are reported against oar angle or drive
    length far more often than against the clock, because that is the
    coordinate the rower actually works in.

    Returns a callable of ``phi`` giving a factor in ``[0, 1]``.
    """
    timing = boat.timing
    span = float(boat.oar_sweep.catch_angle - boat.oar_sweep.finish_angle)
    catch = float(boat.oar_sweep.catch_angle)

    progress = np.linspace(0.0, 1.0, int(samples))
    curve = np.asarray(
        boat.force_profile.magnitude(progress * timing.drive_duration, timing),
        dtype=float)
    curve = np.clip(curve, 0.0, None)

    def shape(angle):
        u = (catch - float(angle)) / span
        return float(np.interp(u, progress, curve, left=0.0, right=0.0))

    return shape


def drag_curve(boat, speeds=None):
    """Hull resistance against speed, as an interpolating callable.

    Computed with the same :func:`~coxswain.hydro.resistance.hull_resistance`
    call the simulator and the scorecard use -- same submerged properties,
    same coefficients, same wave table -- so the three cannot disagree
    about what the drag is.  Tabulated rather than evaluated inside the
    loop because it is smooth in speed and the loop runs thousands of
    steps.
    """
    from ..core.state import State
    from ..hydro.resistance import hull_resistance
    from .simulator import RowingSimulator

    if speeds is None:
        speeds = np.linspace(0.2, 8.0, 40)
    speeds = np.asarray(speeds, dtype=float)

    sim = RowingSimulator(boat, fast=True)
    drag = []
    for speed in speeds:
        y = sim.initial_state(surge_speed=float(speed))
        props = boat.mesh.submerged(np.array([0.0, 0.0, float(y[2])]),
                                    np.asarray(y[3:6], dtype=float),
                                    rho=boat.water.density, gravity=9.81)
        result = hull_resistance(State.from_vector(y).velocity_hull, props,
                                 boat.length, boat.water, boat.resistance,
                                 getattr(boat, "shallow", None),
                                 wave_table=getattr(boat, "wave_table", None))
        force = result[0] if isinstance(result, tuple) else result
        drag.append(abs(float(np.asarray(force)[0])))

    table = np.asarray(drag, dtype=float)

    def resistance(speed):
        return float(np.interp(abs(float(speed)), speeds, table))

    return resistance


@dataclass(frozen=True)
class CoupledRun:
    """A settled run of the reduced model."""

    speed: float                  # mean over the last whole cycles, m/s
    drive_duration: float         # s, an output
    drive_fraction: float
    handle_power: float           # W per rower, mean over the cycle
    thrust_impulse: float         # N s per oar per drive
    cycles: int
    settled: bool                 # speed changed little over the last cycles

    @property
    def efficiency(self) -> float:
        """``R(v) v / P``, the same quantity the scorecard measures."""
        return self._efficiency

    _efficiency: float = 0.0


def settle_coupled(boat, handle_torque: float, oar=None, resistance=None,
                   cycles: int = 40, dt: float = 0.002,
                   start_speed: float = 4.0,
                   water_depth: Optional[float] = None,
                   shape=None) -> CoupledRun:
    """Run the reduced model to steady state at a fixed handle torque.

    ``handle_torque`` is the PEAK magnitude in N m; it is multiplied by
    ``shape(phi)`` and applied as a negative torque through the drive,
    driving ``phi`` down.  ``shape`` defaults to :func:`torque_shape`, the
    measured front-loaded curve; pass ``lambda _phi: 1.0`` for a constant
    pull, which is a study and not a model of a rower.
    """
    from ..crew.oardynamics import OarDynamics

    if oar is None:
        oar = OarDynamics.from_boat(boat)
    if resistance is None:
        resistance = drag_curve(boat)
    if shape is None:
        shape = torque_shape(boat)

    period = float(boat.timing.period)
    n_oars = sum(len(seat.oarlocks) for seat in boat.rig.seats)
    mass = float(boat.total_mass)

    speed = float(start_speed)
    per_cycle = []

    for _cycle in range(int(cycles)):
        angle, rate = float(oar.catch_angle), 0.0
        t = 0.0
        work, impulse = 0.0, 0.0
        speeds = []
        drive_end = None

        while t < period:
            on_drive = drive_end is None
            if on_drive:
                torque = handle_torque * shape(angle)
                accel = float(oar.acceleration(angle, rate, -torque,
                                               speed, water_depth))
                force = float(oar.blade.normal_force(angle, rate, speed,
                                                     water_depth))
                # NO gearing factor.  For the hull-plus-crew system the
                # only external horizontal forces are the blade force and
                # the drag: the rower's pull on the handle, the handle's
                # push back, and the stretcher reaction are all internal.
                # The lever ratio decides how hard the rower must pull for
                # a given blade force, not how much of that force reaches
                # the boat.  Applying `gearing` here as well -- which a
                # first pass did, by analogy with `hull_load` -- charges
                # the same lever twice and cost a factor of 3.2 in
                # efficiency.
                thrust = n_oars * force * np.cos(angle)
                work += abs(torque * rate) * dt
                impulse += force * np.cos(angle) * dt
            else:
                accel, thrust = 0.0, 0.0

            speed = speed + dt * (thrust - resistance(speed)) / mass
            speeds.append(speed)

            if on_drive:
                angle = angle + dt * rate
                rate = rate + dt * accel
                if angle <= oar.finish_angle:
                    drive_end = t + dt
            t += dt

        if drive_end is None:
            drive_end = period      # the pull could not finish the drive
        # Power PER ROWER, not per oar: a sculler drives two.  Counting
        # one oar's work while collecting two oars' thrust flattered the
        # single by a factor of two.
        per_rower = work * n_oars / max(boat.n_seats, 1)
        per_cycle.append((float(np.mean(speeds)), drive_end,
                          per_rower / period, impulse))

    tail = per_cycle[-4:]
    mean_speed = float(np.mean([each[0] for each in tail]))
    drive = float(np.mean([each[1] for each in tail]))
    power = float(np.mean([each[2] for each in tail]))
    impulse = float(np.mean([each[3] for each in tail]))
    drift = abs(tail[-1][0] - tail[0][0]) / max(mean_speed, 1e-9)

    return CoupledRun(
        speed=mean_speed,
        drive_duration=drive,
        drive_fraction=drive / period,
        handle_power=power,
        thrust_impulse=impulse,
        cycles=int(cycles),
        settled=bool(drift < 0.005),
        _efficiency=(resistance(mean_speed) * mean_speed
                     / max(power * boat.n_seats, 1e-9)),
    )


def settle_at_power(boat, watts: float, oar=None, resistance=None,
                    low: float = 20.0, high: float = 4000.0,
                    tolerance: float = 0.005, iterations: int = 30,
                    **kwargs) -> CoupledRun:
    """Settle the reduced model at a stated handle power per rower.

    The natural input to the model is a torque, and the natural input to a
    *comparison* is a power: published race paces belong to crews rowing
    at some wattage, and driving a boat at whatever ``power_scales = 1.0``
    happens to mean is how the regression suite ended up asserting that
    every boat beats race pace (see ``tests/regression/``).

    Bisection, because handle power is monotone in the pull and the model
    is cheap -- about thirty settles, a second or so.
    """
    from ..crew.oardynamics import OarDynamics

    if oar is None:
        oar = OarDynamics.from_boat(boat)
    if resistance is None:
        resistance = drag_curve(boat)

    target = float(watts)
    run = None
    for _step in range(int(iterations)):
        middle = 0.5 * (low + high)
        run = settle_coupled(boat, middle, oar=oar, resistance=resistance,
                             **kwargs)
        if abs(run.handle_power - target) <= tolerance * target:
            return run
        if run.handle_power < target:
            low = middle
        else:
            high = middle
    return run
