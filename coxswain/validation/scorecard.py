"""Run a physics profile against the battery and report one table.

The measurements
----------------
Everything here comes from settling the boat at several power levels and
reading the steady state, because that is the only operating point at which
the identity the whole programme turns on holds:

    R(v) v = eta P

Four operating points give four ``(v, eta)`` pairs, and the *shape* of that
relation is the diagnostic.  With no velocity term in the blade force, ``R``
is set by the force rather than by the speed, so ``eta`` comes out
proportional to ``v`` -- a straight line through the origin.  A blade with
real slip physics has no reason to pass through the origin, so the fitted
intercept is a direct, falsifiable score for the defect.

One settle per operating point serves every ready target, so a profile costs
four integrations per boat rather than four per target.

Provenance
----------
``efficiency_at`` was promoted here from ``tests/test_blade_velocity.py``,
which measured it first and still imports it.  One definition, one place: a
validation harness that measures a quantity slightly differently from the
test that found it is worse than no harness, because the two will drift and
nobody will know which is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .. import physics
from .targets import Target, TARGETS

__all__ = ["Score", "Settled", "settle", "settle_dynamic", "efficiency_at",
           "measure", "run", "table", "OPERATING_POINTS", "DYNAMIC_POINTS"]


#: ``(power_scale, starting_speed)`` pairs spanning the racing range.
#:
#: Four points, because two can only ever fit a line and the question here
#: is whether the relation IS a line.  The starting speeds are rough
#: guesses at where each scale settles; they only shorten the transient,
#: since the average is taken over the last few cycles of a 70 s run.
OPERATING_POINTS = ((0.25, 2.6), (0.45, 3.4), (0.70, 4.4), (0.95, 5.4))

#: Long enough for the transient to leave and for the average to be taken
#: over whole stroke cycles. Matches what docs/TRACKING.md was measured with,
#: so the numbers here are comparable to the ones recorded there.
SETTLE_DURATION = 70.0
SETTLE_DT = 0.01

#: ``(watts_per_rower, starting_speed)`` for a profile whose oar angle is a
#: dynamic state.
#:
#: Stated in watts, not as a power scale, and deliberately.  A power scale
#: is a multiplier on the prescribed handle force, and the only conversion
#: to watts in the project is ``mean_handle_power``, which dots the oarlock
#: force with the handle velocity -- not a conjugate pair under the ideal
#: lever, and listed as an open question in docs/PHYSICS_PROGRAMME.md.  The
#: dynamic oar's power is closed-form and measured, so importing that
#: conversion into it would carry a known-bad number into the new physics.
#: The span matches the speed range of the power-scale points.
DYNAMIC_POINTS = ((80.0, 3.0), (180.0, 4.2), (260.0, 4.8), (360.0, 5.5))

#: Strokes to settle a dynamic-oar run.  Sixteen leaves the stroke-to-stroke
#: drift under 1% on the eight across the whole range of DYNAMIC_POINTS.
SETTLE_STROKES = 16


@dataclass(frozen=True)
class Settled:
    """One boat, run to steady state at one power level."""

    #: The operating point: a power scale for a prescribed-oar profile, and
    #: WATTS PER ROWER for a dynamic-oar one (see ``DYNAMIC_POINTS``).
    scale: float
    #: Mean speed over the last few whole cycles, m/s.
    speed: float
    #: Peak-to-peak surge over those cycles, as a fraction of the mean.
    surge_swing: float
    #: Total crew power at the handle, W.
    crew_power: float
    #: Hull resistance power at the settled speed, W.
    drag_power: float
    #: Force-weighted blade efficiency MEASURED on a dynamic-oar run; ``None``
    #: for a prescribed-oar run, whose level is computed from the schedule.
    blade_efficiency: Optional[float] = None

    @property
    def efficiency(self) -> float:
        """``R(v) v / P`` -- the fraction of handle power that moves the boat."""
        return self.drag_power / max(self.crew_power, 1e-9)

    @property
    def speed_per_watt(self) -> float:
        """m/s per 100 W of TOTAL crew power.

        Total and not per-rower, so the figure is comparable between boat
        classes: an eight and a four that convert power equally well
        should read the same here, which is the comparison worth making.
        The denominator that must not be lost -- see the target's note.
        """
        return self.speed / max(self.crew_power, 1e-9) * 100.0


@dataclass(frozen=True)
class Score:
    """One target, measured against one profile."""

    target: Target
    profile: str
    boat: str
    #: ``None`` when the target could not be run.
    value: Optional[float]
    status: str          # "pass", "fail", "pending", "n/a"
    detail: str = ""

    @property
    def mark(self) -> str:
        return {"pass": "ok", "fail": "FAIL",
                "pending": "--", "n/a": "n/a"}[self.status]


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def settle(boat, scale: float, start: float,
           duration: float = SETTLE_DURATION,
           dt: float = SETTLE_DT) -> Settled:
    """Run ``boat`` at ``scale`` until it settles, and read the steady state.

    The boat is driven dead straight -- the rudder is overridden to zero --
    because a scorecard that lets the coxswain controller intervene is
    measuring the controller as much as the physics.
    """
    from ..crew.exertion import mean_handle_power
    from ..sim.control import Coxswain
    from ..sim.simulator import RowingSimulator

    if _uses_dynamic_oar(boat):
        raise ValueError(
            "settle() drives the prescribed oar through power_scales; this "
            "boat is stamped %r, whose oar angle is a dynamic state -- use "
            "settle_dynamic(), which takes a stated wattage"
            % getattr(boat, "physics_profile", None))

    boat.power_scales = np.full(boat.n_seats, float(scale))
    sim = RowingSimulator(
        boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    result = sim.run(duration=duration, dt=dt, surge_speed=start)

    time_s = np.asarray(result.time)
    speed = np.hypot(*np.asarray(result.velocity)[:2])
    tail = time_s > duration - 4 * boat.timing.period
    mean_speed = float(speed[tail].mean())
    swing = float(np.ptp(speed[tail]) / max(mean_speed, 1e-9))

    crew = mean_handle_power(boat, samples=360) * float(scale) * boat.n_seats
    drag = _drag_at(sim, boat, mean_speed)

    return Settled(scale=float(scale), speed=mean_speed, surge_swing=swing,
                   crew_power=crew, drag_power=drag * mean_speed)


def _uses_dynamic_oar(boat) -> bool:
    """Whether the profile stamped on ``boat`` makes the oar angle a state."""
    name = getattr(boat, "physics_profile", None)
    return name is not None and physics.resolve(name).uses_dynamic_oar


def _drag_at(sim, boat, speed: float) -> float:
    """Hull resistance at ``speed``, computed exactly as the simulator does.

    Same submerged properties, same coefficients, same wave table -- so the
    two cannot disagree about what the drag is.  Shared by both settles so
    the prescribed and dynamic scorecards price drag identically.
    """
    from ..core.state import State
    from ..hydro.resistance import hull_resistance

    y = sim.initial_state(surge_speed=speed)
    props = boat.mesh.submerged(np.array([0.0, 0.0, float(y[2])]),
                                np.asarray(y[3:6], dtype=float),
                                rho=boat.water.density, gravity=9.81)
    res = hull_resistance(State.from_vector(y).velocity_hull, props,
                          boat.length, boat.water, boat.resistance,
                          getattr(boat, "shallow", None),
                          wave_table=getattr(boat, "wave_table", None))
    force = res[0] if isinstance(res, tuple) else res
    return abs(float(np.asarray(force)[0]))


def settle_dynamic(boat, watts: float, start: float,
                   strokes: int = SETTLE_STROKES) -> Settled:
    """Settle a dynamic-oar boat at a stated handle power per rower.

    Driven dead straight, as :func:`settle` is.  ``power_scales`` is set to
    ones so the stated wattage is not silently rescaled per seat, and the
    torque that delivers it is the closed form -- work per drive is
    ``peak * integral(shape dphi)`` whatever the speed.  Crew power is the
    measured handle power of the run itself, not the stated figure, so a
    mismatch between the two would show in the table rather than be
    assumed away.
    """
    from ..sim.control import Coxswain
    from ..sim.dynamic_oar import DynamicOarSimulator

    boat.power_scales = np.ones(boat.n_seats)
    torque = DynamicOarSimulator.peak_torque_for_power(boat, float(watts))
    sim = DynamicOarSimulator(
        boat, peak_torque=torque,
        coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    run = sim.run_strokes(int(strokes), surge_speed=float(start))

    mean_speed = run.settled_speed()
    swing = float(np.mean([each.surge_swing for each in run.strokes[-4:]]))
    rowers = sum(1 for seat in boat.rig.seats if seat.oarlocks)
    crew = run.settled_power() * rowers
    drag = _drag_at(sim, boat, mean_speed)
    return Settled(scale=float(watts), speed=mean_speed, surge_swing=swing,
                   crew_power=crew, drag_power=drag * mean_speed,
                   blade_efficiency=run.settled_blade_efficiency())


def efficiency_at(boat, scale: float, start: float):
    """``(speed, efficiency)`` at one power level.

    Kept as a two-tuple because that is the shape
    ``tests/test_blade_velocity.py`` has always used it in, and the strict
    xfail there is the tripwire that announces the defect being fixed.
    """
    got = settle(boat, scale, start)
    return got.speed, got.efficiency


def sweep(boat, points: Optional[Sequence] = None):
    """Settle ``boat`` at each operating point, by the physics it carries.

    A boat stamped with a dynamic-oar profile is settled at stated
    wattages (``DYNAMIC_POINTS``); anything else at power scales
    (``OPERATING_POINTS``), exactly as before.
    """
    if _uses_dynamic_oar(boat):
        chosen = DYNAMIC_POINTS if points is None else points
        return [settle_dynamic(boat, watts, start) for watts, start in chosen]
    chosen = OPERATING_POINTS if points is None else points
    return [settle(boat, scale, start) for scale, start in chosen]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _fit_efficiency(runs):
    """``(slope, crossing, spread)`` of efficiency against speed.

    ``crossing`` is **where the fitted line reaches zero efficiency, as a
    multiple of the mean speed** -- not the raw y-intercept.

    That choice matters and is worth the paragraph.  "Does the line pass
    through the origin?" is the question, and the y-intercept answers it
    only against some scale you have to supply from outside: -0.008 is
    near zero compared with an efficiency of 0.4, and enormous compared
    with one of 0.001.  The speed at which the line crosses zero is in
    metres per second, so it can be compared against the boat's own speed
    with nothing else brought in.  A line through the origin crosses at
    zero; a blade with real physics crosses far outside the racing range,
    usually at a large negative speed, which is another way of saying it
    never crosses at all.

    ``spread`` is the range of ``eta / v`` as a fraction of its mean: if
    that is a couple of percent the relation is proportional whatever any
    fit says, which makes it the more robust of the two statistics and
    the reason both are carried.
    """
    speeds = np.array([r.speed for r in runs], dtype=float)
    etas = np.array([r.efficiency for r in runs], dtype=float)
    slope, intercept = np.polyfit(speeds, etas, 1)
    mean_speed = float(speeds.mean())
    if abs(slope) < 1e-12:
        crossing = float("inf")      # flat: never reaches zero
    else:
        crossing = abs(-intercept / slope) / max(mean_speed, 1e-9)
    ratio = etas / np.maximum(speeds, 1e-9)
    spread = float(np.ptp(ratio) / max(abs(ratio.mean()), 1e-12))
    return float(slope), float(crossing), spread


def _blade_efficiency_level(boat, speed: float) -> Optional[float]:
    """Force-weighted blade efficiency over the drive, or ``None``.

    Weighted by oar force rather than by time, because the efficiency of
    an instant where the blade is barely loaded does not matter and an
    unweighted mean would let the ends of the drive dominate.

    ``speed`` is the boat's settled speed: the whole reason a blade model
    is worth having is that efficiency depends on it, so it is a required
    argument and not a default that could quietly be wrong.

    ``None`` without a blade model: below tier 1 the efficiency simply
    *is* the oar's lumped constant, so reporting it would be checking a
    constant against itself.
    """
    blade = getattr(boat, "blade_model", None)
    if blade is None:
        return None
    timing = boat.timing
    t = np.linspace(0.0, timing.drive_fraction * timing.period, 400)
    angle = np.asarray(boat.oar_sweep(t, timing), dtype=float)
    rate = np.asarray(boat.oar_sweep.rate(t, timing), dtype=float)
    weight = np.asarray(boat.force_profile.magnitude(t, timing), dtype=float)
    eff = np.asarray(blade.efficiency(angle, rate, float(speed)), dtype=float)
    total = float(weight.sum())
    if total <= 0.0:
        return None
    return float((eff * weight).sum() / total)


def measure(boat, name: str, profile, runs=None):
    """Score every target that can be run against this boat."""
    profile = physics.resolve(profile)
    if runs is None:
        runs = sweep(boat)
    slope, crossing, spread = _fit_efficiency(runs)
    # Single-number targets are quoted at the FASTEST operating point,
    # because that is nearest racing and racing is where the published
    # measurements were taken.  It matters: surge swing is strongly
    # speed-dependent -- 77.4% at 2.81 m/s against 37.6% at 5.93 on the
    # eight -- so a target quoted at an unstated speed is not a target.
    middle = max(runs, key=lambda r: r.speed)

    values = {
        "blade_efficiency_zero_crossing": crossing,
        "blade_efficiency_linearity": spread,
        "speed_per_watt": middle.speed_per_watt,
        "surge_swing": 100.0 * middle.surge_swing,
        # A dynamic-oar boat carries no ``blade_model`` to read the level
        # from, and should not: its efficiency is a property of the run, not
        # of a schedule.  So it is MEASURED there, on the states the boat
        # actually had, and scored against the same band.
        "blade_efficiency_level": (
            middle.blade_efficiency if _uses_dynamic_oar(boat)
            else _blade_efficiency_level(boat, middle.speed)),
    }
    details = {
        "blade_efficiency_zero_crossing":
            "eta reaches 0 at %.3f x mean speed, over %.2f-%.2f m/s"
            % (crossing, min(r.speed for r in runs),
               max(r.speed for r in runs)),
        "blade_efficiency_linearity":
            "eta/v = %.4f +/- %.1f%%"
            % (np.mean([r.efficiency / r.speed for r in runs]),
               50.0 * spread),
        "speed_per_watt":
            "%.2f m/s at %.0f W per rower"
            % (middle.speed, middle.crew_power / boat.n_seats),
        "surge_swing": "at %.2f m/s, rate %.1f" % (middle.speed,
                                                   boat.timing.rate),
        "blade_efficiency_level": (
            "force-weighted over the drive, measured on the run, at %.2f m/s"
            if _uses_dynamic_oar(boat)
            else "force-weighted, at %.2f m/s") % middle.speed,
    }

    scores = []
    for target in TARGETS:
        if not target.implemented:
            scores.append(Score(target, profile.name, name, None,
                                "pending", target.note.split(".")[0]))
            continue
        if profile.blade_tier < target.min_blade_tier:
            scores.append(Score(
                target, profile.name, name, None, "n/a",
                "needs blade tier %d; this profile is tier %d"
                % (target.min_blade_tier, profile.blade_tier)))
            continue
        value = values.get(target.key)
        if value is None:
            scores.append(Score(target, profile.name, name, None, "n/a",
                                "no measurement on this profile"))
            continue
        low, high = target.band
        status = "pass" if low <= value <= high else "fail"
        scores.append(Score(target, profile.name, name, float(value),
                            status, details.get(target.key, "")))
    return scores


def run(profile=physics.SHIPPED, boats=("8+", "4+"), rate=28.0):
    """Score ``profile`` on every boat named, and return a flat list.

    This is the whole point of the module: the same call, the same
    targets, any profile. A change that improves one number while wrecking
    another cannot pass unnoticed, because both are in the table.
    """
    from ..boats import catalog

    resolved = physics.resolve(profile)
    scores = []
    for name in boats:
        boat = resolved.apply(catalog.build(name, rate=rate))
        scores.extend(measure(boat, name, resolved))
    return scores


def table(scores) -> str:
    """The scorecard as plain text, widest column first."""
    rows = [("", "boat", "target", "measured", "band", "note")]
    for score in scores:
        band = ("%.3g - %.3g" % score.target.band
                if score.target.band else "-")
        value = "-" if score.value is None else "%.4g" % score.value
        rows.append((score.mark, score.boat, score.target.key, value,
                     band, score.detail))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    for index, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i])
                             for i, cell in enumerate(row)).rstrip())
        if index == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)
