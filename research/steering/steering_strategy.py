"""Constant rudder versus rudder modulated within the stroke.

The question, from a rower at practice: if the crew has a standing
port/starboard power imbalance, is it better to

* **A -- hold the average.** Set one rudder angle, leave it there, and let
  the boat yaw back and forth within each stroke so long as the heading
  averages out over several strokes.
* **B -- hold the heading.** Correct continuously, using a different rudder
  angle on the drive than on the recovery, so the boat points the same way
  all the time.

There is a real trade-off here, which is why it is worth simulating rather
than arguing about.

Against B: rudder drag goes as the *square* of deflection, so for a given
mean corrective moment, spreading the deflection evenly costs the least
drag.  Concentrating it into part of the cycle needs a bigger peak and
loses more.

For B: rudder force also goes as the square of the *water speed*, and hull
speed is far from constant through the stroke.  It is **slowest just after
the catch and fastest on the recovery** -- the crew's mass slides sternward
during the recovery and momentum pushes the hull forward.  This model puts
the minimum at phase 0.125 (3.48 m/s) and the maximum at phase 0.775
(6.19 m/s), which is the measured shape.

Rudder authority therefore goes as roughly ``(6.19/3.48)^2 = 3.2`` times
higher on the recovery than at the catch.  Steering on the recovery is
cheap; steering at the catch is expensive.  A controller that only holds a
constant angle cannot exploit that; one that modulates within the stroke
can buy its correction where it costs least.

Note this cuts against the intuition that you steer on the drive because
that is when the boat "has power on".  The rudder does not care about the
oars, only about the water going past it -- and there is most of that
during the recovery, which is also when the blades are out and cannot
fight the turn.

Also for B: a yawing boat is not going where it is pointing.  It carries
leeway, which adds induced drag, and its track through the water is longer
than the straight line.

The metric is what actually wins races: **distance made good along the
intended heading, per second**.  Mean speed alone would reward a boat that
sails fast in the wrong direction.
Results
-------
Eight at rate 32, straight target heading, 16 s, measured over the last
60%.  The integration noise floor was checked by halving the step: made
good moved by 1e-6 m/s, so differences above about 1e-5 m/s are real.

::

    split  strategy                made good  yaw swing  cross track
    0.06   A  constant rudder        5.16212     0.720       0.423
    0.06   B  hold heading           5.16304     0.495       0.944
    0.06   B+ hold heading, hard     5.16205     0.367       0.810
    0.06   C  hold the line          5.16118     1.494       0.701
    0.12   A  constant rudder        5.15791     1.181       0.834
    0.12   B  hold heading           5.15837     0.640       1.271
    0.12   B+ hold heading, hard     5.15668     0.469       1.061
    0.12   C  hold the line          5.15494     1.982       0.892

**On speed, it is a wash.**  The whole spread is under 0.06%, which over
the 4,800 m Charles course is well under a second.  B is nominally fastest
and B+ nominally slowest, in the direction the physics predicts -- mild
modulation buys correction cheaply on the recovery, hard modulation pays
the quadratic drag penalty -- but the differences are the same order as a
confound: the trimmed constant rudder still leaves 0.55-1.03 deg of
residual heading drift, and made good is measured along a fixed axis, so
some of A's deficit is simply that it is not quite pointing where it is
being measured.  The honest conclusion is **no measurable speed
difference**, not that any strategy wins.

**On position, the strategies genuinely differ, and not the way intuition
says.**  The constant rudder held the *straightest line* -- 0.42 m of
cross-track wander against 0.94 m for the heading-holder at the same
split.  Holding heading is not holding position: with a standing power
split the boat is pushed sideways as well as turned, so a controller that
nails the compass can still crab steadily off the line.

That is the finding worth taking to the boat.  In a Head race the thing
that costs you is being in the wrong place -- wide on a turn, or into
another crew -- not being a degree off the compass.  If the choice is
between holding the average heading and continuously correcting it,
neither is faster, and the constant angle is easier to hold a line with.

Caveats: this is a *straight* course with a *constant* split, over 16 s.
The Charles is a sequence of bends, and a bend changes the problem -- the
rudder has to produce a sustained turn rather than cancel a bias, and the
crew's power split becomes the primary control rather than a disturbance.
Whether the same answer holds through Weeks and Anderson is a separate
experiment, and the one that actually matters for the race.
"""

from __future__ import annotations

import numpy as np

from coxswain.boats import catalog
from coxswain.sim.control import Coxswain, HeadingController
from coxswain.sim.simulator import RowingSimulator

DURATION = 16.0
STEP = 0.008


def _measure(result, settle=0.4):
    """Distance made good, mean speed, and how much the heading wandered."""
    n = int(len(result.time) * settle)
    time = np.asarray(result.time)[n:]
    x = np.asarray(result.surge)[n:]
    y = np.asarray(result.sway)[n:]
    yaw = np.asarray(result.yaw)[n:]

    span = time[-1] - time[0]
    # made good along the intended heading, which is +x
    made_good = (x[-1] - x[0]) / span
    path = np.hypot(np.diff(x), np.diff(y)).sum() / span
    return {
        "made_good": made_good,
        "path_speed": path,
        "yaw_swing": np.degrees(yaw.max() - yaw.min()),
        "yaw_drift": np.degrees(yaw[-1] - yaw[0]),
        "cross_track": y.max() - y.min(),
    }


def run_constant(boat, split, deflection):
    cox = Coxswain(pressure_split=split,
                   rudder_override=lambda t, state: deflection)
    cox.heading = HeadingController(enabled=False)
    return RowingSimulator(boat, coxswain=cox).run(
        duration=DURATION, dt=STEP, surge_speed=4.6)


def run_tracking(boat, split, gain, rate_gain):
    """Strategy B: correct continuously.

    A PD loop on instantaneous heading.  It naturally produces different
    rudder on the drive than on the recovery, because the heading error it
    is chasing is itself different in the two phases -- which is exactly
    what a coxswain steering through the stroke is doing by hand.
    """
    cox = Coxswain(pressure_split=split)
    cox.heading = HeadingController(target=0.0, gain=gain,
                                    rate_gain=rate_gain, enabled=True)
    return RowingSimulator(boat, coxswain=cox).run(
        duration=DURATION, dt=STEP, surge_speed=4.6)


def run_cross_track(boat, split, position_gain=0.05, gain=6.0,
                    rate_gain=2.5):
    """Strategy C: steer the *line*, not the compass.

    Holding heading is not the same as holding position.  With a standing
    power split the boat is pushed sideways as well as turned, so a
    controller that nails the heading can still crab steadily off the line
    -- and what loses a Head race is being in the wrong place, not pointing
    the wrong way.

    This is a cascade: cross-track error commands a heading offset, and the
    heading loop chases that.  It is what a coxswain actually does when
    they pick a point and hold it.
    """
    cox = Coxswain(pressure_split=split)

    def rudder(t, state):
        cross = float(state.position[1])
        heading = float(state.yaw)
        yaw_rate = float(state.omega_hull[2])
        # Cross-track error commands a heading; the inner loop chases it.
        # Signs follow HeadingController: positive rudder yaws to starboard,
        # so a positive heading error calls for positive rudder.
        target = float(np.clip(-position_gain * cross,
                               -np.radians(8.0), np.radians(8.0)))
        demand = gain * (heading - target) + rate_gain * yaw_rate
        return float(np.clip(demand, -np.radians(25.0), np.radians(25.0)))

    cox.rudder_override = rudder
    cox.heading = HeadingController(enabled=False)
    return RowingSimulator(boat, coxswain=cox).run(
        duration=DURATION, dt=STEP, surge_speed=4.6)


def trim_constant(boat, split, probes=(-0.12, 0.0, 0.12)):
    """Constant rudder angle that leaves no net heading drift.

    Yaw drift is very nearly linear in rudder angle over this range, so a
    least-squares line through three probes beats bisecting with a full
    simulation per step.
    """
    drifts = [_measure(run_constant(boat, split, d))["yaw_drift"]
              for d in probes]
    slope, intercept = np.polyfit(probes, drifts, 1)
    return float(-intercept / slope)


def main():
    boat = catalog.eight(rate=32.0)
    split = 0.06          # starboards ~6% up, ports ~6% down: a real bias

    print("Port/starboard power split: %+.0f%% per side" % (100 * split))
    print("Course: hold heading 0, %.0f s at rate 32\n" % DURATION)

    deflection = trim_constant(boat, split)
    print("A  constant rudder, trimmed to %.3f deg" % np.degrees(deflection))
    a = _measure(run_constant(boat, split, deflection))

    print("B  continuous correction (PD on instantaneous heading)")
    b = _measure(run_tracking(boat, split, gain=6.0, rate_gain=2.5))

    print()
    print("%-16s %12s %12s" % ("", "A constant", "B tracking"))
    for key, label, scale, unit in (
            ("made_good", "made good", 1.0, "m/s"),
            ("path_speed", "path speed", 1.0, "m/s"),
            ("yaw_swing", "yaw swing", 1.0, "deg"),
            ("cross_track", "cross track", 1.0, "m"),
    ):
        print("%-16s %12.4f %12.4f  %s"
              % (label, a[key] * scale, b[key] * scale, unit))

    gain = b["made_good"] - a["made_good"]
    print()
    print("B - A made good: %+.4f m/s  (%+.3f%%)"
          % (gain, 100 * gain / a["made_good"]))
    over_5k = gain / a["made_good"] * (5000.0 / a["made_good"])
    print("over a 5000 m Charles course that is %+.1f s" % -over_5k)


if __name__ == "__main__":
    main()
