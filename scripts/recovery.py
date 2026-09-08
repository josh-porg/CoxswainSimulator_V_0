r"""How many strokes it takes to get a boat back after a roll.

    python scripts/recovery.py

A coxswain's question, and one the model could not answer until the
balance authority became finite.  With a flat 4000 N m available at
every instant the boat sat itself in well under a stroke, which is not
a recovery, it is a servo.

Two recoveries, and they are not the same
-----------------------------------------
**Set.**  The roll returns to something a crew can row through.  The
threshold used here is the angle at which the blade tip reaches the
water -- about 1.3 degrees on this eight -- because that is the point at
which the boat stops costing length.

**Timing.**  The crew come back together.  A roll does not only heel the
boat; it arrives at every rower simultaneously through the hull, and it
lands hardest exactly when their authority is lowest.  Once the catches
have been knocked apart the power is no longer aligned, and that takes
strokes of its own to gather up.  ``CoupledCrew`` carries that channel.

Reported in STROKES rather than seconds, because that is the unit the
correction is actually issued in.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                              # noqa: E402
from coxswain.crew.blade_contact import BladeContact            # noqa: E402
from coxswain.crew.synchronisation import (CoupledCrew,         # noqa: E402
                                           stroke_chain_topology)
from coxswain.sim.control import Coxswain, balance_for_experience  # noqa: E402
from coxswain.sim.simulator import RowingSimulator              # noqa: E402


def set_recovery(experience, heel_degrees=4.0, contact=True, duration=40.0):
    """Strokes until the heel stays under the blade-touch angle."""
    boat = catalog.eight(rate=30, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    touch = BladeContact.from_boat(boat)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0,
                   balance=balance_for_experience(boat, experience))
    sim = RowingSimulator(
        boat, coxswain=cox,
        blade_contact=touch if contact else None)
    state = sim.initial_state(surge_speed=4.4)
    state[3] = np.radians(heel_degrees)
    result = sim.run(duration=duration, dt=0.01, initial_state=state)

    time = np.asarray(result.time)
    roll = np.abs(np.degrees(np.asarray(result.roll)))
    limit = np.degrees(touch.roll_to_touch())
    period = float(boat.timing.period)

    # Measured against the boat's OWN steady roll, not against zero.
    # A rowing eight is never level -- the crew's own movement rolls it
    # every stroke -- so "recovered" means the knock has decayed into
    # the motion that was always there, not that the boat went still.
    # Comparing to an absolute angle instead reports "never" for a boat
    # that recovered perfectly well.
    per_stroke = []
    for start in np.arange(0.0, duration - period, period):
        window = (time >= start) & (time < start + period)
        if window.any():
            per_stroke.append((start, float(roll[window].max())))
    if not per_stroke:
        return float("nan"), limit, float(roll.max())
    baseline = float(np.median([peak for _s, peak in per_stroke[-4:]]))
    threshold = baseline * 1.25 + 0.05
    for start, peak in per_stroke:
        if peak <= threshold:
            return start / period, limit, baseline
    return float("inf"), limit, baseline


def coupled_recovery(skill=0.55, heel=8.0, strokes=16, rate=30.0):
    """Strokes to gather the crew back up, in the boat rather than beside it.

    The whole loop, closed: the hull rolls, the roll reaches every rower
    at once through the shell and knocks their catches apart, the
    misaligned power rolls the boat again, and the crew have to find each
    other from there.  Measured as the spread of the catches, in strokes
    to come back within a fifth of the knock.
    """
    from coxswain.crew.synchronisation import (CoupledCrew,
                                               stroke_chain_topology)
    from coxswain.sim.realtime import FixedStepLoop

    boat = catalog.eight(rate=rate, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    period = float(boat.timing.period)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0,
                   balance=balance_for_experience(boat, skill))
    sim = RowingSimulator(boat, coxswain=cox,
                          blade_contact=BladeContact.from_boat(boat))
    crew = CoupledCrew(n_seats=8, topology=stroke_chain_topology(8),
                       sensory_gain=0.6 + 1.4 * skill,
                       noise=0.05 * (1.0 - skill), seed=11)
    omega = 2.0 * np.pi / period
    trace = []

    def on_step(t, state, dt):
        crew.step(t, dt, omega, hull_roll_rate=float(state[9]))
        boat.phase_offsets = crew.phase_offsets()
        trace.append((t, crew.spread_seconds))

    loop = FixedStepLoop(sim, rate=100.0, on_step=on_step)
    loop.start(sim.initial_state(surge_speed=4.4))
    for _ in range(int(period / 0.02)):
        loop.advance(0.02)
    loop.state[3] = np.radians(heel)
    knocked_at = loop.t
    for _ in range(int(strokes * period / 0.02)):
        loop.advance(0.02)

    after = [(t, spread) for t, spread in trace if t > knocked_at]
    if not after:
        return float("nan"), 0.0, 0.0
    window = [spread for t, spread in after if t < knocked_at + 2.0 * period]
    peak = max(window) if window else 0.0
    calm = float(np.median([spread for _t, spread in after[-200:]]))
    target = calm + 0.2 * max(peak - calm, 1e-9)
    for t, spread in after:
        if t > knocked_at + 0.5 * period and spread <= target:
            return (t - knocked_at) / period, peak * 1000.0, calm * 1000.0
    return float("inf"), peak * 1000.0, calm * 1000.0


def timing_recovery(hit=2.5, rate=30.0, strokes=14, seed=3):
    """Strokes until the crew's catches are back together.

    The roll is applied as a rate through the mechanical channel, which
    is how a real disturbance reaches the crew: instantly, to all of
    them, with no choice about listening.
    """
    boat = catalog.eight(rate=rate, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    omega = 2.0 * np.pi / float(boat.timing.period)
    crew = CoupledCrew(n_seats=8, topology=stroke_chain_topology(8),
                       seed=seed)
    dt = 0.005
    settled = None
    spread0 = None
    history = []
    for step in range(int(strokes * boat.timing.period / dt)):
        t = step * dt
        # One stroke of clean rowing, then the knock, then watch.
        roll_rate = hit if boat.timing.period < t < boat.timing.period + 0.25 \
            else 0.0
        crew.step(t, dt, omega, hull_roll_rate=roll_rate)
        if t > boat.timing.period + 0.25:
            spread = crew.spread_seconds
            if spread0 is None:
                spread0 = spread
            history.append((t, spread))
    if spread0 is None:
        return float("nan"), 0.0
    target = 0.2 * spread0
    for t, spread in history:
        if spread <= target:
            settled = (t - boat.timing.period) / boat.timing.period
            break
    return (settled if settled is not None else float("inf")), spread0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heel", type=float, default=4.0)
    args = parser.parse_args(argv)

    print("Getting the boat back after a %.0f degree knock\n" % args.heel)
    print("SET -- strokes until the blades stop touching")
    print("  %-10s %12s %12s %10s" % ("crew", "with contact", "without",
                                      "own roll"))
    for experience, label in ((0.0, "novice"), (0.55, "club"),
                              (1.0, "ideal")):
        with_c, limit, peak = set_recovery(experience, args.heel, True)
        without, _, _ = set_recovery(experience, args.heel, False)

        def show(value):
            return "  never" if value == float("inf") else "%6.1f" % value

        print("  %-10s %12s %12s %8.2f d"
              % (label, show(with_c), show(without), peak))
    print("  (peak roll is the crew's OWN steady roll, once settled;")
    print("   blades touch below %.2f degrees of heel)" % limit)

    print("")
    print("TIMING, coupled -- the hull knocks the crew apart and they")
    print("        have to find each other again")
    print("  %-10s %10s %12s %12s" % ("crew", "strokes", "knocked to", "settles at"))
    for _skill, _label in ((0.15, "novice"), (0.55, "club"), (0.95, "elite")):
        _back, _peak, _calm = coupled_recovery(skill=_skill, heel=args.heel)
        print("  %-10s %10s %9.0f ms %9.0f ms"
              % (_label, "never" if _back == float("inf") else "%.1f" % _back, _peak, _calm))
    print("")
    print("TIMING, uncoupled -- the oscillator alone")
    strokes, spread = timing_recovery()
    print("  knocked apart by %.0f ms; back within a fifth of that after "
          "%s strokes"
          % (spread * 1000.0,
             "never" if strokes == float("inf") else "%.1f" % strokes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
