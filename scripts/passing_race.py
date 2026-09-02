r"""What the passing rules cost, once yielding actually slows you down.

    python scripts/passing_race.py --boat 4+

`coxswain/river/passing.py` implements the rulebook as a state machine and
`scripts/two_boat.py` prices a wake. Neither, alone, can answer the
question a coxswain has: **what does being told to move over actually
cost?**

The state machine on its own has no physics -- a crew yields, changes a
number, and rows on at the same speed. That makes every yield free, which
is exactly wrong on this river. SOURCES sec. 66-67 and 79 found the line is
worth a great deal, because the centreline *is* the deep water and being
off it puts the hull toward the transcritical drag rise. So a yield is not
a courtesy, it is a **time penalty paid in bathymetry**.

This couples the two. Each crew's speed is read from the river at its
actual lateral position, every step, so:

* a crew forced off the deep line by a yield **slows down**, and the
  amount depends on where on the course it was told to move;
* a crew sitting in another's puddles **slows down**, by
  ``PuddleWake.lateral_overlap``;
* gaps therefore open and close on their own, and the rulebook fires when
  the geometry says it should rather than when a script says so.

What to read off it
-------------------
The interesting number is not the 60-second penalty -- that is in the
rulebook and needs no model. It is the **uncharged** cost: the seconds a
complying crew loses to the yield itself, which appear on no scoresheet
and which a coxswain can influence by choosing *where* to be caught and
*which side* to give.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.hydro.wake import PuddleWake, blade_track          # noqa: E402
from coxswain.river.charles import charles_course                # noqa: E402
from coxswain.river.passing import (Entry, HeadRace,  # noqa: E402
                                    PassingRules)

from course_pacing import build_boat, hull_drag                  # noqa: E402
from export_map import speed_table                               # noqa: E402


def river_speed(course, boat, drag, power=307.0):
    """``(station, lateral) -> m/s`` on the real bathymetry.

    Tabulated in depth, as everywhere else in this project, because the
    passing loop asks for it a few hundred thousand times.
    """
    lookup = speed_table(drag, boat.shallow, boat.n_seats, power)
    length = course.length

    def speed(station, lateral):
        station = float(np.clip(station, 0.0, length))
        limit = max(float(course.half_width_at(station)) - 1.0, 0.5)
        lateral = float(np.clip(lateral, -limit, limit))
        point = np.atleast_1d(np.asarray(
            course.offset_position(station, lateral), float)).ravel()[:2]
        return lookup(float(course.depth_at(point[0], point[1])), 0.0)
    return speed


def with_wake(speed, wake, track, gap_of):
    """Wrap a speed field so a crew in another's puddles is slowed."""
    def slowed(station, lateral, other_lateral=None):
        base = speed(station, lateral)
        if other_lateral is None:
            return base
        gap = gap_of()
        if gap <= 0.0:
            return base
        overlap = float(wake.lateral_overlap(
            gap, abs(lateral - other_lateral), blade_track=track))
        penalty = float(wake.power_penalty(gap)) * overlap
        # Power down by ``penalty``; speed follows as P^(1/3).
        return base * (1.0 - penalty) ** (1.0 / 3.0)
    return slowed


def race(course, boat, drag, interval, gap_speed, leader_line, chaser_line,
         chaser_gain, compliance=1.0, dt=1.0, seed=0):
    """One two-boat race. Returns the log and both entries."""
    speed = river_speed(course, boat, drag)
    wake = PuddleWake(drag=drag(gap_speed), speed=gap_speed,
                      period=boat.timing.period, n_blades=boat.n_seats)
    track = blade_track(boat)

    leader = Entry(bow=1, start=0.0, speed=gap_speed, name="ahead",
                   lateral=leader_line, preferred_lateral=leader_line)
    chaser = Entry(bow=2, start=interval, speed=gap_speed, name="you",
                   lateral=chaser_line, preferred_lateral=chaser_line)

    def gap_now():
        return leader.station - boat.length - chaser.station

    def leader_speed(station, lateral):
        return speed(station, lateral)

    def chaser_speed(station, lateral):
        base = speed(station, lateral) * (1.0 + chaser_gain)
        gap = gap_now()
        if gap <= 0.0:
            return base
        overlap = float(wake.lateral_overlap(
            gap, abs(lateral - leader.lateral), blade_track=track))
        penalty = float(wake.power_penalty(gap)) * overlap
        return base * max(1.0 - penalty, 0.05) ** (1.0 / 3.0)

    leader.speed_fn = leader_speed
    chaser.speed_fn = chaser_speed

    event = HeadRace([leader, chaser], length=course.length,
                     rules=PassingRules(boat_length=boat.length),
                     compliance=compliance, seed=seed)
    log = event.run(dt=dt, limit=6000.0)
    return event, log, leader, chaser


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--gain", type=float, default=0.03,
                        help="how much faster you are, as a fraction")
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 30.0 if args.boat == "4+" else 32.0

    boat = build_boat(args.boat, args.rate)
    drag = hull_drag(boat)
    course = charles_course()
    nominal = 3.895 if args.boat == "4+" else 4.490

    print("Two boats on the surveyed reach, %s" % args.boat)
    print("  they start %.0f s ahead; you are %.1f%% faster"
          % (args.interval, 100 * args.gain))
    print("  yielding moves a crew %.1f m off its line, and the river is"
          % PassingRules(boat_length=boat.length).yield_width)
    print("  read at wherever that leaves them.")
    print()

    cases = [
        ("both on the centreline", 0.0, 0.0),
        ("they hold centre, you sit 1 m off", 0.0, 1.0),
        ("they are 5 m off, you take centre", 5.0, 0.0),
    ]
    print("  %-36s %9s %9s %7s %9s %8s"
          % ("case", "their s", "your s", "yields", "lost m", "lost s"))
    for label, leader_line, chaser_line in cases:
        event, log, leader, chaser = race(
            course, boat, drag, args.interval, nominal,
            leader_line, chaser_line, args.gain, dt=args.dt)
        yields = len(log.of_kind("yield"))
        their_time = (leader.finished if leader.finished is not None
                      else float("nan"))
        your_time = ((chaser.finished - chaser.start)
                     if chaser.finished is not None else float("nan"))
        print("  %-36s %9.1f %9.1f %7d %9.1f %8.2f"
              % (label, their_time, your_time, yields,
                 chaser.lost_to_yield, chaser.lost_to_yield / nominal))

    print()
    print("  'lost' is progress given up to being off the preferred line --")
    print("  the uncharged cost of the manoeuvre, which appears on no")
    print("  scoresheet and is separate from any 60 s penalty.")
    print()

    # The question the rules actually pose: which side to send them.
    print("WHICH SIDE TO SEND THEM")
    print("  A passer names the side. The rulebook does not say which is")
    print("  better and the river does: the passee is pushed toward one")
    print("  bank or the other, and those are not the same depth.")
    print()
    speed = river_speed(course, boat, drag)
    print("  %-10s %9s %9s %9s   %s"
          % ("station", "centre", "port 3.5", "stbd 3.5", "better side"))
    for station in np.linspace(500.0, course.length - 500.0, 10):
        centre = speed(station, 0.0)
        port = speed(station, +3.5)
        starboard = speed(station, -3.5)
        better = "port" if port > starboard else "starboard"
        margin = abs(port - starboard) / centre * 100
        print("  %-10.0f %9.4f %9.4f %9.4f   %s (%.2f%%)"
              % (station, centre, port, starboard, better, margin))
    print()
    print("  Sending them to the shallow side costs them more, which is")
    print("  legal, free, and entirely within the passer's gift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
