r"""What traffic costs a lottery entry on the Charles.

    python scripts/traffic.py --bow 40

A guaranteed entry starts near the front of its event, among crews of its
own speed. **A lottery entry does not.** It is seeded on last year's
result or on nothing at all, which means starting behind boats that are
slower and ahead of boats that are faster, and spending the race passing
and being passed.

The rulebook prices only the failure: 60 s, then 120 s, then
disqualification for not yielding. It says nothing about the cost of
yielding *correctly*, which is the cost a well-behaved crew actually pays
-- and on this river that cost is not the metres of lateral travel. It is
that **the racing line is the deep line** (SOURCES sec. 66-67, 79), so a
crew pushed a boat-width off it is pushed into shallower water, where the
depth Froude number is nearer one and every watt buys less speed.

That coupling is the point of this script. :mod:`coxswain.river.passing`
holds the rules with no physics in it; here the physics is supplied as a
speed field over ``(station, lateral)`` and the two are run together.

What is modelled
----------------
Speed at a point is solved from the crew's power against the real depth
and current at that station and offset, using the same
:class:`~coxswain.crew.pacing.CoursePacing` machinery as everything else.
Tabulated once on a grid because the state machine asks for it at every
step of every boat.

What is not
-----------
Steering dynamics: a crew moves sideways at a fixed rate and pays for
where it ends up, not for the rudder it used getting there. The line
optimiser (§67) prices that separately and the two have not been coupled.
The wash of the boat being passed is also absent -- §97's reflection
geometry says a following crew rows through the leader's wake, and nothing
here charges for it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402
from coxswain.river.charles import charles_course           # noqa: E402
from coxswain.river.passing import (Entry, HeadRace,  # noqa: E402
                                    PassingRules)

from course_pacing import hull_drag                         # noqa: E402

#: The race course, not the surveyed reach.
RACE_LENGTH = 4828.0
#: Rule 12: no non-yield penalty between the start and clearing BU Bridge.
BU_BRIDGE = 700.0


def speed_field(course, boat, power, stations=41, offsets=9,
                half_width=28.0):
    """``(station, lateral) -> m/s`` at fixed power, from the real river.

    Tabulated and bilinearly interpolated.  Solving the power balance
    inside the state machine's inner loop would make a twenty-boat field
    take longer than the race, which is the same lesson the shallow factor
    taught in sec. 66.
    """
    model = CoursePacing([CourseSegment(100.0)], hull_drag(boat),
                         rowers=boat.n_seats, shallow_model=boat.shallow)
    station_grid = np.linspace(0.0, RACE_LENGTH, stations)
    offset_grid = np.linspace(-half_width, half_width, offsets)
    table = np.zeros((stations, offsets))

    for i, station in enumerate(station_grid):
        point = course.position_at(station)
        heading = float(course.heading_at(station))
        normal = np.array([-np.sin(heading), np.cos(heading)])
        tangent = np.array([np.cos(heading), np.sin(heading)])
        for j, offset in enumerate(offset_grid):
            where = np.asarray(point) + normal * offset
            depth = float(course.depth_at(where[0], where[1]))
            current = np.asarray(course.current_at(where[0], where[1]))[:2]
            segment = CourseSegment(length=100.0, depth=max(depth, 0.30),
                                    current=float(np.dot(current, tangent)))
            water = model.speed_for_power(power, segment)
            table[i, j] = water + segment.current

    def lookup(station, lateral):
        s = np.clip(station, 0.0, RACE_LENGTH)
        o = np.clip(lateral, -half_width, half_width)
        i = np.interp(s, station_grid, np.arange(stations))
        j = np.interp(o, offset_grid, np.arange(offsets))
        i0, j0 = int(np.floor(i)), int(np.floor(j))
        i1 = min(i0 + 1, stations - 1)
        j1 = min(j0 + 1, offsets - 1)
        fi, fj = i - i0, j - j0
        return float(
            table[i0, j0] * (1 - fi) * (1 - fj)
            + table[i1, j0] * fi * (1 - fj)
            + table[i0, j1] * (1 - fi) * fj
            + table[i1, j1] * fi * fj)

    lookup.table = table
    lookup.stations = station_grid
    lookup.offsets = offset_grid
    return lookup


def build_field(n, our_bow, interval, spread, seed, speed, lookup):
    """A field seeded worst-first, which is how a lottery entry starts.

    Bow numbers run roughly slowest to fastest in a head race, so a crew
    seeded at ``our_bow`` has slower boats ahead of it and faster boats
    behind.  That is the whole reason traffic costs a lottery entry more
    than a guaranteed one.
    """
    rng = np.random.default_rng(seed)
    # Expected speed rises with bow number; scatter on top of it.
    ramp = np.linspace(-spread, spread, n)
    noise = rng.normal(0.0, spread * 0.45, size=n)
    entries = []
    for index in range(n):
        factor = 1.0 + ramp[index] + noise[index]
        # **Our crew is the same crew wherever it is seeded.**  Letting
        # its speed follow the ramp like everyone else's made a higher bow
        # a faster boat by construction, so "the cost of traffic" came out
        # at MINUS 58 seconds -- traffic making a crew quicker, which is
        # the model comparing different crews and calling it a position
        # effect.
        if index + 1 == our_bow:
            factor = 1.0
        entry = Entry(bow=index + 1, start=index * float(interval),
                      speed=float(speed * factor),
                      name="ours" if index + 1 == our_bow else "")
        entry.speed_fn = (lambda st, lat, f=factor:
                          f * lookup(st, lat))
        entries.append(entry)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boats", type=int, default=25)
    parser.add_argument("--bow", type=int, default=13,
                        help="our seeding; 1 is the front of the event")
    parser.add_argument("--interval", type=float, default=12.0)
    parser.add_argument("--spread", type=float, default=0.05,
                        help="fractional speed spread across the field")
    parser.add_argument("--power", type=float, default=210.0,
                        help="per rower, W -- 210 is the medal pace of "
                             "sec. 102")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args(argv)

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    course = charles_course()
    print("tabulating the speed field over the race course")
    lookup = speed_field(course, boat, args.power)
    table = lookup.table
    centre = table[:, table.shape[1] // 2]
    print("  speed on the centreline: %.3f - %.3f m/s"
          % (centre.min(), centre.max()))
    print("  cost of a boat-width off it: %.3f m/s at the worst station"
          % float(np.max(centre - table[:, table.shape[1] // 2 + 1])))
    print()

    rules = PassingRules(boat_length=13.4, exempt_from_start=BU_BRIDGE)
    reference = None
    ours, penalties, passes_made, passes_taken, lost = [], [], [], [], []

    for run in range(args.runs):
        entries = build_field(args.boats, args.bow, args.interval,
                              args.spread, run, 3.75, lookup)
        race = HeadRace(entries, length=RACE_LENGTH, rules=rules,
                        compliance=1.0, seed=run)
        log = race.run(dt=0.5, limit=4000.0)
        us = race.entries[args.bow]
        results = {r["bow"]: r for r in race.results()}
        ours.append(results[args.bow]["raw"])
        penalties.append(results[args.bow]["penalty"])
        lost.append(us.lost_to_yield)
        passes_made.append(sum(1 for e in log.of_kind("declare")
                               if e["passer"] == args.bow))
        passes_taken.append(sum(1 for e in log.of_kind("declare")
                                if e["passee"] == args.bow))

        if reference is None:
            # The same crew, alone on the river, holding the deep line.
            solo = Entry(bow=1, start=0.0, speed=3.75)
            solo.speed_fn = lambda st, lat: lookup(st, lat)
            alone = HeadRace([solo], length=RACE_LENGTH, rules=rules)
            alone.run(dt=0.5, limit=4000.0)
            reference = alone.results()[0]["raw"]

    ours = np.array(ours)
    print("OUR CREW, bow %d of %d, %d runs" % (args.bow, args.boats,
                                               args.runs))
    print("  alone on the river          %s"
          % ("%d:%05.2f" % (int(reference // 60), reference % 60)))
    print("  in traffic                  %d:%05.2f  +/- %.1f s"
          % (int(ours.mean() // 60), ours.mean() % 60, ours.std()))
    print("  cost of traffic             %+.1f s" % (ours.mean() - reference))
    print("  metres given up to yields   %.1f m" % np.mean(lost))
    print("  passes made / taken         %.1f / %.1f"
          % (np.mean(passes_made), np.mean(passes_taken)))
    print("  penalties                   %.1f s" % np.mean(penalties))
    print()

    print("BY SEEDING -- what the draw is worth")
    print("  %-8s %12s %10s %10s %10s"
          % ("bow", "time", "vs alone", "passes", "passed"))
    for bow in (2, 7, 13, 19, 24):
        if bow > args.boats:
            continue
        times, made, taken = [], [], []
        for run in range(max(args.runs // 2, 8)):
            entries = build_field(args.boats, bow, args.interval,
                                  args.spread, run, 3.75, lookup)
            race = HeadRace(entries, length=RACE_LENGTH, rules=rules,
                            compliance=1.0, seed=run)
            log = race.run(dt=0.5, limit=4000.0)
            row = {r["bow"]: r for r in race.results()}[bow]
            times.append(row["raw"])
            made.append(sum(1 for e in log.of_kind("declare")
                            if e["passer"] == bow))
            taken.append(sum(1 for e in log.of_kind("declare")
                             if e["passee"] == bow))
        mean = float(np.mean(times))
        print("  %-8d %12s %+9.1f s %10.1f %10.1f"
              % (bow, "%d:%05.2f" % (int(mean // 60), mean % 60),
                 mean - reference, np.mean(made), np.mean(taken)))
    print()
    print("  Bow order runs slowest to fastest, so a low bow starts among")
    print("  slower crews and spends the race passing.  A high bow is")
    print("  passed instead, and pays for yielding rather than for")
    print("  overtaking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
