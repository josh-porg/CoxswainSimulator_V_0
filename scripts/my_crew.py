r"""Your HOCR four, against what the category actually takes.

    python scripts/my_crew.py

Everything here is measured or derived from measurement:

* the erg times are yours;
* the watts come from Concept2's own relation, ``P = 2.80 / pace^3``,
  checked against the squad's spreadsheet where it agrees to 0.3 W;
* the boat is the catalogue four at YOUR crew's mass and height, which
  matters -- the published targets assume 68 kg rowers and yours average
  59, and a lighter boat needs less power for the same time;
* the targets come from the real 60-69 fields, 2019-2025.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from coxswain.boats import catalog                                # noqa: E402
from coxswain.crew.blade_contact import BladeContact              # noqa: E402
from coxswain.crew.pacing import CoursePacing, CourseSegment      # noqa: E402
from coxswain.river.charles import (HOCR_COURSE_LENGTH,           # noqa: E402
                                    charles_course)
from course_pacing import build_segments, build_wind, hull_drag   # noqa: E402
from splits import stamp                                          # noqa: E402

LB = 0.45359237

#: seat, name, 5k erg, pounds, metres, rigged side (+1 port, -1 starboard)
CREW = (
    ("stroke", "Marilyn", "23:25", 120.0, 1.664, -1),
    ("3",      "Alex",    "23:41", 120.0, 1.575, +1),
    ("2",      "Lea",     "23:07", 125.0, 1.613, +1),
    ("bow",    "Sheila",  "22:20", 155.0, 1.600, -1),
)
COX_LB = 160.0

#: 2026 central estimates from scripts/targets.py.
TARGETS = (("WIN the category", 19 * 60 + 55.9),
           ("MEDAL (top three)", 20 * 60 + 15.7),
           ("REQUALIFY (top half)", 21 * 60 + 5.9))


def watts(mmss: str) -> float:
    """Concept2's own relation between pace and power."""
    minutes, seconds = mmss.split(":")
    total = int(minutes) * 60 + float(seconds)
    return 2.80 / (total / 5000.0) ** 3


def erg_time(power: float) -> str:
    total = (2.80 / power) ** (1.0 / 3.0) * 5000.0
    return "%d:%04.1f" % (int(total // 60), total % 60)


def build(rate: float = 30.0):
    boat = catalog.coxed_four(
        rate=rate,
        rower_mass=float(np.mean([c[3] for c in CREW])) * LB,
        rower_stature=float(np.mean([c[4] for c in CREW])),
        coxswain_mass=COX_LB * LB)
    course = charles_course()
    segments = build_segments(course, 12, boat, build_wind(0.0, 0.0))
    scale = HOCR_COURSE_LENGTH / sum(s.length for s in segments)
    raced = [CourseSegment(length=s.length * scale, current=s.current,
                           headwind=s.headwind, depth=s.depth,
                           drag_factor=s.drag_factor, label=s.label)
             for s in segments]
    return boat, CoursePacing(raced, hull_drag(boat), rowers=boat.n_seats,
                              shallow_model=boat.shallow)


def time_for(model, power: float) -> float:
    return model.evaluate(np.full(len(model.segments), power)).total_time


def power_for(model, target: float) -> float:
    low, high = 60.0, 600.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if time_for(model, mid) > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args(argv)

    powers = [watts(c[2]) for c in CREW]
    mean = float(np.mean(powers))
    boat, model = build(args.rate)

    print("YOUR CREW")
    print("  %-7s %-9s %-7s %6s %7s %10s %6s"
          % ("seat", "name", "5k", "lb", "height", "watts", "side"))
    for (seat, name, tm, lb, ht, side), power in zip(CREW, powers):
        print("  %-7s %-9s %-7s %6.0f %7.3f %9.1f %6s"
              % (seat, name, tm, lb, ht, power,
                 "port" if side > 0 else "stbd"))
    print("  crew %.1f kg + cox %.1f kg; boat all-up %.1f kg"
          % (sum(c[3] for c in CREW) * LB, COX_LB * LB, boat.total_mass))
    print("  mean %.1f W a rower" % mean)

    now = time_for(model, mean)
    print("\nWHAT THAT PREDICTS ON THE CHARLES")
    print("  %s   (%s per 500 m)"
          % (stamp(now), stamp(now / (HOCR_COURSE_LENGTH / 500.0))))

    print("\nWHAT THE CATEGORY TAKES, for a boat of your weight")
    print("  %-22s %9s %9s %11s %8s"
          % ("target", "time", "W/rower", "5k erg", "gain"))
    print("  %-22s %9s %9.0f %11s %8s"
          % ("you now", stamp(now), mean, erg_time(mean), "--"))
    for label, target in reversed(TARGETS):
        need = power_for(model, target)
        print("  %-22s %9s %9.0f %11s %+7.0f%%"
              % (label, stamp(target), need, erg_time(need),
                 100.0 * (need - mean) / mean))

    print("\n  Power is cubed into speed, so the erg gap always looks worse")
    print("  than the time gap: %+.0f%% of power is %+.0f%% of boat speed."
          % (100.0 * (power_for(model, TARGETS[2][1]) - mean) / mean,
             100.0 * (now / TARGETS[2][1] - 1.0)))

    contact = BladeContact.from_boat(boat)
    print("\nWHAT ELSE IS ON THE TABLE")
    print("  blades touch at %.2f degrees of heel (tip %.2f m out)"
          % (np.degrees(contact.roll_to_touch()), contact.reach))
    rate = abs(float(boat.oar_sweep.rate(0.2 * boat.timing.period,
                                         boat.timing)))
    for heel in (1.5, 2.0, 3.0):
        keep = contact.length_fraction(np.radians(heel), rate,
                                       boat.oar_sweep.total_sweep,
                                       timing=boat.timing)
        print("     at %.1f degrees you keep %.0f%% of the sweep"
              % (heel, 100.0 * keep))
    pace = HOCR_COURSE_LENGTH / now
    print("  steering: every 10 m of extra line costs %.1f s"
          % (10.0 / pace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
