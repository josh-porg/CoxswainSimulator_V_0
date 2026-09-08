r"""Your boat, on the Charles, against what it takes to medal.

    python scripts/hocr_crew.py

A specific Women's Veteran 60+ coxed four, modelled rower by rower
rather than as four copies of an average, and raced against the targets
``scripts/targets.py`` derives from the real fields.

Why rower by rower matters here
-------------------------------
The reference crew in ``targets.py`` is 68 kg at 1.70 m.  This crew runs
54 to 70 kg and 1.575 to 1.664 m.  That is not a detail: a lighter crew
displaces less and drags less, so it needs less power for the same
speed -- and it also HAS less power.  Modelling them as the average of
themselves gets both halves wrong at once.

Erg times to watts
------------------
Concept2's own relation, ``P = 2.80 / pace^3`` with pace in seconds per
metre.  It is the definition the monitor uses, not a fit.

A 5 km piece for this crew runs 22-24 minutes, and the Charles takes
them about the same, so their 5 km power is a fair estimate of what
they can hold for the race.  That is a happier coincidence than it
looks: a 2 km power would need discounting, and by an amount nobody
here has measured.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                                # noqa: E402
from coxswain.crew.anthropometry import RowerAnthropometry        # noqa: E402
from coxswain.crew.exertion import mean_handle_power              # noqa: E402
from coxswain.sim.control import Coxswain                         # noqa: E402
from coxswain.sim.simulator import RowingSimulator                # noqa: E402

LB = 0.45359237
INCH = 0.0254

#: Seat order is STROKE first, as the rig is laid out.
#: ``(name, seat label, pounds, feet, inches, 5k erg time)``
CREW = (
    ("Marilyn", "stroke", 120.0, 5, 5.5, "23:25"),
    ("Alex",    "3",      120.0, 5, 2.0, "23:41"),
    ("Lea",     "2",      125.0, 5, 3.5, "23:07"),
    ("Sheila",  "bow",    155.0, 5, 3.0, "22:20"),
)

COX_LB, COX_FT, COX_IN = 160.0, 5, 8.0

#: Bucket rigged, starboard stroke: S-P-P-S from the stroke seat.
#: In this model +1 is port, so that is (-1, +1, +1, -1).
BUCKET_SPPS = (-1, +1, +1, -1)


def seconds(stamp: str) -> float:
    minutes, secs = stamp.split(":")
    return float(minutes) * 60.0 + float(secs)


def erg_watts(five_k: str) -> float:
    """Concept2: ``P = 2.80 / pace^3``, pace in s/m."""
    pace = seconds(five_k) / 5000.0
    return 2.80 / pace ** 3


def build(rate: float = 30.0):
    people = []
    powers = []
    for _name, _seat, pounds, feet, inches, erg in CREW:
        people.append(RowerAnthropometry(mass=pounds * LB,
                                         stature=feet * 12 * INCH
                                         + inches * INCH,
                                         sex="female"))
        powers.append(erg_watts(erg))
    boat = catalog.coxed_four(
        rate=rate,
        rower_mass=float(np.mean([p.mass for p in people])),
        rower_stature=float(np.mean([p.stature for p in people])),
        coxswain_mass=COX_LB * LB,
        bow_loaded=True,
        anthropometry=people,
        rig_pattern="bucket, stbd stroke",
    )
    return boat, np.asarray(powers)


def race(boat, powers, duration=90.0):
    """Steady speed with each rower at their own power."""
    reference = mean_handle_power(boat, samples=180)
    boat.power_scales = powers / max(reference, 1.0)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=0.01, surge_speed=3.4)
    time = np.asarray(result.time)
    speed = np.hypot(*np.asarray(result.velocity)[:2])
    settled = time > duration - 4 * boat.timing.period
    return float(speed[settled].mean())


def stamp(total: float) -> str:
    return "%d:%04.1f" % (int(total // 60), total % 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--length", type=float, default=4828.0,
                        help="HOCR course length, m")
    args = parser.parse_args(argv)

    boat, powers = build(args.rate)

    print("YOUR BOAT")
    print("  %-9s %-7s %7s %7s %9s %9s" % ("", "seat", "kg", "m", "5k", "W"))
    for (name, seat, pounds, feet, inches, erg), watts in zip(CREW, powers):
        side = "starboard" if BUCKET_SPPS[list(CREW).index(
            (name, seat, pounds, feet, inches, erg))] < 0 else "port"
        print("  %-9s %-7s %7.1f %7.3f %9s %7.0f  %s"
              % (name, seat, pounds * LB,
                 feet * 12 * INCH + inches * INCH, erg, watts, side))
    print("  %-9s %-7s %7.1f  lying down (bow-loader)"
          % ("you", "cox", COX_LB * LB))
    print()
    print("  crew mass %.1f kg + cox %.1f = %.1f kg on the water"
          % (sum(c[2] for c in CREW) * LB, COX_LB * LB,
             (sum(c[2] for c in CREW) + COX_LB) * LB))
    print("  mean crew power %.0f W per rower  (spread %.0f to %.0f)"
          % (powers.mean(), powers.min(), powers.max()))
    print("  bucket rigged S-P-P-S from stroke, starboard stroke")
    print()

    speed = race(boat, powers)
    flat = args.length / speed
    print("PREDICTED, at your measured power")
    print("  still water, no current: %.2f m/s -> %s over %.0f m  (%s /500)"
          % (speed, stamp(flat), args.length, stamp(500.0 / speed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
