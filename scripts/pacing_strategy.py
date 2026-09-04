r"""Is a head race a time trial or a championship race?

    python scripts/pacing_strategy.py --boat 4+

The pacing literature splits cleanly in two (SOURCES sec. 72, 75):

* **When position decides** -- championship 5000 m, a rowing final, a
  bunch sprint -- pacing is tactical. Fast starts, responses to attacks,
  end spurts, and a *worse* mean distribution than a lone rider would
  choose.
* **When the clock decides** -- a time trial, a world-record attempt --
  pacing is even, modulated only by the course.

A Head of the Charles is not obviously either. The clock decides, which
says time trial. But boats start 10-15 s apart and you spend the whole
race among them, which says championship. **This script asks which
literature applies, by computing the things that separate them**, rather
than asserting an answer.

Three questions, three numbers
------------------------------
1. **How much is the reserve worth here?** The 2 km literature is written
   about a six-minute effort where ``W'`` is a large fraction of what you
   spend. Over twenty minutes it is not. If the reserve is small, tactical
   pacing has little to spend.
2. **What does chasing cost?** The defining championship behaviour is
   deviating from your own best schedule to stay with someone. Priced
   directly.
3. **How much of your time is even yours?** Wake and yielding are
   interference from other boats. If they dominate, position matters
   whatever the clock says.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.crew.exertion import (ROWER_ANAEROBIC_WORK,  # noqa: E402
                                    ROWER_CRITICAL_POWER)
from coxswain.crew.pacing import CoursePacing, CourseSegment    # noqa: E402
from coxswain.river.charles import (HOCR_COURSE_LENGTH,  # noqa: E402
                                    charles_course)

from course_pacing import (build_boat, build_segments,  # noqa: E402
                           build_wind, hull_drag)


def reserve_worth(durations):
    """``W'/T`` as watts and as a fraction of critical power."""
    rows = []
    for label, seconds in durations:
        extra = ROWER_ANAEROBIC_WORK / seconds
        rows.append((label, seconds, extra,
                     100.0 * extra / ROWER_CRITICAL_POWER))
    return rows


def chase_cost(model, baseline, fraction, segments_ahead):
    """Seconds lost to surging early to hold contact, at equal work.

    A crew that chases spends power early and pays for it late.  Modelled
    as ``+fraction`` on the opening segments and whatever balances it
    afterwards, so the total work is identical to the flat schedule and
    the comparison is about **distribution only** -- the same discipline
    :meth:`CoursePacing.price_scatter` uses.
    """
    n = len(model.segments)
    powers = np.full(n, baseline, dtype=float)
    powers[:segments_ahead] *= (1.0 + fraction)
    reference = model.evaluate(np.full(n, baseline))
    # Restore the mean exactly, weighting by the flat schedule's own
    # durations so "equal work" means equal work and not equal mean watts.
    powers = powers + (baseline - float(np.average(
        powers, weights=reference.durations)))
    plan = model.evaluate(powers)
    return plan.total_time - reference.total_time, plan


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=250.0)
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 30.0 if args.boat == "4+" else 32.0

    boat = build_boat(args.boat, args.rate)
    drag = hull_drag(boat)
    course = charles_course()
    wind = build_wind(args.wind, args.wind_from)
    segments = build_segments(course, args.segments, boat, wind)
    scale = HOCR_COURSE_LENGTH / sum(s.length for s in segments)
    raced = [CourseSegment(length=s.length * scale, current=s.current,
                           headwind=s.headwind, depth=s.depth,
                           drag_factor=s.drag_factor, label=s.label)
             for s in segments]
    model = CoursePacing(raced, drag, rowers=boat.n_seats,
                         shallow_model=boat.shallow)
    baseline = model.flat_power()
    reference = model.evaluate(np.full(len(raced), baseline))

    print("IS A HEAD RACE A TIME TRIAL OR A CHAMPIONSHIP RACE?")
    print("  %s over %.0f m, modelled at %.1f s" % (args.boat,
                                                    HOCR_COURSE_LENGTH,
                                                    reference.total_time))
    print()

    # -- 1. what the reserve is worth ---------------------------------
    print("1. HOW MUCH IS THE ANAEROBIC RESERVE WORTH?")
    print("   W' = %.0f J per rower; CP = %.1f W"
          % (ROWER_ANAEROBIC_WORK, ROWER_CRITICAL_POWER))
    print()
    print("   %-26s %8s %10s %10s" % ("effort", "seconds", "W'/T", "of CP"))
    for label, seconds, extra, percent in reserve_worth([
            ("2 km final (the 2k lit)", 370.0),
            ("5 km head race", 900.0),
            ("Head of the Charles", reference.total_time),
            ("marathon-length piece", 3600.0)]):
        print("   %-26s %8.0f %9.1f W %9.1f%%" % (label, seconds, extra,
                                                  percent))
    print()
    two_k = ROWER_ANAEROBIC_WORK / 370.0
    here = ROWER_ANAEROBIC_WORK / reference.total_time
    print("   The reserve buys %.1f W in a 2 km final and %.1f W here --"
          % (two_k, here))
    print("   **%.1f times less**.  Tactical pacing is spending the"
          % (two_k / here))
    print("   reserve, and over twenty minutes there is very little to")
    print("   spend.  That is the single clearest reason 2 km tactical")
    print("   advice does not transfer.")
    print()

    # -- 2. what chasing costs ----------------------------------------
    print("2. WHAT DOES CHASING COST?")
    print("   Surging early to hold contact, then paying for it, at")
    print("   identical total work.")
    print()
    print("   %-30s %10s %10s" % ("chase", "extra W", "seconds lost"))
    quarter = max(len(raced) // 4, 1)
    for fraction in (0.02, 0.05, 0.10, 0.15):
        lost, plan = chase_cost(model, baseline, fraction, quarter)
        print("   %-30s %9.1f %11.2f"
              % ("+%.0f%% for the first quarter" % (100 * fraction),
                 baseline * fraction, lost))
    print()
    print("   Chasing is not free and it is not catastrophic.  A crew that")
    print("   goes 10% over for a quarter of the race to stay with someone")
    print("   pays for it, but in seconds, not minutes.")
    print()

    # -- 3. how much of the time is yours -----------------------------
    print("3. HOW MUCH OF YOUR TIME IS ACTUALLY YOURS?")
    print("   From the two-boat work (SOURCES sec. 80, 82), over the reach:")
    print()
    print("   %-38s %12s" % ("channel", "seconds"))
    print("   %-38s %12s" % ("sitting in their puddles", "6.3"))
    print("   %-38s %12s" % ("each yield you are made to give", "~1.0"))
    print("   %-38s %12s" % ("optimal vs flat pacing", "7.1"))
    print("   %-38s %12s" % ("optimal vs centreline (the LINE)", "33.5"))
    print()
    print("   The line dominates everything the other boats do.  Traffic")
    print("   is worth a handful of seconds; steering is worth half a")
    print("   minute.")
    print()

    # -- the answer ----------------------------------------------------
    print("SO WHICH LITERATURE APPLIES?")
    print("  **The time-trial literature**, and it is not close.")
    print()
    print("  A head race has the *form* of a championship race -- you can")
    print("  see your rivals and you race them -- but none of the")
    print("  mechanics.  Tactical pacing works when the reserve is a large")
    print("  fraction of what you spend and position decides the result.")
    print("  Here the reserve is %.0f%% of CP, the clock decides, and the"
          % (100 * here / ROWER_CRITICAL_POWER))
    print("  biggest lever on the course is not another crew at all.")
    print()
    print("  What that means for a coxswain:")
    print("   * Row the course, not the boat in front.  Chasing costs")
    print("     seconds you cannot get back and buys nothing the clock")
    print("     rewards.")
    print("   * Spend the reserve on the WATER (SOURCES sec. 66), not on")
    print("     contact.  It is only worth ~%.0f W; a 1%% power spread"
          % here)
    print("     captures the whole gain.")
    print("   * The exception is genuinely tactical: taking a clean line")
    print("     past a slower crew, and choosing which side to send them")
    print("     (sec. 86).  Those are position decisions that pay in")
    print("     seconds, and they are not pacing decisions at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
