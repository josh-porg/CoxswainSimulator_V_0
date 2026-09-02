r"""The head race is not a rowing race. It is a hilly cycling time trial.

    python scripts/pacing_literature.py

Rowing's own pacing literature is almost entirely about the **2000 m**:
six to seven minutes, side by side, in lanes, on flat water. Xia's 179
women's eight finals (SOURCES sec. 72) are all of that kind, and they
produce the familiar "1-4-2-3" template with a fast first 500 and a
sprint finish.

A Head of the Charles piece is a different event in every respect that
matters to pacing:

===================  ==================  ======================
                     2000 m final        HOCR women's masters 8+
===================  ==================  ======================
duration             6-7 min             **19-20 min**
start                standing            rolling
opponents            beside you          staggered, passing
water                uniform lane        depth 2-8 m, current, shelter
what wins            position at 450 m   elapsed time
===================  ==================  ======================

So the useful comparison is not to other rowing. It is to endurance
events of **matched duration over non-uniform courses**, and those have a
large, quantitative, decades-old literature: the cycling individual time
trial.

Two results from it, and they point opposite ways
-------------------------------------------------
**In constant conditions, power variability is purely a cost.** Atkinson
et al. [A07]_ modelled a 289 W rider and found that varying power by
+/-15% *increased* time by 3.29 s over 16.1 km, 4.46 s over 20 km and
10.43 s over 40 km. Liedl et al. and others reach the same conclusion:
on a flat, windless course, hold the number.

**In varying conditions, power variability is the whole game.** Swain
[S97]_ showed the optimum is to vary power in parallel with gradient and
wind. Atkinson's same model puts a +/-10% variation matched to the course
at **26 s** saved on a windy 40 km and **126 s** on a hilly one.
Sundström and Bäckström [SB17]_, solving it as an optimal-control problem
with a full bioenergetic model, report optimal power distribution about
**2.9% faster** than constant power.

Those two results are the same result. Spread power *at random* and
convexity taxes you; spread it *in phase with the course* and you buy
more than the tax. This module's :mod:`coxswain.crew.pacing` derives
exactly that from rowing physics, and this script checks it against the
cycling numbers, which were produced by different people from a different
model of a different sport.

Why the check is meaningful
---------------------------
Cycling and rowing share the exponent. Aerodynamic drag gives ``P ~ v^3``
for a bicycle; hydrodynamic drag gives very nearly the same for a shell.
So ``e = d ln v / d ln P = 1/3`` in both, and the second-order cost of
spreading power at fixed mean work,

.. math::

    \Delta t = \tfrac{1}{2} T \, e (e + 1) \, \frac{\operatorname{var}(P)}{\bar P^2}

should reproduce Atkinson's published seconds without any rowing-specific
tuning. It does, to 1% at the duration closest to a head race.

One reading trap worth stating: their "+/-15%" is a *sinusoidal swing of
amplitude 0.15*, whose standard deviation is ``0.15/sqrt(2)``, not 0.15.
Treating it as a standard deviation makes the model look twice as
pessimistic as it is.

References
----------
.. [S97] Swain, D.P. (1997) *A model for optimizing cycling performance
   by varying power on hills and in wind*, Med Sci Sports Exerc
   29(8):1104-8.
.. [A07] Atkinson, G., Peacock, O., Passfield, L. (2007) *Variable versus
   constant power strategies during cycling time-trials: prediction of
   time savings using an up-to-date mathematical model*, J Sports Sci
   25(9):1001-9.
.. [SB17] Sundstrom, D., Backstrom, M. (2017) *Optimization of pacing
   strategies for variable wind conditions in road cycling*, Proc IMechE
   Part P 231(3).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                            # noqa: E402
from coxswain.crew.pacing import CoursePacing, CourseSegment   # noqa: E402

#: Atkinson et al. table: distance m, seconds LOST to +/-15% power in
#: constant conditions, for a 289 W rider.
ATKINSON_CONSTANT = ((16100.0, 3.29), (20000.0, 4.46), (40000.0, 10.43))
ATKINSON_POWER = 289.0
ATKINSON_SPEED = 11.11        # m/s, 40 km/h
#: Their variable-condition savings at +/-10% matched to the course.
ATKINSON_VARIABLE = (("40 km, windy", 26.0), ("40 km, hilly", 126.0))
#: Sundstrom and Backstrom's optimal-control result.
SUNDSTROM_GAIN = 0.029

EXPONENT = 1.0 / 3.0          # d ln v / d ln P, both sports


def convexity_loss(duration, spread, exponent=EXPONENT):
    """Second-order time lost to power spread at fixed mean work, s."""
    return 0.5 * duration * exponent * (exponent + 1.0) * spread ** 2


def cycling_model(distance, segments=40):
    """A 289 W time-triallist, as a :class:`CoursePacing`."""
    stiffness = ATKINSON_POWER / ATKINSON_SPEED ** 3
    return CoursePacing(
        [CourseSegment(distance / segments) for _ in range(segments)],
        lambda v: stiffness * v * v,
        rowers=1, critical_power=ATKINSON_POWER * 0.92, capacity=20000.0,
        efficiency=1.0, drag_area=0.0, max_power=900.0)


def check_constant_conditions():
    """Does the rowing model reproduce the cycling paper's seconds?"""
    print("VALIDATION -- Atkinson et al. 2007, constant conditions")
    print("  their +/-15%% is a sinusoidal swing, so sd = 0.15/sqrt(2) = %.4f"
          % (0.15 / np.sqrt(2)))
    print()
    print("  %-12s %9s %11s %12s %9s"
          % ("distance", "duration", "published", "this model", "ratio"))
    spread = 0.15 / np.sqrt(2)
    for distance, published in ATKINSON_CONSTANT:
        model = cycling_model(distance)
        duration = model.evaluate(
            np.full(len(model.segments), ATKINSON_POWER)).total_time
        predicted = convexity_loss(duration, spread)
        print("  %-12s %8.0fs %11.2f %12.2f %9.2f"
              % ("%.1f km" % (distance / 1000.0), duration, published,
                 predicted, predicted / published))
    print()
    print("  Different sport, different authors, different model.  The")
    print("  agreement is the mechanism being confirmed, not a fit -- there")
    print("  is no free parameter in the expression above.")
    print()


def head_race_in_context(args):
    """Where a HOCR piece sits against the events that have been studied."""
    print("DURATION -- what a head race is actually like")
    print("  %-28s %10s %-34s" % ("event", "duration", "optimal strategy"))
    rows = (
        ("2000 m rowing final", "6-7 min", "1-4-2-3, fast start, sprint"),
        ("5000 m run", "14-20 min", "even, small end spurt"),
        ("20 km cycling TT", "~28 min", "constant if flat"),
        ("20 km TT, hilly or windy", "~28 min", "VARY with the course"),
        ("HOCR masters 8+", "19-20 min", "-- this project --"),
    )
    for name, duration, strategy in rows:
        print("  %-28s %10s %-34s" % (name, duration, strategy))
    print()
    print("  A head race is duration-matched to a 20 km cycling time trial")
    print("  and course-matched to a HILLY one.  That is the analogue to")
    print("  reason from, and it is not the 2 km that rowing studies.")
    print()


def rowing_scatter(args):
    """The same cost curve, for this boat on flat water."""
    boat = catalog.eight(rate=32.0, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)
    stiffness = 0.8 * 313.0 * 8 / 4.23 ** 3
    segments = [CourseSegment(args.distance / args.segments)
                for _ in range(args.segments)]
    model = CoursePacing(segments, lambda v: stiffness * v * v)
    baseline = model.flat_power()
    duration = model.evaluate(
        np.full(args.segments, baseline)).total_time

    print("COST OF SCATTER -- this eight, %.1f km of flat water, %.0f s"
          % (args.distance / 1000.0, duration))
    print("  %-10s %14s %13s %13s"
          % ("power sd", "speed-coef sd", "measured s", "convexity s"))
    for spread in (0.02, 0.05, 0.10, 0.15, 0.20):
        lost, achieved = model.price_scatter(spread, samples=args.samples)
        realised = spread * np.sqrt((args.segments - 1) / args.segments)
        print("  %9.0f%% %14.4f %13.3f %13.3f"
              % (100 * spread, achieved, lost,
                 convexity_loss(duration, realised)))
    print()
    print("  Measured and analytic agree, which is the same check the")
    print("  cycling validation above makes, run on the actual boat.")
    print("  A 15% sinusoidal swing costs a few seconds.  That is the price")
    print("  of RANDOM variation; the next section is the value of the")
    print("  deliberate kind.")
    print()


def what_the_course_is_worth():
    """The other side: what varying WITH the course has been worth."""
    print("VALUE OF VARYING WITH THE COURSE -- published cycling results")
    for label, saving in ATKINSON_VARIABLE:
        print("  %-28s %6.0f s saved at +/-10%% matched to the course"
              % (label, saving))
    print("  %-28s %6.1f%% faster, full optimal control"
          % ("Sundstrom and Backstrom", 100 * SUNDSTROM_GAIN))
    print()
    print("  For comparison, on the surveyed Charles reach this project")
    print("  measures 4.4 s from pacing alone (SOURCES sec. 66) and 80-230 s")
    print("  from the LINE (sec. 67).  The ordering matches cycling, where")
    print("  the hilly course is worth five times the windy one: on both,")
    print("  the terrain dominates the schedule.")
    print()
    print("  The rowing analogue of gradient is DEPTH.  A shoal is a hill.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--distance", type=float, default=4800.0,
                        help="head race distance, m")
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args(argv)

    check_constant_conditions()
    head_race_in_context(args)
    rowing_scatter(args)
    what_the_course_is_worth()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
