r"""Where to spend the anaerobic reserve on the Charles.

    python scripts/course_pacing.py

The two-parameter model (:mod:`coxswain.crew.exertion`) gives one power
for the whole race, ``P = CP + W'/T``.  On a course whose current, depth
and shelter all vary that is not the fastest schedule, and
:mod:`coxswain.crew.pacing` says by how much.

The result is counter-intuitive and worth stating plainly: **push hardest
where the boat is slowest.**  Extra watts applied in slow water buy more
seconds than the same watts in fast water, because time is distance over
speed and it is the seconds that are being minimised.  A crew that eases
in the fast stretches and pushes in the slow ones lowers its average speed
where the speed is cheap and raises it where it is dear.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402
from coxswain.hydro.resistance import hull_resistance       # noqa: E402
from coxswain.river.charles import charles_course           # noqa: E402


def hull_drag(boat):
    """``v -> newtons`` for this hull in deep water, still air."""
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)

    def drag(speed):
        force, _ = hull_resistance(
            np.array([float(speed), 0.0, 0.0]), submerged,
            mean_wetted_length=boat.length, water=boat.water,
            coefficients=boat.resistance)
        return abs(float(force[0]))
    return drag


def build_wind(speed: float, wind_from: float):
    """The sheltered wind field, or ``None`` if the data is not to hand."""
    if speed <= 0.0:
        return None
    from coxswain.hydro.canopy import ShelteredWind
    from coxswain.river import charles as charles_module
    from coxswain.river.structures import charles_structures

    return ShelteredWind(charles_structures(), charles_module.charles_channel(),
                         speed, wind_from, height=1.5)


def build_segments(course, n: int, boat, wind=None):
    """Cut the reach into ``n`` segments and read the river on each."""
    from coxswain.hydro.shallow import ShallowWaterModel

    stations = np.linspace(0.0, course.length, n + 1)
    shallow = boat.shallow if boat.shallow is not None else ShallowWaterModel()
    segments = []
    for start, end in zip(stations[:-1], stations[1:]):
        middle = 0.5 * (start + end)
        point = course.position_at(middle)
        heading = float(course.heading_at(middle))
        tangent = np.array([np.cos(heading), np.sin(heading)])
        current = np.asarray(course.current_at(point[0], point[1]))[:2]
        # Positive helps: project the current onto the direction of travel.
        along = float(np.dot(current, tangent))
        # The DEPTH goes in, not a factor evaluated at some reference
        # speed.  The shallow-water correction runs on the depth Froude
        # number, so freezing it at one speed would discard exactly the
        # nonlinearity that makes depth matter to a pacing decision.
        depth = float(course.depth_at(point[0], point[1]))

        headwind = 0.0
        if wind is not None:
            # The sheltered field gives a SPEED at chest height; the
            # component that matters is the one along the direction of
            # travel, and it is the *variation* of that down the reach
            # that rewards a variable schedule.
            local = float(wind.speed_at(point[0], point[1]))
            headwind = -local * float(np.dot(wind._towards, tangent))

        segments.append(CourseSegment(length=float(end - start),
                                      current=along, depth=max(depth, 0.30),
                                      headwind=headwind,
                                      label="%.0f m" % middle))
    return segments


def build_boat(kind: str, rate: float):
    """A masters women's crew in the named hull.

    The catalogue's defaults are heavyweight men; a women's masters crew
    is lighter and shorter, and both change the displacement and so the
    wetted area.  Getting this wrong is not cosmetic -- it was the eight
    that this project was calibrated against, and the boat actually being
    raced is a four.
    """
    if kind in ("4+", "four", "coxed_four"):
        return catalog.coxed_four(rate=rate, rower_mass=68.0,
                                  rower_stature=1.70, coxswain_mass=68.0)
    return catalog.eight(rate=rate, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--statistic", default="median")
    parser.add_argument("--wind", type=float, default=0.0,
                        help="reference wind speed at 10 m, m/s "
                             "(0 disables the wind field)")
    parser.add_argument("--wind-from", type=float, default=250.0,
                        help="meteorological bearing the wind comes from")
    parser.add_argument("--boat", default="8+", choices=["8+", "4+"],
                        help="which hull to pace")
    parser.add_argument("--rate", type=float, default=None,
                        help="stroke rate; defaults to 32 for an eight, "
                             "30 for a four")
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 30.0 if args.boat == "4+" else 32.0

    boat = build_boat(args.boat, args.rate)
    course = charles_course(month=args.month, statistic=args.statistic)
    wind = build_wind(args.wind, args.wind_from)
    segments = build_segments(course, args.segments, boat, wind)
    model = CoursePacing(segments, hull_drag(boat),
                         rowers=boat.n_seats,
                         shallow_model=boat.shallow)

    flat = model.flat_power()
    flat_plan = model.evaluate(np.full(len(segments), flat))
    plan, amplitude = model.optimise(span=200.0, samples=81)

    print("Charles reach, %.0f m in %d segments, %s %s discharge"
          % (course.length, len(segments),
             args.statistic, ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov",
                              "Dec"][args.month]))
    print()
    print("  flat pacing   %7.1f W per rower   %8.2f s" % (flat,
                                                           flat_plan.total_time))
    print("  optimal       %7.1f W mean        %8.2f s"
          % (np.average(plan.powers, weights=plan.durations),
             plan.total_time))
    print("  SAVED                              %8.2f s"
          % (flat_plan.total_time - plan.total_time))
    print("  spread wanted %7.1f W between hardest and easiest water"
          % (plan.powers.max() - plan.powers.min()))
    print("  reserve left  %7.1f J of %.0f (min over the race %.1f)"
          % (plan.reserve[-1], model.capacity, plan.reserve.min()))
    print()

    driver = model.driver(flat)
    print("  %-10s %9s %9s %7s %9s %9s %9s %8s"
          % ("station", "current", "headwind", "depth", "e*k", "flat W",
             "opt W", "delta"))
    for segment, drive, power in zip(segments, driver, plan.powers):
        print("  %-10s %9.3f %9.2f %7.2f %9.5f %9.1f %9.1f %+8.1f"
              % (segment.label, segment.current, segment.headwind,
                 segment.depth, drive, flat, power, power - flat))
    print()
    print("  k is speed through water over speed over ground.  Where k > 1")
    print("  the river is against the boat and the optimum asks for more;")
    print("  where k < 1 it is helping and the optimum eases.  In still")
    print("  water every k is 1 and the schedule is flat, which is the")
    print("  check that nothing course-dependent has been assumed.")
    print()
    print("  Read the amplitude, not just the seconds.  A large spread on a")
    print("  small saving means the optimum is flat-bottomed and the crew")
    print("  can round the schedule off freely; a small spread on a large")
    print("  saving means the opposite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
