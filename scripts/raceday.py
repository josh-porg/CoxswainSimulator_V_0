"""What the wind is doing this morning, and what it will cost.

    python scripts/raceday.py
    python scripts/raceday.py --speed 8 --from 340        # a what-if

This is deliberately **not** a line-picker.  It was built to be one -- read
the stations, re-optimise, hand over a different racing line -- and
``scripts/wind_scenarios.py`` then showed that re-optimising a line for
the forecast is worth between 0.1 and 0.5 seconds, while the wind itself
is worth up to 133.  Steering for shelter is not a decision; the line is
the line.

So what a coxswain can actually use on the morning is the other two
things: **which wind you are really in** -- which is not always what the
nearest station says -- and **what it is going to cost**, so the plan and
the split are set against the right race rather than against still air.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.hydro.canopy import ShelteredWind           # noqa: E402
from coxswain.river import charles, lines, stations       # noqa: E402
from coxswain.river.charts import CourseGeometry          # noqa: E402
from coxswain.river.route import RouteEvaluator           # noqa: E402
from coxswain.river.structures import charles_structures  # noqa: E402
from coxswain.river.trajectory import ReducedModel        # noqa: E402

RACE_LENGTH = 4822.0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=None,
                        help="skip the stations; use this 10 m wind, m/s")
    parser.add_argument("--from", dest="direction", type=float, default=None,
                        help="skip the stations; wind comes FROM this bearing")
    parser.add_argument("--race-time", type=float, default=1140.0,
                        help="your still-air time, seconds")
    args = parser.parse_args(argv)

    if args.speed is not None and args.direction is not None:
        speed, direction = args.speed, args.direction
        note = "hypothetical -- no station consulted"
        readings = []
    else:
        readings = []
        for identifier in stations.STATIONS:
            try:
                readings.append(stations.latest(identifier))
            except Exception as error:                     # noqa: BLE001
                print("  %s did not answer (%s)" % (identifier,
                                                    type(error).__name__))
        if not readings:
            raise SystemExit("no station reported; pass --speed and --from")
        print("stations")
        for observation in readings:
            station = stations.STATIONS[observation.station]
            print("  %-5s %-28s %4.0f km  %4.1f m/s from %03.0f  "
                  "-> open-exposure %4.1f m/s"
                  % (observation.station, station.name[:28],
                     station.distance_km, observation.speed,
                     observation.direction,
                     stations.potential_wind(observation))
                  if not observation.calm else
                  "  %-5s %-28s %4.0f km  calm or variable this cycle"
                  % (observation.station, station.name[:28],
                     station.distance_km))
        print("  (%s)" % readings[0].timestamp[:16])
        print()
        speed, direction, note = stations.charles_reference(readings)

    if not np.isfinite(direction) or speed < 0.3:
        print("over the basin: calm")
        print("  %s" % note)
        return 0
    print("over the basin: %.1f m/s from %03.0f" % (speed, direction))
    print("  %s" % note)
    print()

    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=10)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    gates = CourseGeometry(channel=raster).gates_on_course()
    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    route = lines.legalise(lines.arch_route(course, raster, gates,
                                            margin=4.0),
                           course, raster, gates, margin=4.0)
    reference = RACE_LENGTH / args.race_time

    def score(field):
        ev = RouteEvaluator(course, flow=flow, reference_speed=reference,
                            upstream=True, margin=4.0, minimum_depth=1.2,
                            n_samples=900)
        ev.with_steering(ReducedModel(), raster=raster, gates=gates)
        ev.with_exertion()
        if field is not None:
            ev.with_wind(field, boat=boat)
        return float(ev.evaluate(route).elapsed_clean)

    structures = charles_structures()
    calm = score(None)
    field = ShelteredWind(structures, raster, speed, direction, height=0.43)
    windy = score(field)
    print("what it costs")
    print("  still air on this line        %8.1f s" % calm)
    print("  today                         %8.1f s   %+.1f s" % (windy,
                                                                 windy - calm))
    print()

    print("where on the course the wind bites")
    crew_field = ShelteredWind(structures, raster, speed, direction,
                               height=1.5)
    print("  %8s %10s %10s" % ("station", "at 0.43 m", "felt at 1.5 m"))
    for st in np.linspace(300.0, course.length - 300.0, 8):
        point = course.offset_position(np.array([st]), np.array([0.0]))[0]
        print("  %8.0f %9.2f  %9.2f"
              % (st, field.speed_at(point[0], point[1]),
                 crew_field.speed_at(point[0], point[1])))
    print()
    print("  a forecast wind is quoted at 10 m over open ground.  What the")
    print("  crew is in is the 1.5 m column, and what pushes the boat is the")
    print("  0.43 m one -- the area-weighted height of hull, bodies and oars.")
    print("  Do not brief the crew off the forecast number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
