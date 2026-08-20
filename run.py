#!/usr/bin/env python
"""Run the simulator from the command line.

This is the front door.  Everything it does is also reachable from the
library; it exists so there is one obvious place to start.

Examples
--------
Simulate an eight and print the summary metrics::

    python run.py

A coxed four at rate 36, with plots::

    python run.py --boat 4+ --rate 36 --plot

Write an animation::

    python run.py --movie out/eight.mp4

Sweep the effect of water depth (the Charles is 2-4 m in places)::

    python run.py --depth-sweep

Report the navigable channel extracted from the Charles depth contours,
and the turn radius the river demands::

    python run.py --charles

Compare the steering authority of the rudder against a crew pressure
split.  On the Charles this is not academic: full rudder alone holds a
259 m turn radius and 19% of the reach is tighter than that::

    python run.py --steer

Open the interactive 3-D viewer with a time slider::

    python run.py --view
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from coxswain.boats import catalog
from coxswain.sim.simulator import RowingSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="6-DOF rowing shell simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    parser.add_argument("--boat", default="8+", choices=sorted(catalog.CATALOG),
                        help="boat class from the catalog (default: 8+)")
    parser.add_argument("--rate", type=float, default=32.0,
                        help="stroke rate in strokes/min (default: 32)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="simulated seconds (default: 30)")
    parser.add_argument("--dt", type=float, default=0.004,
                        help="integration step in seconds (default: 0.004)")
    parser.add_argument("--speed", type=float, default=None,
                        help="initial surge speed; default is the boat's "
                             "steady-state estimate")
    parser.add_argument("--depth", type=float, default=None,
                        help="water depth in metres (default: deep water)")

    parser.add_argument("--plot", action="store_true",
                        help="show the standard diagnostic plots")
    parser.add_argument("--view", action="store_true",
                        help="open the interactive 3-D viewer")
    parser.add_argument("--movie", metavar="PATH",
                        help="write an .mp4 or .gif animation")
    parser.add_argument("--frames", type=int, default=72,
                        help="frames in the animation (default: 72)")
    parser.add_argument("--contact-sheet", metavar="PATH",
                        help="write a four-panel still of one stroke")
    parser.add_argument("--depth-sweep", action="store_true",
                        help="report mean speed against water depth")
    parser.add_argument("--charles", action="store_true",
                        help="report the extracted Charles channel and what "
                             "turn radius it demands")
    parser.add_argument("--steer", action="store_true",
                        help="turn radius available from rudder and from "
                             "crew pressure split, in the full model")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the summary table")
    return parser


def summarise(result, boat) -> None:
    """Print the headline numbers for one run."""
    speed = result.mean_speed()
    print(f"\n{boat.name}  --  rate {boat.timing.rate:.0f} spm, "
          f"{result.time[-1]:.0f} s simulated")
    print(f"  total mass            {boat.total_mass:7.1f} kg "
          f"(hull {boat.hull_mass:.0f}, crew {boat.crew_mass:.0f}, "
          f"cox {boat.coxswain_mass:.0f})")
    print(f"  mean speed            {speed:7.3f} m/s"
          f"   ({2000.0 / speed:.1f} s per 2 km)")
    print(f"  speed fluctuation     "
          f"{result.speed_fluctuation_ratio() * 100:7.2f} %"
          f"   (measured 8+: 5-7 %)")
    print(f"  pitch amplitude       "
          f"{np.degrees(result.pitch_amplitude()):7.3f} deg")
    print(f"  roll amplitude        "
          f"{np.degrees(result.roll_amplitude()):7.3f} deg")
    print(f"  heave amplitude       "
          f"{result.heave_amplitude() * 1000:7.1f} mm")


def depth_sweep(boat, args) -> None:
    """Mean speed against water depth -- the Charles is 2-4 m in places."""
    from coxswain.hydro.shallow import ShallowWaterModel

    print(f"\n{boat.name}: effect of water depth at rate {args.rate:.0f}")
    print(f"{'depth (m)':>10}{'mean speed':>12}{'vs deep':>10}"
          f"{'2 km time':>12}{'depth Froude':>14}")
    deep = None
    for depth in (None, 10.0, 6.0, 4.0, 3.0, 2.5, 2.0):
        variant = catalog.build(args.boat, rate=args.rate)
        if depth is not None:
            variant.shallow = ShallowWaterModel(depth=depth)
        result = RowingSimulator(variant).run(
            duration=args.duration, dt=args.dt,
            surge_speed=args.speed or 5.0)
        speed = result.mean_speed()
        if deep is None:
            deep = speed
        label = "deep" if depth is None else f"{depth:.1f}"
        froude = "" if depth is None else \
            f"{speed / np.sqrt(9.80665 * depth):.3f}"
        print(f"{label:>10}{speed:>12.3f}{(speed / deep - 1) * 100:>9.1f}%"
              f"{2000.0 / speed:>12.1f}{froude:>14}")


def charles_report() -> None:
    """The extracted channel, and the turn radius the river demands."""
    from coxswain.river import charles

    channel = charles.charles_channel()
    line = channel.centreline()
    print(f"\nCharles channel, from the CRAB one-foot depth contours")
    print(f"  water area        {channel.water_area / 1e6:6.2f} km2")
    print(f"  navigable area    {channel.navigable_area / 1e6:6.2f} km2")
    length = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
    print(f"  centreline        {length / 1000:6.2f} km")

    depth = np.array([channel.depth[channel.index_of(x, y)] for x, y in line])
    half = channel.half_width_along(line)
    print(f"  depth on the line {np.nanmin(depth):5.2f} - "
          f"{np.nanmax(depth):.2f} m, median {np.nanmedian(depth):.2f}")
    print(f"  half-width        {half.min():5.1f} - {half.max():.1f} m, "
          f"median {np.median(half):.1f}")

    kernel = np.ones(25) / 25
    smooth = np.column_stack([np.convolve(line[:, i], kernel, mode="valid")
                              for i in (0, 1)])
    d1 = np.gradient(smooth, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.hypot(d1[:, 0], d1[:, 1])
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    radius = 1.0 / np.maximum(np.abs(cross) / np.maximum(speed ** 3, 1e-12),
                              1e-9)
    print("\n  turn radius demanded:")
    for pct in (1, 5, 10, 25, 50):
        print(f"    tightest {pct:2d}% of the reach needs "
              f"R <= {np.percentile(radius, pct):6.0f} m")


def steering_report(args) -> None:
    """Turn radius available from each control, in the full 6-DOF model."""
    from coxswain.sim.control import Coxswain

    boat = catalog.build(args.boat, rate=args.rate)
    speed = args.speed or 5.2
    print(f"\n{boat.name}: steady turn radius at {speed:.1f} m/s")
    print(f"{'control':>26}{'yaw rate':>11}{'radius':>10}")

    def steady(rudder=0.0, split=0.0):
        cox = Coxswain(rudder_override=lambda t, s: rudder,
                       pressure_split=split)
        result = RowingSimulator(boat, coxswain=cox).run(
            duration=12.0, dt=0.008, surge_speed=speed)
        rate = float(np.mean(result.omega[2][result.last_cycles(2)]))
        return rate, (speed / abs(rate) if abs(rate) > 1e-6 else float("inf"))

    for label, kwargs in (
        ("rudder 4 deg", dict(rudder=np.radians(4.0))),
        ("rudder 8 deg", dict(rudder=np.radians(8.0))),
        ("rudder 12 deg", dict(rudder=np.radians(12.0))),
        ("split 15%", dict(split=0.15)),
        ("split 30%", dict(split=0.30)),
        ("rudder 12 + split 30%", dict(rudder=np.radians(12.0), split=0.30)),
    ):
        rate, radius = steady(**kwargs)
        print(f"{label:>26}{np.degrees(rate):10.2f}d{radius:9.0f}m")
    print("\n  the Charles demands 103-146 m at its tightest bends")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    boat = catalog.build(args.boat, rate=args.rate)

    if args.charles:
        charles_report()
        return 0

    if args.steer:
        steering_report(args)
        return 0

    if args.depth_sweep:
        depth_sweep(boat, args)
        return 0

    if args.depth is not None:
        from coxswain.hydro.shallow import ShallowWaterModel
        boat.shallow = ShallowWaterModel(depth=args.depth)

    simulator = RowingSimulator(boat)
    result = simulator.run(duration=args.duration, dt=args.dt,
                           surge_speed=args.speed or 5.0)

    if not args.quiet:
        summarise(result, boat)

    if args.plot:
        import matplotlib.pyplot as plt
        from coxswain.viz import plots
        plots.dashboard(result, boat, simulator)
        plt.show()

    if args.movie or args.contact_sheet or args.view:
        from coxswain.viz.scene3d import BoatScene
        scene = BoatScene(boat, result)
        if args.contact_sheet:
            path = scene.contact_sheet(args.contact_sheet)
            print(f"  wrote {path}")
        if args.movie:
            path = scene.write_movie(args.movie, n_frames=args.frames)
            print(f"  wrote {path}")
        if args.view:
            scene.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
