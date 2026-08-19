"""Command-line entry point for visualising a run.

    python -m coxswain.viz --boat 8+ --rate 32 --duration 16
    python -m coxswain.viz --boat 4+ --show-3d
    python -m coxswain.viz --boat 1x --rate 30 --movie out/single.mp4

Writes the 2-D dashboard, a 3-D contact sheet of one stroke and a set of
still views into ``--out``, then optionally opens the interactive scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coxswain.viz",
        description="Simulate a boat and render the standard diagnostics.",
    )
    parser.add_argument("--boat", default="8+",
                        help="catalog name: 8+, 4+, 1x (default: 8+)")
    parser.add_argument("--rate", type=float, default=32.0,
                        help="stroke rate in strokes per minute")
    parser.add_argument("--duration", type=float, default=16.0,
                        help="simulated seconds")
    parser.add_argument("--dt", type=float, default=0.006,
                        help="fixed integration step")
    parser.add_argument("--speed", type=float, default=4.8,
                        help="initial surge speed")
    parser.add_argument("--out", default="out",
                        help="output directory (default: out)")
    parser.add_argument("--show-3d", action="store_true",
                        help="open the interactive 3-D scene with a time slider")
    parser.add_argument("--view", default="iso",
                        help="camera preset for the interactive scene")
    parser.add_argument("--movie", default=None,
                        help="also record the last two strokes to this .mp4/.gif")
    parser.add_argument("--forces", action="store_true",
                        help="draw force arrows in the 3-D views")
    parser.add_argument("--no-3d", action="store_true",
                        help="skip everything needing PyVista")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib
    if not args.show_3d:
        matplotlib.use("Agg", force=True)

    from ..boats import catalog
    from ..sim.simulator import RowingSimulator
    from . import plots

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"building {args.boat} at {args.rate:.0f} spm ...")
    boat = catalog.build(args.boat, rate=args.rate)
    simulator = RowingSimulator(boat)

    print(f"integrating {args.duration:.0f} s at dt={args.dt} ...")
    result = simulator.run(duration=args.duration, surge_speed=args.speed,
                           dt=args.dt)

    if not result.is_finite:
        print("  WARNING: the trajectory contains non-finite values",
              file=sys.stderr)

    print(f"  mean speed          {result.mean_speed():.3f} m/s")
    print(f"  speed fluctuation   {result.speed_fluctuation_ratio() * 100:.0f} %")
    print(f"  heave amplitude     {result.heave_amplitude() * 1000:.1f} mm")
    print(f"  pitch amplitude     {np.degrees(result.pitch_amplitude()):.3f} deg")
    print(f"  roll amplitude      {np.degrees(result.roll_amplitude()):.3f} deg")
    print(f"  distance            {result.distance():.1f} m")

    dashboard_path = out / f"dashboard_{args.boat.replace('+', 'p')}.png"
    plots.save_dashboard(result, boat, str(dashboard_path),
                         simulator=simulator)
    print(f"wrote {dashboard_path}")

    if args.no_3d:
        return 0

    try:
        from .scene3d import BoatScene
    except ImportError as exc:
        print(f"skipping 3-D output: {exc}", file=sys.stderr)
        return 0

    scene = BoatScene(boat, result)
    tag = args.boat.replace("+", "p")

    sheet = out / f"stroke_{tag}.png"
    scene.contact_sheet(str(sheet), n_frames=4, view="side", zoom=1.0,
                        window_size=(1800, 420), simulator=simulator,
                        show_forces=args.forces)
    print(f"wrote {sheet}")

    t_still = float(result.time[-1]) - 0.75 * boat.timing.period
    for view, zoom, size in (("iso", 1.1, (1300, 820)),
                             ("stern", 3.0, (900, 700)),
                             ("top", 1.05, (1500, 500))):
        path = out / f"{view}_{tag}.png"
        scene.snapshot(t_still, str(path), view=view, zoom=zoom,
                       window_size=size, simulator=simulator,
                       show_forces=args.forces)
        print(f"wrote {path}")

    if args.movie:
        print(f"recording {args.movie} ...")
        scene.write_movie(args.movie, n_frames=96, view=args.view, zoom=1.1,
                          simulator=simulator, show_forces=args.forces)
        print(f"wrote {args.movie}")

    if args.show_3d:
        print("opening interactive scene -- drag the slider to scrub time")
        scene.show(view=args.view, simulator=simulator,
                   show_forces=args.forces)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
