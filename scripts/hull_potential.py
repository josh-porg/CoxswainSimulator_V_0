"""Solve the potential flow around the real hull, and see what it changed.

    python scripts/hull_potential.py
    python scripts/hull_potential.py --panels 80 160 320

The vortex-wake model represented the hull by thin-body theory: sources on
the centreline at ``sigma(x) = 2 U db/dx``.  That never enforces the thing
it is supposed to -- that water does not flow through the boat -- it only
approximates it, and for a 30:1 hull the approximation is good.  "Good" is
worth checking rather than assuming when the offsets are already in hand.

This solves the boundary condition properly with a Hess-Smith source panel
method, validates it on a case with an exact answer, and then asks the
only question that matters here: **does the follower feel anything
different?**
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.panels import (SourcePanelBody,  # noqa: E402
                                   circle_nodes,
                                   waterline_from_offsets)
from coxswain.hydro.vortex import ThinBody             # noqa: E402

RACE_LENGTH = 4822.0


def validate(panel_counts):
    """A circle, where the answer is known exactly."""
    print("validation: flow past a cylinder, exact surface speed 2U sin(theta)")
    print("  %8s %14s %14s %14s"
          % ("panels", "max |q| error", "closure", "normal residual"))
    ok = True
    for count in panel_counts:
        body = SourcePanelBody(circle_nodes(1.0, count)).solve(1.0)
        theta = np.arctan2(body.control[:, 1], body.control[:, 0])
        error = float(np.abs(np.abs(body.surface_speed())
                             - np.abs(2.0 * np.sin(theta))).max())
        print("  %8d %14.2e %14.2e %14.2e"
              % (count, error, body.closure(), body.normal_residual()))
        ok = ok and error < 1e-3
    print("  %s" % ("PASS" if ok else "FAIL -- the influence coefficients "
                                      "are wrong"))
    print()
    return ok


def hull(boat, speed, panels):
    nodes = waterline_from_offsets(boat.offsets, panels=panels)
    return SourcePanelBody(nodes).solve(speed)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--panels", type=int, nargs="+",
                        default=[40, 120, 400])
    parser.add_argument("--hull-panels", type=int, default=200)
    parser.add_argument("--out", default="out/hull")
    args = parser.parse_args(argv)

    if not validate(args.panels):
        return 1

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / args.race_time
    body = hull(boat, speed, args.hull_panels)
    thin = ThinBody.from_offsets(boat.offsets, speed)

    print("the real hull: %d panels over %.2f m of waterline, beam %.3f m"
          % (body.n_panels, boat.offsets.length, boat.offsets.max_beam))
    print("  closure %.2e, normal residual %.2e m/s on a %.2f m/s stream"
          % (body.closure(), body.normal_residual(), speed))
    surface = body.surface_speed()
    print("  surface speed runs %.3f to %.3f m/s against a %.3f m/s stream"
          % (np.abs(surface).min(), np.abs(surface).max(), speed))
    print("  peak overspeed %.1f%% -- a shell is fine enough that the water"
          % (100 * (np.abs(surface).max() / speed - 1.0)))
    print("  barely notices it, which is the whole point of the shape.")
    print()

    convergence(boat, speed)
    compare(body, thin, speed)
    path = plot(body, thin, speed, args)
    print("wrote", path)
    return 0


def convergence(boat, speed):
    """Is the hull answer converged in the panel count?"""
    print("convergence on the real hull")
    print("  %8s %14s %16s" % ("panels", "peak |q|/U", "at 3 m abeam"))
    probe = np.array([[-17.3, 3.15], [-34.6, 3.15], [-8.0, 3.15]])
    for count in (50, 100, 200, 400):
        body = hull(boat, speed, count)
        far = body.velocity_at(probe)[:, 0].mean()
        print("  %8d %14.4f %16.5f"
              % (body.n_panels, np.abs(body.surface_speed()).max() / speed,
                 far))
    print()


def compare(body, thin, speed):
    """The question that decides whether the upgrade was worth it."""
    print("panel solution against thin-body, in the water's frame")
    print("  %-22s %12s %12s %10s"
          % ("where", "panel m/s", "thin m/s", "ratio"))
    places = (("3.15 m abeam, at the stern", (-1.0, 3.15)),
              ("3.15 m abeam, one length", (-17.3, 3.15)),
              ("3.15 m abeam, two lengths", (-34.6, 3.15)),
              ("on the centreline, 5 m", (-5.0, 0.0)),
              ("on the centreline, one length", (-17.3, 0.0)),
              ("1 m abeam, amidships", (0.0, 1.0)))
    for label, point in places:
        a = float(body.velocity_at([point])[0, 0])
        b = float(thin.velocity_at([point])[0, 0])
        print("  %-22s %12.5f %12.5f %10s"
              % (label, a, b, "%.2f" % (a / b) if abs(b) > 1e-9 else "-"))
    print()
    print("  Both are potential flow, so both decay fast: a source-sink")
    print("  pair's field falls off as the square of distance and there is")
    print("  no vorticity in either to carry anything downstream.  The")
    print("  disturbance a FOLLOWING crew sits in is almost entirely the")
    print("  vortex wake, not this -- which is worth knowing, because it")
    print("  means the expensive half of the hull model matters least for")
    print("  the question that motivated it.")
    print()


def plot(body, thin, speed, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, RULE = "#16211f", "#dce2e0"
    x = np.linspace(-12.0, 12.0, 460)
    y = np.linspace(-3.0, 3.0, 220)
    gx, gy = np.meshgrid(x, y)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    total = body.velocity_at(points, freestream=True)
    magnitude = np.hypot(total[:, 0], total[:, 1]).reshape(gx.shape) / speed

    inside = np.zeros(gx.shape, dtype=bool)
    nodes = body.nodes
    station = nodes[:, 0]
    beam = np.abs(nodes[:, 1])
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            local = np.interp(gx[i, j], np.sort(station),
                              beam[np.argsort(station)])
            inside[i, j] = abs(gy[i, j]) < local * 0.98
    magnitude = np.ma.array(magnitude, mask=inside)

    figure, (ax, bx) = plt.subplots(2, 1, figsize=(11.0, 6.4),
                                    gridspec_kw={"height_ratios": [3, 2]})
    mesh = ax.pcolormesh(gx, gy, magnitude, cmap="viridis", shading="auto",
                         vmin=0.9, vmax=1.1)
    figure.colorbar(mesh, ax=ax, pad=0.01, label="speed / boat speed")
    ax.plot(nodes[:, 0], nodes[:, 1], color="white", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("along the hull, m")
    ax.set_ylabel("across, m")
    ax.set_title("Potential flow around the waterline, %d panels"
                 % body.n_panels, fontsize=11, color=INK, loc="left")

    order = np.argsort(body.control[:, 0])
    upper = order[body.control[order, 1] >= 0]
    bx.plot(body.control[upper, 0], body.pressure_coefficient()[upper],
            color="#1f5673", lw=1.8, label="panel method")
    bx.axhline(0.0, color="#5c6968", lw=0.8)
    bx.invert_yaxis()
    bx.set_xlabel("along the hull, m (bow to the right)")
    bx.set_ylabel("pressure coefficient")
    bx.set_title("Cp along the waterline", fontsize=11, color=INK, loc="left")
    bx.legend(frameon=False, fontsize=8.5)
    bx.grid(color=RULE, lw=0.6)
    bx.set_axisbelow(True)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)

    figure.tight_layout()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "hull_potential.png")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
