"""What does it cost to row in another crew's water?

    python scripts/wake.py
    python scripts/wake.py --race-time 1140 --radius 0.25 0.30 0.40

There is no published model of shell-to-shell wake interference, so this
one is built from the momentum budget and the turbulent-vortex-ring decay
law, and every number it prints is a prediction rather than a fit.  See
:mod:`coxswain.hydro.wake` for the derivation and the one constant that
has to be measured.

The output is deliberately a band, not a number.  The virtual origin of a
puddle is unknown to within a factor of two and the answer is sensitive
to it, so quoting three figures would be dishonest.  The band is still
useful: it separates "worth a stroke" from "worth half a minute", and it
says clearly which of the two it is.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.hydro.wake import PuddleWake, blade_track  # noqa: E402

RACE_LENGTH = 4822.0
#: How race time answers a change in propulsive power, measured by
#: scripts/time_budget.py on this hull at masters speed: v ~ P^0.498, so
#: a 1% power loss is a 0.498% time loss.
SPEED_EXPONENT = 0.498


def leader(race_time, rate, rower_mass=72.0, stature=1.72):
    """The boat in front: its drag, speed and stroke period."""
    boat = catalog.eight(rate=rate, rower_mass=rower_mass,
                         rower_stature=stature)
    speed = RACE_LENGTH / float(race_time)
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, detail = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                                    mean_wetted_length=boat.length,
                                    water=boat.water,
                                    coefficients=boat.resistance)
    return boat, abs(float(force[0])), speed, detail


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--radius", type=float, nargs="+",
                        default=[0.25, 0.30, 0.40],
                        help="candidate puddle birth radii, m -- the one "
                             "constant the model cannot derive")
    parser.add_argument("--gaps", type=float, nargs="+",
                        default=[5, 10, 17.3, 25, 34.6, 50, 69.2, 100, 150])
    parser.add_argument("--out", default="out/wake")
    args = parser.parse_args(argv)

    boat, drag, speed, detail = leader(args.race_time, args.rate)
    track = blade_track(boat)
    period = 60.0 / args.rate
    reference_radius = PuddleWake(drag=drag, speed=speed,
                                  period=period).radius(17.3 / speed)

    print("the boat in front: %.3f m/s, %.0f N of drag (%.0f W), rate %.0f"
          % (speed, drag, drag * speed, args.rate))
    print("  viscous %.0f N, wave %.0f N, shape %.0f N"
          % (detail["viscous"], detail["wave"], detail["shape"]))
    print()
    print("geometry")
    print("  puddle lines sit %.2f m either side of the centreline, %.2f m "
          "apart" % (track, 2 * track))
    print("  the shell is %.2f m wide at the waterline, so the HULL is never "
          "in them" % 0.57)
    print("  along the track the puddles are %.2f m apart -- one per stroke,"
          % (speed * period))
    print("  each about %.1f m across by the time it is a length astern"
          % (2 * float(reference_radius)))
    print()

    reference = PuddleWake(drag=drag, speed=speed, period=period)
    print("the momentum budget fixes the strength, not a fitted constant")
    print("  each blade must replace %.1f N s of momentum per stroke"
          % reference.impulse)
    for radius in args.radius:
        w = PuddleWake(drag=drag, speed=speed, period=period,
                       initial_radius=radius)
        print("  birth radius %.2f m -> fresh puddle runs aft at %.2f m/s, "
              "decay scale %.2f s" % (radius, w.initial_speed, w.time_scale))
    print("  a real eight shows 0.5-1.0 m/s of blade slip, which is the same")
    print("  water seen from the other side -- so 0.30 m is the plausible one")
    print()

    mid = PuddleWake(drag=drag, speed=speed, period=period)
    print("the gap does not have to be chosen carefully -- in an eight")
    print("  the blades span %.1f m of boat and the puddles repeat every "
          "%.1f m," % (1.22 * 7, mid.spacing()))
    print("  which is the same number to within %.0f%%.  So the eight blades"
          % (100 * abs(mid.blades_engaged() - 1.0)))
    print("  sample a whole phase cycle at once: one or two are always in a")
    print("  puddle, and moving the gap changes which ones, not how many.")
    print("  In a four (%.1f m of a %.1f m cycle) the gap would matter."
          % (1.22 * 3, mid.spacing()))
    print()

    print("time lost over a %.0f s race, by gap (a length is 17.3 m)"
          % args.race_time)
    header = "  %8s" % "gap"
    for radius in args.radius:
        header += " %12s" % ("R0=%.2f m" % radius)
    print(header + " %14s" % "one blade, in")
    for gap in args.gaps:
        line = "  %6.0f m" % gap
        severity = 0.0
        for radius in args.radius:
            w = PuddleWake(drag=drag, speed=speed, period=period,
                           initial_radius=radius)
            line += " %11.1f s" % (w.power_penalty(gap) * SPEED_EXPONENT
                                   * args.race_time)
            if abs(radius - 0.30) < 1e-9:
                severity = 100 * float(w.force_loss(gap / speed))
        print(line + " %13.0f%%" % severity)
    print()
    print("  The last column is how much force one blade loses on the strokes")
    print("  it lands in a puddle -- the 'no grip' a crew reports.  The time")
    print("  columns already fold in that only about one blade in six is in")
    print("  one at any moment.")
    print()

    print("the other half of the wake, which helps")
    for gap in (17.3, 34.6, 69.2, 150.0):
        print("  %6.1f m astern: hull wake is worth %+5.1f s to a hull on the "
              "centreline"
              % (gap, -float(mid.hull_benefit(gap)) * SPEED_EXPONENT
                 * args.race_time))
    print()
    print("  net, directly astern at a length: the hull gains a little and")
    print("  the blades lose more.  Move across about %.1f m and your blades"
          % track)
    print("  straddle the gap between their two puddle lines -- one set")
    print("  inside, over their clean centreline, one set outside them --")
    print("  which is the whole argument for coming alongside rather than")
    print("  sitting on somebody's stern.")
    print()

    figure_path = plot(args, drag, speed, period, track)
    print("wrote", figure_path)
    return 0


def plot(args, drag, speed, period, track):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, RULE = "#16211f", "#dce2e0"
    PALETTE = ["#1f5673", "#2a7f62", "#c8901a"]
    gaps = np.linspace(3.0, 160.0, 400)
    figure, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for colour, radius in zip(PALETTE, args.radius):
        w = PuddleWake(drag=drag, speed=speed, period=period,
                       initial_radius=radius)
        loss = w.power_penalty(gaps) * SPEED_EXPONENT * args.race_time
        ax.plot(gaps, loss, color=colour, lw=1.8,
                label="birth radius %.2f m" % radius)
    w = PuddleWake(drag=drag, speed=speed, period=period)
    helped = -w.hull_benefit(gaps) * SPEED_EXPONENT * args.race_time
    ax.plot(gaps, helped, color="#a2382a", lw=1.4, ls="--",
            label="hull wake, which helps")
    ax.axhline(0.0, color="#5c6968", lw=0.8)
    top = ax.get_ylim()[1]
    for n in (1, 2, 4, 8):
        ax.axvline(17.3 * n, color=RULE, lw=0.9)
        ax.text(17.3 * n, top, "%d" % n, fontsize=7.5, color="#5c6968",
                va="top", ha="center",
                bbox=dict(fc="white", ec="none", pad=1.0))
    ax.text(0.985, 0.06, "gridlines are boat lengths astern",
            transform=ax.transAxes, fontsize=7.5, color="#5c6968", ha="right")
    ax.set_xlabel("gap behind the boat in front, m")
    ax.set_ylabel("time lost over %.0f s, s" % args.race_time)
    ax.set_title("Following costs, and the constant nobody has measured",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)

    # -- where the puddles actually are ----------------------------------
    along = np.linspace(0.0, 3.0 * speed * period, 600)
    bx.axhspan(-0.285, 0.285, color="#c9d6d3", zorder=1)
    bx.text(0.3, 0.0, "hull", fontsize=8, va="center", color=INK, zorder=4)
    for sign in (-1, 1):
        for k in range(1, 4):
            centre = k * speed * period
            r = float(w.radius(centre / speed))
            circle = plt.Circle((centre, sign * track), r, color="#a2382a",
                                alpha=0.45, zorder=3)
            bx.add_patch(circle)
        bx.axhline(sign * track, color="#a2382a", lw=0.8, ls=":", zorder=2)
    bx.plot(along, np.full_like(along, 0.0), color=INK, lw=1.0, zorder=4)
    bx.text(0.3, track + 0.55, "their puddles, %.2f m out" % track, fontsize=8,
            color="#a2382a", zorder=4)
    # The follower, one length astern and on the same line: its hull is in
    # the clean centreline wake and its blades are on the puddle lines.
    follow = 17.3
    bx.plot([follow, follow + 8.5], [0.0, 0.0], color="#1f5673", lw=3.0,
            solid_capstyle="butt", zorder=5)
    for sign in (-1, 1):
        bx.plot(np.linspace(follow, follow + 8.5, 8),
                np.full(8, sign * track), "o", ms=5.0, color="#1f5673",
                zorder=6)
        for k in range(8):
            bx.plot([follow + k * 1.22, follow + k * 1.22],
                    [0.0, sign * track], color="#1f5673", lw=0.6, alpha=0.6,
                    zorder=5)
    bx.text(follow + 0.2, -track - 1.05, "you, a length astern: hull clean, "
            "blades on their lines", fontsize=7.5, color="#1f5673", zorder=6)
    bx.set_xlim(0.0, follow + 10.0)
    bx.set_ylim(-track - 1.4, track + 1.4)
    bx.set_xlabel("distance astern, m")
    bx.set_ylabel("across the river, m")
    bx.set_title("Why the hull is safe and the blades are not", fontsize=10.5,
                 color=INK, loc="left")
    bx.set_aspect("equal", adjustable="box")

    for axis in (ax, bx):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.grid(color=RULE, lw=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "wake.png")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
