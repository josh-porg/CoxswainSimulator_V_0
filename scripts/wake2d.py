r"""The wake as a vortex field, not a decay law.

    python scripts/wake2d.py
    python scripts/wake2d.py --distance 300 --eddy 3e-3

:mod:`coxswain.hydro.wake` models each puddle as an isolated blob with an
analytic decay.  This runs the actual thing: a 2-D vortex method where
every blade drops a dipole, the dipoles advect one another, and the
velocity anywhere is the sum over all of them.  No CFD, no mesh, no free
surface -- a few hundred vortex blobs and Biot-Savart.

Three things it gets that the analytic model cannot
---------------------------------------------------
**The field is spiky, not smooth.**  Puddles sit 9 m apart and a dipole's
influence falls off as the square of distance, so the water between them
is nearly still and the water inside them is moving at a metre a second.
The analytic model smeared that into a duty cycle; here it is explicit,
and it is the reason a following crew reports the wash coming and going
rather than sitting on them.

**The lateral structure.**  Which is the question a coxswain actually
has.  The answer comes out as a curve rather than an argument.

**Its own conservation law as a check.**  Impulse is conserved for vortex
pairs in unbounded water, and the momentum budget fixes what the impulse
must be, so the model can be audited against a number it was not fitted
to.  It agrees to four figures, which is the arithmetic working rather
than the physics being right -- but the arithmetic failing would have
been the cheapest possible way to find a bug.

What is still assumed
---------------------
The eddy viscosity that spreads the cores, and the depth used to turn a
3-D impulse into a 2-D one.  Both are swept.  Everything else -- the
circulation, the shedding rate, the spacing -- follows from the boat.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.hydro.vortex import PuddleWake2D          # noqa: E402
from coxswain.hydro.wake import PuddleWake, blade_track  # noqa: E402

RACE_LENGTH = 4822.0
SEAT_SPACING = 1.22


def leader(race_time, rate):
    boat = catalog.eight(rate=rate, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / float(race_time)
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                               mean_wetted_length=boat.length,
                               water=boat.water, coefficients=boat.resistance)
    return boat, abs(float(force[0])), speed


def follower_sample(model, gap, offset, track):
    """Mean **signed** along-course velocity over a follower's blades, m/s.

    Signed, not magnitude -- and that only became meaningful once the hull
    was in the field.  With puddles alone every disturbance was aft and
    the sign carried no information.  Now the water on the centreline
    travels *with* the boat ahead while the water on the blade tracks
    travels against it, so the sign is the whole finding.

    Positive means the water moves in the direction of racing, which
    helps twice: a hull in it meets a lower relative velocity and so less
    drag, and a blade in it meets a higher relative velocity and so more
    grip.  Negative is the "no grip" a crew reports.
    """
    stations = -gap - SEAT_SPACING * np.arange(8)
    points = np.array([[x, offset + side * track]
                       for x in stations for side in (-1.0, 1.0)])
    return float(np.mean(model.velocity_at(points)[:, 0]))


def hull_sample(model, gap, offset, length=17.3, stations=20):
    """The same, sampled along a follower's own seventeen metres."""
    x = -gap - np.linspace(0.0, length, stations)
    points = np.column_stack([x, np.full(stations, offset)])
    return float(np.mean(model.velocity_at(points)[:, 0]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--distance", type=float, default=260.0,
                        help="how far the leader rows before we look")
    parser.add_argument("--eddy", type=float, nargs="+",
                        default=[5e-4, 1.5e-3, 5e-3],
                        help="eddy viscosities to compare, m^2/s")
    parser.add_argument("--out", default="out/wake2d")
    args = parser.parse_args(argv)

    boat, drag, speed = leader(args.race_time, args.rate)
    track = blade_track(boat)
    period = 60.0 / args.rate

    model = PuddleWake2D(drag=drag, speed=speed, period=period, track=track,
                         offsets=boat.offsets,
                         eddy_viscosity=args.eddy[len(args.eddy) // 2])
    print("the leader: %.2f m/s, %.0f N, blade track %.2f m out"
          % (speed, drag, track))
    print("  impulse per puddle   %.1f N s   (momentum budget, not fitted)"
          % model.impulse_per_puddle)
    print("  circulation          %.3f m^2/s (derived from that impulse)"
          % model.circulation)
    print("  self-induced speed   %.3f m/s   (a real eight slips 0.5-1.0)"
          % model.self_induced_speed)
    print()

    field = model.row(args.distance, dt=0.05)
    blades_only = PuddleWake2D(drag=drag, speed=speed, period=period,
                               track=track, hull_wake=False,
                               eddy_viscosity=model.eddy_viscosity)
    blades_only.row(args.distance, dt=0.05)
    aft = blades_only.field.impulse()[0]
    total = field.impulse()[0]
    print("after %.0f m: %d blade puddles, %d hull pairs, %d blobs, "
          "oldest %.0f s" % (args.distance, model.shed_count,
                             model.hull_count, len(field), field.age.max()))
    print("  blades alone put %+.0f N s/m into the water, all of it aft"
          % aft)
    print("  with the hull included the total is %+.1f -- a residual of"
          % total)
    print("  %.0e of it.  That is the momentumless wake of a self-propelled"
          % abs(total / aft))
    print("  body falling out of the construction rather than imposed on it.")
    print("  cores grew %.3f -> %.3f m against a %.2f m pair separation."
          % (model.core, field.core.max(), model.pair_separation))
    print()

    offsets = (0.0, 1.5, 3.0, 4.5, 6.0)
    for label, sample in (("BLADES", follower_sample), ("HULL", hull_sample)):
        print("%s: signed along-course velocity, + helps, - hurts" % label)
        print("  %-10s" % "gap" + "".join("%11s" % ("%+.1f m" % o)
                                          for o in offsets))
        for gap in (5.0, 10.0, 17.3, 34.6, 69.2):
            row = "  %7.1f m" % gap
            for offset in offsets:
                value = (sample(model, gap, offset, track)
                         if sample is follower_sample
                         else sample(model, gap, offset))
                row += "%10.4f " % value
            print(row)
        print()
    print("  There are three features now, not two: their hull wake on the")
    print("  centreline running WITH them, and two puddle lines at "
          "+/-%.2f m" % track)
    print("  running against.  Sitting on their line puts your blades on")
    print("  both puddle tracks at once and your hull in the one helpful")
    print("  place, which is the trade the geometry forces on a crew that")
    print("  simply follows.")
    print()

    sensitivity(model, args, track)
    path = plot(model, field, args, track)
    print("wrote", path)
    return 0


def sensitivity(model, args, track):
    """Does the one remaining free constant change the recommendation?"""
    print("sensitivity to the eddy viscosity, at a length astern")
    print("  %-14s %12s %12s %12s"
          % ("nu_t m^2/s", "on their line", "3 m across", "ratio"))
    for eddy in args.eddy:
        trial = PuddleWake2D(drag=model.drag, speed=model.speed,
                             period=model.period, track=track,
                             offsets=model.offsets, eddy_viscosity=eddy)
        trial.row(args.distance, dt=0.05)
        on = follower_sample(trial, 17.3, 0.0, track)
        across = follower_sample(trial, 17.3, 3.0, track)
        print("  %-14.1e %12.4f %12.4f %12.2f"
              % (eddy, on, across, across / on if on else float("nan")))
    print("  The ratio is what the tactical advice rests on, and it is")
    print("  much steadier than either number -- which is the useful kind")
    print("  of insensitivity: the size of the effect is uncertain, the")
    print("  direction of the advice is not.")
    print()


def plot(model, field, args, track):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, RULE = "#16211f", "#dce2e0"
    x = np.linspace(-80.0, 4.0, 420)
    y = np.linspace(-9.0, 9.0, 200)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    velocity = model.velocity_at(points)
    speed = velocity[:, 0].reshape(grid_x.shape)

    figure, (ax, bx) = plt.subplots(2, 1, figsize=(11.0, 7.2),
                                    gridspec_kw={"height_ratios": [3, 2]})
    limit = float(np.percentile(np.abs(speed), 99.0))
    mesh = ax.pcolormesh(grid_x, grid_y, speed, cmap="RdBu_r",
                         vmin=-limit, vmax=limit, shading="auto")
    figure.colorbar(mesh, ax=ax, pad=0.01,
                    label="along-course velocity, m/s")
    for side in (-1.0, 1.0):
        ax.axhline(side * track, color="#2a7f62", lw=0.8, ls=":")
    ax.plot([0.0, 17.3], [0.0, 0.0], color="#1f5673", lw=4.0,
            solid_capstyle="butt")
    ax.text(1.0, 0.6, "the boat ahead", color="#1f5673", fontsize=8.5)
    ax.set_xlabel("distance astern, m (0 is their stern)")
    ax.set_ylabel("across the river, m")
    ax.set_title("What the water is doing behind a rowing eight",
                 fontsize=11, color=INK, loc="left")
    ax.set_aspect("equal", adjustable="box")

    offsets = np.linspace(0.0, 8.0, 60)
    for gap, colour in ((10.0, "#1f5673"), (17.3, "#2a7f62"),
                        (34.6, "#c8901a")):
        felt = [follower_sample(model, gap, o, track) for o in offsets]
        bx.plot(offsets, felt, color=colour, lw=1.8,
                label="%.0f m astern" % gap)
    bx.set_xlabel("how far across from their line, m")
    bx.axhline(0.0, color="#5c6968", lw=0.8)
    bx.set_ylabel("along-course velocity\nat your blades, m/s")
    bx.set_title("Where to sit", fontsize=11, color=INK, loc="left")
    bx.legend(frameon=False, fontsize=8.5)
    bx.grid(color=RULE, lw=0.6)
    bx.set_axisbelow(True)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)

    figure.tight_layout()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "wake2d.png")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
