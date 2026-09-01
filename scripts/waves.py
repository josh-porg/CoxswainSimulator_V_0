r"""The waves the boat radiates, and what the river bed does to them.

    python scripts/waves.py
    python scripts/waves.py --rate 32 --depths 1.2 1.8 3.0 6.0

Formaggia, Miglio, Mola and Montano [FMM08]_ close their paper with a
figure of the free surface radiated by a scull oscillating in heave and in
pitch -- the waves the crew's own secondary motion throws away, which they
find accounts for as much as 10% of the total energy dissipation.  This
reproduces that figure for this eight, from their equations, and then asks
the question they set aside: what the depth does.

Their formulation, and the one bit of it that carries the depth
----------------------------------------------------------------
The radiated potential solves a linear problem: Laplace inside, no flow
through the bed, the linearised free-surface condition on top, the body's
generalised normal on the hull, and a radiation condition at infinity::

    -Lap(Psi) = 0                 in the fluid
    dPsi/dZ = 0                   on the bed
    dPsi/dZ - (omega^2/g) Psi = 0 on the free surface
    dPsi/dn = N_s                 on the hull

and the free surface follows from the potential as

.. math::

    \eta(x, y) = \mathrm{Re}\left\{\frac{i\omega}{g}\Psi_s(x, y, 0)\right\}

The decay condition carries the wavenumber, and the wavenumber carries the
depth -- this is the paper's own dispersion relation, equation (20e):

.. math::

    k \tanh(kH) = \frac{\omega^2}{g}

The paper then assumes ``kH >> 1`` and takes ``k = omega^2/g``, which is
the deep-water limit.  **On the Charles that assumption is not safe.**
The relation is solved here rather than assumed, and the difference is
reported.

Reproducing the pattern without a boundary element solve
--------------------------------------------------------
The paper solves the potential problem by BEM.  What sets the *pattern*,
though, is the source distribution on the hull and the outgoing Green's
function, and both are available in closed form: a free-surface source
oscillating at frequency ``omega`` radiates ``H_0^(1)(kr)``, and the
generalised normal gives the strengths.  Heave puts every source in phase
-- concentric rings.  Pitch flips the sign fore and aft -- a dipole with a
null on the beam.  That is exactly the structure of their Figure 11, and
it comes from their equations rather than from a fit to their picture.

What the bed actually does, and where KdV comes in
---------------------------------------------------
Two different depth effects, and they are often confused:

**The radiated waves** shorten as the water shallows, through the
dispersion relation above.  It is a modest effect at Charles depths.

**The steady wake** changes REGIME, at the depth Froude number
``Fh = U / sqrt(gH)``.  Below one the Kelvin wedge widens from its deep
water 19 degrees 28 minutes; at one the wave speed limit equals the boat
speed, energy cannot escape ahead, and the response becomes a train of
upstream solitons -- **that** is the regime the Korteweg-de Vries equation
describes, and it is why KdV keeps coming up in shallow-water ship
literature.  Above one the transverse waves vanish entirely and what is
left is a Mach cone of half-angle ``arcsin(1/Fh)``.

It is not reflection off the bottom.  The bed does not bounce waves back
so much as cap how fast they can travel, and everything else follows from
that cap.

References
----------
.. [FMM08] Formaggia, L., Miglio, E., Mola, A. and Montano, A. (2008)
   *A model for the dynamics of rowing boats*, Int. J. Numerical Methods
   in Fluids.  Equations 19-24 and Figure 11.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402

GRAVITY = 9.80665
RACE_LENGTH = 4822.0
INK, MUTED, RULE = "#16211f", "#5c6968", "#dce2e0"


def wavenumber(omega: float, depth: float, tol: float = 1e-12) -> float:
    """Solve ``k tanh(kH) = omega^2 / g`` for k.

    Newton from the deep-water guess, which is the paper's assumption and
    is also the right starting point for it.
    """
    target = omega ** 2 / GRAVITY
    k = target if depth <= 0 else max(target, 1e-6)
    for _ in range(80):
        t = np.tanh(k * depth)
        f = k * t - target
        derivative = t + k * depth * (1.0 - t ** 2)
        step = f / max(derivative, 1e-12)
        k -= step
        if abs(step) < tol:
            break
    return float(k)


def radiated(boat, omega, depth, mode, extent=60.0, grid=440):
    """Free-surface elevation radiated by the hull oscillating in ``mode``.

    ``mode`` is "heave" or "pitch", the paper's s = 2 and s = 3.  Sources
    are distributed along the waterline with strength given by the
    generalised normal, and each radiates an outgoing cylindrical wave.
    """
    from scipy.special import hankel1

    station = np.asarray(boat.offsets.station, dtype=float)
    half_beam = 0.5 * np.asarray(boat.offsets.beam, dtype=float)
    keep = half_beam > 1e-4
    station, half_beam = station[keep], half_beam[keep]
    # Strength proportional to the local waterplane width, which is what
    # a heaving section displaces; pitch adds the moment arm and its sign.
    strength = 2.0 * half_beam
    if mode == "pitch":
        strength = strength * (-station)
    strength = strength / np.abs(strength).sum()

    k = wavenumber(omega, depth)
    axis = np.linspace(-extent, extent, grid)
    gx, gy = np.meshgrid(axis, axis)
    field = np.zeros(gx.shape, dtype=complex)
    for x0, q in zip(station, strength):
        r = np.hypot(gx - x0, gy)
        field += q * hankel1(0, k * np.maximum(r, 0.35))
    elevation = np.real(1j * omega / GRAVITY * field)
    return gx, gy, elevation, k


def depth_froude(course, speed, samples=600):
    """Depth Froude number along the race line, from the real bathymetry."""
    station = np.linspace(0.0, course.length, samples)
    points = course.offset_position(station, np.zeros_like(station))
    depth = np.asarray(course.depth(points[:, 0], points[:, 1]), dtype=float)
    depth = np.maximum(depth, 0.15)
    return station, depth, speed / np.sqrt(GRAVITY * depth)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--depths", type=float, nargs="+",
                        default=[1.2, 1.8, 3.0, 6.0])
    parser.add_argument("--pattern-depth", type=float, default=3.0)
    parser.add_argument("--out", default="out/waves")
    args = parser.parse_args(argv)

    speed = RACE_LENGTH / args.race_time
    boat = catalog.eight(rate=args.rate, rower_mass=72.0, rower_stature=1.72)
    omega = 2.0 * np.pi / boat.timing.period

    print("the secondary motion radiates at the stroke frequency")
    print("  rate %.0f  ->  T = %.3f s, omega = %.3f rad/s"
          % (args.rate, boat.timing.period, omega))
    print("  the paper's figure 11 used omega = 4.136 rad/s, a racing four")
    print()

    print("the dispersion relation the paper writes down, solved not assumed")
    deep = omega ** 2 / GRAVITY
    print("  deep-water limit k = omega^2/g = %.4f /m  (lambda %.2f m)"
          % (deep, 2 * np.pi / deep))
    print("  %8s %10s %10s %12s %10s"
          % ("depth m", "kH", "k /m", "lambda m", "vs deep"))
    for depth in args.depths:
        k = wavenumber(omega, depth)
        print("  %8.1f %10.2f %10.4f %12.2f %9.1f%%"
              % (depth, k * depth, k, 2 * np.pi / k,
                 100 * (2 * np.pi / k) / (2 * np.pi / deep) - 100))
    print("  kH above about 3 is deep water to a per cent; the Charles runs")
    print("  shallower than that over much of its width, so the radiated")
    print("  wave is measurably shorter than the paper's limit.")
    print()

    critical = speed ** 2 / GRAVITY
    print("the steady wake changes REGIME with depth, which is the bigger "
          "effect")
    print("  at %.2f m/s the critical depth is U^2/g = %.2f m" % (speed,
                                                                  critical))
    print("  %8s %10s %14s %s" % ("depth m", "Fh", "regime", "wedge"))
    for depth in args.depths:
        froude = speed / np.sqrt(GRAVITY * depth)
        if froude < 0.9:
            regime, wedge = "subcritical", "19.5 deg, widening"
        elif froude <= 1.1:
            regime, wedge = "TRANSCRITICAL", "solitons; KdV lives here"
        else:
            regime = "supercritical"
            wedge = "%.0f deg Mach cone" % np.degrees(np.arcsin(1.0 / froude))
        print("  %8.1f %10.2f %14s %s" % (depth, froude, regime, wedge))
    print()

    charles(speed, critical, args)
    path = plot(boat, omega, speed, critical, args)
    print("wrote", path)
    return 0


def charles(speed, critical, args):
    """How much of the race line is shallower than critical?"""
    from coxswain.river import charles as C

    raster = C.charles_channel()
    _, _, race_line, _ = C.hocr_course(raster)
    course = C.charles_course(centreline=race_line, month=10)
    station, depth, froude = depth_froude(course, speed)
    print("the Charles, on the race line")
    print("  depth %.2f to %.2f m, median %.2f" % (depth.min(), depth.max(),
                                                   np.median(depth)))
    print("  depth Froude %.2f to %.2f, median %.2f" % (froude.min(),
                                                        froude.max(),
                                                        np.median(froude)))
    print("  %.0f%% of the line is shallower than the critical %.2f m"
          % (100 * np.mean(depth < critical), critical))
    print("  %.0f%% is transcritical (0.9 < Fh < 1.1), where the linear"
          % (100 * np.mean((froude > 0.9) & (froude < 1.1))))
    print("  wake theory this project uses stops being the right model and")
    print("  a Korteweg-de Vries description takes over.")
    globals()["_CHARLES"] = (station, depth, froude)
    print()


def plot(boat, omega, speed, critical, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.0, 9.2))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30,
                               wspace=0.18)

    for column, mode in enumerate(("heave", "pitch")):
        gx, gy, elevation, k = radiated(boat, omega, args.pattern_depth, mode)
        axis = figure.add_subplot(grid[0, column])
        limit = float(np.percentile(np.abs(elevation), 99.5))
        axis.pcolormesh(gx, gy, np.clip(elevation, -limit, limit),
                        cmap="Spectral_r", shading="auto",
                        vmin=-limit, vmax=limit)
        axis.plot(np.asarray(boat.offsets.station),
                  np.zeros_like(boat.offsets.station), color=INK, lw=2.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("radiated in %s, lambda = %.2f m" % (mode,
                                                            2 * np.pi / k),
                       fontsize=11, color=INK, loc="left")
        axis.set_xlabel("m")
        if column == 0:
            axis.set_ylabel("m")

    station, depth, froude = globals()["_CHARLES"]
    axis = figure.add_subplot(grid[1, :])
    axis.plot(station, froude, color="#1f5673", lw=1.4)
    axis.axhline(1.0, color="#a2382a", lw=1.2)
    axis.axhspan(0.9, 1.1, color="#a2382a", alpha=0.15)
    axis.text(station[-1], 1.0, " critical", color="#a2382a", fontsize=9,
              va="center")
    axis.set_ylabel("depth Froude number")
    axis.set_xlabel("station along the course, m")
    axis.set_title("Where the boat outruns its own waves  "
                   "(critical depth %.2f m at %.2f m/s)" % (critical, speed),
                   fontsize=11, color=INK, loc="left")
    axis.grid(color=RULE, lw=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    figure.suptitle("Waves radiated by the secondary motion, after "
                    "Formaggia et al. figure 11", fontsize=12.5, color=INK,
                    x=0.02, ha="left")
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "waves.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
