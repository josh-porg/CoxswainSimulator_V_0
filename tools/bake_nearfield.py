r"""Bake the hull's near-field surface disturbance, once, for the trainer.

    python tools/bake_nearfield.py

:mod:`coxswain.hydro.nearfield` computes the water the hull pushes about
-- the pile-up at the stem and the drawdown along the midbody -- from a
thin-ship source sheet.  That is a double sum over centreplane panels for
every point of a grid, which is a tenth of a second and therefore not
something to do inside a frame.

It does not have to be.  The source strengths go as ``U`` and the
elevation as ``U * phi_x``, so the whole field is ``U^2 / g`` times a
function of the hull's shape alone.  Baked here per boat class and
sampled as a texture at run time, it costs one lookup and a multiply,
and it still responds correctly to speed because the scaling is exact.

The result is added to the Kelvin construction in
:mod:`coxswain.viz.water`, which is the complementary half: that one is
the wave system radiating away, this one is the flow around the body.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="coxswain/data/nearfield.npz")
    parser.add_argument("--nx", type=int, default=192)
    parser.add_argument("--ny", type=int, default=96)
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from coxswain.boats import catalog
    from coxswain.hydro.nearfield import bake, elevation

    shells = {}
    for name, maker in (("four", catalog.coxed_four),
                        ("eight", catalog.eight)):
        try:
            boat = maker(rate=30.0, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)
        except TypeError:
            boat = maker(rate=30.0)
        east, north, field = bake(boat, nx=args.nx, ny=args.ny)
        peak = elevation(field, 4.5)
        centre = field[np.argmin(np.abs(north))]
        half = 0.5 * boat.length
        ahead = centre[np.argmin(np.abs(east - (half + 0.6)))]
        abaft = centre[np.argmin(np.abs(east - (half - 0.9)))]
        print("%-6s hull %5.2f m, grid %dx%d" % (name, boat.length,
                                                 len(east), len(north)))
        print("   at 4.5 m/s: %+.4f to %+.4f m over the whole field"
              % (peak.min(), peak.max()))
        print("   just ahead of the stem %+.4f m (pile-up), just abaft "
              "%+.4f m (drawdown)"
              % (ahead * 4.5 ** 2 / 9.80665, abaft * 4.5 ** 2 / 9.80665))
        shells["%s_east" % name] = east.astype(np.float32)
        shells["%s_north" % name] = north.astype(np.float32)
        shells["%s_field" % name] = field.astype(np.float32)
        shells["%s_length" % name] = np.float32(boat.length)

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), args.out)
    np.savez_compressed(target, **shells)
    print("\nwrote %s (%.0f kB)" % (target, os.path.getsize(target) / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
