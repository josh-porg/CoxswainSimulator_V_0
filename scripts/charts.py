"""Draw the Charles course charts.

    python scripts/charts.py --out out/charts
    python scripts/charts.py --only arches profiles
    python scripts/charts.py --month 4 --out out/spring

Charts:

  bathymetry  the course over the depth survey, with bridges and arches
  current     the current, resolved across the width of the channel
  profiles    the course straightened out: depth, width and current
  arches      every bridge's arches to scale, against the width of an eight
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river import charts  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw the Charles course charts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--out", default="out/charts",
                        help="directory to write into (default: out/charts)")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=sorted(charts.BUILDERS),
                        help="draw only these charts (default: all four)")
    parser.add_argument("--month", type=int, default=10,
                        help="month for the discharge used by the current "
                             "charts; 10 is regatta month (default: 10)")
    parser.add_argument("--summary", action="store_true",
                        help="also print the arches and the width of the "
                             "course to standard output")
    args = parser.parse_args(argv)

    written = charts.write_all(args.out, month=args.month, which=args.only)
    for path in written:
        print("wrote", path)

    if args.summary:
        _summary()
    return 0


def _summary():
    import numpy as np
    from coxswain.river import bridges, traffic

    geometry = charts.CourseGeometry()
    print()
    print("arches, Boston shore first")
    for gate, metres in geometry.gates_on_course():
        print("  %-20s %5.0f m from the start" % (gate.name, metres))
        racing = bridges.racing_arch(gate, geometry.channel)
        for arch in bridges.bridge_arches(gate, geometry.channel):
            if racing is not None and arch.index == racing.index:
                mark = "conventional line"
            elif arch.legal:
                mark = "legal, untested alternative"
            else:
                mark = "%.0f s penalty" % bridges.WRONG_ARCH_PENALTY
            print("      %d %-16s %5.1f m  %4.1f eights   %s"
                  % (arch.index, arch.label, arch.width, arch.fits(), mark))

    rows = traffic.lane_report(geometry, step=25.0)
    metres, total, lane, usable = rows.T
    print()
    print("course width: min %.0f  median %.0f  max %.0f m"
          % (total.min(), np.median(total), total.max()))
    pinched = lane > 0
    if pinched.any():
        print("with the travel lane taken out, the narrowest usable water is "
              "%.0f m at %.0f m from the start"
              % (usable[pinched].min(), metres[pinched][usable[pinched].argmin()]))
    print("a rowed eight is %.2f m wide; two abreast need %.1f m"
          % (bridges.EIGHT_ROWED_WIDTH, traffic.PASSING_WIDTH))


if __name__ == "__main__":
    raise SystemExit(main())
