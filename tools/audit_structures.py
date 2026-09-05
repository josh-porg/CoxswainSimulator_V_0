r"""Look for buildings whose geometry is wrong, before a render does.

    python tools/audit_structures.py --which seattle

The Space Needle was caught by eye: it rendered as a fat cylinder
because twelve ``building:part`` polygons were all extruded from the
ground instead of from their own ``min_height``.  That is the second
time a geometry fault has been found by looking at a picture rather than
a number, and pictures do not scale to 41,332 footprints.

So this asks the questions that would have caught it, and a few more:

* **Aspect** -- a 160 m tower on a 4 m footprint is a mast or a mistake;
  a 3 m building covering a hectare is a car park roof or a mistake.
* **Stacks** -- footprints that overlap heavily at the same level, which
  means the same building is being drawn twice.
* **Degenerate rings** -- fewer than four points, or near-zero area.
* **Floating** -- a part starting above the ground with nothing under it.
* **In the water** -- a footprint whose centre is inside the racing
  water, which is either a houseboat or a georeferencing error.

Nothing here is fatal on its own.  A tall thin thing really might be a
mast, and Seattle really does have 141 floating homes.  The output is a
list to look at, ranked, not a pass/fail.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def area(ring):
    return 0.5 * abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], 1))
                           - np.dot(ring[:, 1], np.roll(ring[:, 0], 1))))


def audit(structures, water=None, limit: int = 12):
    """Return ``{check: [(score, description), ...]}``."""
    polygons = structures.polygons
    heights = structures.heights
    base = getattr(structures, "base", np.zeros(len(polygons)))
    names = getattr(structures, "names", np.full(len(polygons), ""))
    centres = structures.centres
    found = {}

    spans = np.array([np.hypot(*(p.max(axis=0) - p.min(axis=0)))
                      if len(p) else 0.0 for p in polygons])
    areas = np.array([area(p) if len(p) >= 3 else 0.0 for p in polygons])

    # -- degenerate -----------------------------------------------------
    bad = [(0.0, "%d points, %.1f m2" % (len(p), areas[i]))
           for i, p in enumerate(polygons)
           if len(p) < 4 or areas[i] < 1.0]
    found["degenerate rings"] = bad[:limit], len(bad)

    # -- aspect ---------------------------------------------------------
    tall = np.nonzero((heights > 25.0) & (spans > 0)
                      & (heights / np.maximum(spans, 1e-9) > 6.0))[0]
    rows = sorted(((float(heights[i] / spans[i]),
                    "%.0f m tall on a %.1f m footprint  %s"
                    % (heights[i], spans[i], names[i] or "")) for i in tall),
                  reverse=True)
    found["needle-thin"] = rows[:limit], len(rows)

    squat = np.nonzero((areas > 4000.0) & (heights < 4.0))[0]
    rows = sorted(((float(areas[i]),
                    "%.0f m2 but only %.1f m tall  %s"
                    % (areas[i], heights[i], names[i] or ""))
                   for i in squat), reverse=True)
    found["pancake"] = rows[:limit], len(rows)

    # -- stacked duplicates ---------------------------------------------
    # Two ground-level footprints with nearly the same centre and area
    # are the same building entered twice, and both get drawn.
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    stacked = []
    for a, b in zip(order[:-1], order[1:]):
        if base[a] > 0.5 or base[b] > 0.5:
            continue                       # massing steps, legitimately stacked
        gap = float(np.linalg.norm(centres[a] - centres[b]))
        if gap > 3.0 or min(areas[a], areas[b]) < 50.0:
            continue
        ratio = min(areas[a], areas[b]) / max(areas[a], areas[b], 1e-9)
        if ratio > 0.75:
            stacked.append((ratio, "%.0f and %.0f m2 %.1f m apart, %.0f/%.0f m"
                            % (areas[a], areas[b], gap,
                               heights[a], heights[b])))
    found["drawn twice"] = sorted(stacked, reverse=True)[:limit], len(stacked)

    # -- floating parts --------------------------------------------------
    floating = []
    for index in np.nonzero(base > 0.5)[0]:
        near = structures.near(centres[index][0], centres[index][1], 30.0)
        # Supported if anything nearby spans the level this one starts
        # at.  The first version of this check demanded a *ground-level*
        # neighbour, which flagged 297 perfectly good massing steps: a
        # part at 210 m stands on one at 150 m, which stands on one at
        # 100 m, and none of them touches the ground either.
        under = [j for j in near
                 if j != index
                 and base[j] <= base[index] + 0.5
                 and heights[j] >= base[index] - 1.0]
        if not under:
            floating.append((float(base[index]),
                             "starts at %.0f m with nothing spanning that "
                             "level  %s" % (base[index], names[index] or "")))
    found["floating parts"] = sorted(floating, reverse=True)[:limit], \
        len(floating)

    # -- in the water ----------------------------------------------------
    if water is not None:
        east, north, mask = water
        rows = np.clip(np.searchsorted(north, centres[:, 1]), 0,
                       len(north) - 1)
        cols = np.clip(np.searchsorted(east, centres[:, 0]), 0, len(east) - 1)
        wet = np.nonzero(mask[rows, cols])[0]
        rows_out = sorted(((float(heights[i]),
                            "%.0f m tall, centre in the water  %s"
                            % (heights[i], names[i] or "")) for i in wet),
                          reverse=True)
        found["standing in the water"] = rows_out[:limit], len(rows_out)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", default="seattle",
                        choices=("seattle", "charles"))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(argv)

    from coxswain.river.structures import charles_structures, seattle_structures
    if args.which == "seattle":
        structures = seattle_structures()
        from coxswain.river.seattle import water_mask
        water = water_mask(10.0, names=("Lake Union",))
    else:
        structures = charles_structures()
        water = None

    print("%s: %d footprints, %d starting above the ground"
          % (args.which, len(structures.polygons),
             int((getattr(structures, "base",
                          np.zeros(1)) > 0.5).sum())))
    findings = audit(structures, water=water, limit=args.limit)
    for check, (rows, total) in findings.items():
        print()
        print("%s: %d" % (check, total))
        for _score, text in rows:
            print("   %s" % text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
