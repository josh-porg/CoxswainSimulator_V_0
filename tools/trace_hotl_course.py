r"""Trace the Head of the Lake course off the regatta's own buoy map.

    python tools/trace_hotl_course.py --check out/hotl_trace

Source: ``hotl updated buoy map 2025``, the PNG the regatta publishes on
its course-information page, 2000 x 1259 px, north up, with a scale bar
reading 250 / 500 / 1000 m in the top right.  It draws the racing lane as
a green dash-dot line, the race buoys as orange and yellow discs, the
2-ton channel buoys as red and green diamonds, and START and FINISH as
labels.  Everything the optimiser needs is on it.

Georeferencing, in two steps that check each other
--------------------------------------------------
**Scale** from the bar.  The bar is a black horizontal rule; its length
in pixels over the 1000 m it labels gives metres per pixel outright, and
nothing else on the map is trusted for scale.

**Position** by overlap.  The map's water is one flat light blue
(208, 224, 240), so it thresholds to a clean mask.  That mask is slid
over the OpenStreetMap water mask -- Lake Union, Portage Bay, the
Montlake Cut and Union Bay together -- at the bar's scale, and the
translation that maximises intersection-over-union is the registration.
Rotation is held at zero because the map is north-up and says so.

This is the same recipe that traced Tail of the Lake (SOURCES sec. 100),
where the lesson was that a registration maximising *precision* rather
than IoU games itself by shrinking the scale.  Scale is fixed here by
the bar before the fit begins, so it cannot.

What is traced
--------------
* the racing lane, as an ordered polyline in metres;
* every buoy, with its colour, as a point;
* the start and finish, as the lane's ends.

What is not
-----------
The lane on the map is a **schematic** drawn a few tens of metres wide
at this scale and not a GPS track; it is where the organisers say the
course goes, which is the thing the buoys enforce.  Treat the traced line
as the centreline the rules define, and the buoys as the constraints, as
:mod:`scripts.render_totl` does.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAP = "data/raw/hotl/buoy_map_2025.png"

#: Map colours, from a histogram of the PNG.  Each is ``(low, high)`` in
#: RGB, inclusive.  The water is a single flat tint; the others are
#: anti-aliased and need a band.
COLOURS = {
    "water": ((202, 218, 234), (214, 230, 246)),
    "lane": ((10, 110, 20), (70, 175, 80)),
    "yellow": ((215, 205, 0), (255, 255, 90)),
    "orange": ((215, 120, 30), (255, 185, 110)),
    "red": ((190, 0, 0), (255, 60, 60)),
}

#: The bar labels 1000 m from its left end to its right.
BAR_METRES = 1000.0


def band(image, name):
    low, high = COLOURS[name]
    return np.all((image >= low) & (image <= high), axis=2)


def scale_bar(image):
    """``(metres per pixel, (x0, x1, y))`` from the segmented rule top right.

    The bar is the usual alternating rule: a black segment for 0-250 m,
    a white one for 250-500, black again for 500-1000, with a thin
    underline beneath the 250-1000 portion.  The first version of this
    found the underline -- 267 px -- and called it 1000 m, which made
    every metre 33% too long and put the map's water a lake's width off
    the shoreline (IoU 0.34).  The bar's true extent is the leftmost
    dark pixel to the rightmost on the row that carries the black
    segments, and only that row is read.
    """
    grey = image.mean(axis=2)
    dark = grey < 60.0
    height, width = grey.shape
    best = None
    for y in range(20, height // 10):
        row = dark[y, width // 2:]
        xs = np.nonzero(row)[0]
        if len(xs) < 150:
            continue
        # Segmented bar: two or more dark runs, total length large, and
        # the whole thing spanning less than a fifth of the map.
        breaks = np.nonzero(np.diff(xs) > 1)[0]
        runs = len(breaks) + 1
        extent = xs[-1] - xs[0] + 1
        filled = len(xs)
        if runs >= 2 and extent < width // 5 and filled > 0.5 * extent:
            if best is None or extent > best[0]:
                best = (extent, xs[0] + width // 2, y)
    if best is None:
        raise SystemExit("could not find the scale bar")
    extent, x0, y = best
    return BAR_METRES / extent, (x0, x0 + extent, y)


def register(map_water, metres_per_pixel, osm):
    """Translation (east, north of the map's top-left) maximising IoU."""
    east, north, mask = osm
    resolution = float(east[1] - east[0])
    # Resample the map's water onto the OSM grid resolution.
    step = resolution / metres_per_pixel
    rows = np.arange(0, map_water.shape[0], step).astype(int)
    cols = np.arange(0, map_water.shape[1], step).astype(int)
    small = map_water[np.ix_(rows, cols)]        # row 0 is the map's top

    best = (-1.0, 0, 0)
    # Coarse-to-fine search over where the map's top-left sits on the grid.
    for coarse in (8, 2, 1):
        if best[0] < 0:
            candidates_r = range(-small.shape[0] // 2, mask.shape[0], coarse)
            candidates_c = range(-small.shape[1] // 2, mask.shape[1], coarse)
        else:
            _score, r0, c0 = best
            candidates_r = range(r0 - 3 * coarse, r0 + 3 * coarse + 1, coarse)
            candidates_c = range(c0 - 3 * coarse, c0 + 3 * coarse + 1, coarse)
        for r in candidates_r:
            for c in candidates_c:
                score = _iou(small, mask, r, c)
                if score > best[0]:
                    best = (score, r, c)
    score, r, c = best
    # The map's row 0 is its top, which is the *north* edge; the OSM grid's
    # row 0 is its south edge.  ``r`` is where the map's top row lands,
    # counted from the grid's south row upward.
    top_north = north[0] + (r + small.shape[0]) * resolution
    left_east = east[0] + c * resolution
    return score, left_east, top_north


def _iou(small, mask, r, c):
    """IoU of ``small`` placed with its bottom-left at grid ``(r, c)``."""
    h, w = small.shape
    # small's rows run top-to-bottom; flip so row 0 is south like the grid
    flipped = small[::-1]
    r1, c1 = r + h, c + w
    if r1 <= 0 or c1 <= 0 or r >= mask.shape[0] or c >= mask.shape[1]:
        return 0.0
    gr0, gc0 = max(r, 0), max(c, 0)
    gr1, gc1 = min(r1, mask.shape[0]), min(c1, mask.shape[1])
    a = flipped[gr0 - r:gr1 - r, gc0 - c:gc1 - c]
    b = mask[gr0:gr1, gc0:gc1]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + (flipped.sum() - a.sum()) \
        + (mask.sum() - b.sum())
    return inter / union if union else 0.0


def to_metres(px, py, metres_per_pixel, left_east, top_north):
    """Map pixel to tangent-plane metres."""
    return (left_east + np.asarray(px, float) * metres_per_pixel,
            top_north - np.asarray(py, float) * metres_per_pixel)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", default="out/hotl_trace")
    args = parser.parse_args(argv)

    from PIL import Image

    from coxswain.river.seattle import SHIP_CANAL, water_mask

    os.makedirs(args.check, exist_ok=True)
    image = np.asarray(Image.open(MAP).convert("RGB"), dtype=int)
    print("map %d x %d" % (image.shape[1], image.shape[0]))

    metres_per_pixel, (x0, x1, y) = scale_bar(image)
    print("scale bar: %d px from x=%d to %d at y=%d -> %.3f m/px"
          % (x1 - x0, x0, x1, y, metres_per_pixel))

    map_water = band(image, "water")
    # The legend box and the inset are drawn on the same blue; cut them.
    map_water[700:, :1350] = False        # legend, lower left
    map_water[560:, 1360:] = False        # Pocock inset, lower right
    # Anti-aliased edges of every road and label leave single blue
    # pixels all over the land; an opening removes anything under about
    # 10 m across and leaves the water, which is thousands of pixels.
    from scipy.ndimage import binary_opening
    map_water = binary_opening(map_water, iterations=2)
    print("map water: %d px = %.2f km2 at that scale"
          % (map_water.sum(), map_water.sum() * metres_per_pixel ** 2 / 1e6))

    osm = water_mask(10.0, names=SHIP_CANAL)
    score, left_east, top_north = register(map_water, metres_per_pixel, osm)
    print("registration: IoU %.3f; map top-left at east %.0f, north %.0f"
          % (score, left_east, top_north))

    # -- diagnostic overlay ----------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    east, north, mask = osm
    figure, axis = plt.subplots(figsize=(13, 8))
    ge, gn = np.meshgrid(east, north)
    axis.contour(ge, gn, mask.astype(float), [0.5], colors="#ff2d55",
                 linewidths=1.2)
    ys, xs = np.nonzero(map_water[::4, ::4])
    me, mn = to_metres(xs * 4, ys * 4, metres_per_pixel, left_east, top_north)
    axis.scatter(me, mn, s=0.4, c="#2f7fb5", alpha=0.35, linewidths=0)
    axis.set_aspect("equal")
    axis.set_title("map water (blue) over OSM shoreline (red): IoU %.3f"
                   % score)
    figure.savefig(os.path.join(args.check, "registration.png"), dpi=110,
                   bbox_inches="tight")
    print("wrote %s" % os.path.join(args.check, "registration.png"))

    np.savez(os.path.join(args.check, "registration.npz"),
             metres_per_pixel=metres_per_pixel, left_east=left_east,
             top_north=top_north, iou=score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
