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

The bar was checked, because the water-overlap fit disagreed with it
------------------------------------------------------------------
Left free, the overlap fit preferred 2.94 m/px to the bar's 2.80 -- a 5%
stretch, which over a 4.8 km course is 240 m.  The map is a schematic,
so its cartography can be off by that much and the overlap would follow
the cartography; the bar and the overlap are not independent witnesses
to the same thing.  Point landmarks are: the lane passes under I-5, the
University Bridge and the Montlake Bridge, and runs the length of the
Montlake Cut, all unambiguous on both map and shoreline.  A similarity
fit on those landmarks gives the scale the bar gives, isotropic, and the
traced lane then comes out 4706 m against the regatta's stated 3 miles
-- 2.5% short, which is what chaining the centroids of dashes around a
turn does.  The bar is trusted; the free-scale overlap is not.

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


def register(map_water, metres_per_pixel, osm, metres_per_pixel_y=None):
    """Translation (east, north of the map's top-left) maximising IoU."""
    east, north, mask = osm
    resolution = float(east[1] - east[0])
    # Resample the map's water onto the OSM grid resolution.
    step_y = resolution / (metres_per_pixel_y or metres_per_pixel)
    step = resolution / metres_per_pixel
    rows = np.arange(0, map_water.shape[0], step_y).astype(int)
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
    # Union over the map's own footprint only.  The shoreline mask runs
    # a kilometre further into Lake Washington than the map's east edge;
    # water the map cannot show is not evidence against where it sits.
    # Map water that falls off the grid still counts against it.
    union = np.logical_or(a, b).sum() + (flipped.sum() - a.sum())
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
    # The legend box and the inset are drawn on the same blue; cut them
    # -- and only them.  The first version blanked everything below row
    # 700 on the left, which took the legend and the entire south lobe
    # of Lake Union with it, and the registration then had 2.8 km2 of
    # map water to match against 6 km2 of shoreline.
    map_water[725:1215, 655:1335] = False     # legend box
    map_water[570:1205, 1360:1995] = False    # Pocock Turn inset
    # Anti-aliased edges of every road and label leave single blue
    # pixels all over the land; an opening removes anything under about
    # 10 m across and leaves the water, which is thousands of pixels.
    from scipy.ndimage import binary_opening
    map_water = binary_opening(map_water, iterations=2)
    # Where the map has no data: the legend, the inset, and the part of
    # Lake Union it cuts off at row 823.  Excluded from the overlap and
    # from every colour search.
    valid = np.ones(map_water.shape, dtype=bool)
    valid[725:1215, 655:1335] = False
    valid[570:1205, 1360:1995] = False
    valid[823:, :655] = False
    map_water &= valid
    print("map water: %d px = %.2f km2 at that scale"
          % (map_water.sum(), map_water.sum() * metres_per_pixel ** 2 / 1e6))

    osm = water_mask(10.0, names=SHIP_CANAL)
    score, iou_east, iou_north = register(map_water, metres_per_pixel, osm)
    print("overlap: IoU %.3f; map top-left at east %.0f, north %.0f"
          % (score, iou_east, iou_north))

    # Position from the bridge landmarks, not from the overlap.
    #
    # The overlap put the lane 43 m south of where the landmarks put it,
    # and through the Montlake Cut that is the difference between the
    # centreline and the south wall: 16 of 99 traced points came out on
    # land and only 2 of 16 in the Cut were in its water.  The overlap
    # follows the schematic's cartography, which is off by tens of
    # metres; a bridge is where it is on both.  The overlap is kept as a
    # check and the two must agree to within 60 m.
    lane_px = trace_lane(image, valid)
    left_east, top_north = landmark_translation(image, lane_px,
                                                metres_per_pixel)
    drift = np.hypot(left_east - iou_east, top_north - iou_north)
    print("landmarks: map top-left at east %.0f, north %.0f (%.0f m from "
          "the overlap)" % (left_east, top_north, drift))
    if drift > 60.0:
        raise SystemExit("overlap and landmarks disagree by %.0f m" % drift)

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

    # -- the lane ----------------------------------------------------------
    lane = lane_px
    lane_e, lane_n = to_metres(lane[:, 0], lane[:, 1], metres_per_pixel,
                               left_east, top_north)
    course = np.column_stack([lane_e, lane_n])
    length = float(np.hypot(*np.diff(course, axis=0).T).sum())
    print("lane: %d points, %.0f m (regatta says 3 miles, 4828 m: %+.1f%%)"
          % (len(course), length, 100.0 * (length / 4828.0 - 1.0)))

    # -- the buoys ----------------------------------------------------------
    buoys = []
    for colour, side in (("yellow", "starboard"), ("orange", "port")):
        for px, py in blobs(image, colour, valid):
            e, n = to_metres(px, py, metres_per_pixel, left_east, top_north)
            buoys.append((colour, side, float(e), float(n)))
    print("buoys: %d yellow (starboard), %d orange (port)"
          % (sum(1 for b in buoys if b[0] == "yellow"),
             sum(1 for b in buoys if b[0] == "orange")))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(root, "data", "hotl_course.npy"), course)
    # Same layout Tail of the Lake uses: (keep_to_port?, east, north).
    # HOTL's rule is yellow to starboard, orange to port; a buoy to be
    # kept to port is a limit on the port side.
    np.save(os.path.join(root, "data", "hotl_buoys.npy"),
            np.array([(1.0 if side == "port" else 0.0, e, n)
                      for _c, side, e, n in buoys]))
    print("wrote data/hotl_course.npy and data/hotl_buoys.npy")

    # -- verification drawing over the shoreline --------------------------
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.contour(ge, gn, mask.astype(float), [0.5], colors="#ff2d55",
                 linewidths=1.0)
    axis.plot(course[:, 0], course[:, 1], "-", color="#1f7a4d", lw=2,
              label="racing lane")
    for colour, side, e, n in buoys:
        axis.plot(e, n, "o", color="#ffd60a" if colour == "yellow" else
                  "#ff9248", ms=5, mec="k", mew=0.4)
    axis.plot(course[0, 0], course[0, 1], "o", color="k", ms=9, label="start")
    axis.plot(course[-1, 0], course[-1, 1], "s", color="k", ms=9,
              label="finish")
    axis.set_aspect("equal")
    axis.set_xlim(-1400, 4400)
    axis.set_ylim(-600, 2200)
    axis.legend(loc="lower left")
    axis.set_title("Head of the Lake, traced: lane %.0f m, %d buoys "
                   "(yellow starboard, orange port)" % (length, len(buoys)))
    figure.savefig(os.path.join(args.check, "course.png"), dpi=110,
                   bbox_inches="tight")
    print("wrote %s" % os.path.join(args.check, "course.png"))
    return 0


#: Where the lane crosses two bridges, in the tangent plane.  I-5's
#: Ship Canal Bridge is a straight north-south freeway, so its easting
#: is unambiguous; the lane meets it at the Pocock apex, at the north
#: end of Portage Bay.  The Montlake Bridge crosses the Cut at its
#: midpoint, and the lane hugs the Cut's centreline.
LANDMARKS = {
    "I-5": (788.0, 1500.0),
    "Montlake Bridge": (2126.0, 867.0),
}


def landmark_translation(image, lane_px, metres_per_pixel):
    """``(left_east, top_north)`` from where the lane crosses the bridges.

    Both bridges are found on the map by their road colour crossing the
    lane -- I-5 is the blue band, Montlake Boulevard a red one -- and the
    lane point nearest each crossing is paired with the bridge's known
    position.  Two points over-determine a translation at fixed scale;
    the mean is taken and the spread reported.
    """
    blue = np.all((image >= (0, 90, 150)) & (image <= (60, 140, 210)),
                  axis=2)
    red = band(image, "red")

    def road_x(mask, y0, y1, x0, x1, least):
        columns = mask[y0:y1, x0:x1].sum(axis=0)
        xs = np.nonzero(columns > least)[0] + x0
        if not len(xs):
            raise SystemExit("a bridge landmark was not found on the map")
        return float(xs.mean())

    def lane_near(x, y0, y1):
        pts = lane_px[(lane_px[:, 1] >= y0) & (lane_px[:, 1] <= y1)]
        return pts[int(np.argmin(np.abs(pts[:, 0] - x)))]

    i5 = lane_near(road_x(blue, 150, 330, 690, 780, 60), 150, 330)
    montlake = lane_near(road_x(red, 360, 430, 1100, 1300, 25), 370, 420)

    offsets = []
    for px, (east, north) in ((i5, LANDMARKS["I-5"]),
                              (montlake, LANDMARKS["Montlake Bridge"])):
        offsets.append((east - px[0] * metres_per_pixel,
                        north + px[1] * metres_per_pixel))
    offsets = np.asarray(offsets)
    spread = float(np.hypot(*(offsets[0] - offsets[1])))
    print("landmark pair disagrees by %.0f m (scale error over their "
          "%.0f px baseline)" % (spread, np.hypot(*(montlake - i5))))
    return float(offsets[:, 0].mean()), float(offsets[:, 1].mean())


def blobs(image, colour, valid, smallest=40, largest=250, widest=20):
    """Centroids of the discs of one colour, in pixels.

    Size-bounded so the FINISH banner (a tall yellow dash) and the
    Gasworks Park polygon (green, thousands of pixels) are not buoys.
    """
    from scipy.ndimage import label
    mask = band(image, colour) & valid
    labels, count = label(mask)
    out = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        if (smallest <= len(xs) <= largest
                and (xs.max() - xs.min()) < widest):
            out.append((xs.mean(), ys.mean()))
    return out


#: Where the START and FINISH labels sit on the map, in pixels.  Read
#: off the image once; the lane is chained from the dash nearest the
#: start and must end within a few dashes of the finish.
START_LABEL = (610.0, 325.0)
FINISH_LABEL = (1641.0, 235.0)

#: Farthest a dash may be from the last one and still be the same lane.
DASH_GAP = 90.0


def trace_lane(image, valid):
    """The racing lane as an ordered pixel polyline, start to finish.

    The lane is drawn dash-dot, so it arrives as a hundred separate green
    blobs.  They are chained greedily from the one nearest START, always
    to the nearest unused blob within :data:`DASH_GAP`.  Two things are
    the same green and must not be chained: the Gasworks Park polygon,
    which is thousands of pixels, and the 2-ton channel-buoy diamonds,
    which are off the line.  The size bound removes the park; the gap
    bound leaves the diamonds unchained, and the run must still end
    within reach of FINISH or the trace is refused.
    """
    from scipy.ndimage import label
    mask = band(image, "lane") & valid
    labels, count = label(mask)
    centres = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        if 8 <= len(xs) <= 2000:
            centres.append((xs.mean(), ys.mean()))
    centres = np.asarray(centres)
    used = np.zeros(len(centres), dtype=bool)
    here = int(np.argmin(np.hypot(*(centres - START_LABEL).T)))
    order = [here]
    used[here] = True
    while True:
        gaps = np.hypot(*(centres - centres[order[-1]]).T)
        gaps[used] = np.inf
        nearest = int(np.argmin(gaps))
        if gaps[nearest] > DASH_GAP:
            break
        order.append(nearest)
        used[nearest] = True
    path = centres[order]
    finish_gap = float(np.hypot(*(path[-1] - FINISH_LABEL)))
    if finish_gap > 60.0:
        raise SystemExit("lane trace ended %.0f px from FINISH" % finish_gap)
    print("lane trace: chained %d of %d dashes, ended %.0f px from FINISH"
          % (len(order), len(centres), finish_gap))
    return path


if __name__ == "__main__":
    raise SystemExit(main())
