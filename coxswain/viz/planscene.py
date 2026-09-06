r"""The course from above, as geometry -- with no idea how it is drawn.

A plan view is the coxswain's other picture.  The 3-D seat view answers
"is this line steerable"; the plan answers "where am I on the course, and
what am I about to hit", which is most of what a head race actually
demands: hold the line, take the right arch, keep the buoys on the right
side, do not let the bend push you wide.

Backend-neutral on purpose
--------------------------
Everything here is world-frame metres in numpy arrays.  It imports no
window library and knows nothing about pixels, so the pygame trainer and
a later ModernGL renderer consume the same scene, and swapping one for
the other cannot change what is on the course.

The layers are ordered back to front, which is the only ordering a 2-D
renderer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["PlanLayer", "PlanScene", "boat_outline", "oar_lines",
           "clip_polygon",
           "charles_scene", "seattle_scene", "build_scene"]


@dataclass
class PlanLayer:
    """One drawable set of polylines, all sharing a style."""

    name: str
    polylines: List[np.ndarray] = field(default_factory=list)
    colour: tuple = (120, 120, 120)
    width: float = 1.0
    fill: bool = False
    closed: bool = False
    #: ``(n, 4)`` of ``(xmin, ymin, xmax, ymax)``, built once for culling.
    boxes: Optional[np.ndarray] = None
    #: Skip a polygon whose largest on-screen dimension is under this
    #: many pixels.  A building three pixels across is a smudge that
    #: costs a fill, and there are 81,022 of them on Lake Union.
    min_pixels: float = 0.0
    #: Never draw more than this many in one frame, nearest first.
    limit: Optional[int] = None
    #: Uniform grid over the boxes, ``(i, j) -> [index, ...]``.
    grid: Optional[dict] = None
    cell: float = 250.0

    def index(self, cell: float = 250.0) -> "PlanLayer":
        """Bounding boxes and a uniform grid over them.  Build time only.

        The grid is what makes the cull cheap.  Testing every box against
        the view is one vectorised comparison, which sounds free until
        the layer holds 81,022 buildings: eight passes over an 81k array,
        every frame, was **17 ms of a 17 ms frame budget** and it did not
        care that only two hundred buildings were on screen.  Bucketing
        by 250 m cell turns that into a lookup of the handful of cells
        the window covers.
        """
        if not self.polylines:
            self.boxes = np.zeros((0, 4))
            self.grid, self.cell = {}, cell
            return self
        self.boxes = np.array([[p[:, 0].min(), p[:, 1].min(),
                                p[:, 0].max(), p[:, 1].max()]
                               for p in self.polylines], dtype=float)
        self.cell = float(cell)
        self.grid = {}
        low = np.floor(self.boxes[:, :2] / self.cell).astype(int)
        high = np.floor(self.boxes[:, 2:] / self.cell).astype(int)
        for index in range(len(self.boxes)):
            for i in range(low[index, 0], high[index, 0] + 1):
                for j in range(low[index, 1], high[index, 1] + 1):
                    self.grid.setdefault((i, j), []).append(index)
        return self

    def visible(self, x0: float, y0: float, x1: float, y1: float,
                scale: float = None, centre=None):
        """Indices of the polylines worth drawing this frame.

        A whole-course layer holds tens of thousands of polygons and a
        window holds a few dozen; testing boxes with one vectorised
        comparison is the difference between a trainer and a slideshow.

        ``scale`` (pixels per metre) and ``centre`` enable the size
        filter and the cap, which are what keep the frame budget flat as
        the camera zooms out over a city.
        """
        if self.boxes is None or not len(self.boxes):
            return ()
        if self.grid:
            candidates = set()
            for i in range(int(np.floor(x0 / self.cell)),
                           int(np.floor(x1 / self.cell)) + 1):
                for j in range(int(np.floor(y0 / self.cell)),
                               int(np.floor(y1 / self.cell)) + 1):
                    bucket = self.grid.get((i, j))
                    if bucket:
                        candidates.update(bucket)
            if not candidates:
                return ()
            pool = np.fromiter(candidates, dtype=int, count=len(candidates))
        else:
            pool = np.arange(len(self.boxes))
        box = self.boxes[pool]
        keep = ((box[:, 0] <= x1) & (box[:, 2] >= x0)
                & (box[:, 1] <= y1) & (box[:, 3] >= y0))
        if scale and self.min_pixels > 0.0:
            span = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
            keep &= span * scale >= self.min_pixels
        found = pool[keep]
        box = self.boxes
        if (self.limit is not None and len(found) > self.limit
                and centre is not None):
            middle = 0.5 * (box[found][:, :2] + box[found][:, 2:])
            gap = np.einsum("ij,ij->i", middle - centre, middle - centre)
            found = found[np.argsort(gap)[:self.limit]]
        return found


@dataclass
class PlanScene:
    """A course laid out flat, in metres, ready to be drawn by anything."""

    name: str
    layers: List[PlanLayer] = field(default_factory=list)
    #: The course as drawn, ``(n, 2)``.
    course: Optional[np.ndarray] = None
    #: An optimised line to chase, ``(n, 2)``, or ``None``.
    ghost: Optional[np.ndarray] = None
    #: ``(k, 3)`` of ``(keep_to_port, east, north)``.
    buoys: Optional[np.ndarray] = None
    #: ``(east, north, label)`` worth naming.
    marks: Sequence[tuple] = ()
    #: Where a session starts, and which way it points.
    start: Optional[np.ndarray] = None
    start_heading: float = 0.0

    def layer(self, name: str) -> Optional[PlanLayer]:
        for item in self.layers:
            if item.name == name:
                return item
        return None


def clip_polygon(points: np.ndarray, x0: float, y0: float,
                 x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to a rectangle.

    **Why a plan view needs this.**  A filled polygon costs a scanline
    fill, and a scanline fill sorts every edge on every row.  Lake
    Washington's shoreline is one ring of 2,345 vertices; drawn across a
    700-pixel window that is 1.6 million edge tests a frame, and it
    measured at **14 ms of a 17 ms frame** -- for a single polygon.
    Clipped to the window first it is a dozen vertices and the same
    picture.

    Returns an empty array if nothing survives.
    """
    output = np.asarray(points, dtype=float)
    for axis, bound, keep_greater in ((0, x0, True), (0, x1, False),
                                      (1, y0, True), (1, y1, False)):
        if len(output) == 0:
            return np.zeros((0, 2))
        current = output
        shifted = np.roll(current, -1, axis=0)
        if keep_greater:
            inside = current[:, axis] >= bound
            inside_next = shifted[:, axis] >= bound
        else:
            inside = current[:, axis] <= bound
            inside_next = shifted[:, axis] <= bound
        pieces = []
        for index in range(len(current)):
            a, b = current[index], shifted[index]
            if inside[index]:
                pieces.append(a)
            if inside[index] != inside_next[index]:
                span = b[axis] - a[axis]
                if abs(span) > 1e-12:
                    pieces.append(a + (b - a) * ((bound - a[axis]) / span))
        output = np.asarray(pieces, dtype=float) if pieces             else np.zeros((0, 2))
    return output


# -- the boat -------------------------------------------------------------

def boat_outline(boat) -> np.ndarray:
    """The hull's plan silhouette in the hull frame, ``(n, 2)``.

    Taken from the panel mesh rather than drawn: the waterline of a coxed
    four really is 13.4 m long and about half a metre wide, and a shape
    invented to look like a boat would misrepresent the one thing a plan
    view is for -- how much room this thing actually needs.
    """
    corners = np.asarray(boat.mesh.corners, dtype=float).reshape(-1, 3)[:, :2]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(corners)
        return corners[hull.vertices]
    except Exception:
        # Without scipy, a lens of the right length and beam.
        half_length = float(np.abs(corners[:, 0]).max())
        half_beam = float(np.abs(corners[:, 1]).max())
        angle = np.linspace(0.0, 2.0 * np.pi, 41)
        return np.column_stack([half_length * np.cos(angle),
                                half_beam * np.sin(angle)])


def oar_lines(boat, t: float):
    """``[(handle, lock, blade), ...]`` in the hull frame, plus drive flag.

    Returns ``(lines, is_drive)``.  Same source as the 3-D scene's oars,
    so the two views cannot disagree about where the blades are.
    """
    from ..crew.oarlock import blade_position, handle_position

    lines = []
    for seat in boat.rig.seats:
        for lock in seat.oarlocks:
            handle = handle_position(t, boat.timing, lock, boat.oar_sweep)
            blade = blade_position(t, boat.timing, lock, boat.oar_sweep)
            lines.append(np.array([handle[:2], np.asarray(lock.position)[:2],
                                   blade[:2]], dtype=float))
    return lines, bool(boat.timing.is_drive(t))


# -- course scenes --------------------------------------------------------

WATER = (58, 96, 122)
BANK = (74, 82, 62)
DOCK = (150, 110, 70)
BUILDING = (92, 92, 96)
BRIDGE = (176, 170, 158)


def _capped(layer: PlanLayer) -> PlanLayer:
    """A building layer, size-filtered and capped."""
    layer.min_pixels, layer.limit = 4.0, 220
    return layer.index()


def _rings(polygons):
    return [np.asarray(p, dtype=float) for p in (polygons or [])
            if len(np.asarray(p)) >= 3]


def charles_scene(with_buildings: bool = True) -> PlanScene:
    """Head of the Charles: the reach, its docks, its bridges."""
    from ..river import charles
    from ..river.charts import CourseGeometry
    from ..river.structures import charles_structures

    structures = charles_structures()
    raster = charles.charles_channel()
    geometry = CourseGeometry(channel=charles.rowable_channel(raster))
    line = geometry.line

    layers = [
        PlanLayer("water", _rings(structures.water), WATER, 1.0,
                  fill=True, closed=True).index(),
    ]
    if with_buildings:
        buildings = PlanLayer("buildings", _rings(structures.polygons),
                              BUILDING, 1.0, fill=True, closed=True)
        buildings.min_pixels, buildings.limit = 4.0, 220
        layers.append(buildings.index())
    layers.append(PlanLayer(
        "docks", [np.asarray(p, dtype=float)
                  for _kind, p in charles.load_obstructions() if len(p) > 1],
        DOCK, 3.0).index())

    spans = []
    for gate, _distance in geometry.gates_on_course():
        spans.append(np.array([gate.start, gate.end], dtype=float))
    layers.append(PlanLayer("bridges", spans, BRIDGE, 5.0).index())

    heading = float(np.arctan2(line[1, 1] - line[0, 1],
                               line[1, 0] - line[0, 0]))
    return PlanScene(name="Head of the Charles", layers=layers, course=line,
                     marks=[(float(0.5 * (g.start[0] + g.end[0])),
                             float(0.5 * (g.start[1] + g.end[1])), g.name)
                            for g, _d in geometry.gates_on_course()],
                     start=line[0].copy(), start_heading=heading)


def seattle_scene(race: str = "hotl") -> PlanScene:
    """Tail of the Lake or Head of the Lake, on the ship canal."""
    import os

    from ..river import seattle
    from ..river.structures import seattle_structures

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    stem = "hotl" if race == "hotl" else "totl"
    course = np.load(os.path.join(root, "data", "%s_course.npy" % stem))
    buoy_path = os.path.join(root, "data", "%s_buoys.npy" % stem)
    buoys = np.load(buoy_path) if os.path.exists(buoy_path) else None

    structures = seattle_structures()
    layers = [
        PlanLayer("water", _rings(structures.water), WATER, 1.0,
                  fill=True, closed=True).index(),
        _capped(PlanLayer("buildings", _rings(structures.polygons),
                          BUILDING, 1.0, fill=True, closed=True)),
        PlanLayer("docks", [np.asarray(p, dtype=float)
                            for _kind, p in seattle.load_obstructions()
                            if len(p) > 1], DOCK, 3.0).index(),
    ]
    heading = float(np.arctan2(course[1, 1] - course[0, 1],
                               course[1, 0] - course[0, 0]))
    return PlanScene(
        name="Head of the Lake" if race == "hotl" else "Tail of the Lake",
        layers=layers, course=course, buoys=buoys,
        marks=[(float(course[0, 0]), float(course[0, 1]), "START"),
               (float(course[-1, 0]), float(course[-1, 1]), "FINISH")],
        start=course[0].copy(), start_heading=heading)


def build_scene(race: str) -> PlanScene:
    """``charles``, ``totl`` or ``hotl``."""
    if race == "charles":
        return charles_scene()
    if race in ("totl", "hotl"):
        return seattle_scene(race)
    raise ValueError("unknown race %r" % race)
