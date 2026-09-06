r"""The course as triangles, built once, for a real-time first-person view.

The plan view answers "where am I"; the seat answers "is this steerable".
From 0.55 m off the water in the bow of a four you see almost nothing
except the near bank, the line ahead and the way the horizon swings when
the boat yaws -- and that last one is the whole point, because it is the
only cue a coxswain actually steers on.

Backend-neutral, like :mod:`coxswain.viz.planscene`: this returns numpy
vertex arrays in world metres and imports no GL.  ``scripts/fpv.py``
uploads them; a Panda3D or PyVista backend could take the same arrays.

Built once, drawn every frame
-----------------------------
Everything here is static.  The boat moves, the world does not, so the
whole course goes into a handful of buffers at load time and each frame
is one matrix and a few draw calls.  That is the opposite of what
``RiverScene`` does -- it rebuilds meshes per frame, which is right for a
figure and hopeless at 60 Hz -- and it is why this is a separate module
rather than a flag on that one.

Lidar over water is still not water
-----------------------------------
The elevation model scatters badly over water (SOURCES sec. 105: a naive
threshold called 48% of Lake Union dry land), so land triangles are
emitted only where the **water raster** says there is no water, not where
the DEM says the ground is high.  The water is a flat plane underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = ["MeshPart", "WorldMesh", "land_mesh", "water_plane",
           "building_walls", "ribbon", "line_markers", "buoy_solids",
           "hull_solid", "tree_solids",
           "dock_solids", "bridge_solids",
           "box_solid", "build_world"]


@dataclass
class MeshPart:
    """Triangles with a colour per vertex."""

    name: str
    vertices: np.ndarray                    # (n, 3) float32
    colours: np.ndarray                     # (n, 3) float32
    normals: Optional[np.ndarray] = None    # (n, 3) float32

    @property
    def triangles(self) -> int:
        return len(self.vertices) // 3

    def interleaved(self) -> np.ndarray:
        """``(n, 9)`` of position, normal, colour, ready for one buffer."""
        normals = self.normals
        if normals is None:
            normals = np.tile(np.array([0.0, 0.0, 1.0], dtype="f4"),
                              (len(self.vertices), 1))
        return np.hstack([self.vertices, normals,
                          self.colours]).astype("f4")


@dataclass
class WorldMesh:
    parts: List[MeshPart] = field(default_factory=list)

    def add(self, part: Optional[MeshPart]) -> None:
        if part is not None and len(part.vertices):
            self.parts.append(part)

    @property
    def triangles(self) -> int:
        return sum(p.triangles for p in self.parts)


def _face_normals(vertices: np.ndarray) -> np.ndarray:
    """Flat normals, one per triangle, repeated per vertex."""
    tri = vertices.reshape(-1, 3, 3)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(length, 1e-9)
    return np.repeat(normal, 3, axis=0).astype("f4")


# -- the ground -----------------------------------------------------------

LAND_LOW = np.array([0.31, 0.35, 0.24])
LAND_HIGH = np.array([0.46, 0.47, 0.40])
WATER_COLOUR = np.array([0.16, 0.28, 0.36])


def land_mesh(terrain, wet_at, box, step: float = 8.0) -> Optional[MeshPart]:
    """A heightfield over ``box`` where ``wet_at`` says there is no water.

    ``wet_at(east, north)`` takes 2-D arrays and returns a boolean array.
    Cells with any wet corner are dropped, so the bank ends at the
    waterline instead of diving under it.
    """
    x0, y0, x1, y1 = box
    east = np.arange(x0, x1 + step, step)
    north = np.arange(y0, y1 + step, step)
    if len(east) < 2 or len(north) < 2:
        return None
    grid_x, grid_y = np.meshgrid(east, north)
    height = np.asarray(terrain.height_above_water(grid_x.ravel(),
                                                   grid_y.ravel()))
    height = np.maximum(height.reshape(grid_x.shape), 0.0)
    wet = wet_at(grid_x, grid_y)

    # Two triangles per cell, skipping any cell touching water.
    a = (slice(0, -1), slice(0, -1))
    b = (slice(0, -1), slice(1, None))
    c = (slice(1, None), slice(1, None))
    d = (slice(1, None), slice(0, -1))
    keep = ~(wet[a] | wet[b] | wet[c] | wet[d])
    if not keep.any():
        return None

    def corner(sl):
        return np.stack([grid_x[sl][keep], grid_y[sl][keep],
                         height[sl][keep]], axis=-1)

    pa, pb, pc, pd = corner(a), corner(b), corner(c), corner(d)
    vertices = np.concatenate([
        np.stack([pa, pb, pc], axis=1).reshape(-1, 3),
        np.stack([pa, pc, pd], axis=1).reshape(-1, 3)]).astype("f4")

    # Higher ground reads lighter, which is enough to give the bank shape
    # without a texture.  A photograph belongs here eventually.
    lift = np.clip(vertices[:, 2] / 12.0, 0.0, 1.0)[:, None]
    colours = (LAND_LOW + (LAND_HIGH - LAND_LOW) * lift).astype("f4")
    return MeshPart("land", vertices, colours, _face_normals(vertices))


def water_plane(centre, reach: float = 3000.0,
                level: float = 0.0) -> MeshPart:
    """One big quad.  The horizon is the thing a coxswain steers on."""
    cx, cy = float(centre[0]), float(centre[1])
    corners = np.array([
        [cx - reach, cy - reach, level], [cx + reach, cy - reach, level],
        [cx + reach, cy + reach, level], [cx - reach, cy + reach, level]],
        dtype="f4")
    vertices = corners[[0, 1, 2, 0, 2, 3]]
    colours = np.tile(WATER_COLOUR, (6, 1)).astype("f4")
    normals = np.tile([0.0, 0.0, 1.0], (6, 1)).astype("f4")
    return MeshPart("water", vertices, colours, normals)


# -- things standing on it ------------------------------------------------

def building_walls(polygons, heights, bases=None, box=None,
                   colour=(0.42, 0.41, 0.40),
                   limit: int = 4000) -> Optional[MeshPart]:
    """Extruded footprints -- **walls only**.

    From 0.55 m off the water you never see a roof, so the top faces are
    not emitted: it halves the triangle count and changes nothing you can
    see from the seat.
    """
    walls, kept = [], 0
    base_colour = np.asarray(colour, dtype=float)
    for index, polygon in enumerate(polygons):
        if kept >= limit:
            break
        ring = np.asarray(polygon, dtype=float)
        if len(ring) < 3:
            continue
        if box is not None:
            if (ring[:, 0].max() < box[0] or ring[:, 0].min() > box[2]
                    or ring[:, 1].max() < box[1] or ring[:, 1].min() > box[3]):
                continue
        top = float(heights[index])
        if top < 2.0:
            continue
        low = float(bases[index]) if bases is not None else 0.0
        nxt = np.roll(ring, -1, axis=0)
        for start, end in zip(ring, nxt):
            walls.append([[start[0], start[1], low], [end[0], end[1], low],
                          [end[0], end[1], top]])
            walls.append([[start[0], start[1], low], [end[0], end[1], top],
                          [start[0], start[1], top]])
        kept += 1
    if not walls:
        return None
    vertices = np.asarray(walls, dtype="f4").reshape(-1, 3)
    # Faint per-building variation so a terrace is not one flat slab.
    shade = 0.85 + 0.3 * ((np.arange(len(vertices)) // 6) % 7) / 7.0
    colours = (base_colour[None, :] * shade[:, None]).astype("f4")
    return MeshPart("buildings", vertices, colours, _face_normals(vertices))


def ribbon(points, width: float = 0.30, height: float = 0.03,
           colour=(1.0, 0.57, 0.28)) -> Optional[MeshPart]:
    """A flat strip along a polyline -- the racing line, drawn on the water.

    Narrow on purpose.  A cox has no painted line to follow, so this is
    a training aid and not scenery: at 1.2 m wide it started under the
    bow and filled half the screen, which is not a hint, it is a wall.
    """
    line = np.asarray(points, dtype=float)[:, :2]
    if len(line) < 2:
        return None
    step = np.gradient(line, axis=0)
    normal = np.column_stack([-step[:, 1], step[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    left = line + 0.5 * width * normal
    right = line - 0.5 * width * normal
    quads = []
    for index in range(len(line) - 1):
        # Dashed.  A solid strip is a wall in perspective: you are
        # standing on the near end of it, so it fills the bottom of the
        # screen however narrow it is.  Gaps make it read as a guide.
        if index % 3 == 2:
            continue
        la, lb = left[index], left[index + 1]
        ra, rb = right[index], right[index + 1]
        quads.append([[la[0], la[1], height], [ra[0], ra[1], height],
                      [rb[0], rb[1], height]])
        quads.append([[la[0], la[1], height], [rb[0], rb[1], height],
                      [lb[0], lb[1], height]])
    vertices = np.asarray(quads, dtype="f4").reshape(-1, 3)
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    normals = np.tile([0.0, 0.0, 1.0], (len(vertices), 1)).astype("f4")
    return MeshPart("line", vertices, colours, normals)


def box_solid(centre, half, colour=(0.62, 0.60, 0.56)) -> MeshPart:
    """An axis-aligned box, for bridge decks and other blunt objects."""
    cx, cy, cz = centre
    hx, hy, hz = half
    corner = np.array([[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
                       [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
                       [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
                       [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz]],
                      dtype="f4")
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    vertices = corner[np.asarray(faces).ravel()]
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    return MeshPart("box", vertices, colours, _face_normals(vertices))


def hull_solid(boat, deck: float = 0.30, colour=(0.88, 0.89, 0.86),
               deck_colour=(0.74, 0.76, 0.74),
               cockpit: float = 0.25,
               floor: float = 0.06) -> MeshPart:
    """The shell itself, in the **hull frame**, for the seat view.

    A coxswain in a bow-loader sits 2.4 m behind the bow with their eye
    about a quarter of a metre above the foredeck, so the bow is the one
    piece of the boat always in shot -- and it is the reference the whole
    view is read against.  Without it the horizon swings with nothing to
    swing relative to, which is why the first seat view felt like a
    camera on a stick rather than a boat.

    Returned in hull coordinates: the caller rotates and translates it
    each frame, the same way the oars are handled.
    """
    from .planscene import boat_outline

    ring = np.asarray(boat_outline(boat), dtype=float)
    # boat_outline gives a convex hull whose winding is not guaranteed;
    # order it by angle so the sides and deck come out consistent.
    centre = ring.mean(axis=0)
    order = np.argsort(np.arctan2(ring[:, 1] - centre[1],
                                  ring[:, 0] - centre[0]))
    ring = ring[order]

    faces, shades = [], []
    nxt = np.roll(ring, -1, axis=0)
    for a, b in zip(ring, nxt):
        low_a = [a[0], a[1], 0.0]
        low_b = [b[0], b[1], 0.0]
        top_a = [a[0], a[1], deck]
        top_b = [b[0], b[1], deck]
        faces += [[low_a, low_b, top_b], [low_a, top_b, top_a]]
        shades += [colour] * 6
    # The foredeck, and a floor under the opening.
    #
    # Only about two feet ahead of a coxswain's face is open -- enough
    # for shoulders -- and the rest of the bow is decked over.  Decking
    # the *whole* shell put a surface directly under the eye and filled a
    # third of the screen with the inside of the boat; leaving the
    # opening as a hole was worse, because with no floor you saw the
    # river straight through the hull.  So: decking from ``cockpit``
    # ahead of the seat, and a floor across the gap.
    seat_x = float(boat.rig.coxswain_position[0]) + float(cockpit)
    bow = float(ring[:, 0].max())
    hub = [0.5 * (seat_x + bow), centre[1], deck]
    for a, b in zip(ring, nxt):
        if a[0] < seat_x and b[0] < seat_x:
            continue
        faces.append([hub, [a[0], a[1], deck], [b[0], b[1], deck]])
        shades += [deck_colour] * 3
    # The cockpit floor: the inside of the boat, not the river.
    inside = ring[ring[:, 0] <= seat_x + 0.05]
    if len(inside) >= 3:
        hub_in = [float(inside[:, 0].mean()), centre[1], float(floor)]
        order_in = np.argsort(np.arctan2(inside[:, 1] - centre[1],
                                         inside[:, 0] - hub_in[0]))
        inside = inside[order_in]
        for a, b in zip(inside, np.roll(inside, -1, axis=0)):
            faces.append([hub_in, [a[0], a[1], floor], [b[0], b[1], floor]])
            shades += [(0.22, 0.23, 0.24)] * 3

    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.asarray(shades, dtype="f4")
    return MeshPart("hull", vertices, colours, _face_normals(vertices))


def line_markers(points, spacing: float = 30.0, height: float = 0.55,
                 radius: float = 0.14,
                 colour=(0.95, 0.55, 0.20)) -> Optional[MeshPart]:
    """Small posts along the course, instead of a painted line.

    A ribbon on the water is wrong twice over: a coxswain has no line to
    follow, and in perspective you are standing on the near end of it, so
    however narrow it is it fills the bottom of the screen.  Posts every
    ``spacing`` metres give the same aiming cue, look like the marks a
    regatta actually sets, and leave the water visible.
    """
    line = np.asarray(points, dtype=float)[:, :2]
    if len(line) < 2:
        return None
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(line, axis=0).T))])
    wanted = np.arange(spacing, station[-1], spacing)
    east = np.interp(wanted, station, line[:, 0])
    north = np.interp(wanted, station, line[:, 1])
    parts = [box_solid((x, y, height * 0.5), (radius, radius, height * 0.5),
                       colour=colour) for x, y in zip(east, north)]
    if not parts:
        return None
    vertices = np.concatenate([p.vertices for p in parts])
    colours = np.concatenate([p.colours for p in parts])
    return MeshPart("marks", vertices, colours, _face_normals(vertices))


def buoy_solids(buoys, height: float = 0.45,
                radius: float = 0.22) -> Optional[MeshPart]:
    """The regatta's own marks, coloured by the side they must be passed."""
    if buoys is None or not len(buoys):
        return None
    parts = []
    for keep_to_port, east, north in np.asarray(buoys, dtype=float):
        colour = (0.98, 0.55, 0.18) if keep_to_port else (1.0, 0.84, 0.05)
        parts.append(box_solid((east, north, height * 0.5),
                               (radius, radius, height * 0.5), colour=colour))
    vertices = np.concatenate([p.vertices for p in parts])
    colours = np.concatenate([p.colours for p in parts])
    return MeshPart("buoys", vertices, colours, _face_normals(vertices))


#: Crown colours by growth form, in the order ``TreeStand.FORMS`` uses.
CROWN = ((0.24, 0.34, 0.20), (0.16, 0.26, 0.18), (0.20, 0.31, 0.20),
         (0.30, 0.40, 0.24))
TRUNK = (0.26, 0.20, 0.15)


def tree_solids(stand, box, limit: int = 5000, near=None,
                min_height: float = 3.0) -> Optional[MeshPart]:
    """Trees as a trunk and a low-poly crown.

    The bank of a river is trees, and leaving them out is why the first
    seat view looked like a reservoir.  There are 24,392 of them on the
    Charles and 742,517 on Lake Union, so they are ranked by height and
    capped: a crown is eight triangles and a trunk twelve, and five
    thousand of them is a hundred thousand triangles, which an Intel UHD
    holds without noticing.

    ``near`` is an optional ``(n, 2)`` line -- the course -- used to
    prefer the trees a crew can actually see over the ones on the hill
    behind them.
    """
    points = np.asarray(stand.points, dtype=float)
    heights = np.asarray(stand.heights, dtype=float)
    forms = np.asarray(stand.form, dtype=int)
    keep = ((points[:, 0] >= box[0]) & (points[:, 0] <= box[2])
            & (points[:, 1] >= box[1]) & (points[:, 1] <= box[3])
            & (heights >= min_height))
    index = np.nonzero(keep)[0]
    if not len(index):
        return None
    if near is not None and len(index) > limit:
        from scipy.spatial import cKDTree

        gap = cKDTree(np.asarray(near, dtype=float)).query(points[index])[0]
        # Tall and close beats tall and far, the same rule the 3-D scene
        # uses for which trees are worth drawing at all.
        index = index[np.argsort(-heights[index] / np.maximum(gap, 5.0))]
    index = index[:limit]

    faces, shades = [], []
    for i in index:
        x, y = points[i]
        height = float(heights[i])
        form = int(forms[i]) if i < len(forms) else 0
        crown_colour = CROWN[form % len(CROWN)]
        # Trunk: a square post up to the crown.
        stem = 0.40 * height
        radius = max(0.018 * height, 0.05)
        trunk = box_solid((x, y, 0.5 * stem), (radius, radius, 0.5 * stem),
                          colour=TRUNK)
        faces.append(trunk.vertices)
        shades.append(trunk.colours)
        # Crown: an octahedron, wider for a broadleaf than a conifer.
        spread = (0.16 if form == 1 else 0.30) * height
        top = height
        mid = 0.5 * (stem + top)
        apex = [x, y, top]
        base = [x, y, stem]
        rim = [[x + spread, y, mid], [x, y + spread, mid],
               [x - spread, y, mid], [x, y - spread, mid]]
        crown = []
        for a, b in zip(rim, rim[1:] + rim[:1]):
            crown.append([apex, a, b])
            crown.append([base, b, a])
        faces.append(np.asarray(crown, dtype="f4").reshape(-1, 3))
        shades.append(np.tile(np.asarray(crown_colour, dtype="f4"),
                              (len(crown) * 3, 1)))
    vertices = np.concatenate(faces).astype("f4")
    colours = np.concatenate(shades).astype("f4")
    return MeshPart("trees", vertices, colours, _face_normals(vertices))


def dock_solids(polylines, height: float = 0.9,
                width: float = 1.6) -> Optional[MeshPart]:
    """Boathouse floats and piers as low kerbs on the water.

    These are what a crew actually steers off, and from a seat 0.55 m up
    a float is a real object in the way, not a line on a map.
    """
    parts = []
    for line in polylines:
        points = np.asarray(line, dtype=float)
        if len(points) < 2:
            continue
        for a, b in zip(points[:-1], points[1:]):
            along = b - a
            length = float(np.hypot(*along))
            if length < 0.5:
                continue
            along = along / length
            middle = 0.5 * (a + b)
            parts.append(_slab(middle, along, length, width, height, height))
    if not parts:
        return None
    vertices = np.concatenate([p.vertices for p in parts])
    colours = np.tile(np.array([0.46, 0.36, 0.26], dtype="f4"),
                      (len(vertices), 1))
    return MeshPart("docks", vertices, colours, _face_normals(vertices))


def bridge_solids(race: str, scene) -> Optional[MeshPart]:
    """Decks and piers for the bridges a course goes under.

    From the water these are the landmark -- on the Charles you steer the
    whole race by which arch you are lined up on -- so a plain deck slab
    on piers is worth far more than nothing while the arch geometry from
    :mod:`coxswain.viz.river3d` waits to be shared.
    """
    parts = []
    if race == "charles":
        from ..river import charles
        from ..river.bridges import deck_geometry
        from ..river.charts import CourseGeometry

        geometry = CourseGeometry(channel=charles.charles_channel())
        for gate, _distance in geometry.gates_on_course():
            row = deck_geometry(gate.name)
            if row is None:
                continue
            _form, width, level, depth, _spans, _span, _camber, _src = row
            start = np.asarray(gate.start, dtype=float)
            end = np.asarray(gate.end, dtype=float)
            middle = 0.5 * (start + end)
            span = float(np.hypot(*(end - start)))
            along = (end - start) / max(span, 1e-9)
            # Deck: a slab across the river, plus a pier at each third.
            parts.append(_slab(middle, along, span, width, level, depth))
            for fraction in (0.33, 0.67):
                foot = start + (end - start) * fraction
                parts.append(box_solid((foot[0], foot[1], level * 0.5),
                                       (1.4, 1.4, level * 0.5),
                                       colour=(0.50, 0.49, 0.46)))
    else:
        try:
            from ..river.seattle import canal_bridges

            for bridge in canal_bridges():
                middle = np.asarray(bridge.centre, dtype=float)
                along = np.asarray(bridge.axis, dtype=float)
                parts.append(_slab(middle, along, bridge.length,
                                   max(bridge.width, 8.0), 10.7, 1.6))
        except Exception:
            return None
    if not parts:
        return None
    vertices = np.concatenate([p.vertices for p in parts])
    colours = np.concatenate([p.colours for p in parts])
    return MeshPart("bridges", vertices, colours, _face_normals(vertices))


def _slab(middle, along, span, width, level, depth) -> MeshPart:
    """A deck slab lying along ``along``, centred on ``middle``."""
    across = np.array([-along[1], along[0]])
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        point = (np.asarray(middle[:2], dtype=float)
                 + along * (sx * span * 0.5) + across * (sy * width * 0.5))
        corners.append(point)
    corners = np.asarray(corners)
    top = np.column_stack([corners, np.full(4, level)])
    bottom = np.column_stack([corners, np.full(4, level - depth)])
    faces = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        faces += [[bottom[a], bottom[b], top[b]], [bottom[a], top[b], top[a]]]
    faces += [[top[0], top[1], top[2]], [top[0], top[2], top[3]]]
    faces += [[bottom[0], bottom[2], bottom[1]],
              [bottom[0], bottom[3], bottom[2]]]
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.tile(np.array([0.60, 0.58, 0.54], dtype="f4"),
                      (len(vertices), 1))
    return MeshPart("deck", vertices, colours)


# -- assembling a course --------------------------------------------------

def build_world(race: str = "charles", reach: float = 900.0,
                step: float = 8.0, with_buildings: bool = True,
                guide: bool = True, trees: bool = True):
    """``(WorldMesh, PlanScene)`` for a course.

    ``reach`` is how far either side of the course to build.  The whole
    course goes in at once -- a few hundred thousand triangles, uploaded
    once -- which is well inside what an Intel UHD holds and saves
    streaming until it is needed.
    """
    from .planscene import build_scene

    scene = build_scene(race)
    course = scene.course
    box = (course[:, 0].min() - reach, course[:, 1].min() - reach,
           course[:, 0].max() + reach, course[:, 1].max() + reach)

    if race == "charles":
        from ..river import charles
        from ..river.structures import charles_structures
        from ..river.terrain import charles_terrain

        terrain = charles_terrain()
        raster = charles.charles_channel()
        structures = charles_structures()
    else:
        from ..river.seattle import ship_canal_channel
        from ..river.structures import seattle_structures
        from ..river.terrain import seattle_terrain

        terrain = seattle_terrain()
        raster = ship_canal_channel(resolution=10.0)
        structures = seattle_structures()

    def wet_at(east, north):
        rows = np.clip(np.searchsorted(raster.north, north), 0,
                       len(raster.north) - 1)
        cols = np.clip(np.searchsorted(raster.east, east), 0,
                       len(raster.east) - 1)
        return raster.water[rows, cols]

    mesh = WorldMesh()
    mesh.add(water_plane(course.mean(axis=0)))
    mesh.add(land_mesh(terrain, wet_at, box, step=step))
    if with_buildings:
        mesh.add(building_walls(structures.polygons, structures.heights,
                                getattr(structures, "base", None), box=box))
    docks = scene.layer("docks")
    if docks is not None:
        mesh.add(dock_solids(docks.polylines))
    if trees:
        try:
            if race == "charles":
                from ..river.structures import charles_trees as tree_stand
            else:
                from ..river.structures import seattle_trees as tree_stand
            mesh.add(tree_solids(tree_stand(), box, near=course))
        except Exception as error:            # pragma: no cover
            print("   (no trees: %s)" % str(error)[:60])
    mesh.add(bridge_solids(race, scene))
    mesh.add(buoy_solids(scene.buoys))
    if guide:
        mesh.add(line_markers(course))
    return mesh, scene
