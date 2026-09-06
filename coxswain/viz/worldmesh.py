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
           "building_walls", "photo_colour", "ribbon", "line_markers", "buoy_solids",
           "hull_solid", "tree_solids", "arch_bridge", "cut_walls",
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


def land_mesh(terrain, wet_at, box, step: float = 8.0,
              imagery=None) -> Optional[MeshPart]:
    """A heightfield over ``box`` where ``wet_at`` says there is no water.

    ``wet_at(east, north)`` takes 2-D arrays and returns a boolean array.
    Cells with any wet corner are dropped, so the bank ends at the
    waterline instead of diving under it.

    ``imagery`` is an :class:`~coxswain.river.terrain.Imagery`.  With it,
    every vertex takes the colour of the orthophoto directly above it, so
    the bank is grass where there is grass, tarmac where there is a car
    park and roof where there is a boathouse -- which is what a coxswain
    is actually reading when they place themselves against the shore.
    Without it the fallback is the old height ramp.
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

    lift = np.clip(vertices[:, 2] / 12.0, 0.0, 1.0)[:, None]
    colours = (LAND_LOW + (LAND_HIGH - LAND_LOW) * lift)
    if imagery is not None:
        try:
            photo = np.asarray(imagery.sample(vertices[:, 0],
                                              vertices[:, 1]), dtype=float)
            if photo.max() > 1.5:                 # 0-255 imagery
                photo = photo / 255.0
            good = np.isfinite(photo).all(axis=1) & (photo.sum(axis=1) > 0.02)
            # Keep a little of the height ramp under the photograph: it
            # is a flat overhead image with no shading of its own, and
            # without this a bank reads as wallpaper rather than ground.
            blended = 0.82 * photo + 0.18 * colours
            colours = np.where(good[:, None], blended, colours)
        except Exception as error:                # pragma: no cover
            print("   (terrain photo failed, using the height ramp: %s)"
                  % str(error)[:60])
    return MeshPart("land", vertices, colours.astype("f4"),
                    _face_normals(vertices))


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

#: Keep a sampled colour inside this lightness band.  An orthophoto over
#: a footprint can come back near-black (a slate roof in shadow) or
#: near-white (a bright membrane), and either one, painted on a wall,
#: reads as a hole rather than a building -- a bug Seattle hit first.
PHOTO_RANGE = (0.30, 0.74)
#: Cap on saturation, for the same reason: a photo pixel over a green
#: roof should not give a green building.
MAX_SATURATION = 0.34


def photo_colour(imagery, ring, fallback):
    """Colour of one footprint, sampled from the orthophoto.

    An orthophoto is a picture taken from directly overhead, so what it
    gives is the *roof*.  From a coxswain's seat you see walls, never
    roofs -- but a building's roof and its walls are far more alike than
    either is to a flat grey default, and this is the only per-building
    colour available: of 9,631 buildings on the Charles reach exactly
    **11** carry an OpenStreetMap ``building:colour``.  So the roof
    colour is used for the walls, clamped so it cannot go black, white
    or lurid, and that is what makes Dunster and the Leverett towers
    read as brick at the Weeks turn instead of as grey blocks.
    """
    if imagery is None:
        return fallback
    centre = ring.mean(axis=0)
    # The centroid plus the vertices drawn in towards it, so the samples
    # land on the building and not on the street beside it.
    probes = np.vstack([centre[None, :], centre + 0.55 * (ring - centre)])
    try:
        pixels = np.asarray(imagery.sample(probes[:, 0], probes[:, 1]),
                            dtype=float)
    except Exception:
        return fallback
    if pixels.ndim == 1:
        pixels = pixels[None, :]
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    if not len(pixels):
        return fallback
    colour = np.median(pixels, axis=0)
    if colour.max() > 1.5:                     # 0-255 imagery
        colour = colour / 255.0
    light = float(colour.mean())
    if light < 1e-6:
        return fallback
    low, high = PHOTO_RANGE
    colour = colour * (min(max(light, low), high) / light)
    grey = float(colour.mean())
    spread = float(colour.max() - colour.min())
    if spread > MAX_SATURATION:
        colour = grey + (colour - grey) * (MAX_SATURATION / spread)
    return np.clip(colour, 0.0, 1.0)


def building_walls(polygons, heights, bases=None, box=None,
                   colour=(0.42, 0.41, 0.40), imagery=None,
                   limit: int = 4000) -> Optional[MeshPart]:
    """Extruded footprints -- **walls only**, coloured from the orthophoto.

    From 0.55 m off the water you never see a roof, so the top faces are
    not emitted: it halves the triangle count and changes nothing you can
    see from the seat.

    ``imagery`` is an :class:`~coxswain.river.terrain.Imagery`; when it is
    given each building takes its own colour from the photograph (see
    :func:`photo_colour`), which is what turns the Harvard houses from
    grey blocks into the landmarks a crew steers the Weeks turn by.
    """
    walls, tints, kept = [], [], 0
    base_colour = np.asarray(colour, dtype=float)
    for index, polygon in enumerate(polygons):
        if kept >= limit:
            break
        ring = np.asarray(polygon, dtype=float)
        if len(ring) < 3:
            continue
        # Force the ring counter-clockwise.
        #
        # For a CCW ring the wall winding below gives outward normals; for
        # a clockwise one it gives inward normals, and the building
        # renders inside-out -- back-face culling removes the near wall
        # and leaves the far one, so you see the back of every building
        # through the front of it.  OpenStreetMap does not guarantee a
        # winding, so roughly half of them came out this way.  The
        # shoelace sign says which is which.
        area = float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1)
                            - np.roll(ring[:, 0], -1) * ring[:, 1]))
        if area < 0.0:
            ring = ring[::-1]
        if box is not None:
            if (ring[:, 0].max() < box[0] or ring[:, 0].min() > box[2]
                    or ring[:, 1].max() < box[1] or ring[:, 1].min() > box[3]):
                continue
        top = float(heights[index])
        if top < 2.0:
            continue
        low = float(bases[index]) if bases is not None else 0.0
        nxt = np.roll(ring, -1, axis=0)
        tint = photo_colour(imagery, ring, base_colour)
        for start, end in zip(ring, nxt):
            walls.append([[start[0], start[1], low], [end[0], end[1], low],
                          [end[0], end[1], top]])
            walls.append([[start[0], start[1], low], [end[0], end[1], top],
                          [start[0], start[1], top]])
            tints.extend([tint] * 6)
        kept += 1
    if not walls:
        return None
    vertices = np.asarray(walls, dtype="f4").reshape(-1, 3)
    # Faint per-face variation so a long facade is not one flat slab.
    shade = 0.88 + 0.24 * ((np.arange(len(vertices)) // 6) % 7) / 7.0
    colours = (np.asarray(tints, dtype=float) * shade[:, None]).astype("f4")
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
    # The foredeck and the cockpit floor, as **strips across the boat**.
    #
    # These were fans from a hub on the centreline, and a fan is the
    # wrong primitive for a shape 13 m long and 0.5 m wide: the hub sat
    # 1.3 m ahead of the eye, every triangle came out a sliver radiating
    # from one screen point, and the two nearest ones -- the ones that
    # should cover the deck directly under the camera -- had a vertex
    # behind the near plane and were culled.  The deck was in the buffer
    # and not on the screen.
    #
    # A strip between the port and starboard edges cannot do that.  The
    # rungs are explicit, the winding is written down rather than
    # inherited from a convex hull, and the assertion below holds it:
    # every deck and floor triangle faces up.
    seat_x = float(boat.rig.coxswain_position[0]) + float(cockpit)
    bow, stern = float(ring[:, 0].max()), float(ring[:, 0].min())

    def edges_at(x):
        """``(y_port, y_starboard)`` of the hull at station ``x``."""
        near = ring[np.abs(ring[:, 0] - x) < 1.2]
        if len(near) < 2:
            near = ring[np.argsort(np.abs(ring[:, 0] - x))[:4]]
        return float(near[:, 1].min()), float(near[:, 1].max())

    def strip(x0, x1, height, shade):
        stations = np.linspace(x0, x1, 14)
        for xa, xb in zip(stations[:-1], stations[1:]):
            pa, sa = edges_at(xa)
            pb, sb = edges_at(xb)
            # Counter-clockwise seen from above, so the normal is +z and
            # the deck survives back-face culling: aft-starboard,
            # forward-starboard, forward-port, aft-port.
            faces.append([[xa, pa, height], [xb, pb, height],
                          [xb, sb, height]])
            faces.append([[xa, pa, height], [xb, sb, height],
                          [xa, sa, height]])
            shades.extend([shade] * 6)

    strip(seat_x, bow, deck, deck_colour)
    strip(stern, seat_x, float(floor), (0.22, 0.23, 0.24))

    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.asarray(shades, dtype="f4")
    normals = _face_normals(vertices)
    flat = np.abs(normals[:, 2]) > 0.9
    if flat.any() and normals[flat, 2].min() <= 0.0:
        raise AssertionError(
            "deck or floor triangles wound face-down: they would be "
            "culled and the coxswain would see the river through the boat")
    return MeshPart("hull", vertices, colours, normals)


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


def tree_solids(stand, box, limit: int = 25000, near=None,
                min_height: float = 3.0) -> Optional[MeshPart]:
    """Trees as a trunk and a low-poly crown.

    The bank of a river is trees, and leaving them out is why the first
    seat view looked like a reservoir.  There are 24,392 of them on the
    Charles and 742,517 on Lake Union, so they are ranked by height and
    by distance from the course, and capped.

    **The cap is 25,000, and impostors were measured and rejected.**  A
    crown is eight triangles and a trunk twelve, so every tree on the
    Charles reach comes to 478,860 triangles and 1.3 s of build -- in a
    scene that already renders 682,000 without complaint on an Intel
    UHD.  Swapping distant trees for crossed billboard quads would cut
    that five-fold, but there is nothing to spend the saving on: the
    reach only *has* 24,392 trees and they all fit.  Lake Union, with
    thirty times as many, is where an impostor would earn its keep, and
    there the ranking already keeps the drawn ones near the course.

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


#: Height of the Montlake Cut's walls above the water, m, and the width
#: of the walkway behind them.
WALL_HEIGHT = 2.6
WALKWAY = 4.0


def cut_walls(terrain=None, colour=(0.66, 0.65, 0.62),
              walk=(0.55, 0.54, 0.51)) -> Optional[MeshPart]:
    """The Montlake Cut: concrete walls and a walkway on both banks.

    The Cut is not a shoreline, it is a **channel cut through a
    hill and walled in concrete**, with a pedestrian path along each
    side -- and from a boat that is the whole visual character of the
    place: a 50 m slot between two hard vertical edges.  Drawn from the
    water polygon's own boundary, so the wall stands exactly where the
    mapped waterline is, rather than from the fourteen sparse
    ``barrier=wall`` ways OpenStreetMap happens to carry there.

    Where a bank rises above the wall the terrain mesh takes over behind
    it; the wall is only ever the hard edge at the water.
    """
    try:
        from ..river.seattle import load_water
    except Exception:
        return None
    rings = []
    _origin, bodies = load_water()
    for name, ring in bodies:
        if str(name) != "Montlake Cut":
            continue
        ring = np.asarray(ring, dtype=float)
        if len(ring) >= 4:
            rings.append(ring)
    if not rings:
        return None

    faces, shades = [], []
    for ring in rings:
        nxt = np.roll(ring, -1, axis=0)
        centre = ring.mean(axis=0)
        for a, b in zip(ring, nxt):
            span = float(np.hypot(*(b - a)))
            if span < 0.5 or span > 120.0:
                continue
            along = (b - a) / span
            normal = np.array([-along[1], along[0]])
            # Point the walkway away from the water.
            if float(np.dot(normal, 0.5 * (a + b) - centre)) < 0.0:
                normal = -normal
            top = WALL_HEIGHT
            # The wall face, looking back over the water.
            _quad(faces, shades,
                  [a[0], a[1], 0.0], [b[0], b[1], 0.0],
                  [b[0], b[1], top], [a[0], a[1], top],
                  np.array([-normal[0], -normal[1], 0.0]), colour)
            # The walkway on top of it.
            c = b + normal * WALKWAY
            d = a + normal * WALKWAY
            _quad(faces, shades,
                  [a[0], a[1], top], [b[0], b[1], top],
                  [c[0], c[1], top], [d[0], d[1], top],
                  np.array([0.0, 0.0, 1.0]), walk)
    if not faces:
        return None
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.asarray(shades, dtype="f4")
    return MeshPart("cut walls", vertices, colours, _face_normals(vertices))


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


def _quad(faces, shades, p0, p1, p2, p3, want, colour):
    """Two triangles for a quad, wound so the normal points along ``want``.

    Winding is the thing that goes wrong silently in this file: a face
    wound the wrong way is culled and simply is not there, and the only
    symptom is a hole in a picture.  Rather than reason about vertex
    order at every call site -- which has already produced a foredeck
    that existed in the buffer and not on the screen -- the direction the
    face should look is passed in and the order is derived from it.
    """
    normal = np.cross(np.asarray(p1) - np.asarray(p0),
                      np.asarray(p2) - np.asarray(p0))
    if float(np.dot(normal, want)) < 0.0:
        p0, p1, p2, p3 = p0, p3, p2, p1
    faces.append([p0, p1, p2])
    faces.append([p0, p2, p3])
    shades.extend([colour] * 6)


#: Height of the arch springing as a fraction of the deck height, and
#: how much of the span each pier occupies.
SPRINGING = 0.12
PIER_FRACTION = 0.13


def arch_bridge(start, end, width: float, level: float, depth: float,
                spans: int, colour=(0.72, 0.71, 0.67),
                pier_colour=(0.58, 0.57, 0.54), samples: int = 13):
    """A concrete deck-arch bridge, as one sees it from a boat.

    River Street, Western Avenue, Larz Anderson and the Weeks Footbridge
    are all *concrete arch, deck* -- NBI item 43A/43B code 1/11 for the
    three that are in the inventory, and OpenStreetMap
    ``bridge:structure=arch`` with ``bridge:material=concrete`` for
    Weeks, which is a footbridge and therefore is not (see
    ``DECK_GEOMETRY`` in :mod:`coxswain.river.bridges`).

    From the water such a bridge is **a wall with arch-shaped holes in
    it**, and that is how it is built here: the spandrel face is a run of
    vertical strips whose bottom edge follows the intrados, so the
    openings are cut by the geometry rather than modelled as separate
    ribs.  The soffit closes the underside, which is the surface a crew
    actually passes beneath and looks up at.
    """
    start = np.asarray(start, dtype=float)[:2]
    end = np.asarray(end, dtype=float)[:2]
    length = float(np.hypot(*(end - start)))
    if length < 1.0:
        return None
    along = (end - start) / length
    across = np.array([-along[1], along[0]])
    half = 0.5 * float(width)
    springing = max(SPRINGING * level, 0.4)
    crown = max(level - depth, springing + 0.8)
    rise = crown - springing
    spans = max(int(spans), 1)
    arch = length / spans

    def point(distance, side, height):
        xy = start + along * distance + across * (side * half)
        return [float(xy[0]), float(xy[1]), float(height)]

    def intrados(distance):
        """Height of the underside of the arch at ``distance`` along."""
        u = (distance % arch) / arch
        # A semi-ellipse: vertical at the springing, flat at the crown,
        # which is what a segmental concrete arch looks like.  A parabola
        # leans out of the pier and reads as a culvert.
        return springing + rise * float(np.sqrt(max(0.0,
                                                    1.0 - (2.0 * u - 1.0) ** 2)))

    faces, shades = [], []
    stations = np.linspace(0.0, length, spans * samples + 1)
    for a, b in zip(stations[:-1], stations[1:]):
        za, zb = intrados(a), intrados(b)
        for side in (1.0, -1.0):
            want = np.array([across[0] * side, across[1] * side, 0.0])
            _quad(faces, shades,
                  point(a, side, za), point(b, side, zb),
                  point(b, side, level), point(a, side, level),
                  want, colour)
        # The soffit, closing the two faces underneath.
        _quad(faces, shades,
              point(a, 1.0, za), point(b, 1.0, zb),
              point(b, -1.0, zb), point(a, -1.0, za),
              np.array([0.0, 0.0, -1.0]), colour)

    # Piers: the solid between the springing and the water, at each
    # junction between arches and at the abutments.
    for index in range(spans + 1):
        centre = index * arch
        thick = PIER_FRACTION * arch
        a = max(centre - 0.5 * thick, 0.0)
        b = min(centre + 0.5 * thick, length)
        for side in (1.0, -1.0):
            want = np.array([across[0] * side, across[1] * side, 0.0])
            _quad(faces, shades,
                  point(a, side, 0.0), point(b, side, 0.0),
                  point(b, side, springing + 0.15),
                  point(a, side, springing + 0.15), want, pier_colour)
        for at, want in ((a, -along), (b, along)):
            _quad(faces, shades,
                  point(at, 1.0, 0.0), point(at, -1.0, 0.0),
                  point(at, -1.0, springing + 0.15),
                  point(at, 1.0, springing + 0.15),
                  np.array([want[0], want[1], 0.0]), pier_colour)

    # The deck slab and its parapets.
    for side in (1.0, -1.0):
        want = np.array([across[0] * side, across[1] * side, 0.0])
        _quad(faces, shades,
              point(0.0, side, level), point(length, side, level),
              point(length, side, level + 1.0),
              point(0.0, side, level + 1.0), want, (0.80, 0.79, 0.75))
    _quad(faces, shades,
          point(0.0, 1.0, level + 1.0), point(length, 1.0, level + 1.0),
          point(length, -1.0, level + 1.0), point(0.0, -1.0, level + 1.0),
          np.array([0.0, 0.0, 1.0]), (0.34, 0.34, 0.33))

    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.asarray(shades, dtype="f4")
    return MeshPart("bridge", vertices, colours, _face_normals(vertices))


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
            form, width, level, depth, spans, _span, _camber, _src = row
            start = np.asarray(gate.start, dtype=float)
            end = np.asarray(gate.end, dtype=float)
            if form == "arch":
                built = arch_bridge(start, end, width, level, depth, spans)
                if built is not None:
                    parts.append(built)
                continue
            middle = 0.5 * (start + end)
            span = float(np.hypot(*(end - start)))
            along = (end - start) / max(span, 1e-9)
            # Not an arch: a slab across the river, plus a pier at each
            # third.  Eliot is NBI 4/9, a steel deck truss, and the
            # Grand Junction is a steel trestle.
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

    photo = None
    try:
        if race == "charles":
            from ..river.terrain import charles_imagery as _photo
        else:
            from ..river.terrain import seattle_imagery as _photo
        photo = _photo()
    except Exception as error:                # pragma: no cover
        print("   (no imagery, flat colour throughout: %s)" % str(error)[:60])

    mesh = WorldMesh()
    mesh.add(water_plane(course.mean(axis=0)))
    mesh.add(land_mesh(terrain, wet_at, box, step=step, imagery=photo))
    if with_buildings:
        mesh.add(building_walls(structures.polygons, structures.heights,
                                getattr(structures, "base", None), box=box,
                                imagery=photo))
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
    if race != "charles":
        mesh.add(cut_walls())
    mesh.add(bridge_solids(race, scene))
    mesh.add(buoy_solids(scene.buoys))
    if guide:
        mesh.add(line_markers(course))
    return mesh, scene
