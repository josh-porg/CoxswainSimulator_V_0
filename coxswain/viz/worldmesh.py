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
           "building_walls", "skyline_walls", "roof_rise", "photo_colour", "ribbon", "line_markers", "buoy_solids",
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


#: OSM roof shapes that come to a ridge or a point, by the integer code
#: ``extract_structures.py`` writes.  Everything else is drawn flat.
PITCHED = (2, 3, 4, 5, 6, 8, 9)

#: Rise to use when a roof is tagged pitched but carries no height, as a
#: fraction of the footprint's short span, and its cap in metres.
TAGGED_PITCH = 0.26
TAGGED_RISE_CAP = 3.6


def roof_rise(shape, tagged_rise, ring, height, kind=None):
    """How far a roof rises above the wall top, in metres.

    **Only from the tags.**  A previous version inferred a pitch for any
    small, low building on the grounds that Seattle is mostly houses, and
    it was wrong to: it put an invented gable on 87% of the city,
    including on the lidar outlines of flat-roofed sheds and on
    footprints that are really the ground plan of a building whose actual
    shape is described by its ``building:part`` massing.  A guess applied
    that widely stops being a guess and becomes the model.

    The shape of a building comes from its parts -- which is what the
    older PyVista scene used and what
    :func:`tools.extract_structures.raise_to_parts` produces -- and its
    roof comes from ``roof:shape`` where somebody has surveyed one.
    Where neither exists the building is drawn flat, which is honest.
    """
    if not shape or int(shape) not in PITCHED:
        return 0.0
    if tagged_rise and float(tagged_rise) > 0.0:
        return float(tagged_rise)
    span = min(ring[:, 0].max() - ring[:, 0].min(),
               ring[:, 1].max() - ring[:, 1].min())
    return float(min(TAGGED_PITCH * span, TAGGED_RISE_CAP))


def _ridge(ring, top, rise):
    """Ridge line of a gable: ``(point_a, point_b)`` at ``top + rise``.

    A roof that rises to a single point over the centroid is a pyramid,
    and almost no house has one.  The ridge runs along the footprint's
    **long** axis, which is what makes a row of houses read as a row of
    houses rather than a row of tents.
    """
    centre = ring.mean(axis=0)
    offset = ring - centre
    # Principal axis of the footprint.
    _u, _s, vt = np.linalg.svd(offset, full_matrices=False)
    along = vt[0]
    reach = float(np.abs(offset @ along).max())
    # Pull the ends in, so the roof hips slightly instead of ending in a
    # vertical wall of gable at each end.
    reach *= 0.55
    apex = top + rise
    return (np.array([centre[0] - along[0] * reach,
                      centre[1] - along[1] * reach, apex]),
            np.array([centre[0] + along[0] * reach,
                      centre[1] + along[1] * reach, apex]),
            along, centre)


def building_walls(polygons, heights, bases=None, box=None,
                   colour=(0.42, 0.41, 0.40), imagery=None,
                   near=None, min_height: float = 2.0, roofs=None,
                   kinds=None, ground_at=None,
                   limit: int = 12000) -> Optional[MeshPart]:
    """Extruded footprints -- **walls only**, coloured from the orthophoto.

    From 0.55 m off the water you never see a roof, so the top faces are
    not emitted: it halves the triangle count and changes nothing you can
    see from the seat.

    ``imagery`` is an :class:`~coxswain.river.terrain.Imagery`; when it is
    given each building takes its own colour from the photograph (see
    :func:`photo_colour`), which is what turns the Harvard houses from
    grey blocks into the landmarks a crew steers the Weeks turn by.

    ``near`` is the course.  It decides **which** buildings survive the
    cap, and that matters more than the cap itself: the limit used to be
    4,000 taken in file order, which on the Charles silently dropped
    5,631 of 9,631 and kept an arbitrary set.  What limits a coxswain's
    line of sight on that river is the bank -- trees and the buildings
    behind them -- so the ones nearest the water are the ones that have
    to be there.
    """
    order = range(len(polygons))
    if near is not None and len(polygons) > limit:
        centres = np.array([np.asarray(p, dtype=float).mean(axis=0)
                            if len(p) else (1e9, 1e9) for p in polygons])
        from scipy.spatial import cKDTree

        gap = cKDTree(np.asarray(near, dtype=float)[:, :2]).query(centres)[0]
        order = np.argsort(gap)

    walls, tints, kept = [], [], 0
    base_colour = np.asarray(colour, dtype=float)
    for index in order:
        polygon = polygons[index]
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
        if top < min_height:
            continue
        # **Stand it on the ground, not on the water.**
        #
        # Every building was extruded from z = 0, which is the lake
        # surface, so a house on Capitol Hill at seventy metres was
        # drawn from the waterline and buried to its roof in the hill.
        # The older PyVista scene had this right and it is the whole of
        # what "simple extrusion" was missing: the roof goes at the
        # ground under the middle plus the surveyed height, and the base
        # at the **lowest** ground anywhere under the footprint, so a
        # building on a slope is cut into the hill rather than left
        # hanging off its downhill corner.
        floor = 0.0
        if ground_at is not None:
            try:
                corners = np.asarray(ground_at(ring[:, 0], ring[:, 1]),
                                     dtype=float)
                middle = float(np.asarray(
                    ground_at(*ring.mean(axis=0)), dtype=float).ravel()[0])
                floor = float(np.min(corners))
                top = middle + top
            except Exception:
                floor = 0.0
        low = floor + (float(bases[index]) if bases is not None else 0.0)
        if low >= top - 0.3:
            # An inverted part -- ``min_height`` above its own ``height``.
            #
            # Dropping the base to zero, which is what this used to do,
            # is the worst possible repair: the Space Needle has a 42 m
            # wide halo tagged base 167 / top 162, and grounding it
            # extrudes a 42 m column from the water to 162 m.  That is
            # the convex cylinder, put there by the fix for it.  Keep the
            # piece where it belongs and make it thin instead.
            low = max(top - 1.0, floor)
        nxt = np.roll(ring, -1, axis=0)
        tint = photo_colour(imagery, ring, base_colour)
        for start, end in zip(ring, nxt):
            walls.append([[start[0], start[1], low], [end[0], end[1], low],
                          [end[0], end[1], top]])
            walls.append([[start[0], start[1], low], [end[0], end[1], top],
                          [start[0], start[1], top]])
            tints.extend([tint] * 6)

        # A roof, because a building without one is an open box.
        #
        # The walls used to be the whole model, on the argument that from
        # 0.55 m off the water you never see a roof.  That is true of a
        # building on the far bank and false of one up a hillside, which
        # is most of Seattle: looking up at an open box you see straight
        # through the near wall into the inside of the far one.  Flat
        # unless the tags say otherwise.
        shape = (int(roofs[0][index])
                 if roofs is not None and roofs[0] is not None else 0)
        tagged = (float(roofs[1][index])
                  if roofs is not None and roofs[1] is not None else 0.0)
        this_kind = None if kinds is None else kinds[index]
        pitch = roof_rise(shape, tagged, ring, top - low, this_kind)
        roof_tint = tuple(min(1.0, c * 1.06) for c in np.atleast_1d(tint))
        if pitch > 0.05:
            ridge_a, ridge_b, along, centre = _ridge(ring, top, pitch)
            for start, end in zip(ring, nxt):
                middle = 0.5 * (start + end)
                # Each eave meets the ridge at its own nearest point, so
                # the long sides give two slopes and the ends hip in.
                t = float(np.dot(middle - centre, along))
                reach = float(np.dot(ridge_b[:2] - centre, along))
                t = max(-abs(reach), min(abs(reach), t))
                onto = [centre[0] + along[0] * t, centre[1] + along[1] * t,
                        top + pitch]
                _quad(walls, tints, [start[0], start[1], top],
                      [end[0], end[1], top], onto, onto,
                      np.array([0.0, 0.0, 1.0]), roof_tint)
        else:
            centre = ring.mean(axis=0)
            apex = [centre[0], centre[1], top]
            for start, end in zip(ring, nxt):
                _quad(walls, tints, [start[0], start[1], top],
                      [end[0], end[1], top], apex, apex,
                      np.array([0.0, 0.0, 1.0]), roof_tint)
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
    # Turn any inward-facing triangle around.
    #
    # All twelve of these were wound inward, which back-face culling
    # turns into "the near side of the box is missing and you are looking
    # at the inside of the far side" -- every buoy, every marker post,
    # every pier and every tree trunk in the scene.  Rather than hand-fix
    # a winding table that was already wrong once, the test is made
    # explicit: a face of a convex box must point away from its centre.
    tri = vertices.reshape(-1, 3, 3)
    centre = np.array([cx, cy, cz], dtype="f4")
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - centre) < 0.0
    tri[inward] = tri[inward][:, ::-1]
    vertices = tri.reshape(-1, 3)
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


#: Trees closer to the course than this get the full model; beyond it,
#: a crossed pair of quads.
#:
#: **250 m, not 90.**  At ninety metres this looked right in principle
#: and came out at 41 solid trees in 80,000 on Lake Union -- effectively
#: everything an impostor -- because that shoreline is built up and its
#: trees sit well back from the water.  A crown is still clearly a crown
#: at a couple of hundred metres from a seat half a metre off the water,
#: and that is the distance that decides this, not a round number.
SOLID_WITHIN = 250.0

#: However close they are, no more than this many get the full model, so
#: a wooded bank cannot blow the triangle budget on its own.
SOLID_BUDGET = 14000

#: Crown colours by growth form, in the order ``TreeStand.FORMS`` uses.
CROWN = ((0.24, 0.34, 0.20), (0.16, 0.26, 0.18), (0.20, 0.31, 0.20),
         (0.30, 0.40, 0.24))
TRUNK = (0.26, 0.20, 0.15)


def tree_solids(stand, box, limit: int = 80000, near=None,
                min_height: float = 3.0) -> Optional[MeshPart]:
    """Trees as a trunk and a low-poly crown.

    The bank of a river is trees, and leaving them out is why the first
    seat view looked like a reservoir.  There are 24,392 of them on the
    Charles and 742,517 on Lake Union, so they are ranked by height and
    by distance from the course, and capped.

    **The cap is 80,000, and it is the impostors that pay for it.**  A
    crown is eight triangles and a trunk twelve, so every tree on the
    Charles reach comes to 478,860 triangles -- and the reach only *has*
    24,392, so there they all fit as full models.

    Lake Union has 742,517, and there the hills read as bare with a cap
    that only covers the near bank.  Measured on that course the mix
    comes out at **4.0 triangles a tree**, because almost everything
    beyond the built-up shoreline is further than
    :data:`SOLID_WITHIN` from the water: 80,000 trees cost 320,656
    triangles, which is what 16,000 solid ones used to.

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
    gap = None
    if near is not None:
        from scipy.spatial import cKDTree

        gap = cKDTree(np.asarray(near, dtype=float)).query(points[index])[0]
        if len(index) > limit:
            # Tall and close beats tall and far, the same rule the 3-D
            # scene uses for which trees are worth drawing at all.
            order = np.argsort(-heights[index] / np.maximum(gap, 5.0))
            index, gap = index[order], gap[order]
    index = index[:limit]
    if gap is not None:
        gap = gap[:limit]

    # Two levels of detail, split by distance from the **course**, not
    # from the camera, because the mesh is built once and uploaded once.
    # A tree near the line is a trunk and an eight-facet crown; a far one
    # is a crossed pair of quads -- four triangles against twenty.  That
    # is what pays for the draw distance on Lake Union, where there are
    # 742,517 trees and the hills read as bare without them.
    if gap is None:
        solid = np.ones(len(index), dtype=bool)
    else:
        solid = gap <= SOLID_WITHIN
        if solid.sum() > SOLID_BUDGET:
            # Keep the nearest, drop the rest to impostors.
            cut = np.sort(gap[solid])[SOLID_BUDGET - 1]
            solid &= gap <= cut

    faces, shades = [], []
    for slot, i in enumerate(index):
        x, y = points[i]
        height = float(heights[i])
        form = int(forms[i]) if i < len(forms) else 0
        crown_colour = CROWN[form % len(CROWN)]
        if not solid[slot]:
            # An impostor, and it has to have a **silhouette**.
            #
            # The first version was two crossed rectangles, which from
            # the water looked exactly like what it was: green slabs
            # standing on a hillside.  What makes a distant tree read as
            # a tree is its outline -- a conifer is a triangle, a
            # broadleaf a rounded mass on a stem -- so the impostor is
            # cut to that shape instead.  Same two triangles a plane, no
            # more cost, and it stops being a billboard.
            spread = (0.17 if form == 1 else 0.30) * height
            # Both forms reach the ground.  A crown drawn from 42% of the
            # height up, with no stem under it, hangs in the air -- a row
            # of green lozenges floating over the bank, which is what the
            # first attempt looked like.  Tapering to a point at the foot
            # implies the trunk for nothing.
            foot = 0.0
            for dx, dy in ((spread, 0.0), (0.0, spread)):
                if form == 1:
                    # Conifer: a spire, wide at the foot and pointed.
                    faces.append(np.asarray(
                        [[x - dx, y - dy, foot], [x + dx, y + dy, foot],
                         [x, y, height]], dtype="f4"))
                    shades.append(np.tile(
                        np.asarray(crown_colour, dtype="f4"), (3, 1)))
                else:
                    # Broadleaf: a diamond crown carried on a stem, so
                    # the outline narrows top and bottom.
                    waist = 0.62 * height
                    faces.append(np.asarray(
                        [[x, y, foot], [x + dx, y + dy, waist],
                         [x, y, height]], dtype="f4"))
                    faces.append(np.asarray(
                        [[x, y, foot], [x, y, height],
                         [x - dx, y - dy, waist]], dtype="f4"))
                    shades.append(np.tile(
                        np.asarray(crown_colour, dtype="f4"), (6, 1)))
            continue
        # Trunk: a square post up to the crown.
        stem = 0.40 * height
        radius = max(0.018 * height, 0.05)
        trunk = box_solid((x, y, 0.5 * stem), (radius, radius, 0.5 * stem),
                          colour=TRUNK)
        faces.append(trunk.vertices)
        shades.append(trunk.colours)
        # Crown: a lantern of two rings, not an octahedron.
        #
        # Four rim points give a crown with four corners, and from the
        # water that reads as a cut gem rather than a tree -- which is
        # exactly what it looked like.  Ten points around and two rings
        # up is forty triangles instead of eight, and the silhouette
        # stops having corners in it.  Conifers keep a single ring and a
        # point, because a conifer really is a cone.
        top = height
        apex = [x, y, top]
        base = [x, y, stem]
        crown = []
        if form == 1:
            spread = 0.16 * height
            around = 9
            rim = [[x + spread * np.cos(t), y + spread * np.sin(t),
                    stem + 0.18 * (top - stem)]
                   for t in np.linspace(0.0, 2.0 * np.pi, around,
                                        endpoint=False)]
            for a, b in zip(rim, rim[1:] + rim[:1]):
                crown.append([apex, a, b])
                crown.append([base, b, a])
        else:
            spread = 0.30 * height
            around = 10
            angles = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
            # Two rings at 35% and 70% of the crown, radii following a
            # circle, so the outline is round in every direction.
            rings = []
            for fraction, radius in ((0.34, 0.86), (0.68, 0.62)):
                level = stem + fraction * (top - stem)
                rings.append([[x + spread * radius * np.cos(t),
                               y + spread * radius * np.sin(t), level]
                              for t in angles])
            lower, upper = rings
            for a, b, c, d in zip(lower, lower[1:] + lower[:1],
                                  upper, upper[1:] + upper[:1]):
                crown.append([base, b, a])          # underside
                crown.append([a, b, d])             # waist
                crown.append([a, d, c])
                crown.append([c, d, apex])          # shoulder
        faces.append(np.asarray(crown, dtype="f4").reshape(-1, 3))
        shades.append(np.tile(np.asarray(crown_colour, dtype="f4"),
                              (len(crown) * 3, 1)))
    if not faces:
        return None
    faces = [np.asarray(f, dtype="f4").reshape(-1, 3) for f in faces]
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

#: How far out to look for skyline buildings, m, and how tall one must
#: be to be worth drawing at that range.
SKYLINE_REACH = 6500.0
SKYLINE_HEIGHT = 38.0


def skyline_walls(structures, course, box, imagery=None, bases=None,
                  roofs=None, ground_at=None):
    """Distant towers, drawn because a crew can see them.

    Only buildings **outside** the near box are considered, so nothing is
    drawn twice, and only those over :data:`SKYLINE_HEIGHT`, because at
    four kilometres anything shorter is a smudge on the bank.  They keep
    their photographed colour: the haze that makes distance read comes
    from the fog term in the shader, not from painting them grey here.
    """
    polygons = list(structures.polygons)
    heights = np.asarray(structures.heights, dtype=float)
    if not len(polygons):
        return None
    centres = np.array([np.asarray(p, dtype=float).mean(axis=0)
                        if len(p) else (1e9, 1e9) for p in polygons])
    outside = ~((centres[:, 0] >= box[0]) & (centres[:, 0] <= box[2])
                & (centres[:, 1] >= box[1]) & (centres[:, 1] <= box[3]))
    from scipy.spatial import cKDTree

    gap = cKDTree(np.asarray(course, dtype=float)[:, :2]).query(centres)[0]
    # Tall enough to see, **or** standing well off the ground: a piece
    # from 30 to 37 m is nothing on its own but is the waist of a
    # landmark, and dropping it leaves a gap in the middle of one.
    lifted = (np.zeros(len(heights), dtype=bool) if bases is None
              else np.asarray(bases) > 15.0)
    keep = np.nonzero(outside & (gap < SKYLINE_REACH)
                      & ((heights >= SKYLINE_HEIGHT) | lifted))[0]
    if not len(keep):
        return None
    # Massing and roofs come across, and this is not optional.
    #
    # Passing ``None`` here is what kept the Space Needle a convex
    # cylinder through three attempts at fixing it.  Its massing was
    # correct in the data all along -- saucer floating at 152 m, three
    # legs from the ground -- but it stands 2.1 km from Lake Union, which
    # puts it outside the near box and into this pass, where the bases
    # were being thrown away.  The tallest, most distinctive buildings
    # are exactly the ones a skyline is made of, so this is the pass that
    # can least afford to lose their shape.
    part = building_walls(
        [polygons[i] for i in keep], heights[keep],
        None if bases is None else np.asarray(bases)[keep],
        box=None, imagery=imagery, limit=4000, ground_at=ground_at,
        roofs=None if roofs is None else
        tuple(None if r is None else np.asarray(r)[keep] for r in roofs))
    if part is None:
        return None
    print("   skyline: %d buildings over %.0f m, out to %.1f km"
          % (len(keep), SKYLINE_HEIGHT, gap[keep].max() / 1000.0))
    return MeshPart("skyline", part.vertices, part.colours, part.normals)


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
        def ground_at(east, north):
            return np.maximum(terrain.height_above_water(east, north), 0.0)

        mesh.add(building_walls(structures.polygons, structures.heights,
                                getattr(structures, "base", None), box=box,
                                imagery=photo, near=course,
                                kinds=getattr(structures, "kind", None),
                                ground_at=ground_at,
                                roofs=(getattr(structures, "roof_shape", None),
                                       getattr(structures, "roof_height",
                                               None))))
        # The skyline.
        #
        # Rowing south down Lake Union you are looking straight at
        # downtown Seattle, three to five kilometres away, and it is the
        # single biggest thing in the view -- but it sits far outside the
        # 900 m working box, so the seat view had an empty horizon where
        # a crew sees towers.  A second pass takes only buildings tall
        # enough to subtend a real angle at that range.
        mesh.add(skyline_walls(
            structures, course, box, photo,
            bases=getattr(structures, "base", None),
            ground_at=ground_at,
            roofs=(getattr(structures, "roof_shape", None),
                   getattr(structures, "roof_height", None))))
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
