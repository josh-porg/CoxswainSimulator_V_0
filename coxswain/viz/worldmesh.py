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
           "building_walls", "skyline_walls", "roof_rise", "outlines_over_massing", "photo_colour", "ribbon", "line_markers", "buoy_solids",
           "hull_solid", "tree_solids", "arch_bridge", "truss_bridge", "cut_walls",
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


#: How far the gunwale stands above the deck, m.
GUNWALE = 0.055

#: How far below the waterline the bank is carried, m.  Enough to be
#: under the deepest wave trough and any drawdown beside the hull, and
#: not so much that the whole bed comes with it.
SHELF = 4.0


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

    # The bank used to stop dead at the waterline: every cell touching
    # water was dropped, so the ground ended in a vertical cut at z = 0
    # with nothing under it.  The water is a *wavy* surface, so every
    # trough that dipped below zero opened a gap you could see in
    # through -- under the terrain, out the far side of the world.
    #
    # So the shoreline gets an apron.  Cells with at least one dry
    # corner are kept, and their **wet** corners are dropped to -SHELF,
    # which carries the ground down under the water by one cell all the
    # way round.  Entirely-wet cells are still dropped.
    #
    # This is an apron and not a river bed, deliberately.  The elevation
    # model has no bathymetry in it: over water the lidar returns the
    # *surface*, a dead flat -0.39 m across the whole width of the
    # Charles, which is the thing the original version of this was right
    # to refuse to draw.  Real soundings exist -- data/charles_isobaths.csv
    # -- and if the bed is ever wanted for its own sake that is where it
    # has to come from, not from here.
    height = np.where(wet, -SHELF, height)
    a = (slice(0, -1), slice(0, -1))
    b = (slice(0, -1), slice(1, None))
    c = (slice(1, None), slice(1, None))
    d = (slice(1, None), slice(0, -1))
    keep = ~(wet[a] & wet[b] & wet[c] & wet[d])
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
    """One big quad.  The horizon is the thing a coxswain steers on.

    ``level`` wants to be **below the deepest wave trough**, not at zero.
    The near field is a displaced grid oscillating about z = 0, and with
    this quad at exactly zero every trough was hidden behind it while the
    crests stood through -- water that bumped up and never down.  That is
    not z-fighting, which flickers; it is one surface occluding another.
    Dropped by a fraction of a wave height it disappears under the chop,
    and at the far edge of the patch the step it leaves subtends well
    under a pixel.
    """
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


#: How far a part's base may sit above its own top before the record is
#: treated as broken rather than merely rounded, m.
INVERTED_LIMIT = 12.0

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


def outlines_over_massing(polygons, heights, bases):
    """Ground plans of buildings whose real shape is their parts.

    ``extract_structures`` drops an outline once **two or more** parts
    stand inside it, and draws the parts instead.  An outline with only
    one is kept and merely *raised* to that part's height -- which for
    the Space Needle leaves its 42 m ground plan extruded from the water
    to 162 m, a cylinder standing over the twelve pieces that describe
    the actual tower.  A shaft that starts slender and turns into a
    massive cylinder is that outline, seen from a boat.

    So: a grounded outline that contains a lifted part and reaches the
    same height as it is not a building, it is the footprint of one that
    has already been modelled properly.  Returns the indices to skip.
    """
    bases = (np.zeros(len(polygons)) if bases is None
             else np.asarray(bases, dtype=float))
    heights = np.asarray(heights, dtype=float)
    lifted = np.nonzero(bases > 0.5)[0]
    if not len(lifted):
        return set()
    from matplotlib.path import Path

    centres = np.array([np.asarray(polygons[i], dtype=float).mean(axis=0)
                        for i in lifted])
    tops = heights[lifted]
    drop = set()
    for index, polygon in enumerate(polygons):
        if bases[index] > 0.5 or heights[index] < 12.0:
            continue
        ring = np.asarray(polygon, dtype=float)
        if len(ring) < 3:
            continue
        low, high = ring.min(axis=0), ring.max(axis=0)
        near = np.nonzero((centres[:, 0] >= low[0]) & (centres[:, 0] <= high[0])
                          & (centres[:, 1] >= low[1])
                          & (centres[:, 1] <= high[1]))[0]
        if not len(near):
            continue
        inside = near[Path(ring).contains_points(centres[near])]
        if not len(inside):
            continue
        # Raised to a part's height: within a tenth of one of them.
        if np.any(np.abs(tops[inside] - heights[index])
                  < 0.10 * max(heights[index], 1.0)):
            drop.add(index)
    return drop


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
    skip = outlines_over_massing(polygons, heights, bases)
    if skip:
        print("   %d ground plans dropped in favour of their massing"
              % len(skip))
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
        if index in skip:
            continue
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
            # An inverted part -- ``min_height`` at or above its own
            # ``height`` -- and the right repair depends on how badly.
            #
            # Grounding it, which is what this did first, is the worst
            # option: the Space Needle has a 42 m wide halo tagged base
            # 167 / top 162, and grounding that extrudes a 42 m column
            # from the water to 162 m.  That was the convex cylinder, put
            # there by the fix for the convex cylinder.
            #
            # When the two disagree by a few metres they are the same
            # surface rounded differently, so the piece is drawn across
            # their span.  When they disagree by a hundred and forty --
            # 43 of Seattle's 100 do -- one of the numbers is simply
            # wrong and there is nothing to draw that is not a guess, so
            # nothing is drawn.  A gap is honest; a tower in the wrong
            # place is not.
            if low - top > INVERTED_LIMIT:
                continue
            low, top = min(low, top) - 0.5, max(low, top)
            low = max(low, floor)
        # Drop repeated vertices before anything is built from them.
        # An OSM ring often closes on its own first point, and a few
        # carry duplicates mid-way; each one produces a zero-area wall
        # quad and two more zero-area triangles in the caps.
        keep_pt = np.hypot(*(ring - np.roll(ring, -1, axis=0)).T) > 1e-6
        if keep_pt.sum() >= 3:
            ring = ring[keep_pt]
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
                _tri(walls, tints, [start[0], start[1], top],
                     [end[0], end[1], top], onto,
                     np.array([0.0, 0.0, 1.0]), roof_tint)
        else:
            centre = ring.mean(axis=0)
            apex = [centre[0], centre[1], top]
            for start, end in zip(ring, nxt):
                _tri(walls, tints, [start[0], start[1], top],
                     [end[0], end[1], top], apex,
                     np.array([0.0, 0.0, 1.0]), roof_tint)

        # A floor under anything that floats.
        #
        # Only the top was ever capped, on the reasoning that you cannot
        # see under a building.  You can see under a *part*: the Space
        # Needle's saucer stands at 152 m over open air, and from a boat
        # you are looking up at its underside.  With no floor the near
        # face is culled and you see the inside of the far one -- the
        # same hole the coxswain's cockpit had, three hundred feet up.
        if low > floor + 0.5:
            centre = ring.mean(axis=0)
            hub = [centre[0], centre[1], low]
            for start, end in zip(ring, nxt):
                _tri(walls, tints, [start[0], start[1], low],
                     [end[0], end[1], low], hub,
                     np.array([0.0, 0.0, -1.0]), tuple(
                         0.7 * c for c in np.atleast_1d(tint)))
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


def viewpoint(boat):
    """``(seat, eye_height, facing)`` for whoever is looking out of this boat.

    A coxed boat has a coxswain and the rig says where they sit.  A
    double or a single has nobody, and ``rig.coxswain_position`` is
    ``None`` -- which crashed the trainer outright the first time
    anybody picked one out of the menu, because both the hull builder
    and the camera indexed it without asking.

    So a coxless boat seats you in the stroke's place, **facing the
    stern**, which is where a sculler looks: at the water they have
    already been over.  ``facing`` is +1 toward the bow and -1 astern,
    and it is why steering one of these is a matter of looking over a
    shoulder.
    """
    rig = boat.rig
    if rig.coxswain_position is not None:
        return (np.asarray(rig.coxswain_position, dtype=float),
                float(rig.coxswain_eye_height), 1.0)

    stern_most = min(float(seat.station_x) for seat in rig.seats)
    eye = 0.62
    crew = getattr(boat, "crew", None)
    if crew:
        # Anchor to the rower's own head rather than to the coxswain's
        # eye height, which is measured from a different seat and comes
        # out above the top of their skull.
        head = min(crew, key=lambda m: float(m.rower.station.x_ankle))
        eye = float(head.rower.skeleton(0.0)["head"][2]) + CREW_LIFT - 0.03
    return np.array([stern_most, 0.0, 0.0]), eye, -1.0


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

    # The sides run past the deck.  A shell's gunwale is a rail standing
    # proud of the washboard, not a flush edge, and from the seat it is
    # the line you see the water against -- the one piece of the boat
    # that is always in the frame with the horizon behind it.
    rail = float(deck) + GUNWALE
    faces, shades = [], []
    nxt = np.roll(ring, -1, axis=0)
    for a, b in zip(ring, nxt):
        low_a = [a[0], a[1], 0.0]
        low_b = [b[0], b[1], 0.0]
        top_a = [a[0], a[1], rail]
        top_b = [b[0], b[1], rail]
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
    seat_x = float(viewpoint(boat)[0][0]) + float(cockpit)
    bow, stern = float(ring[:, 0].max()), float(ring[:, 0].min())

    # Each side of the outline as a curve of half-breadth against
    # station, so the deck can be cut to the hull's actual width at any
    # x.  The previous version took the widest point within 1.2 m of a
    # station, which at the stem is the beam a metre back -- so the
    # deck ended in a blunt square and the bow, which on a shell is a
    # knife, read as a barge.  The tip vertices sit at zero beam and are
    # in both curves, so the interpolation closes to a point there.
    _port = ring[ring[:, 1] >= -1e-6]
    _starboard = ring[ring[:, 1] <= 1e-6]
    _port = _port[np.argsort(_port[:, 0])]
    _starboard = _starboard[np.argsort(_starboard[:, 0])]

    def edges_at(x):
        """``(y_starboard, y_port)`` of the hull at station ``x``."""
        return (float(np.interp(x, _starboard[:, 0], _starboard[:, 1])),
                float(np.interp(x, _port[:, 0], _port[:, 1])))

    def graded(x0, x1, count):
        """Stations from ``x0`` to ``x1`` crowded toward both ends.

        Cosine spacing: the same clustering a wing or a hull section is
        panelled with, because the curvature is at the ends.  A shell
        is straight-sided for most of its length and turns through its
        whole entry in the last metre or two, and even stations put one
        vertex in that metre and a dozen along the flat.
        """
        theta = np.linspace(0.0, np.pi, int(count))
        return 0.5 * (x0 + x1) + 0.5 * (x0 - x1) * np.cos(theta)

    # Resample the outline itself on graded stations, so the sides get
    # the same refinement as the deck: the convex hull hands back one
    # vertex at the stem, and the hull walls between it and the next
    # were a single long facet across the whole entry.
    _bow_x, _stern_x = float(ring[:, 0].max()), float(ring[:, 0].min())
    _stations = graded(_stern_x, _bow_x, 60)
    _s_side = np.column_stack([_stations, [edges_at(x)[0] for x in _stations]])
    _p_side = np.column_stack([_stations, [edges_at(x)[1] for x in _stations]])
    # Starboard forward, port aft: counter-clockwise from above, which
    # is the winding the side walls below are built for.
    ring = np.vstack([_s_side, _p_side[::-1][1:-1]])

    def strip(x0, x1, height, shade):
        stations = graded(x0, x1, 22)
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

    # Where the boat is open, and where it is decked over.
    #
    # This used to be one cut at the coxswain's seat: deck ahead of it,
    # floor behind.  In a bow-loader that is right, because the cox is
    # in the bow and everything forward of them is foredeck.  In a
    # stern-coxed eight it puts a deck at gunwale height over the whole
    # length of the crew, and the rowers stand up through it -- eight
    # people sunk to the chest in a lid.
    #
    # A shell is open over its crew and decked at both ends, so that is
    # what gets built: a cockpit spanning the seats (and the coxswain,
    # who sits in it), with the interior floor set below the gunwale
    # rather than down at the keel, and short decks fore and aft.
    seats_x = [float(seat.station_x) for seat in boat.rig.seats]
    cockpit_stern = min(seats_x) - 0.90
    cockpit_bow = max(seats_x) + 0.90
    if boat.rig.coxswain_position is not None:
        cox_x = float(boat.rig.coxswain_position[0])
        cockpit_stern = min(cockpit_stern, cox_x - 0.55)
        # A bow-loader's cockpit reaches just past their head, which is
        # why only a couple of feet ahead of the face is ever open.
        cockpit_bow = max(cockpit_bow, cox_x + float(cockpit))
    cockpit_stern = max(cockpit_stern, stern)
    cockpit_bow = min(cockpit_bow, bow)

    interior = min(float(deck) - 0.10, max(float(floor), 0.12))
    strip(cockpit_bow, bow, deck, deck_colour)
    strip(stern, cockpit_stern, deck, deck_colour)
    strip(cockpit_stern, cockpit_bow, interior, (0.22, 0.23, 0.24))

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
def _sphere(centre, radius, colour, rings: int = 5, segments: int = 8):
    """A low-polygon sphere.  Heads are round; nothing else here is."""
    centre = np.asarray(centre, dtype=float)
    lats = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(rings) + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)

    def point(lat, lon):
        return centre + radius * np.array([np.cos(lat) * np.cos(lon),
                                           np.cos(lat) * np.sin(lon),
                                           np.sin(lat)])

    faces = []
    for i in range(int(rings)):
        for j in range(int(segments)):
            k = (j + 1) % int(segments)
            a = point(lats[i], lons[j])
            b = point(lats[i], lons[k])
            c = point(lats[i + 1], lons[k])
            d = point(lats[i + 1], lons[j])
            faces += [[a, b, c], [a, c, d]]
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    tri = vertices.reshape(-1, 3, 3)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - centre) < 0.0
    tri[inward] = tri[inward][:, ::-1]
    vertices = tri.reshape(-1, 3)
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    return MeshPart("sphere", vertices, colours, _face_normals(vertices))


def _tube(a, b, radius, colour, sides: int = 8):
    """A round prism between two points.

    :func:`_strut` is square, which is right for a truss member and
    wrong for an arm -- at the distance a coxswain sits from the stroke,
    a square limb reads as a plank and catches the light in flat facets.
    Eight sides is enough to lose the corners at that range without
    doubling the crew's triangle count twice over.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    axis = axis / length
    guide = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ guide)) > 0.95:
        guide = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, guide)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(axis, u)

    angles = np.linspace(0.0, 2.0 * np.pi, int(sides), endpoint=False)
    ring = [np.cos(angle) * u * radius + np.sin(angle) * v * radius
            for angle in angles]
    lo = [a + offset for offset in ring]
    hi = [b + offset for offset in ring]
    faces = []
    for i in range(int(sides)):
        j = (i + 1) % int(sides)
        faces += [[lo[i], lo[j], hi[j]], [lo[i], hi[j], hi[i]]]
    for i in range(1, int(sides) - 1):
        faces.append([hi[0], hi[i], hi[i + 1]])
        faces.append([lo[0], lo[i + 1], lo[i]])
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    tri = vertices.reshape(-1, 3, 3)
    centre = 0.5 * (a + b)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - centre) < 0.0
    tri[inward] = tri[inward][:, ::-1]
    vertices = tri.reshape(-1, 3)
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    return MeshPart("tube", vertices, colours, _face_normals(vertices))


#: How far the drawn crew is raised off the kinematic model, m.
#:
#: The model puts the seat plane 0.104 m above the waterline.  A real
#: shell's deck sits three or four inches below the gunwale and the seat
#: an inch or two above the deck, which with this hull's 0.30 m gunwale
#: puts a rower's backside at about 0.25 -- so drawn literally the crew
#: were sunk into the boat up to the waist, rowing from inside it.
#:
#: This is a **render offset only**; nothing in the dynamics reads it.
#: That is a compromise rather than a fix: if the seat really is 0.15 m
#: low then the crew's mass is 0.15 m low too, which flatters the roll
#: inertia and the trim.  Correcting it properly means moving the
#: station in the kinematics and re-checking everything calibrated
#: against it, which is not a thing to do quietly at the same time as
#: making the picture look right.
CREW_LIFT = 0.146

#: Kit and skin.  Muted on purpose: in a stern-coxed boat the crew fill
#: the middle of the frame, and anything saturated there pulls the eye
#: off the bank, which is the thing actually being steered by.
KIT = (0.17, 0.21, 0.28)
SKIN = (0.74, 0.61, 0.53)

#: Girth of each bone, m, and whether it reads as kit or as skin.  A
#: bare skeleton is a wire diagram; these are what make it a body.
BONE_STYLE = (
    (("ankle", "knee"), 0.055, KIT),        # shank
    (("knee", "hip"), 0.078, KIT),          # thigh
    (("hip", "shoulder"), 0.100, KIT),      # trunk, one side each
    (("shoulder", "shoulder"), 0.090, KIT),  # across the chest
    (("neck", "head"), 0.095, SKIN),
    (("shoulder", "elbow"), 0.045, KIT),
    (("elbow", "hand"), 0.038, SKIN),
)


def _bone_style(start: str, end: str):
    """Radius and colour for one bone, by the joints it runs between."""
    a = start.split("_")[0]
    b = end.split("_")[0]
    for (first, second), radius, colour in BONE_STYLE:
        if (a, b) == (first, second) or (a, b) == (second, first):
            return radius, colour
    return 0.05, KIT


#: Oar colours: a pale loom, and a blade dark enough to read against
#: both the water and the sky as it comes round.
LOOM = (0.86, 0.87, 0.83)
BLADE = (0.20, 0.24, 0.30)


def oar_pose(boat, t, water_z: float = 0.0,
             bury: float = 0.09, clear: float = 0.10):
    """``(handle, lock, blade, lift)`` for every oar, in the hull frame.

    The oar model is **planar**: :func:`handle_position` and
    :func:`blade_position` both return points at oarlock height, because
    the dynamics only ever needed the horizontal sweep and blade
    immersion is carried as a factor rather than as geometry.  Drawn
    literally that puts the blades 0.38 m above the water for the whole
    stroke, skimming along and never entering it -- and it is why the
    oars were invisible from the seat before: a horizontal ribbon seen
    from a horizontal eye is a line.

    So the vertical is reconstructed here, for the picture only, from
    the one thing that fixes it: the oar is rigid and pivots about the
    lock.  Put the blade where it belongs -- buried on the drive, clear
    on the recovery -- and the handle height follows from the lever
    ratio.  Nothing in the dynamics reads this.
    """
    from ..crew.oarlock import blade_position, handle_position

    drive = bool(boat.timing.is_drive(t))
    target = water_z - float(bury) if drive else water_z + float(clear)
    poses = []
    for seat in boat.rig.seats:
        for lock in seat.oarlocks:
            handle = np.asarray(handle_position(t, boat.timing, lock,
                                                boat.oar_sweep), dtype=float)
            pivot = np.asarray(lock.position, dtype=float)
            blade = np.asarray(blade_position(t, boat.timing, lock,
                                              boat.oar_sweep), dtype=float)
            inboard = float(np.linalg.norm(handle[:2] - pivot[:2]))
            outboard = max(float(np.linalg.norm(blade[:2] - pivot[:2])), 1e-6)
            drop = pivot[2] - target
            blade = blade.copy()
            blade[2] = target
            lift = drop * inboard / outboard
            handle = handle.copy()
            handle[2] = pivot[2] + lift
            poses.append((handle, pivot, blade, lift, drive))
    return poses


#: Where in the recovery the hands cross the knees and the roll-up
#: starts, as a fraction of the recovery.  Before this the blade is flat
#: on the feather; after it, it rolls square by the catch.
SQUARE_UP_AT = 0.55

#: How much of the *cycle* the feather takes.  It is a flick of the
#: inside wrist at the finish and it is fast -- much faster than the
#: roll up, which is a progressive turn of the handle through the last
#: of the recovery.  Drawing both at the same speed loses the asymmetry,
#: and that asymmetry is one of the things a coxswain watches for.
FEATHER_SPAN = 0.045


def blade_roll(boat, t) -> float:
    """Blade rotation: 0 squared, 1 fully feathered.

    Squared through the drive, because the face has to be square to the
    water to move any.  Feathered off the finish with a flick, held flat
    through the recovery so it runs clear of the water, then rolled back
    square through the last of the recovery, starting where the hands
    cross the knees and finishing at the catch.
    """
    period = float(boat.timing.period)
    if period <= 0.0:
        return 0.0
    phase = (float(t) % period) / period
    drive = float(boat.timing.drive_fraction)
    if phase < drive:
        return 0.0                                   # squared, pulling
    through = (phase - drive) / max(1.0 - drive, 1e-9)
    feather = FEATHER_SPAN / max(1.0 - drive, 1e-9)
    if through < feather:
        return through / feather                     # the flick
    if through < SQUARE_UP_AT:
        return 1.0                                   # flat, running
    # The roll up: smooth, and finished by the catch.
    left = (through - SQUARE_UP_AT) / max(1.0 - SQUARE_UP_AT, 1e-9)
    return float(0.5 * (1.0 + np.cos(np.pi * min(left, 1.0))))


def _blade_surface(root, tip, edge, normal, colour=BLADE):
    """One blade, as a curved hatchet rather than a flat rectangle.

    A modern sweep blade is asymmetric -- deeper below the shaft line
    than above it, squared off at the tip -- and it is *spooned*, curved
    across its width so it holds water.  Drawn as a flat rectangle it
    reads as a paddle off a raft.  The outline below is taken across the
    blade at a few stations, each offset along ``normal`` by a parabolic
    spoon, and closed on both faces so it is still a solid when it turns
    edge on at the feather.
    """
    root = np.asarray(root, dtype=float)
    tip = np.asarray(tip, dtype=float)
    axis = tip - root
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None

    # Fractions along the blade, and its half-width at each: narrow at
    # the neck, widest just short of the tip, squared off at the end.
    profile = ((0.00, 0.026), (0.16, 0.082), (0.42, 0.116),
               (0.74, 0.129), (1.00, 0.124))
    #: How far below the shaft line the blade hangs, against above it.
    #: A cleaver is markedly one-sided -- most of the area is on the
    #: lower edge, which is the part that stays buried.
    low, high = 0.72, 0.28
    spoon = 0.075

    rows = []
    for fraction, half in profile:
        centre = root + axis * fraction
        bow = spoon * 4.0 * fraction * (1.0 - fraction)
        rows.append((centre + normal * bow - edge * (2.0 * half * low),
                     centre + normal * bow + edge * (2.0 * half * high)))

    faces = []
    thickness = normal * 0.016
    for (a0, b0), (a1, b1) in zip(rows[:-1], rows[1:]):
        for offset in (thickness, -thickness):
            p0, p1, p2, p3 = (a0 + offset, b0 + offset,
                              b1 + offset, a1 + offset)
            faces += [[p0, p1, p2], [p0, p2, p3],
                      [p0, p2, p1], [p0, p3, p2]]
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    return MeshPart("blade", vertices, colours, _face_normals(vertices))


def blade_frame(axis, rim, lie, toward_bow, roll):
    """``(edge, dish)`` of a blade rolled ``roll`` of the way to the feather.

    ``rim`` runs across the blade and points up when squared, ``lie`` is
    the horizontal across-blade direction, and ``toward_bow`` is the
    sign that puts the dish's back toward the bow on the drive -- which
    differs between the two sides of the boat, because a port and a
    starboard oar point opposite ways.

    The roll turns with that same sign.  It used to turn the same way
    on both sides, and since the dish sign flips between them, one
    side feathered with its hollow to the sky and the other with its
    hollow to the water: the starboard blades feathered upside down.
    Feathered, the hollow faces up on both sides -- the way every rower
    carries a blade on the recovery -- which is what the test holds.
    """
    axis = np.asarray(axis, dtype=float)
    angle = 0.5 * np.pi * float(roll)
    edge = (np.asarray(rim, dtype=float) * np.cos(angle)
            + np.asarray(lie, dtype=float) * np.sin(angle) * float(toward_bow))
    dish = float(toward_bow) * np.cross(axis, edge)
    return edge, dish


def oar_solids(boat, t):
    """Both looms and both blades of every oar, in the hull frame.

    The blade is squared on the drive and feathered on the recovery,
    which is the clearest cue in the frame for where in the cycle the
    crew is -- more legible than the hands, because it is a whole
    surface turning rather than a small thing moving.
    """
    parts = []
    roll = blade_roll(boat, t)
    for handle, pivot, blade, _lift, drive in oar_pose(boat, t):
        parts.append(_tube(handle, pivot, 0.024, LOOM, sides=6))
        parts.append(_tube(pivot, blade, 0.021, LOOM, sides=6))

        axis = blade - pivot
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            continue
        axis = axis / length
        # The blade's own frame, built with its signs decided rather
        # than inherited from whichever way a cross product happened to
        # come out.  Both of the following were wrong that way, and in
        # opposite directions on the two sides of the boat, which is
        # exactly why a port and a starboard oar are not mirror images
        # of one another.
        #
        # ``rim`` runs across the blade and points UP when squared, so
        # the long side of the profile -- which is taken along -rim --
        # hangs below the shaft, where it belongs.
        vertical = np.array([0.0, 0.0, 1.0])
        rim = vertical - axis * float(axis @ vertical)
        rim /= max(float(np.linalg.norm(rim)), 1e-9)
        if rim[2] < 0.0:
            rim = -rim
        lie = np.cross(axis, rim)
        lie /= max(float(np.linalg.norm(lie)), 1e-9)

        # Which way the spoon is dished.  On the drive the blade pushes
        # water astern, so its hollow face looks astern and its back
        # bulges toward the bow.  Decided once, squared, and carried
        # through the roll -- deciding it per frame would break as soon
        # as the blade feathered and the normal went vertical.
        toward_bow = 1.0 if float(np.cross(axis, rim)[0]) > 0.0 else -1.0

        # Squared the blade hangs DOWN into the water: width vertical,
        # face along the pull.  Feathered it lies flat: width
        # horizontal, face at the sky.
        edge, normal = blade_frame(axis, rim, lie, toward_bow, roll)
        built = _blade_surface(blade - axis * 0.52, blade, edge, normal)
        if built is not None:
            parts.append(built)

    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    vertices = np.concatenate([part.vertices for part in parts])
    colours = np.concatenate([part.colours for part in parts])
    return MeshPart("oars", vertices, colours, _face_normals(vertices))


def crew_solids(boat, t, scale: float = 1.0):
    """Every rower's body in the hull frame at stroke time ``t``.

    Built on :meth:`JointDrivenRower.skeleton` and its ``BONES`` -- the
    same joints the PyVista scene draws and the same chain the dynamics
    are integrated from, so the bodies cannot drift out of step with the
    boat they are driving.  ``skeleton`` resolves both arms in three
    dimensions including the trunk rotation, which matters: a sweep
    rower has both hands on one handle and is wound round toward it, and
    a mirrored figure would read as sculling.

    Why draw them at all: **the crew face the stern, and in a
    stern-coxed boat that means they face you.**  From an eight's seat
    the view is eight bodies coming at you and swinging away, and the
    timing you are reading is written on their fronts.  Without them an
    eight and a bow-loaded four look the same out of the window, and the
    eight is the one where they should not.
    """
    if not getattr(boat, "crew", None):
        return None

    # The kinematics already put the hands ON the handle -- they are
    # solved against it -- so the two agree exactly in the model.  What
    # the model has no vertical for is the oar itself (see
    # :func:`oar_pose`), and once the drawn loom is tilted so the blade
    # reaches the water, a hand has to ride up the tilt to stay on it.
    #
    # By how much depends where along the loom the hand is: a sweep
    # rower's two hands are at different points on it, so lifting both
    # by the handle's own rise pulls the inboard hand off the shaft by
    # about 5 cm.  The lift is therefore interpolated along the loom.
    looms = {}
    poses = oar_pose(boat, float(t))
    index = 0
    for seat_index, seat in enumerate(boat.rig.seats):
        for _lock in seat.oarlocks:
            handle, pivot, _blade, lift, _drive = poses[index]
            looms[seat_index] = (pivot, handle, lift)
            index += 1

    parts = []
    for member in boat.crew:
        rower = member.rower
        joints = dict(rower.skeleton(float(t)))
        loom = looms.get(member.seat_index)
        if loom is not None and loom[2]:
            pivot, handle, lift = loom
            span = max(float(np.linalg.norm(handle[:2] - pivot[:2])), 1e-6)
            hand_rise = {}
            for name in [n for n in joints if n.startswith("hand")]:
                along = float(np.linalg.norm(joints[name][:2] - pivot[:2]))
                rise = lift * along / span
                hand_rise[name.split("_", 1)[-1]] = rise
                joints[name] = joints[name] + np.array([0.0, 0.0, rise])
            # The shoulder does not move, so the elbow takes up about
            # half of whatever its own hand did.
            for name in [n for n in joints if n.startswith("elbow")]:
                rise = hand_rise.get(name.split("_", 1)[-1], 0.0)
                joints[name] = joints[name] + np.array([0.0, 0.0, rise * 0.5])
        # Everything but the hands rises onto the seat.  The hands do
        # not: they are on the loom, and the loom is hung off an oarlock
        # 0.38 m above the water that has not moved.  Lifting them too
        # would take them straight back off the shaft that the whole
        # previous fix put them on.  The elbow splits the difference,
        # which is what an elbow does.
        for name in joints:
            if name.startswith("hand"):
                continue
            rise = CREW_LIFT * (0.5 if name.startswith("elbow") else 1.0)
            joints[name] = joints[name] + np.array([0.0, 0.0, rise])

        for start, end in rower.BONES:
            if start not in joints or end not in joints:
                continue
            if end == "head":
                # A head is a sphere, not a length of pipe.
                parts.append(_tube(joints[start], joints[end], 0.045, SKIN))
                parts.append(_sphere(joints[end] + np.array([0.0, 0.0, 0.05]),
                                     0.105, SKIN))
                continue
            radius, colour = _bone_style(start, end)
            parts.append(_tube(joints[start], joints[end],
                               radius * float(scale), colour))

    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    vertices = np.concatenate([part.vertices for part in parts])
    colours = np.concatenate([part.colours for part in parts])
    return MeshPart("crew", vertices, colours, _face_normals(vertices))


SOLID_WITHIN = 250.0

#: However close they are, no more than this many get the full model, so
#: a wooded bank cannot blow the triangle budget on its own.
SOLID_BUDGET = 14000

#: Crown colours by growth form, in the order ``TreeStand.FORMS`` uses.
CROWN = ((0.24, 0.34, 0.20), (0.16, 0.26, 0.18), (0.20, 0.31, 0.20),
         (0.30, 0.40, 0.24))
TRUNK = (0.26, 0.20, 0.15)


def tree_solids(stand, box, limit: int = 80000, near=None,
                min_height: float = 3.0, ground_at=None,
                shore: float = 0.25) -> Optional[MeshPart]:
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

    # **Stand them on the ground, and not in the river.**
    #
    # Buildings were put on the terrain when the vertical datum was
    # fixed; trees were missed, so every one sat at z = 0 -- the
    # waterline.  On a rising bank they were buried to the knees, and any
    # whose position falls over water (the canopy polygons overlap it)
    # stood *in* the river: 1,753 triangles of conifer spire growing out
    # of the water beside the course, which is exactly what a spike looks
    # like from the seat.  A tree below the shoreline is bad data, not a
    # short tree, so it is dropped rather than floated.
    if ground_at is not None:
        try:
            floor = np.asarray(ground_at(points[index, 0],
                                         points[index, 1]), dtype=float)
        except Exception:                          # pragma: no cover
            floor = np.zeros(len(index))
        dry = floor >= shore
        if not dry.all():
            print("   %d trees dropped for standing in the water"
                  % int((~dry).sum()))
        index, floor = index[dry], floor[dry]
        if gap is not None:
            gap = gap[dry]
        if not len(index):
            return None
    else:
        floor = np.zeros(len(index))

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
        base_z = float(floor[slot])
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
        stem = base_z + 0.40 * height
        radius = max(0.018 * height, 0.05)
        trunk = box_solid((x, y, 0.5 * (base_z + stem)),
                          (radius, radius, 0.5 * (stem - base_z)),
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
        top = base_z + height
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


def _tri(faces, shades, p0, p1, p2, want, colour):
    """One triangle, wound so its normal points along ``want``.

    The fans that cap a roof or floor a part used to go through
    :func:`_quad` with the apex passed twice, which emitted a real
    triangle and a zero-area twin beside it -- 106,821 of the Charles's
    311,650 building triangles, a third of the geometry, doing nothing.
    """
    normal = np.cross(np.asarray(p1) - np.asarray(p0),
                      np.asarray(p2) - np.asarray(p0))
    if float(np.dot(normal, want)) < 0.0:
        p1, p2 = p2, p1
    faces.append([p0, p1, p2])
    shades.extend([colour] * 3)


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


def _strut(a, b, radius, colour):
    """A square prism between two points -- one member of a truss."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    axis = axis / length
    # Any two directions perpendicular to the member.
    guide = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ guide)) > 0.95:
        guide = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, guide)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(axis, u)
    faces = []
    corner = [u * radius + v * radius, u * radius - v * radius,
              -u * radius - v * radius, -u * radius + v * radius]
    lo = [a + c for c in corner]
    hi = [b + c for c in corner]
    for i in range(4):
        j = (i + 1) % 4
        faces += [[lo[i], lo[j], hi[j]], [lo[i], hi[j], hi[i]]]
    faces += [[hi[0], hi[1], hi[2]], [hi[0], hi[2], hi[3]]]
    faces += [[lo[0], lo[2], lo[1]], [lo[0], lo[3], lo[2]]]
    vertices = np.asarray(faces, dtype="f4").reshape(-1, 3)
    # Outward-facing, by construction rather than by hope.
    tri = vertices.reshape(-1, 3, 3)
    centre = 0.5 * (a + b)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    inward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - centre) < 0.0
    tri[inward] = tri[inward][:, ::-1]
    vertices = tri.reshape(-1, 3)
    colours = np.tile(np.asarray(colour, dtype="f4"), (len(vertices), 1))
    return MeshPart("strut", vertices, colours, _face_normals(vertices))


def truss_bridge(start, end, width: float, level: float, depth: float,
                 span: float, colour=(0.55, 0.55, 0.53),
                 pier_colour=(0.52, 0.51, 0.48), piers=None):
    """A steel deck truss, **open**, as Seattle's canal bridges are.

    The Ship Canal Bridge and the Fremont Bridge carry their structure
    below the deck and you look straight through it: two chords with a
    zig-zag web between them, on piers.  Drawing them as a solid slab --
    which is what this did -- gets the deck height right and the whole
    character wrong, because the thing a crew sees approaching one is
    daylight through steelwork.

    This is the construction :mod:`coxswain.viz.river3d` already used for
    the PyVista scene (parallel chords of constant depth, a zig-zag web,
    piers carrying the bottom chord); it is brought over so the two
    renderers agree.  Panel length is a fraction of the span, which is
    what sets how coarse the lattice looks.
    """
    start = np.asarray(start, dtype=float)[:2]
    end = np.asarray(end, dtype=float)[:2]
    length = float(np.hypot(*(end - start)))
    if length < 1.0:
        return None
    along = (end - start) / length
    across = np.array([-along[1], along[0]])
    half = 0.5 * float(width)
    member = max(0.045 * depth, 0.35)
    bottom = level - depth
    parts = [_slab(0.5 * (start + end), along, length, width, level,
                   max(0.12 * depth, 0.5))]

    def at(distance, side, height):
        xy = start + along * distance + across * (side * half)
        return np.array([xy[0], xy[1], height])

    # Chords down each side, and the web between them.
    panel = max(span / 6.0, 8.0)
    stations = np.arange(0.0, length + panel, panel)
    stations = np.clip(stations, 0.0, length)
    for side in (-0.86, 0.86):
        for a, b in zip(stations[:-1], stations[1:]):
            if b - a < 1e-6:
                continue
            parts.append(_strut(at(a, side, bottom), at(b, side, bottom),
                                member, colour))
            # Zig-zag: bottom-to-deck and back, alternating.
            up = at(a, side, level - 0.12 * depth)
            parts.append(_strut(at(a, side, bottom), up, 0.8 * member,
                                colour))
            parts.append(_strut(at(a, side, bottom),
                                at(b, side, level - 0.12 * depth),
                                0.7 * member, colour))
    # Cross-bracing between the two trusses, so it reads as a box.
    for a in stations[::2]:
        parts.append(_strut(at(a, -0.86, bottom), at(a, 0.86, bottom),
                            0.7 * member, colour))
    # Piers under the bottom chord.  Surveyed stations are used when the
    # caller has them -- the Grand Junction trestle's five are measured
    # from OSM ``bridge:support=pier`` polygons and sit at 28, 53, 78, 96
    # and 121 m, which is neither evenly spaced nor the seven that
    # dividing by the span length produces.
    if piers is not None:
        feet = [start + along * float(distance) for distance in piers]
    else:
        count = max(int(round(length / max(span, 1.0))), 1)
        feet = [start + (end - start) * (step / count)
                for step in range(1, count)]
    for foot in feet:
        parts.append(box_solid((foot[0], foot[1], 0.5 * bottom),
                               (0.055 * span + 1.0, 0.055 * span + 1.0,
                                0.5 * bottom), colour=pier_colour))
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    vertices = np.concatenate([p.vertices for p in parts])
    colours = np.concatenate([p.colours for p in parts])
    return MeshPart("bridges", vertices, colours, _face_normals(vertices))


def arch_bridge(start, end, width: float, level: float, depth: float,
                spans: int, colour=(0.72, 0.71, 0.67),
                pier_colour=(0.58, 0.57, 0.54), samples: int = 13,
                piers=None):
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
    # Bay boundaries.  Given explicit pier stations the arches are laid
    # between them and can be unequal, which is what a real bridge is:
    # the navigable span is set by the channel and the side spans take
    # up whatever is left to the abutments.  Without them the length is
    # divided evenly, as before.
    if piers is not None and len(piers):
        inner = sorted(float(p) for p in piers
                       if 0.5 < float(p) < length - 0.5)
        edges = np.array([0.0] + inner + [length])
    else:
        edges = np.linspace(0.0, length, spans + 1)
    arch = length / spans

    def point(distance, side, height):
        xy = start + along * distance + across * (side * half)
        return [float(xy[0]), float(xy[1]), float(height)]

    def intrados(distance):
        """Height of the underside of the arch at ``distance`` along."""
        index = int(np.searchsorted(edges, distance, side="right")) - 1
        index = min(max(index, 0), len(edges) - 2)
        low_edge, high_edge = edges[index], edges[index + 1]
        u = (distance - low_edge) / max(high_edge - low_edge, 1e-6)
        # A semi-ellipse: vertical at the springing, flat at the crown,
        # which is what a segmental concrete arch looks like.  A parabola
        # leans out of the pier and reads as a culvert.
        return springing + rise * float(np.sqrt(max(0.0,
                                                    1.0 - (2.0 * u - 1.0) ** 2)))

    faces, shades = [], []
    # Sampled per bay, so an arch is resolved whatever its width and
    # the springing lands exactly on the pier faces.
    stations = np.unique(np.concatenate([
        np.linspace(edges[i], edges[i + 1], samples + 1)
        for i in range(len(edges) - 1)]))
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


#: Landmarks that the building extract flattens, as
#: ``(lat, lon, roof, shaft_top, belfry_top, dome_top, half_width)``.
#:
#: **Why this table has to exist.** Lowell House carries no ``height``
#: tag in OpenStreetMap, so the extractor falls back to a guess from the
#: building type -- ``height_source = 2`` -- and gets 15 m, which is the
#: house without its tower.  That is the wrong 25 m to lose: the Lowell
#: House bell tower is *the* mark for the Weeks turn, 250 m off the
#: bridge, and a coxswain lines the turn up on it.
#:
#: **The heights here are estimates, not survey.**  They are shaped to
#: the published description of the tower -- a Georgian brick shaft, an
#: open belfry, and the blue and gold onion dome above it -- and they
#: are not measured the way the bridge decks in
#: :mod:`coxswain.river.bridges` are.  Anything that needs a real number
#: should not read them.
LANDMARKS = {
    "Lowell House tower": (42.3707041, -71.1185517,
                           15.0, 28.0, 33.0, 37.0, 4.2),
}

#: Brick, stone and the dome.
BRICK = (0.55, 0.38, 0.33)
STONE = (0.82, 0.80, 0.75)
DOME = (0.24, 0.46, 0.58)


#: How far the abutments are carried past the water's edge, m.
ABUTMENT = 3.0


def _raw_waterway(gate, raster, min_depth: float = 0.6, samples: int = 400):
    """The widest wet run across a gate, **unclamped**.

    :func:`coxswain.river.bridges.waterway` trims its answer to the
    bridge's inventory length, on the reasoning that a bridge cannot
    open wider than it is long.  That is right for working out what a
    crew can steer through and wrong for deciding how long to draw the
    thing: where the two disagree, the drawn bridge has to be at least
    as long as the water, or it ends in mid river.
    """
    from ..river.bridges import _runs

    distance = np.linspace(0.0, gate.span, int(samples))
    points = gate.point_at(distance)
    try:
        if hasattr(raster, "is_navigable"):
            wet = np.array([bool(raster.is_navigable(p[0], p[1]))
                            for p in points])
        else:
            wet = np.array([float(raster.depth_at(p[0], p[1])) >= min_depth
                            for p in points])
    except Exception:
        return None
    runs = _runs(distance, wet)
    if not runs:
        return None
    return max(runs, key=lambda pair: pair[1] - pair[0])


def landmark_solids(race: str) -> Optional[MeshPart]:
    """Towers and spires the building data flattens.  Charles only."""
    if race != "charles" or not LANDMARKS:
        return None
    from ..river.charles import CHARLES_ORIGIN
    from ..river.course import local_tangent_plane

    parts = []
    for _name, row in LANDMARKS.items():
        lat, lon, roof, shaft, belfry, dome, half = row
        east, north = local_tangent_plane(np.array([lat]), np.array([lon]),
                                          CHARLES_ORIGIN)
        x, y = float(east[0]), float(north[0])
        parts.append(box_solid((x, y, 0.5 * (roof + shaft)),
                               (half, half, 0.5 * (shaft - roof)),
                               colour=BRICK))
        # The belfry is open, so it is drawn as corner posts rather than
        # a solid block -- daylight through it is what makes a tower
        # read as a tower and not a chimney.
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                parts.append(_tube((x + sx * half * 0.8, y + sy * half * 0.8,
                                    shaft),
                                   (x + sx * half * 0.8, y + sy * half * 0.8,
                                    belfry), 0.28, STONE, sides=6))
        parts.append(box_solid((x, y, belfry + 0.35),
                               (half * 0.95, half * 0.95, 0.35),
                               colour=STONE))
        parts.append(_sphere((x, y, belfry + 0.6 + 0.45 * (dome - belfry)),
                             0.62 * half, DOME, rings=6, segments=10))
        parts.append(_tube((x, y, dome), (x, y, dome + 2.4), 0.10,
                           (0.78, 0.68, 0.35), sides=6))

    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    vertices = np.concatenate([part.vertices for part in parts])
    colours = np.concatenate([part.colours for part in parts])
    return MeshPart("landmarks", vertices, colours, _face_normals(vertices))


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
        from ..river.bridges import (BRIDGE_STRUCTURE, MEASURED_PIERS,
                                     deck_geometry, derive_piers)
        from ..river.charles import CHARLES_ORIGIN
        from ..river.course import local_tangent_plane
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
                # Trimmed to the *bridge*, not to the line OSM draws.
                #
                # Western Avenue's deck way is 152 m long; the bridge in
                # the National Bridge Inventory is 85.3 m.  The rest is
                # approach roadway over dry land.  Dividing the whole
                # 152 m into three equal arches put the piers 51 m apart
                # when the centre span is 26.8 m -- roughly twice the
                # real spacing, on the bridge a crew lines up an arch on
                # from several hundred metres out.
                #
                # The inventory length, centred on the deck line, puts
                # them within a metre of where laying the centre span
                # symmetrically about the channel does.
                structure = BRIDGE_STRUCTURE.get(gate.name)
                span_length = getattr(structure, "structure_length", None)
                low, high = start, end
                full = float(np.hypot(*(end - start)))
                if span_length and 0.0 < float(span_length) < full:
                    along = (end - start) / max(full, 1e-9)
                    length = float(span_length)
                    # Centred on the CHANNEL, not on the deck line.  The
                    # river does not run under the middle of the road:
                    # centring Western Avenue's 85 m of arches on its
                    # 152 m deck way put them 27 m from where the
                    # navigation side has always had the piers, which is
                    # more than an arch width on the bridge a crew picks
                    # its arch on from several hundred metres out.
                    #
                    # derive_piers lays the centre span symmetrically
                    # about the wet opening; the side spans take up the
                    # rest of the inventory length either side.
                    # The structure has to reach dry land at both
                    # ends.  The inventory length alone does not
                    # guarantee that: River Street's raster water is
                    # 77.9 m wide and its NBI structure_length is 64.0,
                    # a bridge shorter than the river it crosses.  Laid
                    # out on the inventory figure it stopped 7 m short
                    # of each bank and the approach embankments stood in
                    # open water -- the bridge began in the middle of
                    # the river.
                    #
                    # ``waterway`` will not show this, because it clamps
                    # the opening to the structure length: correct for
                    # navigation, where a bridge cannot open wider than
                    # it is long, and useless here.  So the wet run is
                    # measured raw and the structure spans whichever is
                    # longer, with the abutments carried onto the bank.
                    try:
                        piers = derive_piers(gate, geometry.channel)
                    except Exception:
                        piers = ()
                    wet = _raw_waterway(gate, geometry.channel)
                    if wet is None:
                        span_lo, span_hi = (0.5 * (full - length),
                                            0.5 * (full + length))
                    else:
                        span_lo = min(wet[0] - ABUTMENT, 0.5 * (full - length))
                        span_hi = max(wet[1] + ABUTMENT, 0.5 * (full + length))
                    span_lo = max(span_lo, 0.0)
                    span_hi = min(span_hi, full)
                    low = start + along * span_lo
                    high = start + along * span_hi
                    # Pier stations, relative to the structure's own
                    # start, so the navigable arch stays between the
                    # piers the navigation side derives and the side
                    # arches take up the rest.
                    bays = [float(pier.centre) - span_lo for pier in piers]
                built = arch_bridge(low, high, width, level, depth, spans,
                                    piers=bays or None)
                if built is not None:
                    parts.append(built)
                # The arches are the bridge; the rest of the way is the
                # approach, which in life is a road on an embankment
                # running onto the bank.  Trimming the arches to the
                # inventory length without this left the bridge floating
                # in the middle of the river with a gap at each end.
                for far, near in ((start, low), (high, end)):
                    run = float(np.hypot(*(np.asarray(near)
                                           - np.asarray(far))))
                    if run < 1.0:
                        continue
                    middle = 0.5 * (np.asarray(far) + np.asarray(near))
                    heading = (np.asarray(near) - np.asarray(far)) / run
                    # Solid down to the ground, not a floating deck.
                    # An approach is masonry on an embankment running
                    # onto the bank; drawn as a thin slab it hangs in
                    # the air off the end of the arches with daylight
                    # under it, which is the one thing a bridge never
                    # does.  Where the bank is higher than the water
                    # this is simply buried in it.
                    parts.append(_slab(middle, heading, run, width, level,
                                       max(float(level), 0.6)))
                continue
            # Not an arch.  Eliot is NBI 4/9, a steel deck truss, and
            # the Grand Junction is a 149 m steel trestle -- both were
            # drawn as a plain slab on two piers at the thirds, which
            # for the trestle is the one bridge on the reach whose whole
            # character is a long row of legs in the water.  They get
            # the same open steelwork the Seattle bridges do, and where
            # the piers have been surveyed they go where they were
            # measured rather than at even fractions.
            length = float(np.hypot(*(end - start)))
            along = (end - start) / max(length, 1e-9)
            stations = None
            measured = MEASURED_PIERS.get(gate.name)
            if measured:
                lats = np.array([point[0] for point in measured])
                lons = np.array([point[1] for point in measured])
                east, north = local_tangent_plane(lats, lons, CHARLES_ORIGIN)
                stations = sorted(
                    float((np.array([e, n]) - start) @ along)
                    for e, n in zip(east, north))
            built = truss_bridge(start, end, width, level, depth,
                                 span=float(_span or 18.0), piers=stations)
            if built is not None:
                parts.append(built)
    else:
        try:
            from ..river.seattle import canal_bridges

            for bridge in canal_bridges():
                middle = np.asarray(bridge.centre, dtype=float)
                along = np.asarray(bridge.axis, dtype=float)
                level = bridge.deck_height
                depth = bridge.structure_depth
                width = max(bridge.width, 8.0)
                half = 0.5 * bridge.length
                start = middle - along * half
                end = middle + along * half
                if bridge.form == "arch":
                    spans = max(int(round(bridge.length
                                          / max(bridge.main_span, 1.0))), 1)
                    built = arch_bridge(start, end, width, level, depth,
                                        spans)
                    if built is not None:
                        parts.append(built)
                        continue
                if bridge.form == "truss":
                    built = truss_bridge(start, end, width, level, depth,
                                         max(bridge.main_span, 20.0))
                    if built is not None:
                        parts.append(built)
                        continue
                parts.append(_slab(middle, along, bridge.length, width,
                                   level, depth))
                count = max(int(round(bridge.length
                                      / max(bridge.main_span, 1.0))), 1)
                for step in range(1, count):
                    foot = start + (end - start) * (step / count)
                    parts.append(box_solid(
                        (foot[0], foot[1], 0.5 * (level - depth)),
                        (2.2, 2.2, 0.5 * (level - depth)),
                        colour=(0.50, 0.49, 0.46)))
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
                guide: bool = True, trees: bool = True,
                water_level: float = 0.0):
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

    def ground_at(east, north):
        """Height of the ground above the water, never below it."""
        return np.maximum(terrain.height_above_water(east, north), 0.0)

    mesh = WorldMesh()
    mesh.add(water_plane(course.mean(axis=0), level=water_level))
    mesh.add(land_mesh(terrain, wet_at, box, step=step, imagery=photo))
    if with_buildings:
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
            mesh.add(tree_solids(tree_stand(), box, near=course,
                                 ground_at=ground_at))
        except Exception as error:            # pragma: no cover
            print("   (no trees: %s)" % str(error)[:60])
    if race != "charles":
        mesh.add(cut_walls())
    mesh.add(bridge_solids(race, scene))
    mesh.add(landmark_solids(race))
    mesh.add(buoy_solids(scene.buoys))
    if guide:
        mesh.add(line_markers(course))
    return mesh, scene
