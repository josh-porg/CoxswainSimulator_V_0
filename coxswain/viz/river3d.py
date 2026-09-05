"""The boat, in the river, in three dimensions -- and from the cox's seat.

:mod:`coxswain.viz.scene3d` draws the boat over an unbounded plane of
water, which is the right picture for checking whether the crew folds
correctly and the wrong one for asking whether a line is rowable.  This
puts the same boat in the actual Charles: the surveyed bank, the bridges
with their piers and arches, and the line it is trying to hold.

The view that matters
---------------------
``"cox"`` places the camera **at the coxswain's head**, in the stern,
looking forward over the crew.  That is not a gimmick.  Every steering
decision on this river is made from that seat and from that height --
about 0.7 m above the water, low enough that a bridge arch 400 m away is
a slot a few pixels wide and the far bank hides behind the near one.  A
plan view flatters the coxswain enormously: it shows them geometry they
cannot actually see.  Anyone judging whether a line is *steerable* rather
than merely *short* should be looking at it from here.

Building the scenery
--------------------
Terrain comes from the same :class:`~coxswain.river.channel.ChannelRaster`
the dynamics use, so the bank in the picture is the bank in the physics.
Only a window around the boat is built -- a few hundred metres -- because
the whole reach is 12 km of raster and none of it is visible from a seat
0.7 m off the water.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.frames import hull_to_abs
from .scene3d import BoatScene, require_pyvista

__all__ = ["RiverScene"]

_WATER = "#4d7f9c"
_BANK = "#8d8467"
_DECK = "#6b6257"
_PIER = "#3a3631"
_LEGAL = "#1f7a4d"
_PENALTY = "#a2382a"
_LINE = "#6b3fa0"
_SKY_TOP = "#20344a"
_SKY_HORIZON = "#9fb8c8"
_WALL = "#8a8079"
_ROOF = "#5f5a55"
_CANOPY = "#4a6b45"

#: Wall colour per building class, as RGB in 0-1.
#:
#: OpenStreetMap carries ``building:material`` on 1% of Seattle and
#: ``building:colour`` on half a percent, so colouring by material would
#: paint 99 buildings in a hundred the same unknown grey -- which is the
#: grey the render already had.  ``building=*`` is on essentially all of
#: them, so class is what there is to work with, and a class palette
#: plus a per-building jitter is an honest picture of a real city block:
#: it does not claim to know any individual building's colour, only that
#: a houseboat and a downtown tower are not the same object.
_KIND_COLOUR = {
    "other":      (0.55, 0.53, 0.50),
    "house":      (0.68, 0.63, 0.55),
    "apartments": (0.60, 0.58, 0.57),
    "commercial": (0.58, 0.60, 0.63),
    "office":     (0.52, 0.58, 0.64),
    "industrial": (0.56, 0.53, 0.47),
    "civic":      (0.72, 0.70, 0.65),
    "boathouse":  (0.48, 0.40, 0.32),
    "houseboat":  (0.52, 0.45, 0.38),
    "roof":       (0.50, 0.48, 0.46),
    "garage":     (0.53, 0.51, 0.48),
    "retail":     (0.62, 0.59, 0.55),
}

#: Extinction distance for aerial perspective, metres.
#:
#: Colour is mixed toward the horizon as ``1 - exp(-d / _HAZE)``.  This
#: is not decoration: without it the downtown towers 4 km down the lake
#: render at the same contrast as the boathouse 200 m away, and the eye
#: reads them as small and near rather than large and far.  Distance on
#: open water is judged almost entirely by haze.
_HAZE = 2600.0

#: A building this many metres tall is worth drawing at 1 m of distance.
#:
#: Beyond the near window only things that subtend a usable angle are
#: drawn -- ``height / distance > _SKYLINE_ANGLE`` -- which keeps 40,000
#: footprints from being extruded to render a horizon made of towers.
_SKYLINE_ANGLE = 0.012


class RiverScene(BoatScene):
    """A :class:`BoatScene` that also knows where the river is."""

    VIEWS = BoatScene.VIEWS + ("cox", "chase3d", "plan") \
        if hasattr(BoatScene, "VIEWS") else ("cox", "chase3d")

    #: Height of a seated coxswain's eye above the waterline, metres.
    EYE_HEIGHT = 0.70

    #: Altitude for the ``"plan"`` view, metres.
    #:
    #: The inherited ``"top"`` view sits one boat length up, which frames
    #: the hull and nothing else -- right for checking the rig, useless for
    #: watching a line.  This is high enough to hold both banks and a
    #: bridge in frame while the boat is still a recognisable object.
    PLAN_ALTITUDE = 260.0

    #: How far the distant scenery reaches, metres.
    #:
    #: The near window is sized for steering -- a few hundred metres of
    #: bank at full raster resolution.  Everything a crew *recognises* is
    #: further off than that: rowing south down Lake Union the downtown
    #: skyline is 3 to 5 km away and is the thing you steer by.  Drawn
    #: from a coarse resample of the whole elevation tile, so it costs
    #: one mesh for the render rather than one per frame.
    SKYLINE_REACH = 7000.0

    #: Points across the coarse distant surface.
    SKYLINE_SAMPLES = 320

    #: Ceiling on bank height in the near window, metres.
    #:
    #: Guards against a lidar return the bare-earth classifier kept,
    #: which would otherwise put a wall on the bank.  It was 16 m, which
    #: was quietly flattening real ground: the Charles climbs to 50 m
    #: inside the frame and Lake Union is a trench between Queen Anne at
    #: 139 m and Capitol Hill at 133 m.  At 16 m the near window and the
    #: distant surface disagreed about the same hillside, leaving a step
    #: at the edge of the window.
    MAX_BANK_HEIGHT = 60.0

    def __init__(self, boat, result=None, channel=None, gates=(),
                 path=None, window: float = 420.0,
                 show_structures: bool = True, show_skyline: bool = True,
                 structures=None, terrain=None, **kwargs):
        """``structures`` and ``terrain`` default to the Charles.

        They are injectable because nothing in this renderer is actually
        about the Charles -- it draws a boat, a channel raster, some
        footprints and a bank.  Hard-coding the providers made it look
        course-specific, which is why a second matplotlib renderer got
        written for Lake Union instead of this one.  Pass Seattle's and
        the same code draws Seattle.
        """
        super().__init__(boat, result=result, **kwargs)
        self.show_structures = bool(show_structures)
        self.show_skyline = bool(show_skyline)
        self._structures = structures
        self._terrain = terrain
        self.channel = channel
        self.gates = tuple(gates)
        self.path = None if path is None else np.asarray(path, float)[:, :2]
        self.window = float(window)

    # -- scenery ----------------------------------------------------------
    def structures(self):
        """Footprints to draw. Defaults to the Charles."""
        if self._structures is None:
            from ..river.structures import charles_structures
            self._structures = charles_structures()
        return self._structures

    def terrain(self):
        """Bank heights. Defaults to the Charles."""
        if self._terrain is None:
            from ..river.terrain import charles_terrain
            self._terrain = charles_terrain()
        return self._terrain

    def terrain_window(self, centre):
        """Grid indices for a window of raster around ``centre``.

        Split out and cached because the mesh only changes when the boat
        leaves the window.  Rebuilding it per frame is what made a 170
        frame render take longer than the race it was drawing.
        """
        east, north = self.channel.east, self.channel.north
        half = self.window
        ix = np.nonzero((east >= centre[0] - half)
                        & (east <= centre[0] + half))[0]
        iy = np.nonzero((north >= centre[1] - half)
                        & (north <= centre[1] + half))[0]
        step = max(len(ix) // 150, 1)
        return ix[::step], iy[::step]

    def terrain_polydata(self, t: float):
        """Bed and bank around the boat, as a surface with a depth array.

        Land is lifted a little above the waterline so the bank reads as a
        bank from a low camera rather than as a change of colour.
        """
        pv = require_pyvista()
        if self.channel is None:
            return None
        state = self.state_at(t)
        origin = self._origin(state)
        centre = np.asarray(state.position, dtype=float)[:2]

        east, north = self.channel.east, self.channel.north
        anchor = getattr(self, "_terrain_anchor", None)
        if anchor is None or np.linalg.norm(centre - anchor) > 0.35 * self.window:
            self._terrain_anchor = centre.copy()
            self._terrain_index = self.terrain_window(centre)
        ix, iy = self._terrain_index
        if len(ix) < 2 or len(iy) < 2:
            return None

        grid_x, grid_y = np.meshgrid(east[ix], north[iy])
        wet = self.channel.water[np.ix_(iy, ix)]
        depth = np.array(self.channel.depth[np.ix_(iy, ix)], dtype=float)
        # Bank heights from the federal elevation model where it covers,
        # instead of an invented 1.6 m shelf: the Cambridge levee, the
        # Storrow fill and the rise behind both banks are all real now.
        # Capped so the odd building the lidar kept does not put a wall on
        # the bank; the fallback shelf survives for anything off the DEM.
        try:
            bank = self.terrain().height_above_water(
                grid_x.ravel(), grid_y.ravel()).reshape(grid_x.shape)
            height = np.where(wet, 0.0, np.clip(bank, 0.6,
                                                self.MAX_BANK_HEIGHT))
        except Exception:
            height = np.where(wet, 0.0, 1.6)

        points = np.column_stack([
            (grid_x - origin[0]).ravel(),
            (grid_y - origin[1]).ravel(),
            height.ravel() + self.water_level,
        ])
        surface = pv.StructuredGrid()
        surface.points = points
        surface.dimensions = (len(ix), len(iy), 1)
        surface.point_data["depth"] = np.where(wet, depth, np.nan).ravel()
        surface.point_data["land"] = (~wet).astype(np.int8).ravel()
        return surface

    def distant_water(self, east, north):
        """Boolean water over a coarse grid, from polygons not elevation.

        Lidar over water is a specular return and comes back badly
        behaved: over Lake Union the bare-earth product scatters from the
        pool level up past 7 m, so calling anything within half a metre
        of the pool "water" marks **48% of the lake as dry land** -- which
        is exactly what the first render showed, a four sitting on a tan
        plain.  No threshold fixes it either: loose enough to recover the
        lake (pool + 3 m) drowns 5.6% of the surrounding land.

        So the elevation model is not asked.  OpenStreetMap's water
        polygons are, and the DEM is left to do the thing it is good at,
        which is the shape of the ground.
        """
        try:
            structures = self.structures()
        except Exception:
            return None
        polygons = getattr(structures, "water", None)
        if not polygons:
            return None
        from matplotlib.path import Path

        grid_east, grid_north = np.meshgrid(east, north)
        points = np.column_stack([grid_east.ravel(), grid_north.ravel()])
        wet = np.zeros(len(points), dtype=bool)
        for ring in polygons:
            if len(ring) < 3:
                continue
            low, high = ring.min(axis=0), ring.max(axis=0)
            inside = np.nonzero(~wet
                                & (points[:, 0] >= low[0])
                                & (points[:, 0] <= high[0])
                                & (points[:, 1] >= low[1])
                                & (points[:, 1] <= high[1]))[0]
            if not len(inside):
                continue
            wet[inside] = Path(ring).contains_points(points[inside])
        return wet.reshape(grid_east.shape)

    def distant_polydata(self):
        """The whole elevation tile, coarsely, as ground and far water.

        Built once and cached: it does not depend on where the boat is,
        because it is everything the boat is *not* near.  Water comes
        from :meth:`distant_water`, which puts Portage Bay, the ship
        canal and Elliott Bay in the picture without a second survey.
        """
        pv = require_pyvista()
        if self._distant_mesh is not None:
            return self._distant_mesh
        try:
            terrain = self.terrain()
        except Exception:
            self._distant_mesh = False
            return None

        step = max(len(terrain.east) // self.SKYLINE_SAMPLES, 1)
        stride = max(len(terrain.north) // self.SKYLINE_SAMPLES, 1)
        east = terrain.east[::step]
        north = terrain.north[::stride]
        elevation = terrain.elevation[::stride, ::step]
        grid_east, grid_north = np.meshgrid(east, north)

        wet = self.distant_water(east, north)
        if wet is None:
            # Nothing else to go on.  Generous, because on this fallback
            # a lake drawn as land is a worse picture than a low field
            # drawn as water.
            wet = elevation <= terrain.pool + 2.0
        height = np.where(wet, 0.0, np.maximum(elevation - terrain.pool, 0.0))

        surface = pv.StructuredGrid()
        surface.points = np.column_stack([
            grid_east.ravel(), grid_north.ravel(),
            height.ravel() + self.water_level
            # Dropped a hand's breadth so the full-resolution near window
            # sits cleanly on top of it instead of z-fighting.
            - 0.05])
        surface.dimensions = (len(east), len(north), 1)
        surface.point_data["land"] = (~wet).astype(np.int8).ravel()
        self._distant_mesh = surface
        return surface

    def span_actors(self, plotter, t: float):
        """Landmark bridge decks -- scenery, not gates.

        A bridge the course does not pass under still matters: it is how
        a crew knows where they are.  The deck height is taken from the
        elevation model at the two abutments rather than from a tag,
        because OpenStreetMap almost never carries one and the ground at
        each end of a high bridge is, by construction, deck height.
        """
        pv = require_pyvista()
        try:
            structures = self.structures()
            terrain = self.terrain()
        except Exception:
            return
        if not getattr(structures, "spans", None):
            return
        state = self.state_at(t)
        origin = self._origin(state)
        centre = np.asarray(state.position, dtype=float)[:2]

        for name, deck in zip(structures.span_names, structures.spans):
            if len(deck) < 3:
                continue
            length = float(np.hypot(*np.diff(deck, axis=0).T).sum())
            gap = float(np.linalg.norm(deck.mean(axis=0) - centre))
            # Only spans big enough to read at the distance they are.
            if length < 120.0 or gap > self.SKYLINE_REACH:
                continue
            if length / max(gap, 1.0) < 0.02:
                continue
            ends = terrain.at(deck[[0, -1], 0], deck[[0, -1], 1])
            level = float(max(ends.max(), terrain.pool + 3.0))
            points = np.column_stack([
                deck[:, 0] - origin[0], deck[:, 1] - origin[1],
                np.full(len(deck), level)])
            try:
                ribbon = pv.lines_from_points(points).tube(
                    radius=max(4.0, 0.012 * length), n_sides=4)
            except Exception:
                continue
            plotter.add_mesh(ribbon, color=_DECK, name="span-%s" % name,
                             ambient=0.30, diffuse=0.75)

    _distant_mesh = None

    def bridge_actors(self, plotter, t: float):
        """Decks, piers and arch markers within sight of the boat."""
        pv = require_pyvista()
        if not self.gates:
            return
        state = self.state_at(t)
        origin = self._origin(state)
        centre = np.asarray(state.position, dtype=float)[:2]

        from ..river import bridges as _bridges
        for gate, _metres in self.gates:
            middle = 0.5 * (gate.start + gate.end)
            # Only the bridges in sight.  Drawing every arch on the reach
            # put six sets of markers across the frame at once, which from
            # a seat 0.8 m off the water reads as laser beams rather than
            # as bridges.
            if np.linalg.norm(middle - centre) > 320.0:
                continue
            a = np.array([gate.start[0] - origin[0],
                          gate.start[1] - origin[1], 3.6])
            b = np.array([gate.end[0] - origin[0],
                          gate.end[1] - origin[1], 3.6])
            plotter.add_mesh(pv.Tube(pointa=a, pointb=b, radius=1.1), color=_DECK,
                             name="deck-%s" % gate.name)
            for index, pier in enumerate(gate.piers):
                low, high = pier.interval
                p = gate.point_at(0.5 * (low + high))
                base = np.array([p[0] - origin[0], p[1] - origin[1], -1.0])
                top = base + np.array([0.0, 0.0, 4.6])
                plotter.add_mesh(pv.Tube(pointa=base, pointb=top, radius=1.7),
                                 color=_PIER,
                                 name="pier-%s-%d" % (gate.name, index))
            if self.channel is not None:
                for arch in _bridges.bridge_arches(gate, self.channel):
                    low, high = arch.interval
                    colour = _LEGAL if arch.legal else _PENALTY
                    # Uprights at the arch edges, so it reads as a gate to
                    # aim at.  A stripe lying on the water disappears into
                    # the surface from a seat 0.8 m above it -- which is
                    # exactly the height that makes this view worth having.
                    for edge, side in ((low, "a"), (high, "b")):
                        e = gate.point_at(edge)
                        base = np.array([e[0] - origin[0],
                                         e[1] - origin[1], 0.0])
                        plotter.add_mesh(
                            pv.Tube(pointa=base,
                                    pointb=base + np.array([0.0, 0.0, 3.2]),
                                    radius=0.22),
                            color=colour,
                            name="arch-%s-%d%s" % (gate.name, arch.index,
                                                   side))

    def building_colours(self, structures, index, distance):
        """Wall and roof colour for one building, as RGB in 0-1.

        Three inputs, in order of how much they are trusted.  A tagged
        ``building:colour`` wins outright, because somebody looked.
        Failing that the class palette, because ``building=*`` is on
        almost everything.  On top of either, a deterministic jitter
        keyed to the footprint's own position -- so the same building is
        the same colour in every frame, and a street is not one flat
        slab, without pretending to know which house is cream.

        Finally the whole thing is mixed toward the horizon by distance.
        """
        rgb = structures.colour[index]
        if rgb[0] < 0.0:
            kind = structures.KINDS[int(structures.kind[index])]
            rgb = np.array(_KIND_COLOUR.get(kind, _KIND_COLOUR["other"]))
        else:
            rgb = np.asarray(rgb, dtype=float)

        # Jitter from the coordinates themselves: stable across frames
        # and across runs, and free of any per-frame random state.
        east, north = structures.centres[index]
        seed = (np.sin(east * 12.9898 + north * 78.233) * 43758.5453) % 1.0
        rgb = np.clip(rgb * (0.86 + 0.28 * seed), 0.0, 1.0)

        # A roof is seen at a glancing angle from a seat 0.7 m off the
        # water and nearly face-on from the skyline; either way it is the
        # edge that separates one building from the next.
        roof = np.clip(rgb * 0.72, 0.0, 1.0)

        haze = 1.0 - np.exp(-max(distance, 0.0) / _HAZE)
        sky = np.array([0.62, 0.72, 0.78])
        return (rgb + (sky - rgb) * haze, roof + (sky - roof) * haze)

    def _extrude(self, structures, index, terrain, distance):
        """One building as a coloured solid, or ``None``."""
        pv = require_pyvista()
        ring = structures.polygons[index]
        if len(ring) < 4:
            return None
        base = float(terrain.at(*ring.mean(axis=0))[0])
        height = float(structures.heights[index])
        points = np.column_stack([ring[:, 0], ring[:, 1],
                                  np.full(len(ring), base)])
        face = np.concatenate([[len(points)], np.arange(len(points))])
        try:
            solid = pv.PolyData(points, faces=face).extrude(
                (0.0, 0.0, height), capping=True)
        except Exception:
            return None
        wall, roof = self.building_colours(structures, index, distance)
        # The cap sits at the top; the walls are quads spanning the
        # height, so their centres are at mid-height.  Splitting on the
        # centre height is enough to tell them apart and does not need
        # normals computed over the merged mesh.
        try:
            centres = solid.cell_centers().points[:, 2]
        except Exception:
            centres = np.zeros(solid.n_cells)
        is_roof = centres > base + height - max(0.15 * height, 0.2)
        colours = np.where(is_roof[:, None], roof[None, :], wall[None, :])
        solid.cell_data["rgb"] = (colours * 255).astype(np.uint8)
        return solid

    def structure_polydata(self, t: float):
        """Buildings within sight, as one merged mesh with colour.

        Two tiers, because "within sight" means two different things.

        **Near**, out to the terrain window: everything, because at 200 m
        a boathouse is a landmark and the DEM is bare earth, so without
        this the coxswain view is a river through empty fields.

        **Skyline**, out to :attr:`SKYLINE_REACH`: only what subtends a
        real angle, ``height / distance > _SKYLINE_ANGLE``.  Rowing down
        Lake Union you are looking straight at downtown Seattle 4 km
        away, and it is both the most recognisable thing in the frame and
        far too much geometry to draw in full -- 40,000 footprints, of
        which about a hundred are actually visible from the water.

        Everything is merged into a single mesh rather than one actor per
        building.  Forty separate actors per frame is what turned a
        render of the Powerhouse Stretch into a slideshow.
        """
        require_pyvista()
        state = self.state_at(t)
        origin = self._origin(state)
        centre = np.asarray(state.position, dtype=float)[:2]
        key = (round(centre[0] / 60.0), round(centre[1] / 60.0),
               bool(self.show_skyline))
        if self._structure_key == key and self._structure_mesh is not None:
            return self._structure_mesh, origin

        try:
            structures = self.structures()
            terrain = self.terrain()
        except Exception:
            self._structure_key, self._structure_mesh = key, None
            return None, origin

        reach = self.window * 1.4
        near = structures.near(centre[0], centre[1], reach)
        chosen = list(near)
        if self.show_skyline and len(structures.centres):
            offset = structures.centres - centre
            gap = np.hypot(offset[:, 0], offset[:, 1])
            visible = ((gap > reach) & (gap < self.SKYLINE_REACH)
                       & (structures.heights > gap * _SKYLINE_ANGLE))
            # Tallest first, and capped: past a few hundred the silhouette
            # stops changing and the render time does not.
            far = np.nonzero(visible)[0]
            far = far[np.argsort(-structures.heights[far])][:400]
            chosen.extend(far.tolist())

        parts = []
        for index in chosen:
            gap = float(np.linalg.norm(structures.centres[index] - centre))
            solid = self._extrude(structures, index, terrain, gap)
            if solid is not None:
                parts.append(solid)
        if not parts:
            self._structure_key, self._structure_mesh = key, None
            return None, origin
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        self._structure_key, self._structure_mesh = key, merged
        return merged, origin

    def canopy_polydata(self, t: float):
        """Mapped trees within sight, as crowns on the bank.

        Individual OSM trees only -- the park polygons are the better
        record of where canopy *is*, but they say nothing about where the
        trunks are, and a park drawn as a solid green slab looks worse
        than no trees at all.
        """
        pv = require_pyvista()
        state = self.state_at(t)
        centre = np.asarray(state.position, dtype=float)[:2]
        try:
            structures = self.structures()
            terrain = self.terrain()
        except Exception:
            return None
        if not len(structures.trees):
            return None
        offset = structures.trees - centre
        within = np.nonzero(np.einsum("ij,ij->i", offset, offset)
                            < (self.window * 1.2) ** 2)[0]
        crowns = []
        for index in within[:220]:
            east, north = structures.trees[index]
            height = float(structures.tree_heights[index])
            base = float(terrain.at(east, north)[0])
            crowns.append(pv.Sphere(radius=0.30 * height,
                                    center=(east, north,
                                            base + 0.72 * height),
                                    theta_resolution=10, phi_resolution=8))
        if not crowns:
            return None
        return crowns[0].merge(crowns[1:]) if len(crowns) > 1 else crowns[0]

    _structure_key = None
    _structure_mesh = None

    def path_polydata(self, t: float):
        """The planned line, drawn on the water ahead."""
        pv = require_pyvista()
        if self.path is None:
            return None
        state = self.state_at(t)
        origin = self._origin(state)
        centre = np.asarray(state.position, dtype=float)[:2]
        near = np.linalg.norm(self.path - centre, axis=1) < self.window * 1.5
        if near.sum() < 2:
            return None
        points = self.path[near]
        return pv.lines_from_points(np.column_stack([
            points[:, 0] - origin[0], points[:, 1] - origin[1],
            np.full(len(points), self.water_level + 0.05)]))

    # -- camera -----------------------------------------------------------
    def cox_camera(self, t: float):
        """Camera at the coxswain's head, looking up the boat.

        Placed at the rig's own coxswain position rather than a guessed
        one, so the eye height and the seat are the ones the mass matrix
        carries.  A stern-loaded eight puts the coxswain aft facing the
        bow, so the view runs forward over eight backs.
        """
        state = self.state_at(t)
        rotation = hull_to_abs(state.attitude)
        seat = np.asarray(self.boat.rig.coxswain_position, dtype=float).copy()
        seat[2] += self.EYE_HEIGHT

        origin = self._origin(state)
        eye = rotation @ seat + state.position - origin
        # Look up the boat, slightly down onto the water ahead.
        forward = rotation @ np.array([1.0, 0.0, 0.0])
        focus = eye + forward * 60.0 - np.array([0.0, 0.0, 4.0])
        up = rotation @ np.array([0.0, 0.0, 1.0])
        return [tuple(eye), tuple(focus), tuple(up)]

    def _camera(self, plotter, view: str = "iso", zoom: float = 1.0):
        if view == "cox":
            plotter.camera_position = self.cox_camera(self._camera_time)
            plotter.camera.view_angle = 70.0     # roughly human, seated
            # VTK derives the clipping range from the visible bounds only
            # when the camera is reset, and this camera is placed by hand.
            # Left alone the far plane lands a few hundred metres out and
            # silently removes the skyline -- the geometry is built, sent
            # and then clipped, which looks exactly like not having built
            # it.  The near plane stays small because the stroke coach is
            # about a metre from the eye.
            plotter.camera.clipping_range = (0.35, self.SKYLINE_REACH * 1.6)
            return
        if view == "plan":
            state = self.state_at(self._camera_time)
            origin = self._origin(state)
            here = np.asarray(state.position, dtype=float) - origin
            eye = np.array([here[0], here[1], self.PLAN_ALTITUDE])
            # North up, so the picture matches the charts.
            plotter.camera_position = [tuple(eye),
                                       (here[0], here[1], 0.0),
                                       (0.0, 1.0, 0.0)]
            plotter.camera.view_angle = 42.0
            return
        if view == "chase3d":
            state = self.state_at(self._camera_time)
            rotation = hull_to_abs(state.attitude)
            back = rotation @ np.array([-38.0, 0.0, 14.0])
            origin = self._origin(state)
            eye = back + state.position - origin
            focus = state.position - origin
            plotter.camera_position = [tuple(eye), tuple(focus),
                                       (0.0, 0.0, 1.0)]
            plotter.camera.clipping_range = (1.0, self.SKYLINE_REACH * 1.6)
            return
        super()._camera(plotter, view=view, zoom=zoom)

    # -- assembly ---------------------------------------------------------
    def _build(self, plotter, t: float, show_forces=False, simulator=None):
        """Draw the boat, then the river around it.

        The base scene is drawn first so the boat, crew and oars keep the
        colouring and the wet/dry panel test they already had -- this adds
        scenery, it does not reinterpret the boat.
        """
        self._camera_time = float(t)
        super()._build(plotter, t, show_forces, simulator)
        # The base scene draws an unbounded water plane at z = 0.  The
        # terrain below carries the real water surface *and* the bank, so
        # the plane is redundant and z-fights with it.
        try:
            plotter.remove_actor("water", render=False)
        except Exception:
            pass

        # The base scene is lit for a boat on a dark ground.  A river seen
        # from the water wants a sky and a low sun, or the far bank is
        # indistinguishable from the night.
        try:
            plotter.set_background(_SKY_TOP, top=_SKY_HORIZON)
        except TypeError:
            plotter.set_background(_SKY_HORIZON)
        try:
            import pyvista as _pv
            plotter.add_light(_pv.Light(position=(-400.0, -250.0, 180.0),
                                        focal_point=(0.0, 0.0, 0.0),
                                        color="white", intensity=0.55))
        except Exception:
            pass

        if self.show_skyline:
            distant = self.distant_polydata()
            if distant:
                state = self.state_at(t)
                shifted = distant.copy()
                shifted.points = shifted.points - np.array(
                    [self._origin(state)[0], self._origin(state)[1], 0.0])
                plotter.add_mesh(shifted, scalars="land",
                                 cmap=[_WATER, _BANK], show_scalar_bar=False,
                                 name="distant", smooth_shading=True,
                                 ambient=0.45, diffuse=0.60, specular=0.0)

        terrain = self.terrain_polydata(t)
        if terrain is not None:
            plotter.add_mesh(terrain, scalars="land", cmap=[_WATER, _BANK],
                             show_scalar_bar=False, name="terrain",
                             smooth_shading=True, opacity=1.0,
                             ambient=0.35, diffuse=0.75, specular=0.05)
        if self.show_structures:
            solids, origin = self.structure_polydata(t)
            if solids is not None:
                shifted = solids.copy()
                shifted.points = shifted.points - np.array(
                    [origin[0], origin[1], 0.0])
                if "rgb" in shifted.cell_data:
                    plotter.add_mesh(shifted, scalars="rgb", rgb=True,
                                     name="buildings", ambient=0.42,
                                     diffuse=0.70, specular=0.04)
                else:
                    plotter.add_mesh(shifted, color=_WALL, name="buildings",
                                     ambient=0.30, diffuse=0.80,
                                     specular=0.02)
            crowns = self.canopy_polydata(t)
            if crowns is not None:
                shifted = crowns.copy()
                shifted.points = shifted.points - np.array(
                    [origin[0], origin[1], 0.0])
                plotter.add_mesh(shifted, color=_CANOPY, name="trees",
                                 ambient=0.35, diffuse=0.70, specular=0.0)
        self.bridge_actors(plotter, t)
        if self.show_skyline:
            self.span_actors(plotter, t)
        line = self.path_polydata(t)
        if line is not None:
            plotter.add_mesh(line, color=_LINE, line_width=4,
                             name="planned-line")

    _camera_time = 0.0
