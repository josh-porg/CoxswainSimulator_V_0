"""Interactive and recorded 3-D views of a simulated boat.

Built on PyVista, which is imported lazily so the rest of the package
does not depend on VTK.

What is drawn
-------------
Everything is drawn from the *same* geometry the dynamics use, so a
disagreement between the picture and the physics is impossible by
construction -- the hull surface is :attr:`HullMesh.corners`, the rowers
are the joint chain that produces the segment masses, the oarlocks are
the rig's, and the panel colouring is the same submersion test that
computes buoyancy.  That is what makes this a verification tool rather
than an illustration.

* **hull** -- the panelled surface, coloured by whether each panel is
  wet or dry at this instant, which shows heave and trim directly.
* **water** -- a still plane at the water level.
* **crew** -- one polyline per rower through ankle, knee, hip, shoulder,
  elbow and hand, plus spheres at the 12 segment centres of mass sized
  by mass.  If the rowers look folded the wrong way, it is visible.
* **oars** -- handle to oarlock to blade, swept by the oar angle model,
  with the blade coloured while it is buried.
* **force arrows** -- optional; the net force and each named source,
  scaled to a readable length.

Usage
-----
    from coxswain.viz.scene3d import BoatScene
    scene = BoatScene(boat, result)
    scene.show()                     # interactive, scrub with the slider
    scene.write_movie("run.mp4")     # or record it
    scene.snapshot(t=2.0, path="frame.png")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.frames import hull_to_abs
from ..core.state import State

__all__ = ["BoatScene", "SceneStyle", "require_pyvista"]


def require_pyvista():
    """Import PyVista with a message that says how to get it."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "3-D visualisation needs PyVista. Install it with:\n"
            "    pip install pyvista\n"
            "The 2-D charts in coxswain.viz.plots need only matplotlib."
        ) from exc
    return pv


@dataclass
class SceneStyle:
    """Colours and sizes, gathered so a view can be restyled in one place."""

    hull_wet: str = "#2f6f9f"
    hull_dry: str = "#d8dee4"
    water: str = "#2a7fa8"
    water_opacity: float = 0.22
    crew: str = "#e08c2a"
    segment: str = "#b5651d"
    oar_shaft: str = "#3a3a3a"
    blade_buried: str = "#1f7a3a"
    blade_free: str = "#9aa5ad"
    background: str = "#101418"
    force_arrow: str = "#e03b3b"
    segment_scale: float = 0.10
    oar_width: float = 5.0
    crew_width: float = 6.0
    water_extent: float = 14.0


#: Bones of the seated coxswain figure, as index pairs into
#: :func:`_coxswain_figure`.
_COXSWAIN_BONES = ((0, 1), (1, 2), (2, 3), (1, 4), (1, 5))


def _coxswain_figure(seat: np.ndarray):
    """A minimal seated figure at the coxswain's seat, in hull coordinates.

    The coxswain is a real 55 kg of a ~855 kg eight sitting 6 m off the
    centre of mass, and the dynamics has always carried that mass.  Drawing
    them keeps the picture honest about where it sits -- and makes it
    obvious at a glance whether a stern-loaded boat is trimmed down by the
    bow or the stern.
    """
    x, y, z = seat
    return [
        np.array([x, y, z - 0.15]),          # 0 seat
        np.array([x - 0.05, y, z + 0.30]),   # 1 shoulders
        np.array([x - 0.07, y, z + 0.45]),   # 2 neck
        np.array([x - 0.08, y, z + 0.58]),   # 3 head
        np.array([x + 0.45, y + 0.12, z]),   # 4 legs, port
        np.array([x + 0.45, y - 0.12, z]),   # 5 legs, starboard
    ]


class BoatScene:
    """A 3-D scene of one boat over one simulation run.

    Parameters
    ----------
    boat:
        The boat that was simulated -- supplies all geometry.
    result:
        A :class:`~coxswain.sim.results.SimulationResult`.  If omitted,
        the boat is drawn in static trim, which is useful for checking a
        new hull or rig before running anything.
    water_level:
        Still-water height, matching the simulator's.
    follow:
        Draw in a frame that travels with the boat.  Almost always what
        you want: a shell covers 100 m in 20 s, which would otherwise
        shrink to a dot.
    """

    def __init__(self, boat, result=None, water_level: float = 0.0,
                 follow: bool = True, style: Optional[SceneStyle] = None):
        self.boat = boat
        self.result = result
        self.water_level = float(water_level)
        self.follow = bool(follow)
        self.style = style or SceneStyle()
        self._plotter = None
        self._actors = {}

    # -- state lookup -----------------------------------------------------
    @property
    def duration(self) -> float:
        if self.result is None:
            return self.boat.timing.period
        return float(self.result.time[-1])

    def state_at(self, t: float) -> State:
        """Interpolated state at time ``t``; static trim if there is no run."""
        if self.result is None:
            heave, pitch = self.boat.trim_attitude(t)
            return State.create(position=[0.0, 0.0, heave],
                                attitude=[0.0, pitch, 0.0])

        time = self.result.time
        t = float(np.clip(t, time[0], time[-1]))
        vector = np.array([np.interp(t, time, self.result.states[row])
                           for row in range(self.result.states.shape[0])])
        return State.from_vector(vector)

    def _origin(self, state: State) -> np.ndarray:
        """Where to put the camera/scene origin, so the boat stays framed."""
        if not self.follow:
            return np.zeros(3)
        return np.array([state.position[0], state.position[1], 0.0])

    # -- geometry builders -------------------------------------------------
    def deck_polydata(self, t: float):
        """The foredeck and stern deck, as a surface over the hull.

        **Rendering only.**  It adds no buoyancy, no wetted area and no
        mass; the physics mesh is untouched, because a deck that is
        always above the waterline changes none of them.

        A racing shell is not an open trough.  It is decked fore and aft
        with a cockpit cut out of the middle, and a bow-loaded four is
        decked *over the coxswain* as well, leaving only enough opening
        for their head and shoulders.  Without this the hull renders as
        an open shell, and from the one viewpoint that matters -- a
        bow-loader's eye, a hand's breadth above the deck -- you look
        straight down inside the boat and the frame fills with two pale
        walls that are the inside of the hull.

        The cockpit runs from a little behind the sternmost seat to a
        little ahead of the bowmost one, which is where the slides are.
        A bow-loading coxswain sits outside that, under their own deck,
        with a separate opening.
        """
        pv = require_pyvista()
        offsets = self.boat.offsets
        station = np.asarray(offsets.station, dtype=float)
        beam = np.asarray(offsets.beam, dtype=float)
        freeboard = float(getattr(offsets, "freeboard", 0.25))

        seats = np.array([seat.station_x for seat in self.boat.rig.seats],
                         dtype=float)
        if not len(seats):
            return None
        # The slides run about a metre either way from a seat's own
        # station, so the cockpit is wider than the seats themselves.
        open_from = float(seats.min()) - 1.10
        open_to = float(seats.max()) + 0.90

        rig = self.boat.rig
        cockpit = []
        if rig.has_coxswain:
            cox = float(rig.coxswain_position[0])
            if cox > open_to or cox < open_from:
                # Bow-loaded (or stern-loaded) coxswain: their own hole,
                # big enough for a head and shoulders and no bigger.
                cockpit.append((cox - 0.55, cox + 0.35))
        cockpit.append((open_from, open_to))

        def decked(x):
            return not any(low <= x <= high for low, high in cockpit)

        faces, points = [], []
        fine = np.linspace(station[0], station[-1], 121)
        width = np.interp(fine, station, beam)
        for index in range(len(fine) - 1):
            middle = 0.5 * (fine[index] + fine[index + 1])
            if not decked(middle):
                continue
            # Deck edge is slightly inboard of the waterline beam: the
            # topsides tumble home a little on a racing shell.
            a, b = 0.46 * width[index], 0.46 * width[index + 1]
            base = len(points)
            points.extend([
                (fine[index], -a, freeboard), (fine[index], a, freeboard),
                (fine[index + 1], b, freeboard),
                (fine[index + 1], -b, freeboard)])
            faces.extend([4, base, base + 1, base + 2, base + 3])
        if not faces:
            return None
        return pv.PolyData(np.asarray(points, dtype=float),
                           faces=np.asarray(faces, dtype=int))

    def hull_polydata(self, t: float):
        """Hull surface at time ``t``, with a ``wet`` cell array."""
        pv = require_pyvista()
        state = self.state_at(t)
        rot = hull_to_abs(state.attitude)
        corners = self.boat.mesh.corners           # (n_panels, 4, 3)

        points = (corners.reshape(-1, 3) @ rot.T
                  + state.position - self._origin(state))
        n_panels = corners.shape[0]
        faces = np.hstack([
            np.full((n_panels, 1), 4, dtype=np.int64),
            np.arange(n_panels * 4, dtype=np.int64).reshape(n_panels, 4),
        ]).ravel()

        surface = pv.PolyData(points, faces)
        centroid_abs = (self.boat.mesh.centroid @ rot.T + state.position)
        surface.cell_data["wet"] = (
            centroid_abs[:, 2] < self.water_level).astype(np.int8)
        return surface

    def crew_polydata(self, t: float, include_coxswain: bool = True):
        """One polyline per rower through the joint chain, plus mass blobs.

        Returns ``(skeleton, centres, masses)``.  With ``include_coxswain``
        the blobs are exactly the point-mass cloud that
        :meth:`Boat.crew_field` hands to the mass matrix -- coxswain
        included -- so what is drawn is what is simulated.  A coxswain is
        55-ish kg of a 855 kg boat sitting well off the centre of mass;
        leaving them out of the picture while the dynamics carries them is
        precisely the kind of quiet divergence this module exists to
        prevent.
        """
        pv = require_pyvista()
        state = self.state_at(t)
        rot = hull_to_abs(state.attitude)
        shift = state.position - self._origin(state)

        lines, points = [], []

        for member in self.boat.crew:
            rower = member.rower
            joints = rower.skeleton(t)
            index = {}
            for name, local in joints.items():
                index[name] = len(points)
                points.append(local @ rot.T + shift)
            # Two legs, two arms, and a shoulder line: a sweep rower has
            # both hands on one handle, so the arms are neither symmetric
            # nor on the centreline and cannot be drawn as one chain.
            for start, end in rower.BONES:
                lines.append([2, index[start], index[end]])

        if self.boat.rig.has_coxswain:
            seat = np.asarray(self.boat.rig.coxswain_position, dtype=float)
            base = len(points)
            for local in _coxswain_figure(seat):
                points.append(local @ rot.T + shift)
            for start, end in _COXSWAIN_BONES:
                lines.append([2, base + start, base + end])

        skeleton = pv.PolyData()
        if points:
            skeleton = pv.PolyData(np.array(points))
            skeleton.lines = np.hstack([np.array(line) for line in lines])

        if include_coxswain:
            masses, positions, _, _ = self.boat.crew_field(t)
        else:
            masses, positions = [], []
            for member in self.boat.crew:
                position, _, _ = member.rower.segment_state(t)
                positions.append(position)
                masses.append(member.rower.segment_masses)
            masses = np.concatenate(masses) if masses else np.zeros(0)
            positions = (np.vstack(positions) if len(positions)
                         else np.zeros((0, 3)))

        centres = positions @ rot.T + shift if len(positions) \
            else np.zeros((0, 3))
        return skeleton, centres, np.asarray(masses)

    def oar_polydata(self, t: float):
        """Oars as handle-oarlock-blade polylines, with a ``buried`` flag."""
        pv = require_pyvista()
        from ..crew.oarlock import blade_position, handle_position

        state = self.state_at(t)
        rot = hull_to_abs(state.attitude)
        shift = state.position - self._origin(state)

        points, lines, buried = [], [], []
        drive = self.boat.timing.is_drive(t)

        for seat in self.boat.rig.seats:
            for lock in seat.oarlocks:
                handle = handle_position(t, self.boat.timing, lock,
                                         self.boat.oar_sweep)
                blade = blade_position(t, self.boat.timing, lock,
                                       self.boat.oar_sweep)
                base = len(points)
                for node in (handle, lock.position, blade):
                    points.append(node @ rot.T + shift)
                lines.append([3, base, base + 1, base + 2])
                buried.extend([bool(drive)] * 3)

        if not points:
            return pv.PolyData()
        oars = pv.PolyData(np.array(points))
        oars.lines = np.hstack([np.array(line) for line in lines])
        oars.point_data["buried"] = np.array(buried, dtype=np.int8)
        return oars

    def water_polydata(self, t: float):
        """A still plane, sized to the boat and centred under it."""
        pv = require_pyvista()
        state = self.state_at(t)
        origin = self._origin(state)
        extent = self.boat.length + self.style.water_extent
        return pv.Plane(
            center=(origin[0] - state.position[0] if self.follow else 0.0,
                    0.0, self.water_level),
            direction=(0.0, 0.0, 1.0),
            i_size=extent, j_size=extent * 0.6,
            i_resolution=1, j_resolution=1,
        )

    # -- rendering ---------------------------------------------------------
    def _build(self, plotter, t: float, show_forces=False, simulator=None):
        style = self.style

        try:
            plotter.enable_depth_peeling(number_of_peels=8)
        except Exception:  # pragma: no cover - driver dependent
            pass
        plotter.add_mesh(self.water_polydata(t), color=style.water,
                         opacity=style.water_opacity, name="water",
                         show_edges=False, lighting=False)

        hull = self.hull_polydata(t)
        plotter.add_mesh(hull, scalars="wet", name="hull",
                         cmap=[style.hull_dry, style.hull_wet],
                         show_scalar_bar=False, show_edges=True,
                         edge_color="#0b1a26", line_width=0.4,
                         smooth_shading=False)

        deck = self.deck_polydata(t)
        if deck is not None:
            state = self.state_at(t)
            rotation = hull_to_abs(state.attitude)
            moved = deck.copy()
            moved.points = (np.asarray(moved.points) @ rotation.T
                            + state.position - self._origin(state))
            plotter.add_mesh(moved, color=style.hull_dry, name="deck",
                             show_edges=True, edge_color="#0b1a26",
                             line_width=0.4, smooth_shading=False)

        skeleton, centres, masses = self.crew_polydata(t)
        if skeleton.n_points:
            plotter.add_mesh(skeleton, color=style.crew,
                             line_width=style.crew_width, name="crew",
                             render_lines_as_tubes=True)
        if len(centres):
            pv = require_pyvista()
            blobs = pv.PolyData(centres)
            # Radius proportional to the cube root of mass, so the spheres
            # read as equal *density* blobs and a heavy trunk does not
            # swamp the limbs visually.
            blobs.point_data["radius"] = (
                style.segment_scale * (masses / masses.max()) ** (1.0 / 3.0))
            plotter.add_mesh(
                blobs.glyph(scale="radius", geom=pv.Sphere(radius=1.0),
                            orient=False),
                color=style.segment, name="segments", smooth_shading=True)

        oars = self.oar_polydata(t)
        if oars.n_points:
            colour = (style.blade_buried if self.boat.timing.is_drive(t)
                      else style.blade_free)
            plotter.add_mesh(oars, color=colour, line_width=style.oar_width,
                             name="oars", render_lines_as_tubes=True)

        if show_forces and simulator is not None:
            self._add_force_arrows(plotter, t, simulator)

        return plotter

    def _add_force_arrows(self, plotter, t: float, simulator, scale=None):
        """Draw each force source as an arrow from the hull centre of mass."""
        pv = require_pyvista()
        state = self.state_at(t)
        origin = state.position - self._origin(state)
        breakdown = simulator.breakdown(t, state)

        sources = {
            "crew": breakdown.crew_force,
            "oar": breakdown.oar_force,
            "resistance": breakdown.resistance_force,
            "net": breakdown.total_force(),
        }
        largest = max(np.linalg.norm(v) for v in sources.values()) or 1.0
        scale = scale or (0.35 * self.boat.length / largest)

        for index, (name, vector) in enumerate(sources.items()):
            if np.linalg.norm(vector) < 1e-6:
                continue
            arrow = pv.Arrow(start=origin + np.array([0.0, 0.0, 0.6 + 0.25 * index]),
                             direction=vector, scale=np.linalg.norm(vector) * scale)
            plotter.add_mesh(arrow, color=self.style.force_arrow,
                             name=f"force_{name}", opacity=0.85)

    #: Named camera presets.  ``side`` and ``stern`` are the diagnostic
    #: ones: a shell's heave, trim and rower posture are all judged from
    #: the side, and roll and blade depth from astern.  ``iso`` is for
    #: getting your bearings.
    VIEWS = ("iso", "side", "stern", "top", "bow_quarter")

    def _camera(self, plotter, view: str = "iso", zoom: float = 1.0):
        if view not in self.VIEWS:
            raise ValueError(f"view must be one of {self.VIEWS}, got {view!r}")
        length = self.boat.length

        placements = {
            "iso": ((-0.9, -0.75, 0.45), (0.0, 0.0, 1.0)),
            # tight on the crew, level with the waterline: this is the view
            # that shows whether the rowers are folding correctly
            "side": ((0.0, -0.62, 0.05), (0.0, 0.0, 1.0)),
            "stern": ((-0.85, 0.0, 0.10), (0.0, 0.0, 1.0)),
            "top": ((0.0, -0.02, 0.95), (1.0, 0.0, 0.0)),
            "bow_quarter": ((0.75, -0.45, 0.18), (0.0, 0.0, 1.0)),
        }
        offset, up = placements[view]
        position = tuple(component * length / zoom for component in offset)
        plotter.camera_position = [position, (0.0, 0.0, 0.0), up]

    def show(self, t: float = None, show_forces: bool = False,
             simulator=None, scrub: bool = True, view: str = "iso",
             zoom: float = 1.0, **kwargs):
        """Open an interactive window.

        With a run loaded and ``scrub`` set, a time slider rebuilds the
        scene as it moves, so a whole stroke can be stepped through by
        hand -- which is the fastest way to spot a rower folding the wrong
        way or a blade entering at the wrong moment.
        """
        pv = require_pyvista()
        plotter = pv.Plotter(**kwargs)
        plotter.set_background(self.style.background)

        start = 0.0 if t is None else t
        self._build(plotter, start, show_forces, simulator)
        self._camera(plotter, view, zoom)

        if scrub and self.result is not None and t is None:
            def update(value):
                camera = plotter.camera_position
                plotter.clear()
                self._build(plotter, float(value), show_forces, simulator)
                plotter.camera_position = camera

            plotter.add_slider_widget(
                update, [float(self.result.time[0]), float(self.result.time[-1])],
                value=float(self.result.time[0]), title="time [s]",
                style="modern",
            )

        plotter.add_axes()
        plotter.show()
        return plotter

    def snapshot(self, t: float, path: str, window_size=(1400, 900),
                 show_forces: bool = False, simulator=None,
                 view: str = "iso", zoom: float = 1.0, axes: bool = True):
        """Render one frame straight to a PNG, no display needed."""
        pv = require_pyvista()
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background(self.style.background)
        self._build(plotter, t, show_forces, simulator)
        self._camera(plotter, view, zoom)
        if axes:
            plotter.add_axes()
        plotter.screenshot(path)
        plotter.close()
        return path

    def contact_sheet(self, path: str, n_frames: int = 4, view: str = "side",
                      zoom: float = 1.0, t_start: float = None,
                      window_size=(1500, 420), simulator=None,
                      show_forces: bool = False):
        """One image with several instants of a stroke side by side.

        The fastest visual check there is: if the rowers are folding the
        wrong way, or a blade is buried on the recovery, it shows up
        immediately across the sequence.
        """
        pv = require_pyvista()
        timing = self.boat.timing
        period = timing.period

        if t_start is None:
            # Snap to the last catch in the run, so frame 0 really is the
            # catch and the phase labels mean what they say.  Taking
            # `time[-1] - period` instead lands at an arbitrary phase and
            # silently mislabels every frame.
            if self.result is None:
                t_start = 0.0
            else:
                last = float(self.result.time[-1])
                t_start = last - float(timing.phase(last)) * period - period
                t_start = max(float(self.result.time[0]), t_start)

        plotter = pv.Plotter(off_screen=True, shape=(1, n_frames),
                             window_size=window_size, border=False)
        for index in range(n_frames):
            plotter.subplot(0, index)
            plotter.set_background(self.style.background)
            t = t_start + period * index / n_frames
            self._build(plotter, float(t), show_forces, simulator)
            self._camera(plotter, view, zoom)
            phase = float(timing.phase(t))
            label = "drive" if phase < timing.drive_fraction else "recovery"
            plotter.add_text(f"phase {phase:.2f}  {label}", font_size=9,
                             color="w")
        plotter.screenshot(path)
        plotter.close()
        return path

    def write_movie(self, path: str, n_frames: int = 120,
                    t_start: float = None, t_end: float = None,
                    framerate: int = 24, window_size=(1400, 900),
                    show_forces: bool = False, simulator=None,
                    view: str = "iso", zoom: float = 1.0):
        """Record the run to an ``.mp4`` (or ``.gif``).

        Defaults to the last two stroke cycles, which is what you want for
        checking technique -- the opening transient is rarely interesting.
        """
        pv = require_pyvista()

        if self.result is None:
            t_start = 0.0 if t_start is None else t_start
            t_end = self.boat.timing.period if t_end is None else t_end
        else:
            span = 2.0 * self.boat.timing.period
            t_end = float(self.result.time[-1]) if t_end is None else t_end
            t_start = max(float(self.result.time[0]),
                          t_end - span) if t_start is None else t_start

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background(self.style.background)

        if str(path).lower().endswith(".gif"):
            plotter.open_gif(path, fps=framerate)
        else:
            plotter.open_movie(path, framerate=framerate)

        from ..progress import progress
        for t in progress(np.linspace(t_start, t_end, n_frames),
                          total=n_frames, desc="  rendering frames",
                          unit="frame"):
            plotter.clear()
            self._build(plotter, float(t), show_forces, simulator)
            self._camera(plotter, view, zoom)
            plotter.write_frame()

        plotter.close()
        return path
