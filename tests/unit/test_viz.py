"""Tests for the visualisation layer.

The geometry builders are tested hard, because their whole value is that
they draw the *same* geometry the dynamics use -- a picture that disagrees
with the physics is worse than no picture.  The rendering calls themselves
are smoke-tested behind a ``pyvista`` skip, since they need VTK and a
working off-screen GL context.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from coxswain.core.frames import hull_to_abs           # noqa: E402
from coxswain.crew.oarlock import (                    # noqa: E402
    OarAngleSweep,
    blade_position,
    handle_position,
    oar_axis,
)
from coxswain.viz import plots                         # noqa: E402

pyvista = pytest.importorskip("pyvista", reason="3-D scene needs PyVista")

from coxswain.viz.scene3d import BoatScene, SceneStyle  # noqa: E402


@pytest.fixture(scope="module")
def run(eight):
    from coxswain.sim.simulator import RowingSimulator
    simulator = RowingSimulator(eight)
    return simulator, simulator.run(duration=4.0, surge_speed=5.0, dt=0.008)


@pytest.fixture(scope="module")
def scene(eight, run):
    return BoatScene(eight, run[1])


# ==========================================================================
# oar geometry
# ==========================================================================
def test_oar_axis_is_a_unit_vector(eight):
    for t in np.linspace(0, eight.timing.period, 20):
        axis = oar_axis(t, eight.timing, 1)
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-12)


def test_oar_axis_is_horizontal(eight):
    """The shaft model keeps the oar in the hull's horizontal plane."""
    axis = oar_axis(0.3, eight.timing, 1)
    assert axis[2] == pytest.approx(0.0, abs=1e-15)


def test_port_and_starboard_oars_reach_opposite_ways(eight):
    port = oar_axis(0.3, eight.timing, 1)
    starboard = oar_axis(0.3, eight.timing, -1)
    assert port[1] == pytest.approx(-starboard[1])
    assert port[0] == pytest.approx(starboard[0])


def test_blade_is_bow_ward_of_its_oarlock_at_the_catch(eight):
    """Catch angle is positive, so the blade starts towards the bow."""
    lock = eight.rig.seats[0].oarlocks[0]
    blade = blade_position(0.0, eight.timing, lock)
    assert blade[0] > lock.position[0]


def test_blade_sweeps_sternward_through_the_drive(eight):
    """It is that sweep that pushes water aft and the boat forward."""
    lock = eight.rig.seats[0].oarlocks[0]
    at_catch = blade_position(0.0, eight.timing, lock)[0]
    at_finish = blade_position(eight.timing.drive_duration,
                               eight.timing, lock)[0]
    assert at_finish < at_catch


def test_handle_is_on_the_opposite_side_of_the_oarlock_from_the_blade(eight):
    lock = eight.rig.seats[0].oarlocks[0]
    for t in (0.0, 0.3, 0.9):
        handle = handle_position(t, eight.timing, lock)
        blade = blade_position(t, eight.timing, lock)
        to_handle = handle - lock.position
        to_blade = blade - lock.position
        assert to_handle.dot(to_blade) < 0.0


def test_oar_length_is_conserved_over_the_stroke(eight):
    """Handle to blade must always be the oar's full length."""
    lock = eight.rig.seats[0].oarlocks[0]
    for t in np.linspace(0, eight.timing.period, 25):
        span = np.linalg.norm(blade_position(t, eight.timing, lock)
                              - handle_position(t, eight.timing, lock))
        assert span == pytest.approx(lock.oar.length, abs=1e-9)


def test_handle_distance_from_the_oarlock_is_the_inboard(eight):
    lock = eight.rig.seats[0].oarlocks[0]
    handle = handle_position(0.4, eight.timing, lock)
    assert np.linalg.norm(handle - lock.position) == pytest.approx(
        lock.oar.inboard, abs=1e-9)


def test_the_oar_sweep_is_periodic(eight):
    lock = eight.rig.seats[0].oarlocks[0]
    np.testing.assert_allclose(blade_position(0.0, eight.timing, lock),
                               blade_position(eight.timing.period,
                                              eight.timing, lock), atol=1e-9)


def test_custom_sweep_widens_the_arc(eight):
    lock = eight.rig.seats[0].oarlocks[0]
    wide = OarAngleSweep(catch_angle=np.radians(70.0),
                         finish_angle=np.radians(-45.0))
    default_span = abs(blade_position(0.0, eight.timing, lock)[0]
                       - blade_position(eight.timing.drive_duration,
                                        eight.timing, lock)[0])
    wide_span = abs(blade_position(0.0, eight.timing, lock, wide)[0]
                    - blade_position(eight.timing.drive_duration,
                                     eight.timing, lock, wide)[0])
    assert wide_span > default_span


# ==========================================================================
# phase averaging
# ==========================================================================
def test_phase_average_recovers_a_known_waveform():
    period = 2.0
    time = np.linspace(0.0, 20.0, 4000)
    signal = np.sin(2 * np.pi * time / period)
    phase, mean, _ = plots.phase_average(time, signal, period, n_bins=40)
    np.testing.assert_allclose(mean, np.sin(2 * np.pi * phase), atol=0.05)


def test_phase_average_spread_is_only_the_finite_bin_width():
    """For an exactly periodic signal the only scatter is within-bin.

    A bin spans ``2 pi / n_bins`` of phase, over which a unit sinusoid
    varies by at most that much, giving a standard deviation of about
    ``(2 pi / n_bins) / sqrt(12)``.  Anything materially larger would mean
    the binning is smearing cycles together.
    """
    period = 2.0
    time = np.linspace(0.0, 20.0, 4000)
    n_bins = 30
    _, _, deviation = plots.phase_average(
        time, np.cos(2 * np.pi * time / period), period, n_bins=n_bins)
    bound = 1.5 * (2 * np.pi / n_bins) / np.sqrt(12.0)
    assert np.nanmax(deviation) < bound


def test_phase_average_bins_span_the_unit_interval():
    time = np.linspace(0.0, 10.0, 500)
    phase, _, _ = plots.phase_average(time, time, 2.0, n_bins=25)
    assert phase.min() > 0.0 and phase.max() < 1.0
    assert len(phase) == 25


def test_phase_average_falls_back_when_the_run_is_short():
    """A run shorter than the skipped transient must still return values."""
    time = np.linspace(0.0, 0.5, 60)
    _, mean, _ = plots.phase_average(time, np.ones_like(time), 2.0, n_bins=10)
    assert np.isfinite(mean).any()


# ==========================================================================
# scene geometry
# ==========================================================================
def test_hull_polydata_has_one_quad_per_panel(eight, scene):
    surface = scene.hull_polydata(0.5)
    assert surface.n_cells == eight.mesh.n_panels
    assert surface.n_points == eight.mesh.n_panels * 4


def test_hull_panels_are_marked_wet_below_the_waterline(scene):
    surface = scene.hull_polydata(0.5)
    wet = surface.cell_data["wet"]
    assert set(np.unique(wet)) <= {0, 1}
    assert 0 < wet.sum() < len(wet), "some panels wet, some dry"


def test_wet_marking_agrees_with_the_buoyancy_submersion_test(eight, scene):
    """The picture and the physics must use the same waterline.

    This is the test that makes the render a verification tool: the panels
    drawn as wet are exactly the panels the hydrostatics counts.
    """
    t = 0.5
    state = scene.state_at(t)
    rot = hull_to_abs(state.attitude)
    centroid_abs = eight.mesh.centroid @ rot.T + state.position
    expected = (centroid_abs[:, 2] < scene.water_level).astype(np.int8)
    np.testing.assert_array_equal(scene.hull_polydata(t).cell_data["wet"],
                                  expected)


def test_more_submerged_when_the_boat_sits_lower(eight):
    """Sanity: pushing the hull down must wet more panels."""
    from coxswain.sim.results import SimulationResult
    from coxswain.core.state import State

    def wet_count(heave):
        states = np.tile(
            State.create(position=[0.0, 0.0, heave]).to_vector()[:, None],
            (1, 2))
        result = SimulationResult(time=np.array([0.0, 1.0]), states=states,
                                  boat=eight)
        return BoatScene(eight, result).hull_polydata(0.0).cell_data["wet"].sum()

    assert wet_count(-0.05) > wet_count(0.05)


def test_crew_polydata_has_one_polyline_per_seat(eight, scene):
    skeleton, centres, masses = scene.crew_polydata(0.4)
    assert skeleton.n_lines == eight.n_seats
    assert skeleton.n_points == eight.n_seats * 6


def test_crew_segment_blobs_cover_every_segment(eight, scene):
    """12 segments per rower, plus one blob for the coxswain."""
    _, centres, masses = scene.crew_polydata(0.4, include_coxswain=False)
    assert centres.shape == (eight.n_seats * 12, 3)
    assert masses.shape == (eight.n_seats * 12,)
    assert masses.sum() == pytest.approx(eight.crew_mass, rel=1e-9)


def test_blobs_include_the_coxswain_by_default(eight, scene):
    _, centres, masses = scene.crew_polydata(0.4)
    assert len(masses) == eight.n_seats * 12 + 1
    assert masses.sum() == pytest.approx(
        eight.crew_mass + eight.coxswain_mass, rel=1e-9)


def test_crew_positions_match_the_dynamics_field(eight, scene):
    """The blobs drawn must be the point masses the mass matrix uses.

    Compared through the centre of mass, which catches both a missing
    body and a misplaced one.
    """
    t = 0.4
    _, centres, masses = scene.crew_polydata(t)
    total = (masses[:, None] * centres).sum(axis=0) / masses.sum()

    state = scene.state_at(t)
    rot = hull_to_abs(state.attitude)
    expected = (eight.crew_centre_of_mass(t) @ rot.T
                + state.position - scene._origin(state))
    np.testing.assert_allclose(total, expected, atol=1e-9)


def test_oar_polydata_has_one_polyline_per_oar(eight, scene):
    oars = scene.oar_polydata(0.4)
    assert oars.n_lines == eight.rig.n_oars
    assert oars.n_points == eight.rig.n_oars * 3


def test_water_plane_sits_at_the_water_level(scene):
    plane = scene.water_polydata(0.4)
    np.testing.assert_allclose(plane.points[:, 2], scene.water_level,
                               atol=1e-12)


def test_following_frame_keeps_the_boat_near_the_origin(eight, run):
    """Otherwise the boat leaves the frame within a couple of strokes."""
    _, result = run
    scene = BoatScene(eight, result, follow=True)
    late = scene.hull_polydata(float(result.time[-1]))
    assert abs(late.points[:, 0].mean()) < eight.length


def test_static_frame_lets_the_boat_travel(eight, run):
    _, result = run
    scene = BoatScene(eight, result, follow=False)
    start = scene.hull_polydata(float(result.time[0])).points[:, 0].mean()
    late = scene.hull_polydata(float(result.time[-1])).points[:, 0].mean()
    assert late - start > 5.0


def test_scene_without_a_run_draws_static_trim(eight):
    scene = BoatScene(eight, result=None)
    state = scene.state_at(0.0)
    heave, pitch = eight.trim_attitude(0.0)
    assert state.position[2] == pytest.approx(heave)
    assert state.pitch == pytest.approx(pitch)


def test_state_at_interpolates_between_samples(scene, run):
    _, result = run
    midpoint = 0.5 * (result.time[3] + result.time[4])
    interpolated = scene.state_at(midpoint).position[0]
    low, high = result.position[0][3], result.position[0][4]
    assert min(low, high) <= interpolated <= max(low, high)


def test_state_at_clamps_outside_the_run(scene, run):
    _, result = run
    beyond = scene.state_at(float(result.time[-1]) + 100.0)
    np.testing.assert_allclose(beyond.position,
                               result.position[:, -1], atol=1e-9)


def test_unknown_view_is_rejected(scene):
    plotter = pyvista.Plotter(off_screen=True)
    with pytest.raises(ValueError, match="view must be one of"):
        scene._camera(plotter, "underwater")
    plotter.close()


def test_scene_style_is_replaceable(eight, run):
    style = SceneStyle(water_opacity=0.9, segment_scale=0.2)
    scene = BoatScene(eight, run[1], style=style)
    assert scene.style.water_opacity == 0.9


# ==========================================================================
# rendering smoke tests
# ==========================================================================
@pytest.mark.slow
def test_snapshot_writes_a_png(scene, tmp_path):
    path = scene.snapshot(0.5, str(tmp_path / "frame.png"),
                          window_size=(320, 240))
    assert (tmp_path / "frame.png").stat().st_size > 1000


@pytest.mark.slow
@pytest.mark.parametrize("view", BoatScene.VIEWS)
def test_every_view_renders(scene, tmp_path, view):
    scene.snapshot(0.5, str(tmp_path / f"{view}.png"), view=view,
                   window_size=(320, 240))
    assert (tmp_path / f"{view}.png").exists()


@pytest.mark.slow
def test_contact_sheet_starts_at_a_catch(eight, scene, tmp_path, monkeypatch):
    """The phase labels are only honest if frame 0 really is the catch."""
    seen = []
    original = scene._build

    def record(plotter, t, *args, **kwargs):
        seen.append(t)
        return original(plotter, t, *args, **kwargs)

    monkeypatch.setattr(scene, "_build", record)
    scene.contact_sheet(str(tmp_path / "sheet.png"), n_frames=4,
                        window_size=(400, 160))

    phases = [float(eight.timing.phase(t)) for t in seen]
    assert phases[0] == pytest.approx(0.0, abs=1e-6)
    for index, phase in enumerate(phases):
        assert phase == pytest.approx(index / 4, abs=1e-6)


# ==========================================================================
# 2-D charts
# ==========================================================================
@pytest.mark.parametrize("function", [
    plots.trajectory, plots.speed_history, plots.rates,
])
def test_single_axis_charts_draw(eight, run, function):
    import matplotlib.pyplot as plt
    _, result = run
    axis = function(result, eight)
    assert axis.has_data()
    plt.close("all")


def test_stroke_cycle_draws(eight, run):
    import matplotlib.pyplot as plt
    _, result = run
    assert plots.stroke_cycle(result, eight).has_data()
    plt.close("all")


def test_secondary_motions_draws_three_panels(eight, run):
    import matplotlib.pyplot as plt
    _, result = run
    axes = plots.secondary_motions(result, eight)
    assert len(axes) == 3
    plt.close("all")


def test_crew_and_hull_draws(eight, run):
    import matplotlib.pyplot as plt
    _, result = run
    assert plots.crew_and_hull(result, eight).has_data()
    plt.close("all")


def test_recompute_breakdown_reproduces_the_net_force(eight, run):
    """Replaying forces from the stored states must be exact."""
    simulator, result = run
    forces, _, net = plots.recompute_breakdown(result, simulator)
    summed = sum(series for series in forces.values())
    np.testing.assert_allclose(summed, net[:, 0:3], atol=1e-8)


def test_dashboard_builds(eight, run):
    import matplotlib.pyplot as plt
    _, result = run
    figure = plots.dashboard(result, eight)
    assert len(figure.axes) >= 7
    plt.close(figure)


@pytest.mark.slow
def test_save_dashboard_writes_a_file(eight, run, tmp_path):
    simulator, result = run
    path = plots.save_dashboard(result, eight, str(tmp_path / "d.png"),
                                simulator=simulator, dpi=60)
    assert (tmp_path / "d.png").stat().st_size > 5000
