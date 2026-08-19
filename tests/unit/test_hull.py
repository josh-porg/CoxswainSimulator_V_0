"""Unit tests for hull geometry and the submerged-surface integrals."""

import numpy as np
import pytest

from coxswain.core.frames import attitude_from_components
from coxswain.hydro.hull import HullMesh, HullOffsets, parametric_offsets

RHO = 999.1
GRAVITY = 9.81


@pytest.fixture
def eight_offsets():
    return parametric_offsets(length=17.3, max_beam=0.57, max_depth=0.165,
                              fullness=2.6, freeboard=0.26)


@pytest.fixture
def mesh(eight_offsets):
    return HullMesh(eight_offsets)


# --------------------------------------------------------------------------
# offsets
# --------------------------------------------------------------------------
def test_offsets_report_their_dimensions(eight_offsets):
    assert eight_offsets.length == pytest.approx(17.3)
    assert eight_offsets.max_beam == pytest.approx(0.57)
    assert eight_offsets.max_depth == pytest.approx(0.165)


def test_parametric_offsets_taper_to_nothing_at_the_ends(eight_offsets):
    assert eight_offsets.beam[0] == pytest.approx(0.0, abs=1e-12)
    assert eight_offsets.beam[-1] == pytest.approx(0.0, abs=1e-12)
    assert eight_offsets.depth[0] == pytest.approx(0.0, abs=1e-12)


def test_parametric_offsets_peak_amidships(eight_offsets):
    peak = np.argmax(eight_offsets.beam)
    assert eight_offsets.station[peak] == pytest.approx(0.0, abs=0.5)


def test_fullness_controls_parallel_midbody():
    fine = parametric_offsets(17.3, 0.57, 0.165, fullness=1.5)
    full = parametric_offsets(17.3, 0.57, 0.165, fullness=6.0)
    assert full.design_volume() > fine.design_volume()


def test_design_volume_matches_semi_elliptical_sections(eight_offsets):
    expected = np.trapezoid(0.25 * np.pi * eight_offsets.beam
                            * eight_offsets.depth, eight_offsets.station)
    assert eight_offsets.design_volume() == pytest.approx(expected, rel=1e-12)


def test_design_displacement_scales_with_density(eight_offsets):
    assert (eight_offsets.design_displacement(2000.0)
            == pytest.approx(2.0 * eight_offsets.design_displacement(1000.0)))


@pytest.mark.parametrize("kwargs,message", [
    ({"station": np.array([0.0, 1.0]), "beam": np.array([1.0, 1.0]),
      "depth": np.array([1.0, 1.0])}, "at least 3 stations"),
    ({"station": np.array([0.0, 2.0, 1.0]), "beam": np.ones(3),
      "depth": np.ones(3)}, "strictly increasing"),
    ({"station": np.arange(3.0), "beam": np.array([1.0, -1.0, 1.0]),
      "depth": np.ones(3)}, "non-negative"),
    ({"station": np.arange(3.0), "beam": np.ones(3), "depth": np.ones(3),
      "freeboard": 0.0}, "freeboard must be positive"),
])
def test_offsets_validate(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HullOffsets(**kwargs)


def test_offsets_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        HullOffsets(station=np.arange(4.0), beam=np.ones(3), depth=np.ones(3))


def test_parametric_offsets_reject_bad_fullness():
    with pytest.raises(ValueError, match="fullness"):
        parametric_offsets(10.0, 0.5, 0.2, fullness=0.5)


# --------------------------------------------------------------------------
# mesh construction
# --------------------------------------------------------------------------
def test_mesh_has_panels(mesh):
    assert mesh.n_panels > 100
    assert mesh.total_area > 0.0


def test_panel_normals_are_unit_vectors(mesh):
    np.testing.assert_allclose(np.linalg.norm(mesh.normal, axis=1), 1.0,
                               atol=1e-10)


def test_panel_normals_point_outward(mesh):
    """Every normal must have a non-negative radial component."""
    radial = mesh.centroid.copy()
    radial[:, 0] = 0.0
    magnitude = np.linalg.norm(radial, axis=1)
    interesting = magnitude > 1e-6
    projection = np.sum(mesh.normal[interesting] * radial[interesting], axis=1)
    assert (projection >= -1e-9).all()


def test_panel_areas_are_positive(mesh):
    assert (mesh.area > 0).all()


def test_mesh_rejects_too_few_girth_panels(eight_offsets):
    with pytest.raises(ValueError, match="at least 4 girth panels"):
        HullMesh(eight_offsets, n_girth=2)


# --------------------------------------------------------------------------
# hydrostatics
# --------------------------------------------------------------------------
def test_displaced_volume_matches_the_mass_it_floats(mesh):
    mass = 855.0
    heave = mesh.equilibrium_heave(mass, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    assert props.volume == pytest.approx(mass / RHO, rel=1e-3)


def test_buoyancy_balances_weight_at_equilibrium(mesh):
    mass = 855.0
    heave = mesh.equilibrium_heave(mass, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    assert props.buoyancy_force[2] == pytest.approx(mass * GRAVITY, rel=1e-4)


def test_buoyancy_is_purely_vertical_when_upright(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    np.testing.assert_allclose(props.buoyancy_force[:2], 0.0, atol=1e-6)


def test_no_buoyancy_when_the_hull_is_clear_of_the_water(mesh):
    props = mesh.submerged(np.array([0.0, 0.0, 5.0]), np.zeros(3), rho=RHO)
    assert props.buoyancy_force[2] == pytest.approx(0.0, abs=1e-9)
    assert props.wetted_area == pytest.approx(0.0, abs=1e-9)
    assert props.volume == pytest.approx(0.0, abs=1e-9)


def test_sinking_the_hull_increases_buoyancy(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    shallow = mesh.submerged(np.array([0.0, 0.0, heave + 0.02]), np.zeros(3),
                             rho=RHO)
    deep = mesh.submerged(np.array([0.0, 0.0, heave - 0.02]), np.zeros(3),
                          rho=RHO)
    assert deep.buoyancy_force[2] > shallow.buoyancy_force[2]


def test_heave_stiffness_equals_rho_g_waterplane(mesh):
    """The textbook result the mesh must reproduce: dF/dz = -rho g A_wp."""
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    step = 0.01
    up = mesh.submerged(np.array([0.0, 0.0, heave + step]), np.zeros(3),
                        rho=RHO)
    down = mesh.submerged(np.array([0.0, 0.0, heave - step]), np.zeros(3),
                          rho=RHO)
    numerical = (up.buoyancy_force[2] - down.buoyancy_force[2]) / (2 * step)

    at_rest = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    expected = -RHO * GRAVITY * at_rest.plan_area
    assert numerical == pytest.approx(expected, rel=0.05)


def test_roll_produces_a_righting_moment(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    for roll_deg in (1.0, 3.0, 5.0):
        attitude = attitude_from_components(roll=np.radians(roll_deg))
        props = mesh.submerged(np.array([0.0, 0.0, heave]), attitude, rho=RHO)
        assert props.buoyancy_moment[0] < 0.0, "roll moment must oppose roll"


def test_pitch_produces_a_righting_moment(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    for pitch_deg in (0.5, 1.0):
        attitude = attitude_from_components(pitch=np.radians(pitch_deg))
        props = mesh.submerged(np.array([0.0, 0.0, heave]), attitude, rho=RHO)
        assert props.buoyancy_moment[1] < 0.0


def test_pitch_stiffness_far_exceeds_roll_stiffness(mesh):
    """A shell is enormously stiffer in pitch than roll -- it is long and thin."""
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    angle = np.radians(1.0)
    roll = mesh.submerged(np.array([0.0, 0.0, heave]),
                          attitude_from_components(roll=angle), rho=RHO)
    pitch = mesh.submerged(np.array([0.0, 0.0, heave]),
                           attitude_from_components(pitch=angle), rho=RHO)
    assert abs(pitch.buoyancy_moment[1]) > 100 * abs(roll.buoyancy_moment[0])


def test_restoring_moment_is_odd_in_roll(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    angle = np.radians(3.0)
    positive = mesh.submerged(np.array([0.0, 0.0, heave]),
                              attitude_from_components(roll=angle), rho=RHO)
    negative = mesh.submerged(np.array([0.0, 0.0, heave]),
                              attitude_from_components(roll=-angle), rho=RHO)
    assert positive.buoyancy_moment[0] == pytest.approx(
        -negative.buoyancy_moment[0], rel=1e-6)


# --------------------------------------------------------------------------
# surface measures
# --------------------------------------------------------------------------
def test_surface_measures_are_positive_and_ordered(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    assert props.wetted_area > props.plan_area > props.lateral_area \
        > props.transverse_area > 0.0


def test_transverse_area_is_of_order_beam_times_draft(mesh):
    """|Gamma_X| must be a midship section, not the hull's side profile.

    The legacy code used ``length * draft`` here, about 23 times too big.
    """
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    section = 0.25 * np.pi * 0.57 * 0.165
    assert 0.3 * section < props.transverse_area < 2.0 * section


def test_plan_area_is_of_order_the_waterplane(mesh):
    """|Gamma_Z| is an area, not a girth: the legacy value was ~4x too small."""
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    rectangle = 17.3 * 0.57
    assert 0.4 * rectangle < props.plan_area < rectangle


def test_wetted_area_grows_as_the_hull_sinks(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    shallow = mesh.submerged(np.array([0.0, 0.0, heave + 0.03]), np.zeros(3),
                             rho=RHO)
    deep = mesh.submerged(np.array([0.0, 0.0, heave - 0.03]), np.zeros(3),
                          rho=RHO)
    assert deep.wetted_area > shallow.wetted_area


def test_submerged_fraction_is_between_zero_and_one(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3), rho=RHO)
    assert 0.0 < props.submerged_fraction < 1.0


# --------------------------------------------------------------------------
# smoothness -- the integrator depends on it
# --------------------------------------------------------------------------
def test_buoyancy_varies_smoothly_with_heave(mesh):
    """No panel-switching steps: the ODE solver would chatter on them."""
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    offsets = np.linspace(-0.03, 0.03, 400)
    force = np.array([
        mesh.submerged(np.array([0.0, 0.0, heave + d]), np.zeros(3),
                       rho=RHO).buoyancy_force[2]
        for d in offsets
    ])
    steps = np.abs(np.diff(force))
    assert steps.max() < 5.0 * np.median(steps) + 20.0


def test_buoyancy_is_monotone_in_heave(mesh):
    heave = mesh.equilibrium_heave(855.0, rho=RHO)
    offsets = np.linspace(-0.05, 0.05, 120)
    force = np.array([
        mesh.submerged(np.array([0.0, 0.0, heave + d]), np.zeros(3),
                       rho=RHO).buoyancy_force[2]
        for d in offsets
    ])
    assert (np.diff(force) <= 1e-6).all()


# --------------------------------------------------------------------------
# equilibrium solver
# --------------------------------------------------------------------------
def test_equilibrium_heave_converges(mesh):
    for mass in (400.0, 700.0, 900.0):
        heave = mesh.equilibrium_heave(mass, rho=RHO)
        props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                               rho=RHO)
        assert props.buoyancy_force[2] == pytest.approx(mass * GRAVITY,
                                                        rel=1e-3)


def test_heavier_boats_float_lower(mesh):
    assert mesh.equilibrium_heave(900.0, rho=RHO) < \
        mesh.equilibrium_heave(500.0, rho=RHO)


def test_equilibrium_rejects_an_overloaded_hull(mesh):
    with pytest.raises(ValueError, match="cannot support"):
        mesh.equilibrium_heave(50000.0, rho=RHO)


def test_denser_water_floats_the_hull_higher(mesh):
    fresh = mesh.equilibrium_heave(855.0, rho=999.1)
    salt = mesh.equilibrium_heave(855.0, rho=1025.9)
    assert salt > fresh


# --------------------------------------------------------------------------
# convergence
# --------------------------------------------------------------------------
def test_results_converge_with_panel_count(eight_offsets):
    volumes = []
    for n_girth in (12, 24, 48):
        mesh = HullMesh(eight_offsets, n_girth=n_girth)
        heave = mesh.equilibrium_heave(855.0, rho=RHO)
        props = mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                               rho=RHO)
        volumes.append(props.volume)
    assert max(volumes) - min(volumes) < 0.005 * np.mean(volumes)
