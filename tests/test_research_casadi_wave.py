"""The CasADi models get the research model's wave drag too.

Research boats carry a depth-aware wave table.  The trajectory and stochastic
optimisers (``SixDofModel``, ``StrokeResolvedModel``) cannot call a Python
table inside a CasADi graph, so it is sampled onto a differentiable surface
(:class:`~coxswain.river.hydro_casadi.WaveSurface`).  Boats without one keep
the constant wave coefficient and the smoothed shallow factor unchanged.
"""

import inspect

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro import resistance as np_resistance
from coxswain.hydro.finite_depth_michell import FiniteDepthWaveTable
from coxswain.hydro.michell import elliptical_offsets
from coxswain.hydro.shallow import ShallowWaterModel
from coxswain.river import hydro_casadi

casadi = pytest.importorskip("casadi")

GRAVITY = 9.80665


@pytest.fixture(scope="module")
def research_boat():
    boat = catalog.build("8+", rate=32.0)
    x, z, half = elliptical_offsets(boat.offsets, stations=161, levels=21)
    boat.wave_table = FiniteDepthWaveTable(station=x, level=z, half_beam=half)
    return boat


@pytest.fixture(scope="module")
def surface(research_boat):
    return hydro_casadi.WaveSurface.from_table(
        research_boat.wave_table,
        speeds=np.arange(2.0, 7.5 + 1e-9, 0.05),
        depths=np.geomspace(1.2, 20.0, 40))


@pytest.fixture(scope="module")
def submerged(research_boat):
    return research_boat.mesh.submerged(
        np.zeros(3), np.zeros(3), rho=research_boat.water.density,
        gravity=GRAVITY, water_level=0.0)


def evaluate(expression_of):
    import casadi as ca
    u, d = ca.SX.sym("u"), ca.SX.sym("d")
    return ca.Function("f", [u, d], [expression_of(u, d)])


def test_plain_boats_get_no_surface():
    assert hydro_casadi.wave_function_for(catalog.build("4+", rate=30.0)) \
        is None


def test_research_boats_share_one_surface(research_boat, surface):
    research_boat.wave_table._casadi_surface = surface
    first = hydro_casadi.wave_function_for(research_boat)
    assert first is surface
    assert hydro_casadi.wave_function_for(research_boat) is first


def test_surface_matches_the_table_off_node(research_boat, surface):
    table = research_boat.wave_table
    f = evaluate(lambda u, d: surface(u, d))
    worst = 0.0
    for depth in np.geomspace(1.6, 6.0, 9):
        critical = np.sqrt(GRAVITY * depth)
        for speed in np.unique(np.concatenate([
                np.arange(3.53, 6.5, 0.211),
                critical * np.array([0.93, 0.99, 1.0, 1.02, 1.09])])):
            if not 2.0 < speed < 7.5:
                continue
            exact = table.at_depth(speed, depth)
            worst = max(worst, abs(float(f(speed, depth)) - exact) / exact)
    assert worst < 0.02, worst


def test_deep_surface_is_michells_integral(research_boat, surface):
    """Against the integral itself, not against another interpolant.

    Off every grid node, 3.0-7.4 m/s.  Below 3 m/s the hull's humps are
    close enough together that any sampled curve has to be judged against
    its sampling, and that is the table's test, not this one.
    """
    import casadi as ca
    from coxswain.hydro.michell import MichellWave

    u = ca.SX.sym("u")
    g = ca.Function("g", [u], [surface(u)])
    x, z, half = elliptical_offsets(research_boat.offsets, stations=161,
                                    levels=21)
    direct = MichellWave(station=x, level=z, half_beam=half,
                         quadrature="trapezoid")
    speeds = np.arange(3.013, 7.4, 0.0419)
    exact = direct.resistance(speeds)
    got = np.array([float(g(s)) for s in speeds])
    assert np.max(np.abs(got - exact) / exact) < 0.01


@pytest.mark.parametrize("depth", [2.5, 4.0, None])
@pytest.mark.parametrize("speed", [4.0, 5.2, 6.0])
def test_casadi_resistance_matches_numpy_with_the_research_table(
        research_boat, surface, submerged, speed, depth):
    import casadi as ca
    boat = research_boat
    shallow = None if depth is None else ShallowWaterModel(depth=depth)
    expected, _ = np_resistance.hull_resistance(
        np.array([speed, 0.0, 0.0]), submerged, boat.length, boat.water,
        boat.resistance, shallow, wave_table=boat.wave_table)
    coefficients = boat.resistance
    got = ca.DM(hydro_casadi.hull_resistance(
        speed, 0.0, 0.0,
        wetted_area=submerged.wetted_area,
        transverse_area=submerged.transverse_area,
        plan_area=coefficients.wave_area(submerged),
        lateral_area=submerged.lateral_area,
        mean_wetted_length=boat.length, depth=depth,
        density=boat.water.density,
        kinematic_viscosity=boat.water.kinematic_viscosity,
        shape=coefficients.shape, wave=coefficients.wave,
        friction_zero=coefficients.friction_zero,
        form_factor=coefficients.form_factor,
        wave_function=surface))
    assert float(got[0]) == pytest.approx(float(expected[0]), rel=0.015)


def test_gradient_is_finite_through_critical(surface):
    import casadi as ca
    depth = 2.6
    u = ca.SX.sym("u")
    gradient = ca.Function("dg", [u], [ca.gradient(surface(u, depth), u)])
    for speed in np.sqrt(GRAVITY * depth) * np.array([0.95, 1.0, 1.05]):
        assert np.isfinite(float(gradient(speed)))


def test_without_a_surface_the_constant_coefficient_path_is_unchanged():
    import casadi as ca
    speed, depth = 5.2, 2.6
    got = float(ca.DM(hydro_casadi.hull_resistance(
        speed, 0.0, 0.0, wetted_area=3.0, transverse_area=0.05,
        plan_area=6.0, lateral_area=1.5, mean_wetted_length=17.0,
        depth=depth))[0])
    q = 0.5 * 1000.0 * speed ** 2
    friction = float(ca.DM(hydro_casadi.friction_coefficient(
        speed * 17.0 / 1.0e-6, 0.075)))
    factor = float(ca.DM(hydro_casadi.shallow_water_factor(speed, depth)))
    expected = -(q * 0.05 * 0.01 + q * 3.0 * friction
                 + q * 6.0 * 0.02 * factor)
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.slow
def test_a_research_sixdof_model_rows_against_its_own_wave_drag(
        research_boat, surface):
    """Built and evaluated, not just read: the same hull and state, one
    model on the constant coefficient and one on the research surface."""
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel

    shipped_boat = catalog.eight(rate=32.0)
    surrogate = HullSurrogate.from_boat(shipped_boat, n_heave=17, n_pitch=9,
                                        n_roll=9)
    research_boat.wave_table._casadi_surface = surface

    def surge(boat):
        model = SixDofModel(boat, surrogate=surrogate)
        state = np.zeros(13)
        state[6] = 5.2
        state[12] = model.anaerobic_capacity
        return np.array(model.function()(state, [0.0, 0.0, 1.0],
                                         0.2)).ravel()[6]

    plain = surge(shipped_boat)
    research = surge(research_boat)
    assert np.isfinite(research)
    assert research != pytest.approx(plain, abs=1e-6)


@pytest.mark.parametrize("module_name, cls_name", [
    ("coxswain.river.sixdof", "SixDofModel"),
    ("coxswain.river.strokemodel", "StrokeResolvedModel"),
])
def test_both_casadi_models_pass_the_boats_wave_function(module_name,
                                                         cls_name):
    import importlib
    cls = getattr(importlib.import_module(module_name), cls_name)
    assert "wave_function_for(self.boat)" in inspect.getsource(cls.derivative)
