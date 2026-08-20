"""The CasADi hydrodynamics must agree with the numpy originals.

Checked over a grid of states, not at one point -- the whole reason for
this module is that a single-point linearisation was not good enough.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro import appendages as np_appendages
from coxswain.hydro import resistance as np_resistance
from coxswain.hydro.shallow import ShallowWaterModel
from coxswain.river import hydro_casadi as ca_hydro

casadi = pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def submerged(boat):
    return boat.mesh.submerged(np.zeros(3), np.zeros(3),
                               rho=boat.water.density, gravity=9.80665,
                               water_level=0.0)


# --------------------------------------------------------------------------
# friction line
# --------------------------------------------------------------------------
@pytest.mark.parametrize("reynolds", [1e5, 1e6, 8.6e6, 5e7, 1e8])
def test_friction_matches_numpy(reynolds):
    import casadi as ca

    expected = np_resistance.friction_coefficient(reynolds, 0.075)
    got = float(ca.DM(ca_hydro.friction_coefficient(reynolds, 0.075)))
    assert got == pytest.approx(expected, rel=1e-9)


def test_friction_uses_base_ten():
    """Base ten, as ITTC 1957 is defined.

    The legacy code read the paper's 'log' as natural and understated
    C_f about eightfold.
    """
    import casadi as ca

    value = float(ca.DM(ca_hydro.friction_coefficient(1e6, 0.075)))
    assert value == pytest.approx(0.075 / (6.0 - 2.0) ** 2, rel=1e-9)


def test_friction_stays_finite_at_zero_reynolds():
    import casadi as ca

    assert np.isfinite(float(ca.DM(ca_hydro.friction_coefficient(0.0))))


# --------------------------------------------------------------------------
# hull resistance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("u", [3.0, 5.2, 6.5])
@pytest.mark.parametrize("v", [0.0, 0.25, -0.4])
def test_hull_resistance_matches_numpy_in_deep_water(boat, submerged, u, v):
    import casadi as ca

    expected, _ = np_resistance.hull_resistance(
        np.array([u, v, 0.0]), submerged, boat.length, boat.water,
        boat.resistance, None)

    got = ca_hydro.hull_resistance(
        u, v, 0.0,
        wetted_area=submerged.wetted_area,
        transverse_area=submerged.transverse_area,
        plan_area=boat.resistance.wave_area(submerged),
        lateral_area=submerged.lateral_area,
        mean_wetted_length=boat.length,
        depth=None,
        density=boat.water.density,
        kinematic_viscosity=boat.water.kinematic_viscosity,
        shape=boat.resistance.shape, wave=boat.resistance.wave,
        friction_zero=boat.resistance.friction_zero,
        form_factor=boat.resistance.form_factor,
        cross_flow_lateral=boat.resistance.cross_flow_lateral,
        cross_flow_vertical=boat.resistance.cross_flow_vertical,
    )
    got = np.array(ca.DM(got)).ravel()
    np.testing.assert_allclose(got[:2], expected[:2], rtol=1e-6, atol=1e-6)


def test_hull_resistance_opposes_the_motion(boat, submerged):
    import casadi as ca

    for u in (2.0, 5.0):
        got = np.array(ca.DM(ca_hydro.hull_resistance(
            u, 0.0, 0.0, submerged.wetted_area, submerged.transverse_area,
            boat.resistance.wave_area(submerged), submerged.lateral_area,
            boat.length))).ravel()
        assert got[0] < 0.0


def test_sway_resistance_is_quadratic(boat, submerged):
    """Cross-flow drag goes as v|v|, so doubling v quadruples it."""
    import casadi as ca

    def sway(v):
        return float(np.array(ca.DM(ca_hydro.hull_resistance(
            5.2, v, 0.0, submerged.wetted_area, submerged.transverse_area,
            boat.resistance.wave_area(submerged), submerged.lateral_area,
            boat.length))).ravel()[1])

    assert sway(0.4) == pytest.approx(4.0 * sway(0.2), rel=1e-6)


def test_shallow_water_raises_resistance(boat, submerged):
    import casadi as ca

    def total(depth):
        return float(np.array(ca.DM(ca_hydro.hull_resistance(
            5.2, 0.0, 0.0, submerged.wetted_area, submerged.transverse_area,
            boat.resistance.wave_area(submerged), submerged.lateral_area,
            boat.length, depth=depth))).ravel()[0])

    assert abs(total(3.0)) > abs(total(30.0))


def test_shallow_factor_peaks_near_critical():
    import casadi as ca

    def factor(froude):
        speed = froude * np.sqrt(9.80665 * 3.0)
        return float(ca.DM(ca_hydro.shallow_water_factor(speed, 3.0)))

    assert factor(1.0) > factor(0.5)
    assert factor(1.0) > factor(1.6)
    assert factor(1.0) == pytest.approx(3.0, rel=1e-6)


# --------------------------------------------------------------------------
# lifting surfaces
# --------------------------------------------------------------------------
@pytest.mark.parametrize("v", [0.0, 0.2, -0.35])
@pytest.mark.parametrize("yaw_rate", [0.0, 0.04, -0.06])
def test_skeg_matches_numpy(boat, v, yaw_rate):
    import casadi as ca

    skeg = [s for s in boat.appendages if not s.controllable][0]
    expected_force, expected_moment = np_appendages.surface_load(
        skeg, np.array([5.2, v, 0.0]), yaw_rate, 0.0, boat.water)

    force, moment = ca_hydro.surface_load(skeg, 5.2, v, yaw_rate, 0.0,
                                          boat.water.density)
    np.testing.assert_allclose(np.array(ca.DM(force)).ravel(), expected_force,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.array(ca.DM(moment)).ravel(),
                               expected_moment, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("deflection", [0.0, 0.1, -0.15])
def test_rudder_matches_numpy(boat, deflection):
    import casadi as ca

    rudder = [s for s in boat.appendages if s.controllable][0]
    expected_force, expected_moment = np_appendages.surface_load(
        rudder, np.array([5.2, 0.1, 0.0]), 0.02, deflection, boat.water)

    force, moment = ca_hydro.surface_load(rudder, 5.2, 0.1, 0.02, deflection,
                                          boat.water.density)
    np.testing.assert_allclose(np.array(ca.DM(force)).ravel(), expected_force,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.array(ca.DM(moment)).ravel(),
                               expected_moment, rtol=1e-8, atol=1e-9)


def test_appendage_sum_matches_numpy(boat):
    import casadi as ca

    expected_force = np.zeros(3)
    expected_moment = np.zeros(3)
    for surface in boat.appendages:
        f, m = np_appendages.surface_load(surface, np.array([5.2, 0.15, 0.0]),
                                          0.03, 0.08, boat.water)
        expected_force += f
        expected_moment += m

    force, moment = ca_hydro.appendage_loads(boat.appendages, 5.2, 0.15, 0.03,
                                             0.08, boat.water.density)
    np.testing.assert_allclose(np.array(ca.DM(force)).ravel(), expected_force,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.array(ca.DM(moment)).ravel(),
                               expected_moment, rtol=1e-8, atol=1e-9)


def test_skeg_damps_yaw(boat):
    """A surface aft of the centre of mass must oppose the yaw rate.

    The sign of the ``r x_ac / V`` term decides whether the skeg damps
    rotation or amplifies it.
    """
    import casadi as ca

    skeg = [s for s in boat.appendages if not s.controllable][0]
    _, moment = ca_hydro.surface_load(skeg, 5.2, 0.0, 0.05, 0.0,
                                      boat.water.density)
    assert float(ca.DM(moment)[2]) < 0.0


def test_weathervane_is_stabilising(boat):
    """Crabbing to port must give a moment that turns the bow to port."""
    import casadi as ca

    force, moment = ca_hydro.appendage_loads(boat.appendages, 5.2, 0.3, 0.0,
                                             0.0, boat.water.density)
    assert float(ca.DM(moment)[2]) > 0.0


# --------------------------------------------------------------------------
# differentiability -- the reason this module exists
# --------------------------------------------------------------------------
def test_everything_is_differentiable(boat, submerged):
    """Symbolic derivatives must exist and be finite across the range."""
    import casadi as ca

    u = ca.MX.sym("u")
    v = ca.MX.sym("v")
    rate = ca.MX.sym("r")
    delta = ca.MX.sym("d")

    resistance = ca_hydro.hull_resistance(
        u, v, 0.0, submerged.wetted_area, submerged.transverse_area,
        boat.resistance.wave_area(submerged), submerged.lateral_area,
        boat.length, depth=3.0)
    force, moment = ca_hydro.appendage_loads(boat.appendages, u, v, rate,
                                             delta)
    total = ca.vertcat(resistance[0] + force[0], resistance[1] + force[1],
                       moment[2])

    jacobian = ca.Function("J", [u, v, rate, delta],
                           [ca.jacobian(total, ca.vertcat(u, v, rate, delta))])
    for state in ((5.2, 0.0, 0.0, 0.0), (4.0, 0.3, 0.05, 0.1),
                  (6.0, -0.2, -0.03, -0.12)):
        value = np.array(jacobian(*state))
        assert np.all(np.isfinite(value)), state
        assert np.abs(value).max() > 0.0
