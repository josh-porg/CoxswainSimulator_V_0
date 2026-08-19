"""Unit tests for frame conventions and attitude kinematics.

These lock down the conventions that the legacy code violated:
  * which direction ``hull_to_abs`` rotates;
  * that index 0 of an attitude vector is roll and index 2 is yaw;
  * that the inertia tensor congruence uses hull->abs on the left.
"""

import numpy as np
import pytest

from coxswain.core import frames
from coxswain.core.frames import PITCH, ROLL, YAW


# --------------------------------------------------------------------------
# rotation matrix direction and properties
# --------------------------------------------------------------------------
def test_identity_attitude_gives_identity_matrix():
    np.testing.assert_allclose(frames.hull_to_abs(np.zeros(3)), np.eye(3),
                               atol=1e-15)


@pytest.mark.parametrize("attitude", [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.3, -0.2, 1.1]),
    np.array([-1.2, 0.4, -2.7]),
    np.array([np.pi / 3, np.pi / 7, -np.pi / 5]),
])
def test_rotation_is_orthonormal_with_unit_determinant(attitude):
    rot = frames.hull_to_abs(attitude)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("attitude", [
    np.array([0.3, -0.2, 1.1]),
    np.array([-1.2, 0.4, -2.7]),
])
def test_abs_to_hull_is_the_inverse_of_hull_to_abs(attitude):
    fwd = frames.hull_to_abs(attitude)
    back = frames.abs_to_hull(attitude)
    np.testing.assert_allclose(back @ fwd, np.eye(3), atol=1e-12)


def test_yaw_rotates_bow_towards_absolute_y():
    """A 90 deg yaw must take the hull x axis (bow) onto absolute +Y."""
    attitude = frames.attitude_from_components(yaw=np.pi / 2)
    bow_hull = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(frames.hull_to_abs(attitude) @ bow_hull,
                               [0.0, 1.0, 0.0], atol=1e-12)


def test_positive_pitch_lifts_the_bow():
    """Right-handed pitch about +y (port) raises the bow in absolute Z."""
    attitude = frames.attitude_from_components(pitch=np.radians(10.0))
    bow_hull = np.array([1.0, 0.0, 0.0])
    bow_abs = frames.hull_to_abs(attitude) @ bow_hull
    assert bow_abs[2] == pytest.approx(-np.sin(np.radians(10.0)), abs=1e-12)


def test_positive_roll_lowers_the_port_side():
    """Right-handed roll about +x takes the port unit vector downwards."""
    attitude = frames.attitude_from_components(roll=np.radians(10.0))
    port_hull = np.array([0.0, 1.0, 0.0])
    port_abs = frames.hull_to_abs(attitude) @ port_hull
    assert port_abs[2] == pytest.approx(np.sin(np.radians(10.0)), abs=1e-12)


def test_rotation_composes_as_yaw_pitch_roll():
    """hull_to_abs must equal Rz(yaw) Ry(pitch) Rx(roll)."""
    phi, theta, psi = 0.21, -0.37, 1.05

    def rot_x(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                         [0, np.sin(a), np.cos(a)]])

    def rot_y(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]])

    def rot_z(a):
        return np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    expected = rot_z(psi) @ rot_y(theta) @ rot_x(phi)
    actual = frames.hull_to_abs(np.array([phi, theta, psi]))
    np.testing.assert_allclose(actual, expected, atol=1e-13)


# --------------------------------------------------------------------------
# skew / unskew
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_skew_reproduces_the_cross_product(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(frames.skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_skew_is_antisymmetric():
    v = np.array([1.0, -2.0, 3.5])
    np.testing.assert_allclose(frames.skew(v).T, -frames.skew(v), atol=1e-15)


def test_unskew_inverts_skew():
    v = np.array([0.4, -1.7, 2.2])
    np.testing.assert_allclose(frames.unskew(frames.skew(v)), v, atol=1e-15)


def test_skew_squared_identity():
    """S(v)S(v) == v v^T - |v|^2 I, the identity the mass matrix relies on."""
    v = np.array([0.4, -1.7, 2.2])
    expected = np.outer(v, v) - v.dot(v) * np.eye(3)
    np.testing.assert_allclose(frames.skew(v) @ frames.skew(v), expected,
                               atol=1e-13)


# --------------------------------------------------------------------------
# Euler-angle ordering -- the legacy swap
# --------------------------------------------------------------------------
def test_pure_body_roll_rate_drives_only_the_roll_index():
    rates = frames.euler_rates_from_body(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert rates[ROLL] == pytest.approx(1.0)
    assert rates[PITCH] == pytest.approx(0.0)
    assert rates[YAW] == pytest.approx(0.0)


def test_pure_body_pitch_rate_drives_only_the_pitch_index():
    rates = frames.euler_rates_from_body(np.zeros(3), np.array([0.0, 1.0, 0.0]))
    assert rates[PITCH] == pytest.approx(1.0)
    assert rates[ROLL] == pytest.approx(0.0)
    assert rates[YAW] == pytest.approx(0.0)


def test_pure_body_yaw_rate_drives_only_the_yaw_index():
    rates = frames.euler_rates_from_body(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert rates[YAW] == pytest.approx(1.0)
    assert rates[ROLL] == pytest.approx(0.0)
    assert rates[PITCH] == pytest.approx(0.0)


@pytest.mark.parametrize("attitude", [
    np.zeros(3),
    np.array([0.25, -0.15, 0.8]),
    np.array([-0.6, 0.3, 2.0]),
])
@pytest.mark.parametrize("seed", range(3))
def test_euler_rate_conversion_round_trips(attitude, seed):
    rng = np.random.default_rng(seed)
    body = rng.normal(size=3)
    euler = frames.euler_rates_from_body(attitude, body)
    back = frames.body_rates_from_euler_rates(attitude, euler)
    np.testing.assert_allclose(back, body, atol=1e-12)


def test_euler_rates_accepts_absolute_omega_and_agrees_with_body_form():
    attitude = np.array([0.25, -0.15, 0.8])
    omega_body = np.array([0.3, -0.2, 0.5])
    omega_abs = frames.hull_to_abs(attitude) @ omega_body
    np.testing.assert_allclose(
        frames.euler_rates(attitude, omega_abs),
        frames.euler_rates_from_body(attitude, omega_body),
        atol=1e-12,
    )


def test_euler_rates_raise_at_gimbal_lock():
    attitude = frames.attitude_from_components(pitch=np.pi / 2)
    with pytest.raises(ValueError, match="gimbal lock"):
        frames.euler_rates_from_body(attitude, np.array([0.0, 0.0, 1.0]))


def test_euler_kinematics_match_finite_differenced_rotation_matrix():
    """dR/dt must equal S(omega_abs) R -- the defining property of omega."""
    attitude = np.array([0.21, -0.13, 0.77])
    omega_abs = np.array([0.4, -0.25, 0.9])
    step = 1e-7

    rates = frames.euler_rates(attitude, omega_abs)
    rot_plus = frames.hull_to_abs(attitude + step * rates)
    rot_minus = frames.hull_to_abs(attitude - step * rates)
    numerical = (rot_plus - rot_minus) / (2 * step)

    analytic = frames.skew(omega_abs) @ frames.hull_to_abs(attitude)
    np.testing.assert_allclose(numerical, analytic, atol=1e-6)


# --------------------------------------------------------------------------
# inertia congruence
# --------------------------------------------------------------------------
def test_inertia_rotation_uses_the_hull_to_abs_congruence():
    inertia = np.diag([10.0, 500.0, 900.0])
    attitude = np.array([0.26, 0.35, 0.52])
    rot = frames.hull_to_abs(attitude)

    expected = rot @ inertia @ rot.T
    np.testing.assert_allclose(frames.rotate_inertia_to_abs(inertia, attitude),
                               expected, atol=1e-12)


def test_inertia_rotation_differs_from_the_reversed_congruence():
    """Guards the specific legacy bug: R @ I @ R.T with R = abs->hull.

    With a realistic (non-isotropic) shell inertia the two differ by
    hundreds of kg m^2, which is why the bug stayed hidden while
    ``hull_inertia`` was ``diag(10, 500, 500)``.
    """
    inertia = np.diag([10.0, 500.0, 900.0])
    attitude = np.array([0.26, 0.35, 0.52])
    rot = frames.abs_to_hull(attitude)

    wrong = rot @ inertia @ rot.T
    right = frames.rotate_inertia_to_abs(inertia, attitude)
    assert np.abs(wrong - right).max() > 100.0


def test_inertia_rotation_preserves_eigenvalues():
    inertia = np.diag([10.0, 500.0, 900.0])
    attitude = np.array([0.26, 0.35, 0.52])
    rotated = frames.rotate_inertia_to_abs(inertia, attitude)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(rotated)),
                               [10.0, 500.0, 900.0], atol=1e-10)


@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0),
    (0.5, 0.5),
    (-0.5, -0.5),
    (2 * np.pi, 0.0),
    (2 * np.pi + 0.3, 0.3),
    (-2 * np.pi - 0.3, -0.3),
    (np.pi - 0.01, np.pi - 0.01),
    (-np.pi + 0.01, -np.pi + 0.01),
])
def test_wrap_to_pi_maps_into_the_principal_branch(angle, expected):
    assert frames.wrap_to_pi(angle) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("angle", [np.pi, -np.pi, 3 * np.pi, -3 * np.pi])
def test_wrap_to_pi_at_the_branch_cut_returns_magnitude_pi(angle):
    """Sign at the cut is float-noise dependent; only the magnitude is defined."""
    assert abs(frames.wrap_to_pi(angle)) == pytest.approx(np.pi, abs=1e-12)


def test_wrap_to_pi_is_vectorised():
    wrapped = frames.wrap_to_pi(np.array([0.0, 2 * np.pi + 0.3, -2 * np.pi - 0.3]))
    np.testing.assert_allclose(wrapped, [0.0, 0.3, -0.3], atol=1e-12)
