"""Unit tests for the generalised mass matrix and moving-mass reactions.

The three sign errors that made the legacy 6-DOF model diverge are each
pinned by a dedicated test here, cross-checked against Formaggia et al.
(2009) eq. (14) and against an independent brute-force particle model.
"""

import numpy as np
import pytest

from coxswain.core import frames
from coxswain.core.rigid_body import (
    MovingMassField,
    assemble_mass_matrix,
    coupling_matrix,
    first_moment,
    gyroscopic_moment,
    moving_mass_inertia,
    moving_mass_reaction,
    solve_accelerations,
)


@pytest.fixture
def crew_cloud():
    """Four 85 kg rowers, 12 segments each, spread over +-4 m -- an eight-ish
    layout.  This is the configuration that drove the legacy inertia
    negative."""
    rng = np.random.default_rng(0)
    seats = np.array([4.0, 2.0, 0.0, -2.0])
    mass, position = [], []
    for seat_x in seats:
        for _ in range(12):
            mass.append(85.0 / 12.0)
            position.append([seat_x + rng.uniform(-0.3, 0.3),
                             rng.uniform(-0.2, 0.2),
                             rng.uniform(0.2, 0.9)])
    return np.array(mass), np.array(position)


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def test_first_moment_matches_explicit_sum(crew_cloud):
    mass, position = crew_cloud
    expected = sum(m * r for m, r in zip(mass, position))
    np.testing.assert_allclose(first_moment(mass, position), expected,
                               atol=1e-10)


def test_coupling_matrix_is_skew_symmetric(crew_cloud):
    mass, position = crew_cloud
    coupling = coupling_matrix(mass, position)
    np.testing.assert_allclose(coupling.T, -coupling, atol=1e-12)


def test_coupling_matrix_equals_sum_of_skews(crew_cloud):
    mass, position = crew_cloud
    expected = sum(m * frames.skew(r) for m, r in zip(mass, position))
    np.testing.assert_allclose(coupling_matrix(mass, position), expected,
                               atol=1e-10)


def test_moving_mass_inertia_is_positive_semi_definite(crew_cloud):
    """The crew must ADD rotational inertia.  The legacy code subtracted it."""
    mass, position = crew_cloud
    inertia = moving_mass_inertia(mass, position)
    eigenvalues = np.linalg.eigvalsh(inertia)
    assert eigenvalues.min() > 0.0, "crew inertia contribution must be positive"


def test_moving_mass_inertia_is_symmetric(crew_cloud):
    mass, position = crew_cloud
    inertia = moving_mass_inertia(mass, position)
    np.testing.assert_allclose(inertia, inertia.T, atol=1e-10)


def test_moving_mass_inertia_matches_negative_skew_squared(crew_cloud):
    """-sum m S(r)S(r) is the identity the docstring claims."""
    mass, position = crew_cloud
    expected = -sum(m * frames.skew(r) @ frames.skew(r)
                    for m, r in zip(mass, position))
    np.testing.assert_allclose(moving_mass_inertia(mass, position), expected,
                               atol=1e-9)


def test_moving_mass_inertia_of_single_particle_on_x_axis():
    """A point mass at (d, 0, 0) contributes m d^2 about y and z, 0 about x."""
    mass = np.array([10.0])
    position = np.array([[3.0, 0.0, 0.0]])
    np.testing.assert_allclose(moving_mass_inertia(mass, position),
                               np.diag([0.0, 90.0, 90.0]), atol=1e-12)


def test_moving_mass_inertia_reduces_to_the_papers_planar_scalar(crew_cloud):
    """Paper eq. (14b) uses the scalar `sum m |x|^2` for pitch about Y.

    For motion confined to the symmetry plane (y = 0) the yy component of
    the tensor must equal that scalar exactly.
    """
    mass, position = crew_cloud
    planar = position.copy()
    planar[:, 1] = 0.0
    tensor = moving_mass_inertia(mass, planar)
    scalar = float((mass * np.einsum("ki,ki->k", planar, planar)).sum())
    assert tensor[1, 1] == pytest.approx(scalar, rel=1e-12)


def test_moving_mass_inertia_of_empty_cloud_is_zero():
    inertia = moving_mass_inertia(np.zeros(0), np.zeros((0, 3)))
    np.testing.assert_allclose(inertia, np.zeros((3, 3)))


# --------------------------------------------------------------------------
# the assembled mass matrix
# --------------------------------------------------------------------------
def test_mass_matrix_is_symmetric(crew_cloud):
    mass, position = crew_cloud
    matrix = assemble_mass_matrix(500.0, np.diag([10.0, 500.0, 900.0]),
                                  mass, position)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)


def test_mass_matrix_is_positive_definite_for_a_realistic_eight(crew_cloud):
    """The regression guard for the original blow-up.

    With the legacy signs this matrix had two eigenvalues near -1471.
    """
    mass, position = crew_cloud
    matrix = assemble_mass_matrix(936.0, np.diag([10.0, 500.0, 500.0]),
                                  mass, position)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() > 0.0, (
        f"mass matrix not positive definite: eigenvalues {eigenvalues}"
    )


def test_mass_matrix_block_signs_match_the_paper(crew_cloud):
    """Force equation couples to omega_dot via -A; moment equation via +A."""
    mass, position = crew_cloud
    coupling = coupling_matrix(mass, position)
    matrix = assemble_mass_matrix(936.0, np.eye(3), mass, position)

    np.testing.assert_allclose(matrix[0:3, 3:6], -coupling, atol=1e-10)
    np.testing.assert_allclose(matrix[3:6, 0:3], coupling, atol=1e-10)


def test_mass_matrix_translational_block_is_total_mass(crew_cloud):
    mass, position = crew_cloud
    matrix = assemble_mass_matrix(936.0, np.eye(3), mass, position)
    np.testing.assert_allclose(matrix[0:3, 0:3], 936.0 * np.eye(3), atol=1e-12)


def test_mass_matrix_rotational_block_exceeds_bare_hull_inertia(crew_cloud):
    """Adding crew must increase, never decrease, the rotational inertia."""
    mass, position = crew_cloud
    hull = np.diag([10.0, 500.0, 500.0])
    matrix = assemble_mass_matrix(936.0, hull, mass, position)
    assert np.all(np.diag(matrix[3:6, 3:6]) > np.diag(hull))


def test_mass_matrix_with_no_crew_is_block_diagonal():
    matrix = assemble_mass_matrix(100.0, np.diag([1.0, 2.0, 3.0]),
                                  np.zeros(0), np.zeros((0, 3)))
    np.testing.assert_allclose(matrix[0:3, 3:6], np.zeros((3, 3)), atol=1e-14)
    np.testing.assert_allclose(matrix[3:6, 0:3], np.zeros((3, 3)), atol=1e-14)
    np.testing.assert_allclose(matrix[3:6, 3:6], np.diag([1.0, 2.0, 3.0]))


def test_mass_matrix_recovers_rigid_body_about_an_offset_point():
    """Cross-check against textbook Newton-Euler about a non-CoM point.

    For a rigid body of mass m with centre of mass at offset c from the
    reference point, the generalised mass matrix must be

        [[ m I ,  -m S(c) ],
         [ m S(c),  I_ref ]]

    with I_ref = I_cm - m S(c)S(c) (parallel axis).  Model the body as a
    single point mass and check every block.
    """
    m, c = 40.0, np.array([1.5, -0.4, 0.7])
    matrix = assemble_mass_matrix(m, np.zeros((3, 3)), np.array([m]),
                                  c.reshape(1, 3))

    np.testing.assert_allclose(matrix[0:3, 0:3], m * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(matrix[0:3, 3:6], -m * frames.skew(c), atol=1e-12)
    np.testing.assert_allclose(matrix[3:6, 0:3], m * frames.skew(c), atol=1e-12)
    np.testing.assert_allclose(matrix[3:6, 3:6],
                               -m * frames.skew(c) @ frames.skew(c), atol=1e-12)


# --------------------------------------------------------------------------
# solve_accelerations
# --------------------------------------------------------------------------
def test_solve_accelerations_matches_direct_solve(crew_cloud):
    mass, position = crew_cloud
    matrix = assemble_mass_matrix(936.0, np.diag([10.0, 500.0, 500.0]),
                                  mass, position)
    force = np.array([100.0, -50.0, 20.0, 5.0, -3.0, 8.0])
    np.testing.assert_allclose(solve_accelerations(matrix, force),
                               np.linalg.solve(matrix, force), atol=1e-9)


def test_solve_accelerations_rejects_an_indefinite_mass_matrix():
    """A negative rotational inertia must fail loudly, not silently."""
    matrix = np.eye(6)
    matrix[5, 5] = -1500.0
    with pytest.raises(ValueError, match="not positive definite"):
        solve_accelerations(matrix, np.ones(6))


def test_solve_accelerations_free_body_accelerates_along_the_force():
    matrix = assemble_mass_matrix(200.0, np.eye(3), np.zeros(0),
                                  np.zeros((0, 3)))
    force = np.array([400.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    accel = solve_accelerations(matrix, force)
    np.testing.assert_allclose(accel[0:3], [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(accel[3:6], np.zeros(3), atol=1e-12)


# --------------------------------------------------------------------------
# moving-mass reaction terms
# --------------------------------------------------------------------------
def test_reaction_is_zero_for_a_stationary_crew():
    field = MovingMassField(mass=np.array([10.0, 20.0]),
                            position=np.array([[1.0, 0.0, 0.5],
                                               [-1.0, 0.2, 0.4]]),
                            velocity=np.zeros((2, 3)),
                            acceleration=np.zeros((2, 3)))
    force, moment = moving_mass_reaction(field, np.zeros(3))
    np.testing.assert_allclose(force, np.zeros(3), atol=1e-14)
    np.testing.assert_allclose(moment, np.zeros(3), atol=1e-14)


def test_reaction_opposes_crew_acceleration():
    """Crew accelerating towards the bow pushes the hull towards the stern."""
    field = MovingMassField(mass=np.array([500.0]),
                            position=np.array([[0.0, 0.0, 0.0]]),
                            velocity=np.zeros((1, 3)),
                            acceleration=np.array([[2.0, 0.0, 0.0]]))
    force, _ = moving_mass_reaction(field, np.zeros(3))
    np.testing.assert_allclose(force, [-1000.0, 0.0, 0.0], atol=1e-12)


def test_reaction_includes_coriolis_with_the_correct_factor_of_two():
    field = MovingMassField(mass=np.array([1.0]),
                            position=np.zeros((1, 3)),
                            velocity=np.array([[1.0, 0.0, 0.0]]),
                            acceleration=np.zeros((1, 3)))
    omega = np.array([0.0, 0.0, 1.0])
    force, _ = moving_mass_reaction(field, omega)
    np.testing.assert_allclose(force, -2.0 * np.cross(omega, [1.0, 0.0, 0.0]),
                               atol=1e-12)


def test_reaction_includes_centrifugal_term():
    field = MovingMassField(mass=np.array([1.0]),
                            position=np.array([[2.0, 0.0, 0.0]]),
                            velocity=np.zeros((1, 3)),
                            acceleration=np.zeros((1, 3)))
    omega = np.array([0.0, 0.0, 3.0])
    force, _ = moving_mass_reaction(field, omega)
    # omega x (omega x r) = -|omega|^2 r for r perpendicular to omega
    np.testing.assert_allclose(force, [18.0, 0.0, 0.0], atol=1e-12)


def test_reaction_moment_is_r_cross_force_per_particle():
    field = MovingMassField(mass=np.array([3.0]),
                            position=np.array([[1.0, 2.0, -0.5]]),
                            velocity=np.array([[0.1, 0.0, 0.2]]),
                            acceleration=np.array([[0.4, -0.3, 0.1]]))
    omega = np.array([0.05, -0.02, 0.3])
    force, moment = moving_mass_reaction(field, omega)
    np.testing.assert_allclose(moment, np.cross(field.position[0], force),
                               atol=1e-12)


def test_reaction_omega_uses_angular_velocity_not_attitude():
    """Regression guard: legacy passed the yaw ANGLE where omega belongs.

    Doubling omega must quadruple the centrifugal contribution; a function
    that had been handed an angle would show no such scaling with rate.
    """
    field = MovingMassField(mass=np.array([1.0]),
                            position=np.array([[2.0, 0.0, 0.0]]),
                            velocity=np.zeros((1, 3)),
                            acceleration=np.zeros((1, 3)))
    slow, _ = moving_mass_reaction(field, np.array([0.0, 0.0, 1.0]))
    fast, _ = moving_mass_reaction(field, np.array([0.0, 0.0, 2.0]))
    assert fast[0] == pytest.approx(4.0 * slow[0], rel=1e-12)


# --------------------------------------------------------------------------
# gyroscopic term
# --------------------------------------------------------------------------
def test_gyroscopic_moment_vanishes_about_a_principal_axis():
    inertia = np.diag([10.0, 500.0, 900.0])
    moment = gyroscopic_moment(inertia, np.array([0.0, 0.0, 1.3]))
    np.testing.assert_allclose(moment, np.zeros(3), atol=1e-12)


def test_gyroscopic_moment_is_perpendicular_to_omega():
    inertia = np.diag([10.0, 500.0, 900.0])
    omega = np.array([0.4, -0.2, 0.7])
    moment = gyroscopic_moment(inertia, omega)
    assert moment.dot(omega) == pytest.approx(0.0, abs=1e-12)


def test_gyroscopic_moment_does_no_work():
    """omega . (-omega x I omega) == 0, so it cannot change kinetic energy."""
    inertia = np.diag([12.0, 480.0, 860.0])
    omega = np.array([0.3, 0.5, -0.2])
    assert gyroscopic_moment(inertia, omega).dot(omega) == pytest.approx(
        0.0, abs=1e-12)


# --------------------------------------------------------------------------
# MovingMassField plumbing
# --------------------------------------------------------------------------
def test_field_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="position"):
        MovingMassField(mass=np.ones(3), position=np.zeros((2, 3)),
                        velocity=np.zeros((3, 3)),
                        acceleration=np.zeros((3, 3)))


def test_field_to_abs_rotates_every_component():
    attitude = np.array([0.2, -0.1, 0.6])
    rot = frames.hull_to_abs(attitude)
    field = MovingMassField(mass=np.array([2.0]),
                            position=np.array([[1.0, 0.5, -0.2]]),
                            velocity=np.array([[0.3, 0.0, 0.1]]),
                            acceleration=np.array([[-0.4, 0.2, 0.0]]))
    rotated = field.to_abs(rot)

    np.testing.assert_allclose(rotated.position[0], rot @ field.position[0],
                               atol=1e-12)
    np.testing.assert_allclose(rotated.velocity[0], rot @ field.velocity[0],
                               atol=1e-12)
    np.testing.assert_allclose(rotated.acceleration[0],
                               rot @ field.acceleration[0], atol=1e-12)
    np.testing.assert_allclose(rotated.mass, field.mass)


def test_field_total_mass():
    field = MovingMassField(mass=np.array([1.0, 2.0, 3.0]),
                            position=np.zeros((3, 3)),
                            velocity=np.zeros((3, 3)),
                            acceleration=np.zeros((3, 3)))
    assert field.total_mass == pytest.approx(6.0)


def test_empty_field_produces_zero_reaction():
    force, moment = moving_mass_reaction(MovingMassField.empty(),
                                         np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(force, np.zeros(3))
    np.testing.assert_allclose(moment, np.zeros(3))
