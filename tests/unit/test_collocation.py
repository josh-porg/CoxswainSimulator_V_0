"""Unit tests for the phase-locked mesh and the transcription schemes."""

import numpy as np
import pytest

from coxswain.crew.stroke import StrokeTiming
from coxswain.river.collocation import (HermiteSimpson, MeshInterval,
                                        RadauIIA, phase_locked_mesh,
                                        radau_points)

casadi = pytest.importorskip("casadi")


@pytest.fixture
def timing():
    return StrokeTiming(rate=32.0)


# --------------------------------------------------------------------------
# the mesh
# --------------------------------------------------------------------------
def test_every_catch_and_finish_is_an_interval_boundary(timing):
    """The point of the whole mesh.

    A uniform mesh straddles the catch, which forces one control value
    across both the drive and the recovery -- and no amount of refinement
    fixes that, because the boundary is in the wrong place.
    """
    mesh = phase_locked_mesh(timing, n_strokes=4)
    boundaries = {round(interval.start, 12) for interval in mesh}
    boundaries |= {round(interval.end, 12) for interval in mesh}

    for stroke in range(4):
        catch = round(stroke * timing.period, 12)
        finish = round(stroke * timing.period + timing.drive_duration, 12)
        assert catch in boundaries, f"catch {stroke}"
        assert finish in boundaries, f"finish {stroke}"


def test_mesh_is_contiguous_and_covers_the_horizon(timing):
    mesh = phase_locked_mesh(timing, n_strokes=3)
    for left, right in zip(mesh[:-1], mesh[1:]):
        assert left.end == pytest.approx(right.start, abs=1e-12)
    assert mesh[0].start == pytest.approx(0.0)
    assert mesh[-1].end == pytest.approx(3 * timing.period)


def test_drive_and_recovery_are_labelled(timing):
    mesh = phase_locked_mesh(timing, n_strokes=2, drive_intervals=5,
                             recovery_intervals=3)
    drive = [i for i in mesh if i.phase == "drive"]
    recovery = [i for i in mesh if i.phase == "recovery"]
    assert len(drive) == 10
    assert len(recovery) == 6


def test_each_phase_can_be_refined_independently(timing):
    """A bend needs resolution on the drive, where the split acts, and
    much less on the recovery, where only the rudder does anything."""
    mesh = phase_locked_mesh(timing, n_strokes=1, drive_intervals=12,
                             recovery_intervals=2)
    drive = [i for i in mesh if i.phase == "drive"]
    recovery = [i for i in mesh if i.phase == "recovery"]
    assert drive[0].duration < recovery[0].duration / 4


def test_interval_durations_sum_to_the_phase_durations(timing):
    mesh = phase_locked_mesh(timing, n_strokes=1)
    drive = sum(i.duration for i in mesh if i.phase == "drive")
    recovery = sum(i.duration for i in mesh if i.phase == "recovery")
    assert drive == pytest.approx(timing.drive_duration, rel=1e-12)
    assert recovery == pytest.approx(timing.recovery_duration, rel=1e-12)


def test_mesh_rejects_nonsense(timing):
    with pytest.raises(ValueError, match="at least one stroke"):
        phase_locked_mesh(timing, n_strokes=0)
    with pytest.raises(ValueError, match="at least one interval"):
        phase_locked_mesh(timing, n_strokes=1, drive_intervals=0)


# --------------------------------------------------------------------------
# Radau
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stages", [1, 2, 3, 4, 5])
def test_radau_includes_the_right_endpoint(stages):
    """What makes the scheme stiffly accurate."""
    points, _, _ = radau_points(stages)
    assert points[0] == pytest.approx(0.0)
    assert points[-1] == pytest.approx(1.0)
    assert len(points) == stages + 1


@pytest.mark.parametrize("stages", [1, 2, 3, 4, 5])
def test_radau_weights_are_a_partition_of_the_interval(stages):
    _, _, weights = radau_points(stages)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_radau_quadrature_is_exact_to_its_order(stages):
    """Order 2s-1 means polynomials up to degree 2s-2 integrate exactly."""
    points, _, weights = radau_points(stages)
    for degree in range(2 * stages - 1):
        exact = 1.0 / (degree + 1)
        quadrature = float(np.sum(weights * points ** degree))
        assert quadrature == pytest.approx(exact, abs=1e-10), degree


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_radau_differentiation_annihilates_constants(stages):
    """d/dt of a constant is zero, whatever the basis."""
    _, differentiation, _ = radau_points(stages)
    assert np.abs(differentiation.sum(axis=0)).max() < 1e-9


def test_radau_order_beats_hermite_simpson():
    assert RadauIIA(3).order > HermiteSimpson.order
    assert RadauIIA(2).order == HermiteSimpson.order


def test_radau_rejects_zero_stages():
    with pytest.raises(ValueError, match="at least one stage"):
        RadauIIA(0)


# --------------------------------------------------------------------------
# the schemes integrate correctly
# --------------------------------------------------------------------------
def test_hermite_simpson_defects_vanish_on_an_exact_solution():
    """Cubic Hermite is exact for a linear system integrated exactly.

    Take x' = -x, whose solution is an exponential; on a fine mesh the
    defect must be small and shrink at third order.
    """
    import casadi as ca

    def dynamics(state, control, time):
        return -state

    errors = []
    for n in (10, 20, 40):
        step = 1.0 / n
        times = np.linspace(0.0, 1.0, n + 1)
        exact = np.exp(-times).reshape(1, -1)
        state = ca.DM(exact)
        control = ca.DM.zeros(1, n + 1)
        control_mid = ca.DM.zeros(1, n)
        residual = HermiteSimpson.defects(dynamics, state, control,
                                          control_mid, times,
                                          np.full(n, step))
        errors.append(float(ca.norm_inf(residual)))

    # third order: halving the step should cut the defect by about 8
    assert errors[1] < errors[0] / 5.0
    assert errors[2] < errors[1] / 5.0


def test_hermite_simpson_defect_is_zero_for_a_constant_solution():
    import casadi as ca

    def dynamics(state, control, time):
        return 0.0 * state

    n = 5
    state = ca.DM.ones(1, n + 1) * 3.0
    residual = HermiteSimpson.defects(
        dynamics, state, ca.DM.zeros(1, n + 1), ca.DM.zeros(1, n),
        np.linspace(0.0, 1.0, n + 1), np.full(n, 0.2))
    assert float(ca.norm_inf(residual)) == pytest.approx(0.0, abs=1e-12)


def _radau_solve(scheme, dynamics, x0, n_intervals, horizon=1.0):
    """Integrate with Radau by solving the defect system as an NLP root.

    Radau is implicit, so the stage values are unknowns; this solves them
    rather than substituting a known answer, which is what actually
    exercises :meth:`RadauIIA.defects`.
    """
    import casadi as ca

    step = horizon / n_intervals
    times = np.linspace(0.0, horizon, n_intervals + 1)
    durations = np.full(n_intervals, step)

    state = ca.MX.sym("x", 1, n_intervals + 1)
    stages = [ca.MX.sym(f"k{k}", 1, scheme.n_stages)
              for k in range(n_intervals)]
    control = ca.DM.zeros(1, n_intervals)

    residual = scheme.defects(dynamics, state, stages, control, times,
                              durations)
    unknowns = ca.vertcat(ca.vec(state), *[ca.vec(s) for s in stages])
    system = ca.vertcat(state[0, 0] - x0, residual)

    solver = ca.rootfinder("r", "newton",
                           {"x": unknowns, "p": ca.MX.sym("p", 0, 1),
                            "g": system})
    guess = ca.DM.ones(unknowns.shape[0], 1) * x0
    solution = np.array(solver(guess, ca.DM(0, 1))).ravel()
    return solution[:n_intervals + 1]


@pytest.mark.parametrize("stages", [2, 3])
def test_radau_integrates_an_ode_correctly(stages):
    """The test the quadrature checks do not give.

    Exact weights and an exact differentiation matrix do not prove the
    defect assembly is right -- an index slip in the stage loop passes
    every algebraic check and still integrates the wrong problem.
    """
    scheme = RadauIIA(stages)

    def dynamics(state, control, time):
        return -2.0 * state

    got = _radau_solve(scheme, dynamics, 1.0, n_intervals=20)
    expected = np.exp(-2.0 * np.linspace(0.0, 1.0, 21))
    np.testing.assert_allclose(got, expected, atol=2e-4)


def test_radau_converges_faster_than_hermite_simpson():
    """Order 2s-1: three stages give fifth order against Hermite-Simpson's
    third, which is the entire reason for carrying the scheme."""
    def dynamics(state, control, time):
        return -2.0 * state

    errors = []
    for n in (4, 8):
        got = _radau_solve(RadauIIA(3), dynamics, 1.0, n_intervals=n)
        exact = np.exp(-2.0 * np.linspace(0.0, 1.0, n + 1))
        errors.append(np.abs(got - exact).max())

    # fifth order: halving the step should cut the error by about 32
    assert errors[1] < errors[0] / 16.0, errors


def test_radau_runs_on_a_phase_locked_mesh(timing):
    """The two pieces have to work together: the mesh is non-uniform, and
    Radau must not assume equal steps."""
    scheme = RadauIIA(3)
    mesh = phase_locked_mesh(timing, n_strokes=1, drive_intervals=3,
                             recovery_intervals=2)
    durations = np.array([interval.duration for interval in mesh])
    assert durations.std() > 1e-3, "mesh must actually be non-uniform"

    import casadi as ca

    times = np.concatenate([[mesh[0].start],
                            [interval.end for interval in mesh]])
    n = len(mesh)
    state = ca.MX.sym("x", 1, n + 1)
    stages = [ca.MX.sym(f"k{k}", 1, scheme.n_stages) for k in range(n)]
    residual = scheme.defects(lambda x, u, t: -x, state, stages,
                              ca.DM.zeros(1, n), times, durations)
    assert residual.shape[0] == n * (scheme.n_stages + 1)
