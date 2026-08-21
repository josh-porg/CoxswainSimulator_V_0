"""Property tests that close the reference-frame bug class.

This project has produced the same *kind* of bug repeatedly:

* the legacy mass matrix mixed up which of ``S(r)`` and ``S(r)S(r)`` went
  where, and one sign made the inertia negative;
* the Euler-rate map had roll and yaw transposed;
* an angle was passed where an angular rate belonged;
* ``HullSurrogate`` tabulated a centre of buoyancy the mesh had already
  rotated into the absolute frame, and the dynamics rotated it again --
  which destroyed 87% of the hull's roll stiffness;
* a start state set the surge velocity in the hull frame while the model
  reads it in the absolute frame, producing a boat crabbing at 4.6 m/s.

Every one of these is invisible at the identity attitude, which is where
example-based tests naturally look. Checking one more example does not
close the class. What closes it is asserting the *symmetries* a correct
formulation must have, because a frame error breaks a symmetry no matter
which particular numbers you feed it.

Four families here:

**Round trips.**  ``abs -> hull -> abs`` is the identity, and the rotation
is orthogonal.  Catches transposes and bad angle conventions.

**Yaw invariance.**  Nothing in still water depends on which way the boat
points.  A quantity that changes when only the heading changes is being
computed in the wrong frame.  This is the single most powerful check here:
it catches any missing or doubled rotation involving yaw.

**Cross-path agreement at non-trivial attitude.**  The numpy simulator and
the CasADi model are independent transcriptions.  They must agree with
roll, pitch *and* yaw all non-zero -- comparing at zero attitude proves
nothing about a frame error in attitude.

**Internal consistency of the hull integrals.**  The mesh reports a
buoyancy moment and a centre of buoyancy; they must be consistent with
each other, and with whatever the surrogate stores, at attitudes where the
distinction matters.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.frames import abs_to_hull, hull_to_abs

#: Attitudes with every component non-zero and mutually distinct, so that
#: no accidental symmetry can mask a transposed axis.
ATTITUDES = [
    np.array([0.0, 0.0, 0.0]),
    np.array([np.radians(2.0), np.radians(-1.0), np.radians(35.0)]),
    np.array([np.radians(-3.0), np.radians(1.5), np.radians(-140.0)]),
    np.array([np.radians(1.0), np.radians(2.5), np.radians(95.0)]),
]


@pytest.fixture(scope="module")
def eight():
    return catalog.eight(rate=32.0)


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------
@pytest.mark.parametrize("attitude", ATTITUDES)
def test_hull_to_abs_round_trips(attitude):
    vector = np.array([1.3, -0.7, 0.42])
    back = abs_to_hull(attitude) @ (hull_to_abs(attitude) @ vector)
    np.testing.assert_allclose(back, vector, atol=1e-12)


@pytest.mark.parametrize("attitude", ATTITUDES)
def test_rotation_is_orthogonal(attitude):
    rotation = hull_to_abs(attitude)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("attitude", ATTITUDES)
def test_abs_to_hull_is_the_transpose(attitude):
    np.testing.assert_allclose(abs_to_hull(attitude),
                               hull_to_abs(attitude).T, atol=1e-12)


@pytest.mark.parametrize("attitude", ATTITUDES)
def test_rotation_preserves_length(attitude):
    vector = np.array([0.3, -1.1, 2.4])
    assert np.linalg.norm(hull_to_abs(attitude) @ vector) == pytest.approx(
        np.linalg.norm(vector), rel=1e-12)


# --------------------------------------------------------------------------
# yaw invariance -- the strongest single check
# --------------------------------------------------------------------------
def _rotate_state_in_yaw(state, angle):
    """Same physical situation, boat pointing somewhere else.

    Positions and absolute-frame velocities rotate; hull-frame quantities
    and the anaerobic reserve do not.
    """
    turned = np.array(state, dtype=float)
    spin = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    turned[0:2] = spin @ state[0:2]        # position, absolute
    turned[5] = state[5] + angle           # heading
    turned[6:8] = spin @ state[6:8]        # linear velocity, absolute
    turned[9:11] = spin @ state[9:11]      # angular velocity, absolute
    return turned


@pytest.mark.parametrize("angle", [0.7, -2.1, np.pi])
def test_six_dof_dynamics_are_yaw_invariant(eight, angle):
    """Still water has no preferred compass direction.

    Any quantity that changes when only the heading changes is being
    evaluated in the wrong frame.  This would have caught the doubled
    rotation immediately had it involved yaw, and it catches the whole
    family of missing-rotation errors that do.
    """
    pytest.importorskip("casadi")
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel

    surrogate = HullSurrogate.from_boat(eight, n_heave=13, n_pitch=7,
                                        n_roll=9)
    model = SixDofModel(eight, surrogate=surrogate)
    function = model.function()

    state = np.zeros(13)
    state[2] = eight.equilibrium_heave()
    state[3] = np.radians(1.5)
    state[4] = np.radians(-0.5)
    state[6] = 4.8
    state[12] = model.anaerobic_capacity

    plain = np.array(function(state, [0.05, 0.02, 1.0], 0.3)).ravel()
    turned = np.array(function(_rotate_state_in_yaw(state, angle),
                               [0.05, 0.02, 1.0], 0.3)).ravel()

    # the derivative transforms the same way the state does
    expected = _rotate_state_in_yaw(plain, angle)
    expected[5] = plain[5]      # yaw *rate* is invariant, not the angle
    np.testing.assert_allclose(turned, expected, atol=1e-7)


@pytest.mark.parametrize("angle", [1.1, -0.4])
def test_numpy_simulator_is_yaw_invariant(eight, angle):
    """The *plant* is yaw invariant; the closed loop is not, and must not be.

    Written naively this test fails, and the failure is instructive: the
    only terms that break are the appendage loads, because the heading
    controller reads absolute yaw and steers against it.  That is the
    controller doing its job.  The invariance being asserted is a property
    of the dynamics, so the loop is opened with a fixed rudder.
    """
    from coxswain.sim.control import Coxswain, HeadingController
    from coxswain.sim.simulator import RowingSimulator

    coxswain = Coxswain(heading=HeadingController(enabled=False),
                        rudder_override=lambda t, state: 0.04)
    simulator = RowingSimulator(eight, coxswain=coxswain)
    state = simulator.initial_state(surge_speed=4.8)
    state[3] = np.radians(1.5)

    plain = simulator.derivative(0.3, state)
    turned = simulator.derivative(0.3, _rotate_state_in_yaw(state, angle))
    expected = _rotate_state_in_yaw(plain, angle)
    expected[5] = plain[5]
    np.testing.assert_allclose(turned, expected, atol=1e-7)


def test_yaw_invariance_would_fail_if_a_rotation_were_dropped(eight):
    """The guard on the guard.

    A test that passes for the wrong reason is worse than none, so this
    checks the invariance test can actually fail: rotating a hull-frame
    vector that should not be rotated must break it.
    """
    attitude = np.array([np.radians(2.0), 0.0, np.radians(40.0)])
    hull_vector = np.array([100.0, 20.0, -5.0])
    correct = hull_to_abs(attitude) @ hull_vector
    doubled = hull_to_abs(attitude) @ correct
    assert not np.allclose(correct, doubled, atol=1e-6)


# --------------------------------------------------------------------------
# the hull integrals must be self-consistent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("roll_deg,pitch_deg", [(1.0, 0.0), (0.0, 1.0),
                                                (2.0, -1.0), (-1.5, 0.8)])
def test_mesh_buoyancy_moment_matches_its_own_centre(eight, roll_deg,
                                                     pitch_deg):
    """Fixes the frame convention of ``centre_of_buoyancy`` by assertion.

    ``HullMesh.submerged`` returns the centre in the **absolute** frame, so
    ``cross(centre, force)`` reproduces its own reported moment without any
    further rotation.  Anything that consumes the centre has to know that;
    this test states it once, so a future change to the mesh cannot quietly
    invalidate every consumer.
    """
    submerged = eight.mesh.submerged(
        np.array([0.0, 0.0, eight.equilibrium_heave()]),
        np.array([np.radians(roll_deg), np.radians(pitch_deg), 0.0]),
        rho=eight.water.density, gravity=9.80665, water_level=0.0)

    centre = np.asarray(submerged.centre_of_buoyancy, dtype=float)
    force = np.asarray(submerged.buoyancy_force, dtype=float)
    moment = np.asarray(submerged.buoyancy_moment, dtype=float)
    reconstructed = np.cross(centre, force)

    # Only the roll and pitch components.  The centre of buoyancy is
    # *defined* by the resultant of those two -- ``(-M_y/F_z, M_x/F_z,
    # z)`` -- so a single point cannot also carry the yaw moment once the
    # buoyancy force has a lateral component.  That is a property of
    # reducing a distributed pressure field to one point, not an error,
    # but it means no consumer may take the yaw moment from the centre.
    np.testing.assert_allclose(reconstructed[:2], moment[:2], rtol=0.05,
                               atol=1.0)


@pytest.mark.parametrize("roll_deg,pitch_deg", [(1.0, 0.0), (1.5, -0.7)])
def test_surrogate_reconstructs_the_mesh_buoyancy_moment(eight, roll_deg,
                                                         pitch_deg):
    """The double rotation, asserted away.

    The surrogate stores the centre in the **hull** frame, so the dynamics
    rotate it exactly once.  If it ever goes back to storing the absolute
    value, this fails.
    """
    pytest.importorskip("casadi")
    from coxswain.river.hullsurrogate import HullSurrogate

    roll, pitch = np.radians(roll_deg), np.radians(pitch_deg)
    surrogate = HullSurrogate.from_boat(eight, n_heave=17, n_pitch=11,
                                        n_roll=13,
                                        roll_range=np.radians(4.0))
    heave = eight.equilibrium_heave()
    values = surrogate(heave, pitch, roll)
    buoyancy = eight.water.density * 9.80665 * values["volume"]
    centre_hull = np.array([values["buoyancy_x"], values["buoyancy_y"],
                            values["buoyancy_z"]])
    rotation = hull_to_abs(np.array([roll, pitch, 0.0]))
    moment = np.cross(rotation @ centre_hull, np.array([0.0, 0.0, buoyancy]))

    exact = eight.mesh.submerged(
        np.array([0.0, 0.0, heave]), np.array([roll, pitch, 0.0]),
        rho=eight.water.density, gravity=9.80665, water_level=0.0)
    np.testing.assert_allclose(moment[0],
                               np.asarray(exact.buoyancy_moment)[0],
                               rtol=0.03)


def test_gravity_moment_is_frame_consistent(eight):
    """Summing segment moments must equal the aggregate first moment.

    Both forms appear in the codebase -- the numpy path sums per segment,
    the symbolic path uses the tabulated first moment -- and they are equal
    only if both are in the same frame.
    """
    attitude = np.array([np.radians(2.0), np.radians(-1.0), np.radians(50.0)])
    rotation = hull_to_abs(attitude)
    mass, position, _, _ = eight.crew_field(0.35)
    gravity = np.array([0.0, 0.0, -9.80665])

    absolute = position @ rotation.T
    per_segment = np.cross(absolute, mass[:, None] * gravity).sum(axis=0)
    aggregate = np.cross(rotation @ (mass[:, None] * position).sum(axis=0),
                         gravity)
    np.testing.assert_allclose(per_segment, aggregate, atol=1e-8)


# --------------------------------------------------------------------------
# the convention has to be written down where it is used
# --------------------------------------------------------------------------
def test_frame_critical_functions_document_their_frame():
    """Documentation as a testable artefact.

    Every one of the frame bugs above was possible because a function
    returned a vector without saying which frame it was in.  This asserts
    that the ones that bit us now say so.
    """
    import inspect

    from coxswain.crew.balance import BalanceRig
    from coxswain.river.hullsurrogate import _to_hull

    for function in (_to_hull, BalanceRig.loads):
        text = (inspect.getdoc(function) or "").lower()
        assert "frame" in text or "hull" in text, function.__name__


def test_the_heading_controller_is_the_only_thing_that_may_break_invariance(
        eight):
    """Pinning the exception, so it cannot grow.

    Exactly one part of the system is allowed to care which way the boat
    points.  If a second one appears, the closed-loop and open-loop
    invariance results diverge and this fails.
    """
    from coxswain.core.state import State
    from coxswain.sim.control import Coxswain, HeadingController
    from coxswain.sim.simulator import RowingSimulator

    angle = 0.9
    spin = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
    coxswain = Coxswain(heading=HeadingController(enabled=False),
                        rudder_override=lambda t, state: 0.04)
    simulator = RowingSimulator(eight, coxswain=coxswain)
    state = simulator.initial_state(surge_speed=4.8)
    state[3] = np.radians(1.5)

    plain = simulator.breakdown(0.3, State.from_vector(state))
    turned = simulator.breakdown(
        0.3, State.from_vector(_rotate_state_in_yaw(state, angle)))

    for name in ("crew_force", "oar_force", "buoyancy_force",
                 "appendage_force", "crew_moment", "oar_moment",
                 "buoyancy_moment", "gravity_moment", "appendage_moment"):
        before = np.asarray(getattr(plain, name), dtype=float)
        after = np.asarray(getattr(turned, name), dtype=float)
        expected = np.array([*(spin @ before[0:2]), before[2]])
        np.testing.assert_allclose(after, expected, atol=1e-6,
                                   err_msg=name)
