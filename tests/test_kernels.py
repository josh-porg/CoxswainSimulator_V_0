"""The compiled path must be the same boat as the interpreted one.

``HullMesh.use_fast`` routes the wetted-surface sweep through a Numba
kernel.  It sums in a different order than ``ndarray.sum``, so it is
**not** bit-identical and cannot be -- which is exactly why it needs a
test of its own.  The failure this project cannot have is a trainer that
teaches a boat the analysis does not describe, and that failure would
arrive silently, as a slow drift rather than an exception.

So: the two paths are held to round-off of each other, per quantity and
over a whole trajectory, and the default path is held to the golden
trajectory bit-for-bit by ``tests/test_stepwise.py``.
"""

import os

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "golden_trajectory.npz")


def make_boat():
    return catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)


def test_the_kernel_agrees_with_numpy_on_one_pose():
    boat = make_boat()
    sim = RowingSimulator(boat, coxswain=Coxswain())
    state = sim.initial_state(surge_speed=4.0)
    position, attitude = state[0:3], state[3:6]

    boat.mesh.use_fast = False
    slow = boat.mesh.submerged(position, attitude)
    boat.mesh.use_fast = True
    fast = boat.mesh.submerged(position, attitude)

    for field in ("wetted_area", "transverse_area", "lateral_area",
                  "plan_area", "volume", "submerged_fraction"):
        a, b = getattr(slow, field), getattr(fast, field)
        assert a == pytest.approx(b, rel=1e-11), field
    for field in ("buoyancy_force", "buoyancy_moment", "centre_of_buoyancy"):
        a, b = getattr(slow, field), getattr(fast, field)
        assert np.allclose(a, b, rtol=1e-11, atol=1e-9), field


def test_the_kernel_agrees_across_poses():
    """Including heeled, pitched and yawed, where the panel clipping
    at the waterline actually does something."""
    boat = make_boat()
    poses = [
        (np.array([0.0, 0.0, 0.00]), np.array([0.00, 0.00, 0.0])),
        (np.array([0.0, 0.0, -0.05]), np.array([0.12, 0.03, 0.4])),
        (np.array([0.0, 0.0, 0.04]), np.array([-0.20, -0.05, -1.1])),
        (np.array([0.0, 0.0, 0.10]), np.array([0.35, 0.10, 2.0])),
    ]
    for position, attitude in poses:
        boat.mesh.use_fast = False
        slow = boat.mesh.submerged(position, attitude)
        boat.mesh.use_fast = True
        fast = boat.mesh.submerged(position, attitude)
        assert slow.volume == pytest.approx(fast.volume, rel=1e-11), attitude
        assert np.allclose(slow.buoyancy_force, fast.buoyancy_force,
                           rtol=1e-11, atol=1e-9), attitude
        assert np.allclose(slow.buoyancy_moment, fast.buoyancy_moment,
                           rtol=1e-10, atol=1e-8), attitude


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="no golden trajectory")
def test_the_two_paths_do_not_drift_apart_over_a_trajectory():
    """Round-off per step must stay round-off after 600 of them.

    A per-step difference of 1e-15 that compounded would be a different
    boat by the end of a race; this is the assertion that it does not.
    """
    golden = np.load(GOLDEN)
    fast = RowingSimulator(make_boat(), coxswain=Coxswain(),
                           fast=True).run(duration=12.0, dt=0.02,
                                          initial_state=golden["initial"])
    difference = np.abs(fast.states - golden["states"])
    scale = np.maximum(np.abs(golden["states"]).max(axis=1), 1e-9)
    assert (difference.max(axis=1) / scale).max() < 1e-9


def test_fast_is_off_unless_asked_for():
    """The studies must stay bit-reproducible by default."""
    boat = make_boat()
    RowingSimulator(boat, coxswain=Coxswain())
    assert boat.mesh.use_fast is False


def test_the_kernel_is_actually_compiled():
    """A silent fall back to the interpreted reference would look like a
    working system that is ten times slower, which is the kind of thing
    that goes unnoticed until the frame rate is the complaint."""
    from coxswain.hydro import _hullkernel
    if not _hullkernel.HAVE_NUMBA:
        pytest.skip("numba is not installed")
    assert hasattr(_hullkernel.submerged_kernel, "py_func")
