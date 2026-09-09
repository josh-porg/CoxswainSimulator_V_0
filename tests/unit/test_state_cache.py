r"""A State's rotation is built once, and it is the same rotation.

``velocity_hull`` and ``omega_hull`` rebuilt the attitude matrix from
its three angles on every access -- a dozen times per derivative.  A
State is immutable, so the matrix is memoised on the instance.  What
must not change is the arithmetic: the golden trajectory is held bit
for bit, so the hull-frame products must be the SAME floats, not merely
close ones.  ``abs_to_hull`` is defined as ``hull_to_abs(...).T``, which
is what makes that true.
"""

from __future__ import annotations

import numpy as np

from coxswain.core import frames
from coxswain.core.state import State


def _state(seed: int) -> State:
    rng = np.random.default_rng(seed)
    return State.from_vector(np.concatenate([
        rng.normal(0.0, 5.0, 3), rng.normal(0.0, 0.3, 3),
        rng.normal(0.0, 3.0, 3), rng.normal(0.0, 0.5, 3)]))


def test_the_rotation_is_the_same_matrix_and_built_once():
    state = _state(1)
    first = state.rot_hull_to_abs
    second = state.rot_hull_to_abs
    assert first is second, "memoised on the instance"
    assert np.array_equal(first, frames.hull_to_abs(state.attitude))


def test_hull_frame_products_are_bit_identical_to_the_old_path():
    for seed in range(20):
        state = _state(seed)
        old_v = frames.abs_to_hull(state.attitude) @ state.velocity
        old_w = frames.abs_to_hull(state.attitude) @ state.omega
        assert np.array_equal(state.velocity_hull, old_v)
        assert np.array_equal(state.omega_hull, old_w)


def test_each_state_has_its_own_cache():
    a, b = _state(3), _state(4)
    assert not np.array_equal(a.rot_hull_to_abs, b.rot_hull_to_abs)
    # and a State built from the same vector agrees exactly
    c = State.from_vector(a.to_vector())
    assert np.array_equal(a.rot_hull_to_abs, c.rot_hull_to_abs)


def test_the_state_is_still_frozen():
    state = _state(5)
    try:
        state.position = np.zeros(3)
    except Exception:
        return
    raise AssertionError("State must stay immutable")
