"""The lateral structure of a shell's wake.

Everything else in :mod:`coxswain.hydro.wake` is a centreline quantity,
which answers "how bad is it to sit on a stern" and cannot answer "which
side should I go".  These tests pin the geometry that does.
"""

import numpy as np
import pytest

from coxswain.hydro.wake import PuddleWake

TRACK = 2.30


def wake(**kwargs):
    base = dict(drag=175.0, speed=3.9, period=2.0, n_blades=4)
    base.update(kwargs)
    return PuddleWake(**base)


# --------------------------------------------------------------------------
# spreading
# --------------------------------------------------------------------------
def test_puddles_grow_with_age():
    model = wake()
    radii = [float(model.puddle_radius(t)) for t in (0.0, 5.0, 20.0, 40.0)]
    assert np.all(np.diff(radii) > 0.0)
    assert radii[0] == pytest.approx(model.initial_radius, rel=1e-9)


def test_spreading_matches_the_core_law():
    """``d(sigma^2)/dt = 4 nu_t``, the same law the blob model uses.

    If these two diverge, the analytic and vortex models disagree about
    how fast a puddle grows and neither answer can be trusted.
    """
    model = wake()
    age = 30.0
    expected = np.sqrt(model.initial_radius ** 2
                       + 4.0 * model.eddy_viscosity * age)
    assert float(model.puddle_radius(age)) == pytest.approx(expected,
                                                            rel=1e-12)


# --------------------------------------------------------------------------
# the lateral overlap
# --------------------------------------------------------------------------
def test_on_their_line_is_full_overlap():
    """Both your blade tracks sit in both of theirs."""
    assert float(wake().lateral_overlap(40.0, 0.0, TRACK)) == \
        pytest.approx(1.0, abs=1e-9)


def test_less_than_a_metre_clears_the_puddles():
    """The headline tactical result, and it is pure geometry.

    The tracks are narrow, so a coxswain does not need to abandon the
    racing line to get clean water -- which matters on a river where the
    line is worth far more than the wake.
    """
    model = wake()
    assert float(model.lateral_overlap(40.0, 1.0, TRACK)) == \
        pytest.approx(0.0, abs=1e-9)


def test_there_is_a_second_wake_lane_at_twice_the_blade_track():
    """Your port blades land in their starboard puddles.

    Not obvious, and a coxswain drifting out to "get clear" can row
    straight back into dirty water on the way.
    """
    model = wake()
    clear = float(model.lateral_overlap(40.0, 3.0, TRACK))
    lane = float(model.lateral_overlap(40.0, 2.0 * TRACK, TRACK))
    assert clear == pytest.approx(0.0, abs=1e-9)
    assert lane > 0.3


def test_the_second_lane_is_weaker_than_sitting_on_their_line():
    """One track of yours is in it, not two."""
    model = wake()
    assert (float(model.lateral_overlap(40.0, 2.0 * TRACK, TRACK))
            < float(model.lateral_overlap(40.0, 0.0, TRACK)))


def test_far_enough_out_is_clean_again():
    model = wake()
    assert float(model.lateral_overlap(40.0, 4.0 * TRACK, TRACK)) == \
        pytest.approx(0.0, abs=1e-9)


def test_overlap_is_bounded():
    model = wake()
    for gap in (5.0, 40.0, 200.0):
        for separation in np.linspace(0.0, 12.0, 40):
            value = float(model.lateral_overlap(gap, separation, TRACK))
            assert 0.0 <= value <= 1.0


def test_the_clear_band_widens_as_puddles_spread():
    """Further back the puddles are bigger, so the clean gap is narrower."""
    model = wake()
    near = float(model.lateral_overlap(10.0, 0.55, TRACK))
    far = float(model.lateral_overlap(400.0, 0.55, TRACK))
    assert far >= near


def test_it_is_symmetric_in_side():
    model = wake()
    for separation in (0.5, 2.0, 4.6):
        assert float(model.lateral_overlap(40.0, separation, TRACK)) == \
            pytest.approx(
                float(model.lateral_overlap(40.0, -separation, TRACK)),
                abs=1e-12)


def test_a_wider_rig_moves_the_second_lane_out():
    """The lane is at twice the blade track, so it follows the rig."""
    model = wake()
    narrow, wide = 2.0, 3.0
    assert float(model.lateral_overlap(40.0, 2.0 * narrow, narrow)) > 0.3
    assert float(model.lateral_overlap(40.0, 2.0 * wide, wide)) > 0.3
    assert float(model.lateral_overlap(40.0, 2.0 * wide, narrow)) == \
        pytest.approx(0.0, abs=1e-9)
