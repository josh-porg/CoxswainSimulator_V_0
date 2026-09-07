r"""Every boat the menu offers must actually get as far as a frame.

The menu offered four shells.  Two of them -- the double and the single
-- crashed on launch, because a coxless boat has ``coxswain_position``
set to ``None`` and both the hull builder and the camera indexed it
without asking.  Nothing caught it: the packaging tests choose courses,
not boats, and every render check had been run on a four or an eight.

These build each boat and drive the parts that read the rig, with no GL
context anywhere, so the whole set runs in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.viz.menu import BOATS, build_boat

BOAT_KEYS = [key for key, _label, _seats, _coxed in BOATS]


@pytest.mark.parametrize("key", BOAT_KEYS)
def test_the_boat_can_be_built_and_drawn(key):
    from coxswain.viz.worldmesh import (crew_solids, hull_solid,
                                        oar_solids, viewpoint)

    boat, made = build_boat(key, 30.0)
    assert made == key, "the menu offers %s but the catalog built %s" % (
        key, made)

    hull = hull_solid(boat)
    assert len(hull.vertices) > 0

    seat, eye_height, facing = viewpoint(boat)
    assert np.isfinite(seat).all()
    assert 0.2 < eye_height < 1.2, eye_height
    assert facing in (-1.0, 1.0)

    for part in (crew_solids(boat, 0.3), oar_solids(boat, 0.3)):
        assert part is not None
        assert np.isfinite(part.vertices).all()


@pytest.mark.parametrize("key", BOAT_KEYS)
def test_the_camera_sits_in_the_boat(key):
    """The eye must be inside the hull's length and above the water."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from fpv import seat_camera

    boat, _made = build_boat(key, 30.0)
    state = np.zeros(12)
    eye, target, up = seat_camera(state, boat)
    assert np.isfinite(eye).all()
    assert 0.3 < eye[2] < 1.2, eye
    # Facing along the boat, one way or the other.
    forward = np.asarray(target) - np.asarray(eye)
    assert abs(forward[0]) > 0.9, forward


def test_a_coxless_boat_faces_astern_and_a_coxed_one_does_not():
    """Which way you look is the whole difference between the two.

    A sculler sits facing the stern and glances over a shoulder; a
    coxswain faces the bow.  Getting this backwards would put a single
    sculler staring at the water ahead of a boat travelling the other
    way, which no amount of scenery would make sensible.
    """
    from coxswain.viz.worldmesh import viewpoint

    for key, _label, _seats, coxed in BOATS:
        boat, _made = build_boat(key, 30.0)
        _seat, _height, facing = viewpoint(boat)
        assert facing == (1.0 if coxed else -1.0), key
