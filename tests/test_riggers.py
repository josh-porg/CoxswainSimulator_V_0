r"""The riggers: every oarlock carried off the hull on the side it rows.

There were none.  The oar pivoted at a point 0.83 m off the centreline
of a hull whose gunwale is 0.3 m out, with nothing drawn between -- and
on a bucket-rigged four the riggers are exactly what a coxswain looks
at to read the rig.  These hold that each lock has a frame, that it
starts on the gunwale and ends on the pin, and that it is on the pin's
side and not a mirror of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.viz.worldmesh import (GUNWALE, RIGGER, hull_solid,
                                    rigger_solids)


def _hocr_four():
    people = [RowerAnthropometry(mass=m, stature=s, sex="female")
              for m, s in ((54.4, 1.664), (54.4, 1.575),
                           (56.7, 1.613), (70.3, 1.600))]
    return catalog.coxed_four(
        rate=30.0, rower_mass=float(np.mean([p.mass for p in people])),
        rower_stature=float(np.mean([p.stature for p in people])),
        coxswain_mass=72.6, bow_loaded=True, anthropometry=people,
        rig_pattern="bucket, stbd stroke")


def _per_lock(boat, deck=0.30):
    """Rigger vertices grouped by the oarlock they belong to."""
    mesh = rigger_solids(boat, deck)
    assert mesh is not None
    locks = [lock for seat in boat.rig.seats for lock in seat.oarlocks]
    groups = []
    for lock in locks:
        pin = np.asarray(lock.position, dtype=float)
        # Everything within the rigger's own footprint of this pin.
        near = ((np.abs(mesh.vertices[:, 0] - pin[0]) < 0.6)
                & (np.sign(mesh.vertices[:, 1]) == np.sign(pin[1])))
        groups.append((lock, mesh.vertices[near]))
    return groups


def test_every_oarlock_has_a_rigger_and_it_is_on_the_locks_side():
    boat = _hocr_four()
    for lock, verts in _per_lock(boat):
        assert len(verts) > 0, lock
        pin = np.asarray(lock.position, dtype=float)
        # All of it on the pin's side of the centreline.
        assert np.all(np.sign(verts[:, 1]) == np.sign(pin[1])), lock
        # It reaches the pin...
        assert np.abs(verts[:, 1]).max() >= abs(pin[1]) - 0.03, lock
        # ...and it starts at the gunwale, well inboard of the pin.
        assert np.abs(verts[:, 1]).min() < 0.5 * abs(pin[1]), lock


def test_the_bucket_puts_the_middle_pair_of_riggers_on_one_side():
    """S-P-P-S from the stroke: seats 1 and 2 rig to port together."""
    boat = _hocr_four()
    sides = []
    for lock, verts in _per_lock(boat):
        sides.append(int(np.sign(verts[:, 1].mean())))
    assert sides == [-1, +1, +1, -1], sides


def test_a_rigger_runs_from_the_rail_up_to_the_pin():
    boat = _hocr_four()
    deck = 0.30
    for lock, verts in _per_lock(boat, deck):
        pin = np.asarray(lock.position, dtype=float)
        low = float(verts[:, 2].min())
        high = float(verts[:, 2].max())
        # Feet on the rail (within a tube radius), head at the pin.
        assert abs(low - (deck + GUNWALE)) < 0.06, (lock, low)
        assert high >= pin[2] - 0.02, (lock, high, pin)


def test_the_riggers_are_part_of_the_hull_mesh():
    """They ride with the hull, so they must be IN the hull part --
    a separate part would need its own transform every frame, and the
    first time someone forgot, the riggers would stay behind."""
    boat = _hocr_four()
    with_riggers = hull_solid(boat)
    bare = rigger_solids(boat)
    assert len(with_riggers.vertices) > len(bare.vertices)
    rigger = np.all(np.abs(with_riggers.colours - np.asarray(RIGGER))
                    < 1e-6, axis=1)
    assert rigger.sum() == len(bare.vertices)


@pytest.mark.parametrize("build, seats", [
    (lambda: catalog.eight(rate=32.0), 8),
    (lambda: catalog.double_scull(rate=30.0), 2),
])
def test_every_shell_gets_one_rigger_per_oarlock(build, seats):
    boat = build()
    locks = sum(len(seat.oarlocks) for seat in boat.rig.seats)
    groups = _per_lock(boat)
    assert len(groups) == locks
    assert all(len(v) > 0 for _l, v in groups)
    # A scull has two riggers per seat, one each side.
    if locks == 2 * seats:
        per_seat = {}
        for lock, verts in groups:
            per_seat.setdefault(round(float(lock.position[0]), 2),
                                set()).add(int(np.sign(verts[:, 1].mean())))
        assert all(s == {-1, +1} for s in per_seat.values()), per_seat
