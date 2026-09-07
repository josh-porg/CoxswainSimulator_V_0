r"""Both blades feather hollow-up, and both drive hollow-astern.

The starboard blades were feathering upside down: the roll turned the
same way on both sides of the boat while the dish's sign flips between
them, so one side came round hollow-to-the-sky and the other
hollow-to-the-water.  This checks every oar in an eight at both ends of
the roll.
"""

from __future__ import annotations

import numpy as np


def _frames(boat, roll, t=0.3):
    from coxswain.viz.worldmesh import blade_frame, oar_pose

    out = []
    for handle, pivot, blade, _lift, _drive in oar_pose(boat, t):
        axis = blade - pivot
        axis = axis / np.linalg.norm(axis)
        vertical = np.array([0.0, 0.0, 1.0])
        rim = vertical - axis * float(axis @ vertical)
        rim = rim / np.linalg.norm(rim)
        if rim[2] < 0.0:
            rim = -rim
        lie = np.cross(axis, rim)
        lie = lie / np.linalg.norm(lie)
        toward_bow = 1.0 if float(np.cross(axis, rim)[0]) > 0.0 else -1.0
        edge, dish = blade_frame(axis, rim, lie, toward_bow, roll)
        out.append((float(pivot[1]), edge, dish))
    return out


def test_every_blade_feathers_hollow_up():
    from coxswain.boats import catalog

    boat = catalog.eight(rate=30, rower_mass=68, rower_stature=1.7,
                         coxswain_mass=68)
    sides = set()
    for side_y, edge, dish in _frames(boat, roll=1.0):
        # Feathered: the width lies flat, and the dish's back bulges
        # DOWN, so the hollow faces the sky.
        assert abs(edge[2]) < 0.15, edge
        assert dish[2] < -0.9, (side_y, dish)
        sides.add(np.sign(side_y))
    assert sides == {-1.0, 1.0}          # both sides were checked


def test_every_blade_drives_hollow_astern():
    from coxswain.boats import catalog

    boat = catalog.eight(rate=30, rower_mass=68, rower_stature=1.7,
                         coxswain_mass=68)
    for side_y, edge, dish in _frames(boat, roll=0.0):
        # Squared: width vertical, back of the dish toward the bow.
        assert edge[2] > 0.9, edge
        assert dish[0] > 0.8, (side_y, dish)
