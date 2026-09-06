"""The first-person world, as geometry, with no GL involved.

``coxswain.viz.worldmesh`` imports no graphics library on purpose, so the
mesh it builds can be checked without a display -- which matters here
because the machine this was written on has no display at all, and the
windowed path can only be exercised by a person.  What *can* be pinned is
the geometry: that triangles wind the right way, that the ground stops at
the waterline, and that the camera sits where a coxswain's head is.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.viz.worldmesh import (MeshPart, WorldMesh, box_solid,
                                    building_walls, line_markers,
                                    water_plane)


def test_a_mesh_part_interleaves_for_one_buffer():
    part = box_solid((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    blob = part.interleaved()
    assert blob.shape == (len(part.vertices), 9)
    assert blob.dtype == np.float32
    assert part.triangles == len(part.vertices) // 3


def test_the_water_plane_faces_up():
    """If it faced down, back-face culling would eat the river."""
    part = water_plane((0.0, 0.0), reach=100.0)
    triangle = part.vertices[:3]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    assert normal[2] > 0.0


def test_the_water_plane_covers_the_boat():
    part = water_plane((500.0, -200.0), reach=1000.0)
    assert part.vertices[:, 0].min() <= -400.0
    assert part.vertices[:, 0].max() >= 1400.0
    assert np.allclose(part.vertices[:, 2], 0.0)


def test_buildings_are_walls_and_not_roofs():
    """Twelve triangles for a four-sided building: two per wall, no top.

    From 0.55 m off the water a roof is never visible, and skipping the
    top faces halves the count for nothing you can see.
    """
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    part = building_walls([square], [12.0])
    assert part is not None
    assert part.triangles == 8            # 4 walls x 2
    assert part.vertices[:, 2].max() == pytest.approx(12.0)
    assert part.vertices[:, 2].min() == pytest.approx(0.0)


def test_a_short_building_is_skipped():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    assert building_walls([square], [1.0]) is None


def test_buildings_outside_the_box_are_skipped():
    near = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    far = near + 5000.0
    part = building_walls([near, far], [10.0, 10.0],
                          box=(-100.0, -100.0, 100.0, 100.0))
    assert part.triangles == 8            # only the near one


def test_the_markers_follow_the_course_at_a_spacing():
    line = np.column_stack([np.linspace(0.0, 300.0, 60), np.zeros(60)])
    part = line_markers(line, spacing=30.0)
    assert part is not None
    # 30 m spacing over 300 m, excluding the start: nine posts.
    assert part.triangles == 9 * 12
    assert part.vertices[:, 2].min() >= 0.0


def test_markers_need_a_line():
    assert line_markers(np.zeros((1, 2))) is None


def test_the_land_stops_at_the_waterline():
    """Cells touching water are dropped, because lidar over water is not
    water (SOURCES §105) and a bank that dives under the river looks
    exactly like a bank that is wrong."""
    from coxswain.viz.worldmesh import land_mesh

    class FlatTerrain:
        def height_above_water(self, east, north):
            return np.full(np.shape(east), 3.0)

    def wet_at(east, north):
        return east > 50.0            # water to the east

    part = land_mesh(FlatTerrain(), wet_at, (0.0, 0.0, 100.0, 40.0), step=10.0)
    assert part is not None
    # Nothing is emitted past the waterline, allowing for the cell that
    # straddles it being dropped whole.
    assert part.vertices[:, 0].max() <= 50.0


def test_the_seat_is_where_a_coxswains_head_is():
    """Bow-loader: the cox is forward of the crew, lying back, and the
    eye is above the seat.  If this drifts the whole view is wrong and
    nothing else would say so."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from fpv import seat_camera

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    state = np.zeros(12)
    eye, target, up = seat_camera(state, boat)
    # Forward of amidships, and above the water.
    assert eye[0] > 3.0
    assert eye[2] == pytest.approx(boat.rig.coxswain_position[2]
                                   + boat.rig.coxswain_eye_height)
    assert np.allclose(up, [0.0, 0.0, 1.0])
    assert target[0] > eye[0]                 # looking forward


def test_the_view_rolls_with_the_boat():
    """The horizon tipping is most of what says the boat is alive."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from fpv import seat_camera

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    level = np.zeros(12)
    heeled = np.zeros(12)
    heeled[3] = np.radians(8.0)               # roll
    assert not np.allclose(seat_camera(level, boat)[2],
                           seat_camera(heeled, boat)[2])


def test_the_view_turns_with_the_boat():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from fpv import seat_camera

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    straight = np.zeros(12)
    turned = np.zeros(12)
    turned[5] = np.radians(90.0)              # yaw
    ahead_straight = seat_camera(straight, boat)[1] - seat_camera(straight, boat)[0]
    ahead_turned = seat_camera(turned, boat)[1] - seat_camera(turned, boat)[0]
    assert ahead_straight[0] == pytest.approx(1.0, abs=1e-6)
    assert ahead_turned[1] == pytest.approx(1.0, abs=1e-6)


def test_the_projection_is_a_projection():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from fpv import look_at, perspective

    matrix = perspective(70.0, 1.6, 0.25, 2600.0)
    assert matrix[3, 2] == pytest.approx(-1.0)
    view = look_at([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    # A point straight ahead lands in front of the camera.
    ahead = view @ np.array([10.0, 0.0, 1.0, 1.0])
    assert ahead[2] < 0.0
