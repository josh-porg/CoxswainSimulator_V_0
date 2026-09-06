"""The plan view's geometry, with no window involved.

``coxswain.viz.planscene`` is deliberately backend-neutral -- world-frame
metres and numpy, no pygame -- so it can be tested without a display and
reused by a later ModernGL renderer.  These are the pieces that were
measured and are worth keeping measured: the polygon clip that took the
water layer from 14 ms a frame to 1.7, and the grid cull that keeps the
frame budget flat over 81,022 buildings.
"""

import numpy as np
import pytest

from coxswain.viz.planscene import (PlanLayer, boat_outline, clip_polygon,
                                    oar_lines)


def square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# -- clipping -------------------------------------------------------------

def test_a_polygon_inside_the_box_is_unchanged():
    inner = square(2.0, 2.0, 8.0, 8.0)
    out = clip_polygon(inner, 0.0, 0.0, 10.0, 10.0)
    assert len(out) == 4
    assert np.allclose(np.sort(out, axis=0), np.sort(inner, axis=0))


def test_a_polygon_outside_the_box_disappears():
    out = clip_polygon(square(20.0, 20.0, 30.0, 30.0), 0.0, 0.0, 10.0, 10.0)
    assert len(out) == 0


def test_a_polygon_larger_than_the_box_becomes_the_box():
    """The case that mattered: a shoreline ring covering the whole
    window, clipped down to its corners."""
    out = clip_polygon(square(-100.0, -100.0, 100.0, 100.0),
                       0.0, 0.0, 10.0, 10.0)
    assert len(out) == 4
    assert out[:, 0].min() == pytest.approx(0.0)
    assert out[:, 0].max() == pytest.approx(10.0)
    assert out[:, 1].min() == pytest.approx(0.0)
    assert out[:, 1].max() == pytest.approx(10.0)


def test_clipping_preserves_area_inside_the_box():
    """A straddling polygon keeps exactly the part that was showing."""
    def area(points):
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    out = clip_polygon(square(5.0, 5.0, 25.0, 25.0), 0.0, 0.0, 10.0, 10.0)
    assert area(out) == pytest.approx(25.0)      # the 5..10 square


def test_clipping_a_many_vertex_ring_keeps_it_closed():
    angle = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
    ring = np.column_stack([50.0 * np.cos(angle), 50.0 * np.sin(angle)])
    out = clip_polygon(ring, -10.0, -10.0, 10.0, 10.0)
    assert len(out) >= 4
    assert len(out) < len(ring)                  # the whole point
    assert out[:, 0].min() >= -10.0 - 1e-9
    assert out[:, 0].max() <= 10.0 + 1e-9


# -- culling --------------------------------------------------------------

def test_the_grid_finds_the_same_polygons_as_a_full_scan():
    rng = np.random.default_rng(7)
    polygons = []
    for _ in range(400):
        x, y = rng.uniform(-2000, 2000, 2)
        polygons.append(square(x, y, x + rng.uniform(5, 60),
                               y + rng.uniform(5, 60)))
    gridded = PlanLayer("test", polygons).index(cell=250.0)
    plain = PlanLayer("test", polygons).index(cell=250.0)
    plain.grid = None                             # force the full scan

    for _ in range(20):
        cx, cy = rng.uniform(-2000, 2000, 2)
        box = (cx - 300, cy - 300, cx + 300, cy + 300)
        assert sorted(gridded.visible(*box)) == sorted(plain.visible(*box))


def test_the_size_filter_drops_what_would_be_a_smudge():
    polygons = [square(0.0, 0.0, 1.0, 1.0),       # 1 m
                square(10.0, 10.0, 40.0, 40.0)]   # 30 m
    layer = PlanLayer("test", polygons).index()
    layer.min_pixels = 4.0
    box = (-100.0, -100.0, 100.0, 100.0)
    # At 0.5 px/m the small one is half a pixel and the big one is 15.
    assert list(layer.visible(*box, scale=0.5)) == [1]
    # Zoomed in, both earn their fill.
    assert len(layer.visible(*box, scale=10.0)) == 2


def test_the_cap_keeps_the_nearest():
    polygons = [square(x, 0.0, x + 5.0, 5.0) for x in (0., 50., 100., 200.)]
    layer = PlanLayer("test", polygons).index()
    layer.limit = 2
    found = layer.visible(-500.0, -500.0, 500.0, 500.0,
                          centre=np.array([0.0, 0.0]))
    assert sorted(found) == [0, 1]


# -- the boat -------------------------------------------------------------

def test_the_hull_outline_is_a_boat_shaped_thing():
    from coxswain.boats import catalog

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    hull = boat_outline(boat)
    assert hull.shape[1] == 2 and len(hull) >= 4
    length = hull[:, 0].max() - hull[:, 0].min()
    beam = hull[:, 1].max() - hull[:, 1].min()
    assert length == pytest.approx(boat.length, rel=0.05)
    assert 3.0 < length / beam < 60.0, "a shell is long and thin"


def test_the_oars_come_from_the_same_place_as_the_3d_scene():
    from coxswain.boats import catalog

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    lines, drive = oar_lines(boat, 0.0)
    assert len(lines) == sum(len(s.oarlocks) for s in boat.rig.seats)
    for oar in lines:
        assert oar.shape == (3, 2)               # handle, lock, blade
    assert isinstance(drive, bool)
    # The blade must move over a stroke, or the picture is a still life.
    later = oar_lines(boat, 0.6 * boat.timing.period)[0]
    assert not np.allclose(lines[0], later[0])
