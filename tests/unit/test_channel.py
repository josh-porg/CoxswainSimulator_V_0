"""Unit tests for channel extraction from depth contours."""

import numpy as np
import pytest

from coxswain.river import channel


@pytest.fixture
def ring_points():
    """A filled annulus: sampled water with a hole an alpha shape must keep.

    Concentric rings every 8 m from r=40 to r=120, mimicking contour lines
    spaced across a channel.  The hole inside r=40 has no samples at all,
    which is the situation the Charles presents wherever the river bends
    around land.
    """
    points = []
    for radius in np.arange(40.0, 121.0, 8.0):
        count = max(int(2 * np.pi * radius / 4.0), 8)
        angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        points.append(np.column_stack([radius * np.cos(angle),
                                       radius * np.sin(angle)]))
    return np.vstack(points)


# --------------------------------------------------------------------------
# alpha shape
# --------------------------------------------------------------------------
def test_alpha_shape_excludes_the_hole_of_an_annulus(ring_points):
    """A convex hull would fill the middle; an alpha shape must not.

    This is the property the Charles needs: the contour vertices trace the
    banks, and a convex hull of them spans land wherever the river bends.
    """
    query = np.array([[0.0, 0.0], [80.0, 0.0], [200.0, 0.0]])
    inside = channel.alpha_shape_mask(ring_points, query, alpha=30.0)
    assert not inside[0], "centre of the annulus must be outside"
    assert inside[1], "the ring itself must be inside"
    assert not inside[2], "far outside must be outside"


def test_larger_alpha_fills_more(ring_points):
    query = np.array([[0.0, 0.0]])
    assert not channel.alpha_shape_mask(ring_points, query, alpha=30.0)[0]
    assert channel.alpha_shape_mask(ring_points, query, alpha=500.0)[0]


# --------------------------------------------------------------------------
# raster construction
# --------------------------------------------------------------------------
@pytest.fixture
def straight_channel():
    """A 600 m straight channel, 3 m deep in the middle, shoaling to 0.3."""
    east = np.arange(0.0, 600.0, 8.0)
    offsets = np.linspace(-45.0, 45.0, 25)
    points, depths = [], []
    for x in east:
        for y in offsets:
            points.append([x, y])
            depths.append(max(0.3, 3.4 - 3.1 * (abs(y) / 45.0) ** 2))
    return channel.build_channel(np.array(points), np.array(depths),
                                 resolution=4.0, alpha=40.0)


def test_water_mask_covers_the_surveyed_extent(straight_channel):
    assert straight_channel.water.any()
    assert straight_channel.water_area > 0.7 * (600.0 * 90.0)


def test_navigable_is_a_subset_of_water(straight_channel):
    assert np.all(straight_channel.navigable <= straight_channel.water)


def test_navigable_excludes_the_shallow_edges(straight_channel):
    """The 1.22 m threshold must cut off the shoaling banks."""
    assert straight_channel.navigable_area < straight_channel.water_area


def test_depth_is_nan_outside_the_water(straight_channel):
    outside = ~straight_channel.water
    assert np.all(np.isnan(straight_channel.depth[outside]))


def test_clearance_is_zero_outside_and_positive_in_the_middle(
        straight_channel):
    assert straight_channel.clearance[~straight_channel.navigable].max() == 0.0
    assert straight_channel.clearance.max() > 10.0


def test_navigable_keeps_only_the_largest_component():
    """A deep pocket cut off behind a shoal is not navigable."""
    points, depths = [], []
    for x in np.arange(0.0, 300.0, 6.0):
        for y in np.arange(-40.0, 40.0, 6.0):
            points.append([x, y])
            # main channel plus an isolated deep pool far to the north
            deep = 3.0 if abs(y) < 20.0 else 0.4
            depths.append(deep)
    for x in np.arange(100.0, 140.0, 6.0):
        for y in np.arange(120.0, 160.0, 6.0):
            points.append([x, y])
            depths.append(4.0)
    raster = channel.build_channel(np.array(points), np.array(depths),
                                   resolution=4.0, alpha=40.0)
    from scipy.ndimage import label
    _, count = label(raster.navigable)
    assert count == 1


# --------------------------------------------------------------------------
# centreline
# --------------------------------------------------------------------------
def test_centreline_stays_navigable(straight_channel):
    line = straight_channel.centreline()
    assert all(straight_channel.is_navigable(x, y) for x, y in line)


def test_centreline_runs_down_the_middle(straight_channel):
    """Greatest clearance on a symmetric channel is the axis."""
    line = straight_channel.centreline()
    assert np.abs(line[:, 1]).max() < 12.0


def test_centreline_spans_the_reach(straight_channel):
    line = straight_channel.centreline()
    assert line[:, 0].max() - line[:, 0].min() > 500.0


def test_half_width_along_is_positive_and_capped(straight_channel):
    line = straight_channel.centreline()
    widths = straight_channel.half_width_along(line, cap=30.0)
    assert np.all(widths > 0.0)
    assert np.all(widths <= 30.0)


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------
def test_crop_shrinks_the_raster(straight_channel):
    points = np.array([[100.0, 0.0], [200.0, 0.0]])
    cropped = straight_channel.crop(points, margin=50.0)
    assert cropped.depth.size < straight_channel.depth.size
    assert cropped.east.min() >= 100.0 - 50.0 - cropped.resolution
    assert cropped.east.max() <= 200.0 + 50.0 + cropped.resolution


def test_crop_preserves_values(straight_channel):
    points = np.array([[100.0, 0.0], [200.0, 0.0]])
    cropped = straight_channel.crop(points, margin=40.0)
    x, y = 150.0, 0.0
    assert cropped.is_navigable(x, y) == straight_channel.is_navigable(x, y)
    assert cropped.clearance_at(x, y) == pytest.approx(
        straight_channel.clearance_at(x, y), abs=1e-9)


def test_crop_rejects_a_region_off_the_raster(straight_channel):
    with pytest.raises(ValueError, match="does not overlap"):
        straight_channel.crop(np.array([[99999.0, 99999.0]]), margin=1.0)
