r"""Tiling and frustum culling of the static world.

The world pass measured vertex-bound -- unchanged at a quarter of the
pixels -- and from the seat about half the world is behind the camera.
Culling by tile is only correct if every triangle lands in exactly one
tile, every tile's sphere holds all of its vertices, and the plane test
keeps what the camera can see.  These hold each of those.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="module")
def fpv():
    import fpv as module
    return module


def _part(n_tri=4000, seed=0):
    from coxswain.viz.worldmesh import MeshPart

    rng = np.random.default_rng(seed)
    centres = rng.uniform(-2000.0, 2000.0, (n_tri, 1, 2))
    tri = centres + rng.uniform(-4.0, 4.0, (n_tri, 3, 2))
    z = rng.uniform(0.0, 40.0, (n_tri, 3, 1))
    vertices = np.concatenate([tri, z], axis=2).reshape(-1, 3).astype("f4")
    colours = rng.uniform(0, 1, (len(vertices), 3)).astype("f4")
    normals = np.tile(np.array([0, 0, 1], "f4"), (len(vertices), 1))
    return MeshPart("test", vertices, colours, normals)


def test_every_triangle_is_in_exactly_one_tile_and_ranges_are_contiguous(fpv):
    part = _part()
    tiles = fpv.tile_part(part, 350.0)
    assert tiles.total == len(part.vertices)
    assert tiles.count.sum() == tiles.total
    # ranges tile the buffer end to end with no gap and no overlap
    order = np.argsort(tiles.first)
    first, count = tiles.first[order], tiles.count[order]
    assert first[0] == 0
    assert np.array_equal(first[1:], first[:-1] + count[:-1])
    assert np.all(count % 3 == 0), "a range never splits a triangle"


def test_the_packed_buffer_is_the_same_triangles_reordered(fpv):
    """Same multiset of vertices: nothing dropped, nothing invented."""
    part = _part(600, seed=3)
    tiles = fpv.tile_part(part, 200.0)
    rec = np.frombuffer(tiles.packed, dtype=[("p", "f4", 3), ("n", "i1", 4),
                                             ("c", "u1", 4)])
    got = np.sort(rec["p"].round(3).view([("x", "f4"), ("y", "f4"), ("z", "f4")]),
                  axis=0)
    want = np.sort(np.asarray(part.vertices, "f4").round(3).view(
        [("x", "f4"), ("y", "f4"), ("z", "f4")]), axis=0)
    assert np.array_equal(got, want)


def test_every_tile_sphere_holds_every_vertex_in_its_range(fpv):
    part = _part(2000, seed=5)
    tiles = fpv.tile_part(part, 300.0)
    rec = np.frombuffer(tiles.packed, dtype=[("p", "f4", 3), ("n", "i1", 4),
                                             ("c", "u1", 4)])
    pts = rec["p"]
    for k in range(len(tiles.first)):
        a, b = int(tiles.first[k]), int(tiles.first[k] + tiles.count[k])
        d = np.linalg.norm(pts[a:b] - tiles.centre[k], axis=1)
        assert d.max() <= tiles.radius[k] + 1e-3, k


def _look(eye, target, fov=60.0, aspect=1.7, near=0.25, far=2600.0):
    """A view-projection matrix looking from eye at target, +z up."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = r, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    t = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = t / aspect
    proj[1, 1] = t
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj @ view


def test_frustum_planes_keep_what_is_in_front_and_drop_what_is_behind(fpv):
    vp = _look(eye=(0, 0, 2), target=(100, 0, 2))
    planes = fpv.frustum_planes(vp)
    assert planes.shape == (6, 4)

    class T:
        pass

    tiles = T()
    tiles.centre = np.array([[300.0, 0.0, 5.0],      # ahead: visible
                             [-300.0, 0.0, 5.0],     # behind: culled
                             [300.0, 400.0, 5.0],    # far off to the side
                             [30.0, 20.0, 5.0]])     # ahead, inside the cone
    tiles.radius = np.array([50.0, 50.0, 50.0, 50.0])
    tiles.total = 4
    seen = set(fpv.visible_tiles(tiles, planes).tolist())
    assert 0 in seen and 3 in seen
    assert 1 not in seen
    assert 2 not in seen


def test_a_sphere_straddling_a_plane_is_kept(fpv):
    """Conservative: a tile half in view is drawn, never dropped."""
    vp = _look(eye=(0, 0, 2), target=(100, 0, 2))
    planes = fpv.frustum_planes(vp)

    class T:
        pass

    tiles = T()
    # centred just behind the eye, but big enough to reach in front of it
    tiles.centre = np.array([[-20.0, 0.0, 2.0]])
    tiles.radius = np.array([60.0])
    tiles.total = 1
    assert fpv.visible_tiles(tiles, planes).tolist() == [0]
