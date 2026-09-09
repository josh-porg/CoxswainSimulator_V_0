r"""The packed vertex layout and the sky's place in the draw order.

Both are bandwidth and overdraw decisions measured on an integrated
GPU, and both would revert silently: a float layout still draws the
same picture, a sky drawn first still fills the frame.  These pin what
the measurement chose.
"""

from __future__ import annotations

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_packed_is_twenty_bytes_and_round_trips_to_eight_bits():
    from coxswain.viz.worldmesh import MeshPart

    rng = np.random.default_rng(1)
    n = 300
    vertices = rng.uniform(-500.0, 500.0, (n, 3)).astype("f4")
    normals = rng.normal(size=(n, 3)).astype("f4")
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    colours = rng.uniform(0.0, 1.0, (n, 3)).astype("f4")
    part = MeshPart("t", vertices, colours, normals)

    raw = part.packed()
    assert len(raw) == n * MeshPart.PACKED_STRIDE
    assert MeshPart.PACKED_STRIDE == 20
    rec = np.frombuffer(raw, dtype=[("p", "f4", 3), ("n", "i1", 4),
                                    ("c", "u1", 4)])
    # position is exact
    assert np.array_equal(rec["p"], vertices)
    # normal direction to better than a degree
    back = rec["n"][:, :3].astype("f4") / 127.0
    cosine = (back * normals).sum(axis=1) / np.linalg.norm(back, axis=1)
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))).max() < 1.0
    # colour to half a step of 8 bits
    assert np.abs(rec["c"][:, :3] / 255.0 - colours).max() <= 0.5 / 255.0 + 1e-6
    # and the float layout is still there for the buffers that use it
    assert part.interleaved().shape == (n, 9)


def test_a_part_without_normals_packs_an_up_normal():
    from coxswain.viz.worldmesh import MeshPart

    part = MeshPart("flat", np.zeros((3, 3), "f4"), np.ones((3, 3), "f4"))
    rec = np.frombuffer(part.packed(), dtype=[("p", "f4", 3), ("n", "i1", 4),
                                              ("c", "u1", 4)])
    assert np.array_equal(rec["n"][:, :3], [[0, 0, 127]] * 3)


def test_the_static_world_is_uploaded_packed_and_the_shader_unpacks_it():
    text = source("scripts", "fpv.py")
    assert 'ctx.buffer(part.packed())' in text
    assert '"3f 4i1 4u1", "in_pos", "in_normal"' in text
    assert '"3f 8x1", "in_pos"' in text, "the shadow pass reads the same buffer"
    assert "uniform float colour_scale;" in text
    assert "v_colour = in_colour * colour_scale;" in text
    assert "v_normal = normalize(in_normal);" in text
    # set for the packed world, and back to 1.0 for the float boat buffer
    world = text.index('program["colour_scale"].value = 1.0 / 255.0')
    boat = text.index('program["colour_scale"].value = 1.0   # float buffer')
    assert world < boat
    assert '"3f 3f 3f", "in_pos", "in_normal"' in text, "the boat is still float"


def test_the_sky_is_drawn_last_where_the_water_is_plain():
    text = source("scripts", "fpv.py")
    assert "sky_last = scene_fbo is None" in text
    # first-path draw is under `if not sky_last`, last-path under `if sky_last`
    first = text.index("if not sky_last:")
    last = text.index("if sky_last:")
    assert first < last
    tail = text[last:last + 400]
    assert 'ctx.depth_func = "<="' in tail, "1.0 against a cleared 1.0 needs LEQUAL"
    assert 'ctx.depth_func = "<"' in tail, "and it must be put back"
    assert "sky_vao.render()" in tail
