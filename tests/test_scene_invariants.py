r"""Things the scene must keep doing, that only ever got checked by eye.

Each of these was verified once, by looking at a render or reading the
source, and then nothing held it.  A shadow pass that quietly stops
including the buildings, or a sky-noise amplitude nudged from 0.045 to
0.45, would look like a rendering change nobody could bisect.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def source():
    with open(os.path.join(ROOT, "scripts", "fpv.py"), encoding="utf-8") as f:
        return f.read()


def test_everything_in_the_world_casts_a_shadow():
    """Buildings and trees must shadow each other, not only the ground.

    The shadow map is baked from ``shadow_casters``, and that list has to
    be built from EVERY mesh part.  Building it from the terrain alone
    would still produce shadows -- the ground would be striped where the
    sun is blocked by nothing -- and would look like a subtle lighting
    bug rather than a missing pass.

    Checked structurally: the caster is appended in the same loop body,
    from the same buffer, as the drawable.  One append each, so the two
    lists cannot drift apart.
    """
    text = source()
    loop = text[text.index("for part in mesh.parts:"):]
    loop = loop[:loop.index("\n\n")]
    assert loop.count("static.append(") == 1, loop
    assert loop.count("shadow_casters.append(") == 1, loop
    # And the bake draws all of them.
    assert "for caster in shadow_casters:" in text


def test_the_world_actually_contains_things_to_cast():
    """The loop above is only worth having if the parts exist."""
    from coxswain.viz.worldmesh import build_world

    mesh, _scene = build_world("charles", reach=350.0, step=12.0,
                               water_level=-0.15, guide=False)
    names = {part.name for part in mesh.parts}
    assert any("land" in n for n in names), names
    assert any("building" in n or "wall" in n for n in names), names


def test_the_sky_noise_stays_under_the_gradient():
    """Texture, not weather.

    The zenith-to-horizon gradient is the sky's shape; the simplex is
    meant to sit under it so an overcast reads as sky rather than as a
    wall.  A few percent does that.  Anything approaching the gradient
    itself would be a different sky.
    """
    text = source()
    match = re.search(r"base \*= 1\.0 \+ ([0-9.]+) \* sky_overcast", text)
    assert match, "the sky noise term has moved or gone"
    assert float(match.group(1)) <= 0.08, match.group(1)
    # And it is gated on overcast, so a clear day barely takes any.
    assert "if (sky_overcast > 0.0)" in text


def test_the_bow_foam_is_tied_to_the_hull_and_the_speed():
    """A bow wave that does not know the hull is a white rectangle.

    It was exactly that once: a smoothstep box at the stem times a
    uniform white.  What makes it a bow wave is that it follows the
    waterline and dies when the boat does.
    """
    text = source()
    body = text[text.index("float waterline_foam(vec2 p) {"):]
    body = body[:body.index("\n}")]
    assert "speed" in body, "foam must scale with boat speed"
    assert "hull_frame" in body, "foam must be in the hull's own frame"
    assert "half_beam" in body or "beam" in body, body[:200]


@pytest.mark.parametrize("name", ["SplashSystem"])
def test_the_splash_pool_exists_and_empties(name):
    """Droplets at the catch, and none that live for ever."""
    import numpy as np

    import fpv

    pool = getattr(fpv, name)()
    pool.spawn(np.array([0.0, 0.0, 0.0]), speed=1.5, t=0.0)
    assert len(pool.as_uniform(0.05)) > 0
    assert len(pool.as_uniform(30.0)) == 0
