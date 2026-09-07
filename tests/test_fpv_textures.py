"""The water shader's textures must not share a unit with the HUD.

A 2.5 m wave appeared beside the hull, in the boat frame, only in the
live window and never in a headless render.  The near-field map was
bound to texture unit 0 once at start-up; the HUD overlay is blitted from
unit 0 every frame; so from the second frame on the water shader sampled
the HUD -- white text reads 1.0, times U^2/g -- as the hull's surface.
Headless mode never draws the HUD, which is why every height-field probe
came back clean.  This pins the fix: every water texture on its own unit,
none of them zero, and re-bound before the water is drawn.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source():
    with open(os.path.join(ROOT, "scripts", "fpv.py"), encoding="utf-8") as f:
        return f.read()


def test_water_textures_are_not_on_the_hud_unit():
    src = _source()
    near = re.search(r"^NEAR_UNIT\s*=\s*(\d+)", src, re.M)
    wave = re.search(r"^WAVE_UNIT\s*=\s*(\d+)", src, re.M)
    assert near and wave
    assert int(near.group(1)) != 0
    assert int(wave.group(1)) != 0
    assert int(near.group(1)) != int(wave.group(1))
    # The HUD keeps unit 0.
    assert "hud_texture.use(0)" in src


def test_water_textures_are_rebound_every_frame():
    """Binding once at set-up is what went wrong; the draw must re-bind."""
    src = _source()
    draw = src[src.index("    def draw(state, t):"):src.index("water_vao.render()")]
    assert "near_tex.use(NEAR_UNIT)" in draw
    assert "wave_tex.use(WAVE_UNIT)" in draw
