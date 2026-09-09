r"""What each graphics tier actually switches off, pinned at the source.

A rower reported the trainer unusably sluggish on an old gaming laptop.
The answer is tiers that genuinely remove work, and these hold each
tier to the specific things it was asked to drop -- screen-space
reflection and refraction, the 16-tap shadow filter, the height-
integrated fog, the far skyline, solid trees, the draw distance -- so a
later change cannot quietly hand a cost back to the lowest tier.

Nothing here needs a GPU: the tier table and the shader source are
both plain data.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

from coxswain.viz.menu import QUALITY_TIERS as TIERS, tier_settings as tier_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _source():
    with open(os.path.join(ROOT, "scripts", "fpv.py"), encoding="utf-8") as f:
        return f.read()


def _tier(name):
    return tier_for(name)


# ---------------------------------------------------------------------------
# the two low tiers drop what they were asked to drop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["ultra", "minimal"])
def test_the_low_tiers_use_the_plain_water_with_no_reflection_or_refraction(name):
    """No rich water, so no scene texture, no depth texture, no march.

    Screen-space reflection is a ray march against the scene's depth and
    refraction a second sample of the scene behind the surface; both
    live only in WATER_FRAGMENT_RICH, and the plain shader declares
    neither sampler.
    """
    tier = _tier(name)
    assert not tier.rich_water
    assert tier.reflect_steps == 0
    src = _source()
    plain = src[src.index("WATER_FRAGMENT = "):src.index("WATER_FRAGMENT_RICH = ")]
    assert "sampler2D scene" not in plain
    assert "reflect_steps" not in plain
    assert "refract_scale" not in plain


@pytest.mark.parametrize("name", ["ultra", "minimal"])
def test_the_low_tiers_take_one_shadow_tap_not_sixteen(name):
    tier = _tier(name)
    assert tier.shadow == "single"
    src = _source()
    # the gate exists in the shader, and the single-tap branch is real
    assert "uniform int shadow_taps" in src
    assert "if (shadow_taps <= 1)" in src
    # and the tier drives it
    assert 'shadow_taps=1 if args.tier.shadow == "single" else 16' in src


@pytest.mark.parametrize("name", ["ultra", "minimal"])
def test_the_low_tiers_use_the_simple_fog(name):
    tier = _tier(name)
    assert tier.fog == "simple"
    src = _source()
    assert "uniform int fog_simple" in src
    assert "if (fog_simple == 1) return fog_density * distance;" in src


@pytest.mark.parametrize("name", ["ultra", "minimal"])
def test_the_low_tiers_draw_every_tree_as_a_sprite_and_no_skyline(name):
    tier = _tier(name)
    assert tier.trees == "impostor", "every tree two triangles"
    assert tier.skyline is False, "no far buildings"
    assert tier.samples == 0, "no multisampling"


def test_the_draw_distance_scales_with_the_tier():
    reach = {t.key: t.reach for t in TIERS}
    assert reach["ultra"] < reach["minimal"] < reach["standard"] <= reach["high"]
    assert reach["ultra"] <= 400.0
    step = {t.key: t.step for t in TIERS}
    assert step["ultra"] >= step["minimal"] >= step["standard"]


def test_ultra_minimal_is_the_cheapest_on_every_axis():
    ultra = _tier("ultra")
    for other in TIERS:
        if other.key == "ultra":
            continue
        assert ultra.water_divisions <= other.water_divisions, other.key
        assert ultra.reach <= other.reach, other.key
        assert ultra.physics_hz <= other.physics_hz, other.key
        assert ultra.shadow_size <= (other.shadow_size or 4096), other.key


def test_the_high_tiers_keep_the_rich_water():
    for name in ("standard", "high"):
        tier = _tier(name)
        assert tier.rich_water
        assert tier.reflect_steps > 0
        assert tier.shadow == "pcf"
        assert tier.fog == "full"
        assert tier.trees == "full"
        assert tier.skyline is True


def test_no_tier_defaults_to_a_render_scale_below_one():
    """Measured on an Intel UHD: 0.75 cost more than it saved."""
    for tier in TIERS:
        assert tier.render_scale == 1.0, tier.key


def test_every_tier_is_reachable_from_the_menu():
    from coxswain.viz.menu import quality_choices

    keys = [k for k, _label in quality_choices()]
    for tier in TIERS:
        assert tier.key in keys, tier.key
    assert keys[0] == "ultra", "the cheapest tier is first in the list"
