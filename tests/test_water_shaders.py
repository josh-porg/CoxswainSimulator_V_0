r"""Both water shaders must compile, and the tiers must differ.

A GLSL error does not exist until a context tries to build it, so a
typo in a shader is invisible to every other test in this suite and
shows up as a black window on somebody else's machine.  These build
both programs against a real (headless) context.

Skipped where no GL context can be created, which is most CI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():
    try:
        context = moderngl.create_standalone_context()
    except Exception as error:                      # pragma: no cover
        pytest.skip("no GL context: %s" % error)
    yield context
    context.release()


def test_both_water_shaders_compile(ctx):
    """The minimal and the rich fragment shaders both build."""
    import fpv

    for name in ("WATER_FRAGMENT", "WATER_FRAGMENT_RICH"):
        program = ctx.program(vertex_shader=fpv.WATER_VERTEX,
                              fragment_shader=getattr(fpv, name))
        assert program is not None, name
        program.release()


def test_the_land_sky_and_shadow_shaders_compile(ctx):
    """Every other program the trainer builds, built here first.

    The water shaders were the only ones compile-tested, and the land
    shader grew a shadow lookup and the sky a noise function without any
    test building them -- so a GLSL slip in either would have shown up
    as a black window on somebody else's machine.
    """
    import fpv

    for vertex, fragment in (("VERTEX_SHADER", "FRAGMENT_SHADER"),
                             ("SKY_VERTEX", "SKY_FRAGMENT"),
                             ("SHADOW_VERTEX", "SHADOW_FRAGMENT")):
        program = ctx.program(vertex_shader=getattr(fpv, vertex),
                              fragment_shader=getattr(fpv, fragment))
        assert program is not None, fragment
        program.release()


def test_the_rich_shader_takes_the_scene_it_reads(ctx):
    """It is a second pass, so it must actually declare the inputs.

    If these ever stop being uniforms the two-pass plumbing in ``main``
    silently stops feeding it and the water goes back to guessing.
    """
    import fpv

    program = ctx.program(vertex_shader=fpv.WATER_VERTEX,
                          fragment_shader=fpv.WATER_FRAGMENT_RICH)
    try:
        for name in ("scene", "scene_depth", "viewport", "near_plane",
                     "far_plane", "refract_scale", "reflect_steps"):
            assert name in program, name
    finally:
        program.release()


def test_quality_presets_pick_a_shader_tier():
    """The two low tiers keep the plain shader; the two high ones ask
    for the rich one.

    Trees are no longer DROPPED at the low tiers -- they are drawn as
    two-triangle impostors, which is what was asked for ("turn every
    tree into a sprite") and costs almost nothing; the old table
    removed them outright.  ``quality_settings`` still reports them as
    present, because they are.
    """
    from coxswain.viz.menu import QUALITY, quality_settings, tier_settings

    keys = [row[0] for row in QUALITY]
    assert keys[:2] == ["ultra", "minimal"], keys
    for key in ("ultra", "minimal"):
        _divisions, trees, rich, particles = quality_settings(key)
        assert rich is False, key
        assert trees is True, "sprites, not nothing"
        assert particles is False, key
        assert tier_settings(key).trees == "impostor", key
    for key in keys[2:]:
        _divisions, trees, rich, _particles = quality_settings(key)
        assert rich is True, key
        assert trees is True, key
    # And the grid genuinely gets bigger as you go up.
    sizes = [quality_settings(key)[0] for key in keys]
    assert sizes == sorted(sizes), sizes


def test_splash_droplets_are_ballistic_and_scale_with_catch_speed():
    """A slammed catch throws more and further than a placed one, and
    every droplet dies -- no leak that would eventually fill the pool
    with permanent points."""
    import fpv

    calm = fpv.SplashSystem()
    calm.spawn(np.array([0.0, 0.0, 0.0]), speed=0.1, t=0.0)
    hard = fpv.SplashSystem()
    hard.spawn(np.array([0.0, 0.0, 0.0]), speed=2.0, t=0.0)

    soon_calm = calm.as_uniform(0.05)
    soon_hard = hard.as_uniform(0.05)
    assert len(soon_calm) and len(soon_hard)
    # Distance travelled from the origin, in the horizontal plane.
    reach_calm = np.hypot(soon_calm[:, 0], soon_calm[:, 1]).max()
    reach_hard = np.hypot(soon_hard[:, 0], soon_hard[:, 1]).max()
    assert reach_hard > reach_calm

    assert len(calm.as_uniform(10.0)) == 0, "nothing survives its lifetime"


def test_a_lazy_catch_barely_splashes():
    """The near-zero-speed floor exists so a boat sitting still, or a
    blade barely moving at the catch, does not throw water."""
    import fpv

    still = fpv.SplashSystem()
    still.spawn(np.array([1.0, 1.0, 0.0]), speed=0.0, t=0.0)
    assert len(still.as_uniform(0.02)) == 0


def test_the_water_patch_fades_at_its_own_edge():
    """The seam that reads as a diagonal line on the water.

    The detailed water is a square patch that follows the boat, with a
    flat plane beyond it.  They met at a hard edge -- waves one side,
    glass the other -- and because the patch is world-axis-aligned and
    travels with the boat, that edge reads from any oblique angle as a
    straight DIAGONAL lying on the water and moving along with you.

    It never showed from the seat, where the edge is past the horizon,
    which is why it survived several attempts to reproduce it.

    Three things have to fade together or the seam survives in one of
    them: the height, the interpolated slope, and the per-pixel sea
    slope.  Fading the height alone leaves the surface flat while the
    SHADING still shows chop.
    """
    import fpv

    vertex = fpv.WATER_VERTEX
    assert "patch_reach" in vertex, "the fade needs the patch half-width"
    assert "smoothstep(0.80, 1.0, edge)" in vertex
    # Height and the interpolated slope both scaled by it.
    assert "h *= blend;" in vertex
    assert "* blend;" in vertex

    # And the per-pixel slope, in whichever shader carries water_normal.
    body = fpv.WATER_FRAGMENT_RICH
    assert "float blend" in body, "water_normal must take the fade"
    assert "sea_slope(world.xy, span, roughness) * blend" in body


def test_the_fade_uses_the_water_patch_not_the_world_reach():
    """Two very different numbers, and using the wrong one is silent.

    The water patch is 110 m; the world is built 900 m either side of
    the course.  Fading at 900 never fires inside a 110 m patch, so the
    seam stays exactly where it was and the fix looks like it did
    nothing.
    """
    import fpv

    assert "patch_reach=float(WATER_REACH)" in _source(), (
        "the uniform must be the water patch's reach, not args.reach")
    assert fpv.WATER_REACH < 200.0


def _source():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "scripts", "fpv.py"),
              encoding="utf-8") as handle:
        return handle.read()
