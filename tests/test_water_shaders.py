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
    """Minimal keeps the old shader; the others ask for the new one."""
    from coxswain.viz.menu import QUALITY, quality_settings

    keys = [row[0] for row in QUALITY]
    assert "minimal" in keys
    divisions, trees, rich = quality_settings("minimal")
    assert rich is False
    assert trees is False
    for key in keys:
        if key == "minimal":
            continue
        _divisions, _trees, rich = quality_settings(key)
        assert rich is True, key
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
