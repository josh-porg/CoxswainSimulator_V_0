r"""What has to hold for the trainer to run on ANY machine.

Not "fast" -- fast is measured, by ``--bench``, and recorded in
TRACKING.  These are the things that make a slow machine *usable*
rather than frozen, and the per-pixel work the low tiers must not be
paying for.  Each was found by measurement on an integrated GPU and
would come back silently.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# the loop cannot spiral
# ---------------------------------------------------------------------------
class _SlowSimulator:
    """A step that costs nothing but records how often it was asked."""

    def __init__(self):
        self.calls = 0

    def step(self, state, t, dt):
        self.calls += 1
        return state


def test_a_frame_that_fell_far_behind_takes_a_bounded_number_of_steps():
    """A quarter of a second late at 60 Hz is fifteen steps.  A machine
    that got there is a machine that cannot afford fifteen steps, so the
    loop takes a few and lets the rest go, and says how much it let go."""
    from coxswain.sim.realtime import FixedStepLoop

    sim = _SlowSimulator()
    loop = FixedStepLoop(sim, rate=60.0)
    loop.start(np.zeros(12))
    taken = loop.advance(0.25)
    assert taken == loop.max_steps
    assert sim.calls == loop.max_steps
    # what was dropped is the rest of the quarter second, to a step
    assert loop.dropped == pytest.approx(0.25 - loop.max_steps / 60.0,
                                         abs=1.0 / 60.0)
    # and the loop is not carrying it forward into the next frame
    assert loop.advance(0.0) == 0


def test_a_frame_on_time_is_not_touched_by_the_cap():
    from coxswain.sim.realtime import FixedStepLoop

    sim = _SlowSimulator()
    loop = FixedStepLoop(sim, rate=60.0)
    loop.start(np.zeros(12))
    for _ in range(30):
        loop.advance(1.0 / 60.0)
    assert sim.calls == 30
    assert loop.dropped == 0.0


def test_simulated_time_never_runs_faster_than_the_wall_even_when_capped():
    """Dropping time must not become gaining it: after a capped frame
    ``t`` has advanced by exactly the steps taken."""
    from coxswain.sim.realtime import FixedStepLoop

    sim = _SlowSimulator()
    loop = FixedStepLoop(sim, rate=60.0)
    loop.start(np.zeros(12))
    loop.advance(0.5)
    assert loop.t == pytest.approx(loop.max_steps / 60.0)


# ---------------------------------------------------------------------------
# the low tiers do not pay for detail they do not draw
# ---------------------------------------------------------------------------
def test_low_tiers_switch_the_sky_noise_off():
    """The dome's simplex is evaluated for every water pixel too, through
    the reflection.  Measured at up to 1 ms of sky and a share of a
    5-7 ms water pass on an integrated part."""
    from coxswain.viz.menu import tier_settings

    assert tier_settings("ultra").sky_detail is False
    assert tier_settings("minimal").sky_detail is False
    assert tier_settings("standard").sky_detail is True
    assert tier_settings("high").sky_detail is True


def test_the_shaders_actually_honour_the_detail_uniforms():
    text = source("scripts", "fpv.py")
    assert "uniform int sky_detail;" in text
    assert re.search(r"if \(sky_overcast > 0\.0 && sky_detail > 0\)", text)
    assert "uniform int water_detail;" in text
    assert "if (water_detail == 0) {" in text
    # and both are SET from the tier, not left at the GLSL default of 0
    assert "sky_detail=1 if getattr(args.tier, \"sky_detail\", True) else 0" in text
    assert "water_detail=0 if not args.tier.rich_water else 1" in text


def test_the_hud_is_uploaded_only_when_it_changes():
    """A full-window RGBA upload plus font rendering every frame was
    about 3 ms on an integrated part, for text that changes a few times
    a second."""
    text = source("scripts", "fpv.py")
    assert "_hud_key = (tuple(lines), knob, _map_key)" in text
    assert "if _hud_changed:" in text
    # the menu path draws a different picture and must reset the key
    assert "draw.hud_last = None          # the menu overwrote the HUD" in text
    # exactly one unconditional per-frame HUD write must be gone: the
    # game path's write now sits under the change check
    # Anchored FROM the key: the menu path has its own _hud_blit(ctx)
    # earlier in the file, and searching from the top sliced backwards
    # to an empty string that passed nothing.
    start = text.index("_hud_key = (tuple(lines), knob, _map_key)")
    block = text[start:text.index("_hud_blit(ctx)", start)]
    assert block, "empty slice: the anchors crossed over"
    assert "hud_texture.write(" in block
    assert block.index("if _hud_changed:") < block.index("hud_texture.write(")
