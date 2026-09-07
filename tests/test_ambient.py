r"""The ambient bed: quiet, seamless, and responsive to the wind.

It is scenery under the stroke, and the tests hold it to that: it must
never approach the stroke's level, it must loop without a click, and a
dead calm must be nearly silent because a dead calm is.
"""

from __future__ import annotations

import numpy as np


def test_the_bed_is_quiet_and_loops_cleanly():
    from coxswain.viz.ambient import AMBIENT_VOLUME, RATE, synthesise_ambient

    wave = synthesise_ambient(6.0)
    assert np.isfinite(wave).all()
    assert np.abs(wave).max() <= 0.95
    # A fifth of the stroke's 0.85, before the per-wind scaling.
    assert AMBIENT_VOLUME <= 0.2
    # The loop point: the last few ms and the first few ms must meet
    # without a step, or every 24 s there is a click on the water.
    tail = wave[-int(0.004 * RATE):]
    head = wave[:int(0.004 * RATE)]
    assert abs(float(tail.mean()) - float(head.mean())) < 0.05


def test_a_calm_is_nearly_silent_and_a_breeze_is_not():
    from coxswain.viz.ambient import synthesise_ambient

    calm = synthesise_ambient(0.0)
    breeze = synthesise_ambient(9.0)
    rms = lambda w: float(np.sqrt(np.mean(w * w)))
    assert rms(breeze) > 1.15 * rms(calm)
    # And the calm still has the water at the bank in it: not zero.
    assert rms(calm) > 0.02


def test_the_player_survives_having_no_device():
    """No sound device is a reason to have no sound, not to crash."""
    import os

    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from coxswain.viz.ambient import AmbientAudio

    player = AmbientAudio(4.0)
    # Either it played on the dummy driver or it explained why not.
    assert player.available or player.reason
    player.set_wind(9.0)          # a big change rebuilds; must not raise
    player.set_wind(9.5)          # a small one re-levels
    player.stop()
