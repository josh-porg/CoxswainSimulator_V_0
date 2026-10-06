"""short_horizon.py on synthetic sessions: a planted 3-5 stroke response is found, the boat's own
momentum is not mistaken for one, and no calls effect gives a chance p-value."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_horizon as SH  # noqa: E402


def session(rng, effect, chase=0.0, n=600, n_calls=60):
    """Per-stroke speed as AR(1) noise; a call at random strokes adds ``effect`` m/s from the
    2nd stroke after it for six strokes. ``chase``: calls placed where the boat had just dipped
    and was already recovering (the coxswain calling what they feel)."""
    t = np.cumsum(np.full(n, 2.0))
    e = rng.normal(0, 0.04, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + e[i]
    if chase:
        idx = np.argsort(np.diff(y))[:n_calls] + 2
        idx = idx[(idx > 10) & (idx < n - 12)]
    else:
        idx = rng.choice(np.arange(10, n - 12), n_calls, replace=False)
    for j in idx:
        y[j + 1:j + 7] += effect
    sp = 4.0 + y
    ra = np.full(n, 30.0) + rng.normal(0, 0.3, n)
    ev = [(float(t[j] - 0.5), "E", "0", "") for j in sorted(idx)]
    return dict(id="syn", kind="race", boat="8+", t=t, speed=sp, rate=ra,
                beta_s=SH.ar_fit(sp), beta_r=SH.ar_fit(ra), episodes=ev)


def test_planted_response_is_found():
    rng = np.random.default_rng(1)
    s = [session(rng, 0.05), session(rng, 0.05)]
    r = SH.summarise(s, lambda e: e[1] == "E", n_null=300)
    assert r["speed_beyond"]["p_window"] < 0.01
    assert 0.03 < r["speed_beyond"]["window_3_5"] < 0.07


def test_no_effect_gives_chance():
    rng = np.random.default_rng(2)
    ps = [SH.summarise([session(rng, 0.0)], lambda e: e[1] == "E", n_null=200)["speed_beyond"]["p_window"]
          for _ in range(8)]
    assert np.median(ps) > 0.1


def test_momentum_is_not_a_call_effect():
    """Calls placed on the boat's sharpest drops: against the strokes before, the raw comparison
    shows a large change that is the boat's own trajectory; the momentum-adjusted one does not."""
    rng = np.random.default_rng(3)
    r = SH.summarise([session(rng, 0.0, chase=1.0)], lambda e: e[1] == "E", n_null=300)
    assert abs(r["speed"]["window_3_5"]) > 0.01
    assert abs(r["speed"]["window_3_5"]) > 2 * abs(r["speed_beyond"]["window_3_5"])
