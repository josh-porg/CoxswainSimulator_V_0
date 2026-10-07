"""call_properties on synthetic sessions: episodes and their union of codes, a planted response is
found, and a property defined from the boat's state is not mistaken for an effect when it is
recomputed at the shifted strokes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import call_properties as CP  # noqa: E402


def synthetic(rng, effect=0.0, n=700, calls=80, chase=False):
    t = np.cumsum(np.full(n, 2.0))
    e = rng.normal(0, 0.04, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + e[i]
    if chase:      # calls placed right after the boat's sharpest rises
        idx = np.argsort(-np.diff(y))[:calls] + 1
        idx = idx[(idx > 12) & (idx < n - 25)]
    else:
        idx = rng.choice(np.arange(12, n - 25), calls, replace=False)
    for j in idx:
        y[j:j + 6] += effect
    s = dict(t=t, speed=4.0 + y, rate=np.full(n, 30.0) + rng.normal(0, 0.3, n), meta={}, phrases=[])
    s["dps"] = s["speed"] * 2.0
    CP.prepare(s)
    lo, hi = s["valid"]
    js = np.sort(idx[(idx >= lo) & (idx < hi)])
    return s, js


def test_episodes_union():
    phrases = [dict(i=0, t=0.0, t_end=0.5, text="big press"), dict(i=1, t=1.0, t_end=1.2, text="legs now"),
               dict(i=2, t=9.0, t_end=9.5, text="good"), dict(i=3, t=10.0, t_end=10.1, text="three")]
    codes = {0: dict(zip(CP.FIELDS, "power dec proc y direct - n n neu expl up".split())),
             1: dict(zip(CP.FIELDS, "power dec proc y direct int n n neu expl up".split())),
             2: dict(zip(CP.FIELDS, "morale ass task n - - n n chal - -".split()))}
    eps = CP.episodes(phrases, codes)
    assert [e["target"] for e in eps] == ["power", "morale", "count"]
    assert eps[0]["focus"] == "int" and eps[0]["words"] == 4 and eps[0]["clauses"] == 2


def test_planted_response_found():
    rng = np.random.default_rng(1)
    s, js = synthetic(rng, effect=0.05)
    obs, mu, sd, p = CP.permutation_test(lambda jj: np.array([np.nanmean(CP.gather([s], jj, "speed", 3, 5))]),
                                         [s], [js], 400, 0)
    assert p[0] < 0.01 and 0.03 < obs[0] - mu[0] < 0.07


def test_relational_property_recomputed():
    """No call effect; calls placed after rises. Contingency recomputed at shifted strokes gives chance."""
    ps = []
    for seed in range(6):
        rng = np.random.default_rng(10 + seed)
        s, js = synthetic(rng, effect=0.0, chase=True)

        def stat(jj):
            pre = np.array([CP.contingency(s, j) for j in jj[0]])
            y = CP.gather([s], jj, "speed", 3, 5)
            c = pre > 0
            return np.array([np.nanmean(y[c]) - np.nanmean(y[~c])]) if c.any() and (~c).any() else np.array([0.0])

        ps.append(CP.permutation_test(stat, [s], [js], 200, seed, local=True)[3][0])
    assert np.median(ps) > 0.1


def test_holm():
    adj = CP.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
