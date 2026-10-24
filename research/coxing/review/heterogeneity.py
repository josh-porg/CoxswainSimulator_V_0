"""Is there between-call heterogeneity in the speed response to explain?

Per-call response = mean speed beyond the boat's own momentum (AR(4) forecast residual) over
strokes 3-5 after the call. The same statistic at every stroke of the session (no call) gives the
noise distribution. Var(calls) - Var(noise) estimates the variance of true per-call effects."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "review", "analysis")))   # data and outputs live here (gitignored)
import os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "callmodel"))
import short_horizon as SH

ROOT = os.path.join(REPO, "data", "local", "coxing")
sess = SH.load_sessions(ROOT, ["race_04", "race_05", "practice_01", "practice_02"])
rng = np.random.default_rng(0)

def win(s, times):
    _, res = SH.responses(s["t"], s["speed"], s["beta_s"], times)
    return res[:, 2:5].mean(1)

calls, noise, fn = [], [], []
for s in sess:
    times = np.array([e[0] for e in s["episodes"]])
    w = win(s, times); ok = ~np.isnan(w)
    calls.append(w[ok]); fn += [e[1] for e, k in zip(s["episodes"], ok) if k]
    nz = win(s, s["t"][5:-10] - 0.3); noise.append(nz[~np.isnan(nz)])
c = np.concatenate(calls); n = np.concatenate(noise); fn = np.array(fn)
print("calls n=%d mean %+.1f sd %.1f mm/s; noise n=%d mean %+.1f sd %.1f" % (len(c), 1e3*c.mean(), 1e3*c.std(ddof=1), len(n), 1e3*n.mean(), 1e3*n.std(ddof=1)))
# bootstrap the variance difference
B = []
for _ in range(4000):
    cb = rng.choice(c, len(c)); nb = rng.choice(n, len(n))
    B.append(cb.var(ddof=1) - nb.var(ddof=1))
B = np.array(B); d = c.var(ddof=1) - n.var(ddof=1)
print("var diff %.1f (mm/s)^2, 95%% CI %.1f..%.1f; implied true-effect sd %.1f mm/s" % (1e6*d, 1e6*np.percentile(B, 2.5), 1e6*np.percentile(B, 97.5), 1e3*np.sqrt(max(d, 0))))
# Levene-type permutation: is |deviation| larger for calls than for random strokes in the same sessions?
for f in "ETRISPMK":
    x = c[fn == f]
    if len(x) > 10:
        print("  %s n=%3d mean %+5.1f sd %5.1f" % (f, len(x), 1e3*x.mean(), 1e3*x.std(ddof=1)))
# reliability: per-function means split by odd/even calls
print("share of calls with response > +1 noise sd: %.2f (noise: %.2f); < -1 sd: %.2f (noise %.2f)" % (
    np.mean(c > n.std()), np.mean(n > n.std()), np.mean(c < -n.std()), np.mean(n < -n.std())))
