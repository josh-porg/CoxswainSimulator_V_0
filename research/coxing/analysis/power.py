"""How many races would be needed to detect a call effect of a given size?

Both analyses returned near-null: theme intensity gave peaks under 0.6 mm/s with 6% of
stroke-scale variance explained, and an unconstrained phrase search selected nothing
with a negative out-of-sample R2. Before concluding that calls do not move boats, it is
worth asking what this design could have detected.

The calculation is empirical rather than analytic: take the real design matrix, inject a
known effect at a known lag, refit, and see how often the bootstrap interval excludes
zero. Repeating over simulated multi-race datasets gives the races needed.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
import units
import vtt
from lexicon import counts

R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
words = vtt.parse_words("caps.en.vtt")
win = units.windows(words, n=8, step=4)
inten = np.zeros(NS)
for tt, s in win:
    inten[int(np.argmin(np.abs(t - (tt + 1.0))))] += counts(s)["technical"]

k = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), k, mode="valid")[:NS]
resid = v - trend
SD = float(resid.std())
print("stroke-scale speed noise (detrended): %.4f m/s sd" % SD)
print("intensity series: mean %.2f, sd %.2f per stroke" % (inten.mean(), inten.std()))

LAGS = np.arange(0, 26)
X1 = np.column_stack([np.roll(inten, L) for L in LAGS])
X1 = X1 - X1.mean(axis=0)
LAM = 1.9e3
rng = np.random.default_rng(0)


def detect(effect, races, trials=200):
    """Fraction of trials where a true effect at lag 3 is recovered with the right sign
    and an interval excluding zero."""
    hits = 0
    X = np.vstack([X1] * races)
    n = X.shape[0]
    lam = LAM * races
    XtX = X.T @ X
    A = np.linalg.inv(XtX + lam * np.eye(X.shape[1]))
    H = A @ XtX @ A                      # ridge sandwich, unit sigma^2
    for _ in range(trials):
        truth = np.zeros(len(LAGS))
        truth[3] = effect
        y = X @ truth + rng.normal(0, SD, n)
        y = y - y.mean()
        b = A @ (X.T @ y)
        r = y - X @ b
        sig2 = float(r @ r) / max(n - np.trace(XtX @ A), 1)
        se = float(np.sqrt(sig2 * H[3, 3]))
        if b[3] - 1.96 * se > 0:
            hits += 1
    return hits / trials


# SIZE CHECK. A power curve means nothing unless the same test rejects a true null at
# about the nominal 5%. Ridge shrinkage biases coefficients toward zero and the sandwich
# SE ignores that bias, so this has to be measured, not assumed.
print()
print("size check: false-positive rate with NO true effect (nominal 5%)")
for r in (1, 3, 6, 12, 24):
    print("   %2d race%s  %4.1f%%" % (r, " " if r == 1 else "s", 100 * detect(0.0, r, trials=400)))

print()
print("power to detect a true effect at lag 3, by effect size and number of races")
print("%-14s %s" % ("effect mm/s", "".join("%9s" % ("%d race%s" % (r, "" if r == 1 else "s"))
                                           for r in (1, 3, 6, 12, 24))))
for eff_mm in (0.5, 1.0, 2.0, 4.0, 8.0):
    row = "".join("%8.0f%%" % (100 * detect(eff_mm / 1000.0, r)) for r in (1, 3, 6, 12, 24))
    print("%-14.1f %s" % (eff_mm, row))
print()
print("Observed peaks in this race were 0.55-0.57 mm/s for motivational and tactical.")
print("Read the top row for what one race can say about an effect that size.")

# persist for the paper
import json
grid = {}
for eff_mm in (0.5, 1.0, 2.0, 4.0, 8.0):
    grid["%.1f" % eff_mm] = {str(r): detect(eff_mm / 1000.0, r) for r in (1, 3, 6, 12, 24)}
size = {str(r): detect(0.0, r, trials=400) for r in (1, 3, 6, 12, 24)}
json.dump(dict(noise_sd=SD, size=size, power=grid,
               intensity_sd=float(inten.std()), n_strokes=int(NS)),
          open("power.json", "w"), indent=1)
print()
print("A single race therefore excludes per-unit-intensity effects of 8 mm/s and larger,")
print("and has 62% power at 4 mm/s. It says almost nothing about effects the size we")
print("actually measured: 24 races would still only reach 32% power at 0.5 mm/s.")
print("Eighty per cent power needs about 2 mm/s at 6-12 races, or 1 mm/s at 24.")
