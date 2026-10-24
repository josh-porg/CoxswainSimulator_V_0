"""Which PHRASES move the boat, without imposing the three-theme taxonomy.

The theme split may not be the thing that determines whether a call works. Two calls
coded identically -- both technical, say -- can differ completely in effect. This asks
the data directly: which specific words and phrases, spoken at stroke s, predict a
change in boat speed over the following few strokes?

Every phrase enters one joint elastic-net regression, so overlapping speech is handled
the same way the theme model handles it, and a phrase only earns a coefficient if it
predicts something its neighbours do not. Selection and the penalty come from
cross-validation; significance from a moving-block bootstrap.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import ElasticNetCV

sys.path.insert(0, "elite")
import vtt

R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
OFF = 1.0
KERNEL = (1, 6)          # strokes after the phrase over which a response is expected
MIN_COUNT = 6
STOP = set("""a an the and or but if of to in on at is are was were be been being
it its this that these those i you he she we they me him her us them my your his
our their there here then than so as for with from by up down out off over under
now just do does did done have has had will would can could should may might must
okay ok yeah yep yes no not what when where who how why all any some more most
very too also only even still back again one two three four five six seven eight
nine ten s t re ve ll m d""".split())

words = vtt.parse_words("caps.en.vtt")
stroke_of = lambda tt: int(np.argmin(np.abs(t - (tt + OFF))))

# candidate phrases: unigrams and bigrams, lower-cased, stripped of punctuation
def norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum() or ch == "'")


toks = [(stroke_of(tt), norm(w)) for tt, w in words]
toks = [(s, w) for s, w in toks if w]
uni = Counter(w for _, w in toks)
bi = Counter(" ".join([toks[i][1], toks[i + 1][1]]) for i in range(len(toks) - 1)
             if toks[i + 1][0] - toks[i][0] <= 2)
cands = [w for w, c in uni.items() if c >= MIN_COUNT and w not in STOP]
cands += [w for w, c in bi.items() if c >= MIN_COUNT and not all(x in STOP for x in w.split())]
cands = sorted(set(cands))
print("candidate phrases with >= %d occurrences: %d" % (MIN_COUNT, len(cands)))

# occurrence series, then convolved with the response kernel
occ = {p: np.zeros(NS) for p in cands}
for i, (s, w) in enumerate(toks):
    if w in occ:
        occ[w][s] += 1
    if i < len(toks) - 1:
        b = w + " " + toks[i + 1][1]
        if b in occ and toks[i + 1][0] - s <= 2:
            occ[b][s] += 1
ker = np.zeros(NS)
ker[KERNEL[0]:KERNEL[1] + 1] = 1.0 / (KERNEL[1] - KERNEL[0] + 1)
X = np.column_stack([np.convolve(occ[p], ker)[:NS] for p in cands])
keep = X.sum(axis=0) > 0
X, cands = X[:, keep], [p for p, k in zip(cands, keep) if k]

# CONTROL: how much the coxswain is speaking at all. Frequent function words are
# proxies for this, which is why an uncontrolled fit selected "we" and "at" with
# impossible magnitudes. Partial it out of both sides before the phrase fit.
dens = np.zeros(NS)
for s_i, _ in toks:
    dens[s_i] += 1.0
dens = np.convolve(dens, ker)[:NS]

k61 = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), k61, mode="valid")[:NS]
y = v - trend
X = X - X.mean(axis=0)
y = y - y.mean()
D = np.column_stack([dens - dens.mean(), np.ones(NS)])
P = D @ np.linalg.pinv(D)
X = X - P @ X
y = y - P @ y
print("speech density partialled out of %d phrase columns and the response" % X.shape[1])

mdl = ElasticNetCV(l1_ratio=[.5, .7, .9, .95], cv=5, max_iter=20000, random_state=0, n_jobs=None)
mdl.fit(X, y)
b = mdl.coef_
nz = np.flatnonzero(b)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet as _EN
oos = cross_val_score(_EN(alpha=mdl.alpha_, l1_ratio=mdl.l1_ratio_, max_iter=20000),
                      X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
print("elastic net: alpha %.3g, l1_ratio %.2f, %d of %d phrases selected"
      % (mdl.alpha_, mdl.l1_ratio_, len(nz), len(cands)))
print("  in-sample R2 %.3f   OUT-OF-SAMPLE R2 %.3f (5-fold, mean of %s)"
      % (mdl.score(X, y), oos.mean(), np.array2string(oos, precision=2)))
if oos.mean() <= 0:
    print("  -> the model does not predict held-out strokes. Any phrase list below is noise.")

rng = np.random.default_rng(1)
BLOCK, NB = 40, 200
starts = np.arange(0, NS - BLOCK)
from sklearn.linear_model import ElasticNet
boots = np.zeros((NB, X.shape[1]))
for i in range(NB):
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, NS // BLOCK)])
    idx = idx[idx < NS]
    m = ElasticNet(alpha=mdl.alpha_, l1_ratio=mdl.l1_ratio_, max_iter=20000)
    m.fit(X[idx], y[idx])
    boots[i] = m.coef_

freq = {p: (uni[p] if " " not in p else bi[p]) for p in cands}
rows = []
for j in nz:
    lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
    stable = float(np.mean(np.sign(boots[:, j]) == np.sign(b[j])))
    rows.append((b[j], cands[j], freq[cands[j]], lo, hi, stable))
rows.sort(reverse=True)

print()
print("%-22s %6s %11s %22s %8s" % ("phrase", "n", "effect mm/s", "95% CI", "sign %"))
print("-" * 76)
for c, p, n, lo, hi, st in rows:
    if st < 0.80:
        continue
    print("%-22s %6d %11.2f  [%+7.2f,%+7.2f] %7.0f%%"
          % (p, n, 1000 * c, 1000 * lo, 1000 * hi, 100 * st))
print()
print("Only phrases whose bootstrap sign agreed with the point estimate in at least")
print("80%% of resamples are shown. Effect is mm/s averaged over strokes %d-%d after." % KERNEL)
json.dump([[p, float(c), int(n), float(lo), float(hi), float(st)] for c, p, n, lo, hi, st in rows],
          open("phrases.json", "w"), indent=1)
