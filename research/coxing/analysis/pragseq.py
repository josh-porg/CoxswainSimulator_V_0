"""Is inform -> request -> reinforce -> feedback a real sequence?

The coxswain describes their own method as a cycle. The earlier n-gram test found the
THEME sequence to be memoryless, but themes describe what a call is about, not what it
is doing. If the pragmatic cycle is real, the pragmatic sequence should be predictable
from its own history where the theme sequence was not.

Tested two ways: a first-order Markov model scored against a null that keeps each
label's frequency but destroys its order, and the same for skip-grams at lag 2 and 3.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import numpy as np

L = json.load(open("labels_llm.json"))
ORDER = ["inform", "request", "reinforce", "feedback", "count", "other"]
rng = np.random.default_rng(11)


def seq(race, drop_other=True):
    s = [d["pragmatic"] for d in L[race]]
    return [x for x in s if not (drop_other and x == "other")]


def trans(s):
    M = np.zeros((len(ORDER), len(ORDER)))
    for a, b in zip(s, s[1:]):
        M[ORDER.index(a), ORDER.index(b)] += 1
    return M


def loglik(s, M, alpha=0.5):
    P = (M + alpha) / (M + alpha).sum(1, keepdims=True)
    return sum(np.log(P[ORDER.index(a), ORDER.index(b)]) for a, b in zip(s, s[1:]))


def cv_loglik(s, nfold=5, lag=1):
    """Held-out log-likelihood of predicting label t from label t-lag."""
    n = len(s); e = np.linspace(0, n, nfold + 1).astype(int)
    tot = 0.0
    for i in range(nfold):
        te = set(range(e[i], e[i + 1]))
        tr = [(s[j - lag], s[j]) for j in range(lag, n) if j not in te]
        M = np.zeros((len(ORDER), len(ORDER)))
        for a, b in tr:
            M[ORDER.index(a), ORDER.index(b)] += 1
        P = (M + 0.5) / (M + 0.5).sum(1, keepdims=True)
        for j in range(lag, n):
            if j in te:
                tot += np.log(P[ORDER.index(s[j - lag]), ORDER.index(s[j])])
    return tot


for race in ("race1", "race2"):
    s = seq(race)
    print("=== %s: %d utterances (excluding 'other')" % (race, len(s)))
    M = trans(s)
    P = M / np.maximum(M.sum(1, keepdims=True), 1)
    print("%-10s %s" % ("from \\ to", " ".join("%9s" % o[:8] for o in ORDER[:5])))
    for i, o in enumerate(ORDER[:5]):
        if M[i].sum() == 0:
            continue
        print("%-10s %s   (n=%d)" % (o, " ".join("%8.2f " % P[i, j] for j in range(5)), int(M[i].sum())))
    base = np.bincount([ORDER.index(x) for x in s], minlength=len(ORDER)) / len(s)
    print("   marginal: %s" % "  ".join("%s %.2f" % (o[:5], base[i]) for i, o in enumerate(ORDER[:5])))
    for lag in (1, 2, 3):
        obs = cv_loglik(s, lag=lag)
        null = []
        for _ in range(400):
            sh = list(s); rng.shuffle(sh)
            null.append(cv_loglik(sh, lag=lag))
        null = np.array(null)
        p = float(np.mean(null >= obs))
        z = (obs - null.mean()) / null.std()
        print("   lag %d: held-out loglik %+.1f vs shuffled %+.1f+-%.1f   z=%+.2f  p=%.4f%s"
              % (lag, obs, null.mean(), null.std(), z, p, "*" if p < 0.05 else ""))
    print()
