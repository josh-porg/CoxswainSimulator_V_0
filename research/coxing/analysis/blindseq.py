"""Does the sequence structure survive a blind relabelling?

The first labelling prompt described the coxswain's inform/request/reinforce/feedback
cycle and gave their own example, so the agent knew to look for it. This second pass
used neutral speech-act categories, never mentioned a cycle, and was told explicitly not
to assume repeating structure. If the sequence is in the speech rather than the prompt,
it should still be predictable here.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import numpy as np

B = json.load(open("labels_blind.json"))
Lm = json.load(open("labels_llm.json"))
ACTS = ["assertive", "directive", "intensifier", "commissive", "expressive",
        "enumerative", "other"]
rng = np.random.default_rng(23)


def cv_loglik(s, acts, nfold=5, lag=1):
    n = len(s); e = np.linspace(0, n, nfold + 1).astype(int); tot = 0.0
    for i in range(nfold):
        te = set(range(e[i], e[i + 1]))
        M = np.zeros((len(acts), len(acts)))
        for j in range(lag, n):
            if j not in te:
                M[acts.index(s[j - lag]), acts.index(s[j])] += 1
        P = (M + 0.5) / (M + 0.5).sum(1, keepdims=True)
        for j in range(lag, n):
            if j in te:
                tot += np.log(P[acts.index(s[j - lag]), acts.index(s[j])])
    return tot


for race in ("race1", "race2"):
    s = [d["act"] for d in B[race] if d["act"] != "other"]
    print("=== %s blind labels: %d utterances" % (race, len(s)))
    print("   counts:", {a: s.count(a) for a in ACTS if s.count(a)})
    for lag in (1, 2, 3):
        obs = cv_loglik(s, ACTS, lag=lag)
        null = []
        for _ in range(400):
            sh = list(s); rng.shuffle(sh)
            null.append(cv_loglik(sh, ACTS, lag=lag))
        null = np.array(null)
        z = (obs - null.mean()) / null.std()
        p = float(np.mean(null >= obs))
        print("   lag %d: %+.1f vs shuffled %+.1f+-%.1f   z=%+.2f  p=%.4f%s"
              % (lag, obs, null.mean(), null.std(), z, p, "*" if p < 0.05 else ""))
    print()

# agreement between the two labellings on race 1
print("=== cross-labelling agreement, race 1")
pr = [d["pragmatic"] for d in Lm["race1"]]
bl = [d["act"] for d in B["race1"]]
PR = ["inform", "request", "reinforce", "feedback", "count", "other"]
M = np.zeros((len(PR), len(ACTS)), int)
for a, b in zip(pr, bl):
    M[PR.index(a), ACTS.index(b)] += 1
print("%-11s %s" % ("cycle \\ act", " ".join("%11s" % a[:10] for a in ACTS)))
for i, a in enumerate(PR):
    print("%-11s %s" % (a, " ".join("%11d" % x for x in M[i])))
exp = {"inform": "assertive", "request": "directive", "reinforce": "intensifier",
       "count": "enumerative"}
hit = sum(M[PR.index(k), ACTS.index(v)] for k, v in exp.items())
tot = sum(M[PR.index(k)].sum() for k in exp)
print()
print("mapping inform->assertive, request->directive, reinforce->intensifier,")
print("count->enumerative agrees on %d of %d (%.0f%%) of those utterances"
      % (hit, tot, 100 * hit / tot))
