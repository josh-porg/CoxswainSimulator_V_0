"""Positive control: can the pipeline find coupling that is known to be there?

Synthetic sessions are generated with the same shape as the real archive -- a few
synchronised sessions and many call-only transcripts -- from a process whose coupling is
set by hand:

  boat   b_t = phi * b_{t-1} + beta * (calls of the target channel in the last 3 steps) + noise
  calls  each channel self-excites; the target channel's rate rises when b is low
         (strength gamma), which is the coxswain reacting to the boat

With beta = gamma = 0 there is no coupling, and both coupling p-values should be
uniform. With them set, the p-values should fall. The run reports both.

    python selftest.py [--beta 0.4] [--gamma 1.0] [--sync 3] [--steps 600]
"""
import argparse
import tempfile

import numpy as np

from encode import KEYWORD_CHANNELS
from evaluate import cross_validate, summarise
from sessions import Grid

K = len(KEYWORD_CHANNELS)
TARGET = KEYWORD_CHANNELS.index("power")


def simulate(T, beta, gamma, rng, with_boat=True, phi=0.8):
    b = np.zeros(T)
    C = np.zeros((T, K))
    base = rng.uniform(0.02, 0.15, K)
    h = np.zeros(K)
    for t in range(1, T):
        recent = C[max(0, t - 3):t, TARGET].sum()
        b[t] = phi * b[t - 1] + beta * recent + rng.normal(0, 0.6)
        h = 0.7 * h + C[t - 1]
        logit = np.log(base / (1 - base)) + 0.8 * h
        logit[TARGET] += -gamma * b[t - 1]
        C[t] = rng.random(K) < 1 / (1 + np.exp(-logit))
    C[:, -1] = np.maximum(C[:, -1], C[:, :-1].max(1))     # "any" covers every call
    z = (b - b.mean()) / b.std()
    boat = np.column_stack([z, z * 0.5 + rng.normal(0, 0.5, T)])
    bm = np.ones_like(boat) if with_boat else np.zeros_like(boat)
    return boat * bm, bm, C, np.ones(T)


def make(n_sync, n_calls_only, steps, beta, gamma, seed):
    rng = np.random.default_rng(seed)
    grids = []
    for i in range(n_sync + n_calls_only):
        sync = i < n_sync
        boat, bm, C, cm = simulate(steps, beta, gamma, rng, with_boat=sync)
        grids.append(Grid("s%02d" % i, "synthetic", np.arange(steps) * 2.0, boat, bm, C, cm,
                          list(KEYWORD_CHANNELS)))
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sync", type=int, default=3)
    ap.add_argument("--calls-only", type=int, default=20)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--nulls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for name, beta, gamma in (("no coupling", 0.0, 0.0), ("coupled", a.beta, a.gamma)):
        grids = make(a.sync, a.calls_only, a.steps, beta, gamma, a.seed)
        acc, null = cross_validate(grids, K=5, n_null=a.nulls, seed=a.seed,
                                   tcfg=dict(steps=a.train_steps), log=lambda *_: None)
        r = summarise(acc, null)
        print("%-12s beta=%.2f gamma=%.2f" % (name, beta, gamma))
        for m in ("linear", "transformer"):
            print("  %-12s C->B %+.4f (p %.3f)   B->C %+.4f (p %.3f)   self boat %+.3f calls %+.3f"
                  % (m, r[m]["CtoB"], r[m]["p_CtoB"], r[m]["BtoC"], r[m]["p_BtoC"],
                     r[m]["self_boat"], r[m]["self_calls"]))


if __name__ == "__main__":
    main()
