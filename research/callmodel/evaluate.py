"""Cross-validated scoring of the transformer against a linear reference.

Everything is held-out log-likelihood on the same test segments:

  reference   no history: boat Gaussian at the training mean and variance, calls at base rates
  linear      a multivariate Hawkes-style autoregression: each stream's history summarised
              by exponentially decayed sums at tau = 2, 8, 30 steps, plus the boat's last two
              readings; ridge for the boat, logistic for each call channel, regularisation
              tuned on the validation segments. The own-past model and the both-pasts model
              are fitted separately, as in a Granger test.
  transformer one network; own-past and both-pasts are the same weights with the other
              stream's inputs removed (see model.py)

Self-prediction is scored on every test segment. Coupling is scored only on synchronised
test segments, where both streams exist. Its null circularly shifts the other stream
within each test session, destroying cross-alignment and keeping each stream's own
structure; the p-value is the share of shifted gains at least as large as the observed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import train as T

TAUS = (2.0, 8.0, 30.0)


# --------------------------------------------------------------------------- linear
def _decay(X):
    out = np.zeros((len(X), X.shape[1] * len(TAUS)))
    for i, tau in enumerate(TAUS):
        a = np.exp(-1.0 / tau)
        h = np.zeros(X.shape[1])
        for t in range(len(X)):
            h = a * h + X[t]                     # includes row t: the input at time t
            out[t, i * X.shape[1]:(i + 1) * X.shape[1]] = h
    return out


def design(seg, use_boat=True, use_calls=True, shift_boat=0, shift_calls=0):
    """Features at input rows [lo, hi-1) and targets at rows [lo+1, hi), restricted to
    targets from score_lo on."""
    g, lo, hi = seg.g, seg.lo, seg.hi
    boat, bm = g.boat, g.boat_mask
    calls, cm = g.calls, g.call_mask
    if shift_boat:
        boat, bm = np.roll(boat, shift_boat, 0), np.roll(bm, shift_boat, 0)
    if shift_calls:
        calls, cm = np.roll(calls, shift_calls, 0), np.roll(cm, shift_calls, 0)
    parts = []
    if use_boat:
        b = (boat * bm)[lo:hi]
        m = bm[lo:hi]
        lag1 = b
        lag2 = np.vstack([np.zeros((1, b.shape[1])), b[:-1]])
        parts += [lag1, lag2, m, _decay(np.hstack([b, m]))]
    if use_calls:
        c = (calls * cm[:, None])[lo:hi]
        parts += [cm[lo:hi, None], _decay(np.hstack([c, cm[lo:hi, None]]))]
    X = np.hstack(parts) if parts else np.zeros((hi - lo, 1))
    X = X[:-1]
    rows = np.arange(lo + 1, hi)
    keep = rows >= seg.score_lo
    tgt = rows[keep]
    return X[keep], g.boat[tgt], g.boat_mask[tgt], g.calls[tgt], g.call_mask[tgt]


def _stack(segs, **kw):
    parts = [design(s, **kw) for s in segs]
    return [np.concatenate(z, 0) for z in zip(*parts)]


class Linear:
    LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)
    CS = (0.01, 0.1, 1.0)

    def __init__(self, use_boat, use_calls):
        self.kw = dict(use_boat=use_boat, use_calls=use_calls)

    # ---- boat: one ridge per channel on rows where the target is observed
    @staticmethod
    def _ridge(X, y, lam):
        mx, my = X.mean(0), y.mean()
        Xc = X - mx
        b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - my))
        return mx, my, b, float(np.mean((y - my - Xc @ b) ** 2)) + 1e-6

    @staticmethod
    def _gll(par, X, y):
        mx, my, b, s2 = par
        r = y - ((X - mx) @ b + my)
        return -0.5 * (np.log(2 * np.pi * s2) + r ** 2 / s2)

    # ---- calls: one logistic per channel on rows where the call stream is covered
    @staticmethod
    def _logit(X, y, C):
        mx, sx = X.mean(0), X.std(0) + 1e-9
        if y.sum() < 3 or y.sum() > len(y) - 3:
            return ("rate", float(np.clip(y.mean(), 1e-3, 1 - 1e-3)))
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit((X - mx) / sx, y)
        return ("model", (mx, sx, m))

    @staticmethod
    def _bll(par, X, y):
        kind, p = par
        if kind == "rate":
            q = np.full(len(y), p)
        else:
            mx, sx, m = p
            q = np.clip(m.predict_proba((X - mx) / sx)[:, 1], 1e-4, 1 - 1e-4)
        return y * np.log(q) + (1 - y) * np.log(1 - q)

    def fit(self, fit_segs, val_segs):
        X, B, BM, C, CM = _stack(fit_segs, **self.kw)
        Xv, Bv, BMv, Cv, CMv = _stack(val_segs, **self.kw) if val_segs else (None,) * 5
        self.boat, self.calls = [], []
        for j in range(B.shape[1]):
            o = BM[:, j] > 0
            if o.sum() < 10:
                self.boat.append(None)
                continue
            if Xv is not None and (BMv[:, j] > 0).sum() > 5:
                ov = BMv[:, j] > 0
                lam = max(self.LAMS, key=lambda l: self._gll(self._ridge(X[o], B[o, j], l),
                                                             Xv[ov], Bv[ov, j]).sum())
            else:
                lam = 10.0
            self.boat.append(self._ridge(X[o], B[o, j], lam))
        o = CM > 0
        ov = CMv > 0 if Xv is not None else None
        for j in range(C.shape[1]):
            if ov is not None and ov.sum() > 20 and Cv[ov, j].sum() > 0:
                Cbest = max(self.CS, key=lambda c: self._bll(self._logit(X[o], C[o, j], c),
                                                             Xv[ov], Cv[ov, j]).sum())
            else:
                Cbest = 0.1
            self.calls.append(self._logit(X[o], C[o, j], Cbest))
        return self

    def score(self, seg, shift_boat=0, shift_calls=0):
        X, B, BM, C, CM = design(seg, shift_boat=shift_boat, shift_calls=shift_calls, **self.kw)
        tot = dict(boat=0.0, boat_n=float(BM.sum()), calls=0.0, calls_n=float(CM.sum()))
        for j, par in enumerate(self.boat):
            o = BM[:, j] > 0
            if par is not None and o.any():
                tot["boat"] += float(self._gll(par, X[o], B[o, j]).sum())
        o = CM > 0
        if o.any():
            for j, par in enumerate(self.calls):
                tot["calls"] += float(self._bll(par, X[o], C[o, j]).sum())
        return tot


class Reference:
    """No history: training mean and variance for the boat, base rates for the calls."""

    def fit(self, fit_segs, val_segs):
        _, B, BM, C, CM = _stack(fit_segs, use_boat=False, use_calls=False)
        self.b = [(B[BM[:, j] > 0, j].mean(), B[BM[:, j] > 0, j].var() + 1e-6)
                  for j in range(B.shape[1])]
        self.p = np.clip(C[CM > 0].mean(0), 1e-3, 1 - 1e-3)
        return self

    def score(self, seg, **_):
        _, B, BM, C, CM = design(seg, use_boat=False, use_calls=False)
        tot = dict(boat=0.0, boat_n=float(BM.sum()), calls=0.0, calls_n=float(CM.sum()))
        for j, (m, v) in enumerate(self.b):
            o = BM[:, j] > 0
            tot["boat"] += float((-0.5 * (np.log(2 * np.pi * v) + (B[o, j] - m) ** 2 / v)).sum())
        o = CM > 0
        tot["calls"] += float((C[o] * np.log(self.p) + (1 - C[o]) * np.log(1 - self.p)).sum())
        return tot


# --------------------------------------------------------------------------- CV
def _add(acc, key, d):
    a = acc.setdefault(key, dict(boat=0.0, boat_n=0.0, calls=0.0, calls_n=0.0))
    for k in a:
        a[k] += d[k]


def _shift(T_, rng):
    lo = max(2, T_ // 5)
    return int(rng.integers(lo, T_ - lo)) if T_ - lo > lo else 0


def cross_validate(grids, K=5, n_null=50, tcfg=None, seed=0, log=print):
    """Return summed held-out log-likelihoods for every model and condition."""
    tcfg = dict(tcfg or {})
    verbose = tcfg.pop("verbose", False)
    rng = np.random.default_rng(seed)
    n_boat = grids[0].boat.shape[1]
    n_calls = grids[0].calls.shape[1]
    acc = {}
    null = {m: {"CtoB": np.zeros(n_null), "BtoC": np.zeros(n_null)}
            for m in ("linear", "transformer")}
    for k in range(K):
        fit_segs, val_segs, test_segs = T.split(grids, k, K, seed=seed)
        sync = [s for s in test_segs if s.g.synchronised]
        log("fold %d/%d: %d training segments, %d test (%d synchronised)"
            % (k + 1, K, len(fit_segs), len(test_segs), len(sync)))

        ref = Reference().fit(fit_segs, val_segs)
        lin_own_b = Linear(True, False).fit(fit_segs, val_segs)
        lin_own_c = Linear(False, True).fit(fit_segs, val_segs)
        lin_both = Linear(True, True).fit(fit_segs, val_segs)
        net = T.fit(fit_segs, val_segs, n_boat, n_calls, seed=seed + k,
                    log=log if verbose else None, **tcfg)

        for s in test_segs:
            _add(acc, "reference", ref.score(s))
            _add(acc, "linear_own_boat", lin_own_b.score(s))
            _add(acc, "linear_own_calls", lin_own_c.score(s))
            _add(acc, "transformer_own_boat", T.score(net, s, drop_calls=True))
            _add(acc, "transformer_own_calls", T.score(net, s, drop_boat=True))
        for s in sync:
            _add(acc, "sync_linear_own_boat", lin_own_b.score(s))
            _add(acc, "sync_linear_own_calls", lin_own_c.score(s))
            _add(acc, "sync_linear_both", lin_both.score(s))
            _add(acc, "sync_transformer_own_boat", T.score(net, s, drop_calls=True))
            _add(acc, "sync_transformer_own_calls", T.score(net, s, drop_boat=True))
            _add(acc, "sync_transformer_both", T.score(net, s))

        for i in range(n_null):
            for s in sync:
                sh = _shift(len(s.g), rng)
                lb = lin_both.score(s, shift_calls=sh)
                lc = lin_both.score(s, shift_boat=sh)
                tb = T.score(net, s, shift_calls=sh)
                tc = T.score(net, s, shift_boat=sh)
                null["linear"]["CtoB"][i] += lb["boat"]
                null["linear"]["BtoC"][i] += lc["calls"]
                null["transformer"]["CtoB"][i] += tb["boat"]
                null["transformer"]["BtoC"][i] += tc["calls"]
    return acc, null


def summarise(acc, null):
    """Per-step gains in nats. Self gains are against the reference; coupling gains are
    both-pasts minus own-past on synchronised test segments."""
    per = lambda d, s: d[s] / max(d[s + "_n"], 1.0)
    out = {"n_boat_cells": acc["reference"]["boat_n"], "n_call_steps": acc["reference"]["calls_n"]}
    ref = acc["reference"]
    for m in ("linear", "transformer"):
        out[m] = {
            "self_boat": per(acc[m + "_own_boat"], "boat") - per(ref, "boat"),
            "self_calls": per(acc[m + "_own_calls"], "calls") - per(ref, "calls"),
        }
        if "sync_%s_both" % m in acc:
            both = acc["sync_%s_both" % m]
            ob, oc = acc["sync_%s_own_boat" % m], acc["sync_%s_own_calls" % m]
            cb = per(both, "boat") - per(ob, "boat")
            bc = per(both, "calls") - per(oc, "calls")
            nb = null[m]["CtoB"] / max(both["boat_n"], 1.0) - per(ob, "boat")
            nc = null[m]["BtoC"] / max(both["calls_n"], 1.0) - per(oc, "calls")
            out[m].update(
                CtoB=cb, p_CtoB=float((np.sum(nb >= cb) + 1) / (len(nb) + 1)),
                BtoC=bc, p_BtoC=float((np.sum(nc >= bc) + 1) / (len(nc) + 1)),
                sync_boat_cells=both["boat_n"], sync_call_steps=both["calls_n"])
    return out
