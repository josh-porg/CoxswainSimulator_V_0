"""Splits and training for the coupled transformer.

A Segment is a run of rows of one session: inputs are rows [lo, hi), and targets are
scored only from row `score_lo` onward, so a held-out block can be scored with the
session's earlier rows as context without ever training on it. Held-out rows never
appear as inputs to training: a training segment that follows a test block starts
fresh after it.

Folds:
  synchronised sessions  one contiguous block per fold, as in the linear analysis
  single-stream sessions whole sessions per fold
"""
from dataclasses import dataclass

import numpy as np
import torch

from model import CoupledTransformer, build_inputs, step_loglik


@dataclass
class Segment:
    g: object          # sessions.Grid
    lo: int
    hi: int
    score_lo: int = None

    def __post_init__(self):
        if self.score_lo is None:
            self.score_lo = self.lo + 1


def _blocks(T, k, K):
    e = np.linspace(0, T, K + 1).astype(int)
    return int(e[k]), int(e[k + 1])


def split(grids, k, K, val_frac=0.15, min_len=12, seed=0):
    """Train, validation and test segments for fold k of K."""
    rng = np.random.default_rng(seed)
    single = [g for g in grids if not g.synchronised]
    order = rng.permutation(len(single))
    test_single = {single[i].sid for i in order[k::K]}

    train, test = [], []
    for g in grids:
        T = len(g)
        if g.synchronised:
            a, b = _blocks(T, k, K)
            test.append(Segment(g, 0, b, max(a, 1)))
            train += [Segment(g, 0, a), Segment(g, b, T)]
        elif g.sid in test_single:
            test.append(Segment(g, 0, T))
        else:
            train.append(Segment(g, 0, T))
    train = [s for s in train if s.hi - s.lo >= min_len]

    fit, val = [], []
    for s in train:
        n = s.hi - s.lo
        cut = s.hi - int(round(val_frac * n))
        if cut - s.lo >= min_len and s.hi - cut >= 2:
            fit.append(Segment(s.g, s.lo, cut))
            val.append(Segment(s.g, s.lo, s.hi, cut))
        else:
            fit.append(s)
    return fit, val, test


def _tensors(g, lo, hi, shift_boat=0, shift_calls=0):
    boat, bm = g.boat[lo:hi], g.boat_mask[lo:hi]
    calls, cm = g.calls[lo:hi], g.call_mask[lo:hi]
    if shift_boat:
        boat, bm = np.roll(g.boat, shift_boat, 0)[lo:hi], np.roll(g.boat_mask, shift_boat, 0)[lo:hi]
    if shift_calls:
        calls = np.roll(g.calls, shift_calls, 0)[lo:hi]
        cm = np.roll(g.call_mask, shift_calls, 0)[lo:hi]
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).unsqueeze(0)
    return f(boat), f(bm), f(calls), f(cm)


def sample_batch(segs, ctx, n, rng):
    """n random windows of up to ctx+1 rows, weighted by segment length."""
    lens = np.array([s.hi - s.lo for s in segs], float)
    pick = rng.choice(len(segs), size=n, p=lens / lens.sum())
    L = int(min(ctx + 1, lens[pick].min()))
    out = []
    for i in pick:
        s = segs[i]
        a = int(rng.integers(s.lo, s.hi - L + 1))
        out.append(_tensors(s.g, a, a + L))
    return [torch.cat(z, 0) for z in zip(*out)]


def window_loglik(model, boat, bm, calls, cm, drop_boat=False, drop_calls=False):
    """Log-likelihood of rows 1..L-1 of a window given rows 0..L-2."""
    N = boat.shape[0]
    db = torch.full((N,), drop_boat) if isinstance(drop_boat, bool) else drop_boat
    dc = torch.full((N,), drop_calls) if isinstance(drop_calls, bool) else drop_calls
    x = build_inputs(boat[:, :-1], bm[:, :-1], calls[:, :-1], cm[:, :-1], db, dc)
    pred = model(x)
    return step_loglik(pred, boat[:, 1:], bm[:, 1:], calls[:, 1:], cm[:, 1:])


@torch.no_grad()
def score(model, seg, drop_boat=False, drop_calls=False, shift_boat=0, shift_calls=0):
    """Summed log-likelihood over the segment's scored targets, with sliding context."""
    model.eval()
    ctx = model.ctx
    tot = dict(boat=0.0, boat_n=0.0, calls=0.0, calls_n=0.0)
    t = seg.score_lo
    while t < seg.hi:
        e = min(seg.hi, t + ctx // 2)                 # targets [t, e)
        a = max(seg.lo, e - 1 - ctx)                  # inputs [a, e-1)
        boat, bm, calls, cm = _tensors(seg.g, a, e, shift_boat, shift_calls)
        bll, cll = window_loglik(model, boat, bm, calls, cm, drop_boat, drop_calls)
        keep = slice(t - a - 1, None)                 # prediction at input i targets i+1
        tot["boat"] += float(bll[0, keep].sum())
        tot["boat_n"] += float(bm[0, 1:][keep].sum())
        tot["calls"] += float(cll[0, keep].sum())
        tot["calls_n"] += float(cm[0, 1:][keep].sum())
        t = e
    return tot


def _val_loss(model, val):
    s = [score(model, v) for v in val]
    n = sum(x["boat_n"] + x["calls_n"] for x in s)
    return -sum(x["boat"] + x["calls"] for x in s) / max(n, 1.0)


def fit(fit_segs, val_segs, n_boat, n_calls, ctx=64, d=32, layers=2, heads=2, dropout=0.1,
        steps=1500, batch=32, lr=1e-3, weight_decay=1e-2, p_drop=0.3, eval_every=100,
        patience=5, seed=0, log=None):
    """Train with stream dropout; keep the weights with the best validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CoupledTransformer(n_boat, n_calls, d, heads, layers, ctx, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = np.inf, None, 0
    for step in range(1, steps + 1):
        model.train()
        boat, bm, calls, cm = sample_batch(fit_segs, ctx, batch, rng)
        db = torch.rand(batch) < p_drop
        dc = (torch.rand(batch) < p_drop) & ~db       # never blank both streams
        bll, cll = window_loglik(model, boat, bm, calls, cm, db, dc)
        n = bm[:, 1:].sum() + cm[:, 1:].sum()
        loss = -(bll.sum() + cll.sum()) / n.clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 and val_segs:
            v = _val_loss(model, val_segs)
            if log:
                log("  step %5d  train %.4f  val %.4f" % (step, float(loss.detach()), v))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
