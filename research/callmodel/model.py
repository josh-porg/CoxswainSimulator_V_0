"""A causal transformer over the coupled call and boat streams.

The joint process factorises one step at a time,

    p(C, B) = prod_t  p(C_t | C_<t, B_<t) * p(B_t | B_<t, C_<t),

and one network carries both factors. At each grid step the input is the boat reading
(z-scored, zero where missing), its observation mask, the call channels and the call
stream's coverage mask. From the history up to step t the network emits a Gaussian for
the boat at t+1 and a Bernoulli for each call channel at t+1.

Stream dropout. During training, the whole boat input or the whole call input of a
window is removed at random (values and masks set to zero), exactly as when that stream
was never recorded. The same network then represents a stream conditioned on its own
past alone and conditioned on both pasts, so

    coupling  C -> B  =  loglik(boat | both pasts)  -  loglik(boat | boat past only)

is measured within one model, and sessions that recorded only one stream (a transcript
without boat data, a cox-box log without audio) train the self-terms without special
handling.
"""
import math

import torch
import torch.nn as nn


def build_inputs(boat, boat_mask, calls, call_mask, drop_boat=None, drop_calls=None):
    """Concatenate one batch of windows into input features.

    boat, boat_mask (N, L, B); calls (N, L, K); call_mask (N, L).
    drop_boat / drop_calls: optional (N,) bool, True removes that stream from the window.
    """
    bm = boat_mask.clone()
    cm = call_mask.clone()
    c = calls.clone()
    if drop_boat is not None:
        bm[drop_boat] = 0.0
    if drop_calls is not None:
        cm[drop_calls] = 0.0
    b = boat * bm
    c = c * cm.unsqueeze(-1)
    return torch.cat([b, bm, c, cm.unsqueeze(-1)], dim=-1)


class CoupledTransformer(nn.Module):
    def __init__(self, n_boat=2, n_calls=10, d=32, heads=2, layers=2, ctx=64, dropout=0.1):
        super().__init__()
        self.n_boat, self.n_calls, self.ctx = n_boat, n_calls, ctx
        self.inp = nn.Linear(2 * n_boat + n_calls + 1, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.boat_mu = nn.Linear(d, n_boat)
        self.boat_logvar = nn.Linear(d, n_boat)
        self.call_logit = nn.Linear(d, n_calls)

    def forward(self, x):
        """x (N, L, F) -> predictions for step t+1 from inputs up to t."""
        L = x.shape[1]
        if L > self.ctx:
            raise ValueError("window %d exceeds context %d" % (L, self.ctx))
        h = self.inp(x) + self.pos(torch.arange(L, device=x.device))
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        h = self.norm(self.body(h, mask=causal))
        return self.boat_mu(h), self.boat_logvar(h).clamp(-6.0, 4.0), self.call_logit(h)


def step_loglik(pred, boat_t, boat_mask_t, calls_t, call_mask_t):
    """Per-cell log-likelihoods of the targets under the predictions.

    Returns (boat_ll, calls_ll): boat_ll (N, L, B) zero where the target is unobserved;
    calls_ll (N, L) summed over channels, zero where the call stream is not covered.
    """
    mu, logvar, logit = pred
    boat_ll = -0.5 * (math.log(2 * math.pi) + logvar + (boat_t - mu) ** 2 / logvar.exp())
    boat_ll = boat_ll * boat_mask_t
    calls_ll = -nn.functional.binary_cross_entropy_with_logits(logit, calls_t, reduction="none")
    calls_ll = calls_ll.sum(-1) * call_mask_t
    return boat_ll, calls_ll
