"""Fit and score the coupled model on every session under a data root.

    python run.py --root <root> [--folds 5] [--nulls 50] [--steps 1500] [--out results.json]

Adding data means adding session folders (see sessions.py or ingest.py); nothing here
changes. The report gives, in nats per step on held-out data:

  self   how much each stream's own past predicts it, against no history
  C->B   how much the call history adds to predicting the boat, given the boat's past
  B->C   how much the boat history adds to predicting the calls, given the calls' past

for the transformer and for the linear reference, with circular-shift p-values for the
two coupling terms.
"""
import argparse
import json
import time

import numpy as np

from evaluate import cross_validate, summarise
from sessions import list_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--encoder", default="keyword", choices=["keyword", "labels"])
    ap.add_argument("--label-field")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nulls", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    sessions = list_sessions(a.root)
    grids = [s.to_grid(a.encoder, a.label_field) for s in sessions]
    grids = [g for g in grids if len(g) > 10]
    sync = [g for g in grids if g.synchronised]
    print("%d sessions, %d synchronised; %d grid steps, %d with boat data"
          % (len(grids), len(sync), sum(len(g) for g in grids),
             int(sum(g.boat_mask[:, 0].sum() for g in grids))))
    for g in sync:
        print("  synchronised %-10s %4d steps, %4d boat cells, %4d call steps"
              % (g.sid, len(g), int(g.boat_mask[:, 0].sum()), int(g.call_mask.sum())))

    t0 = time.time()
    acc, null = cross_validate(grids, K=a.folds, n_null=a.nulls, seed=a.seed,
                               tcfg=dict(steps=a.steps, ctx=a.ctx, d=a.d, layers=a.layers,
                                         verbose=a.verbose))
    res = summarise(acc, null)
    res["config"] = vars(a)
    res["seconds"] = round(time.time() - t0, 1)

    print()
    print("held-out nats per step            linear     transformer")
    for key, name in (("self_boat", "boat, own past vs none"),
                      ("self_calls", "calls, own past vs none")):
        print("  %-30s %+8.4f   %+8.4f" % (name, res["linear"][key], res["transformer"][key]))
    if "CtoB" in res["transformer"]:
        for key, name in (("CtoB", "C->B, calls into boat"), ("BtoC", "B->C, boat into calls")):
            print("  %-30s %+8.4f   %+8.4f   (p = %.3f, %.3f)"
                  % (name, res["linear"][key], res["transformer"][key],
                     res["linear"]["p_" + key], res["transformer"]["p_" + key]))
    print("(%.0f s)" % res["seconds"])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.item()
                  if isinstance(o, np.generic) else str(o))


if __name__ == "__main__":
    main()
