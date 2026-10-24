"""Exact p-values for the two LLM categories that survived, with enough draws to
clear a Bonferroni threshold of 0.05/23 = 0.0022."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import numpy as np
from llm_irf import AX, y, lagmat, blocked_oos, NS

rng = np.random.default_rng(77)
NDRAW = 3000
for k in ("target=catch", "theme=technical"):
    s = AX[k]
    obs = blocked_oos(lagmat(s), y)
    null = np.empty(NDRAW)
    for i in range(NDRAW):
        null[i] = blocked_oos(lagmat(np.roll(s, int(rng.integers(26, NS - 26)))), y)
    hits = int((null >= obs).sum())
    p = (hits + 1) / (NDRAW + 1)
    print("%-18s n=%3d  OOS R2 %+.4f   null 95th %+.4f max %+.4f"
          % (k, int(s.sum()), obs, np.percentile(null, 95), null.max()))
    print("   %d of %d shifts reached it -> p = %.5f   (Bonferroni 23 tests: %s)"
          % (hits, NDRAW, p, "PASSES" if p < 0.05 / 23 else "fails"))
