"""Announced pushes against executed ones: does a call that sets up a push ("in two, let's take a 10") have a later
response than a call that is the push ("press", "drive", "go")? Directive power calls, by session.

    .venv/Scripts/python.exe research/coxing/announced_pushes.py [session ...]

Speed beyond the boat's own momentum (call_properties) at strokes 1-15 after the call, mean over calls, with the
spread a mean of that many random strokes has in the same session (the global shift null, 2000 shifts) at
strokes 3-5 and 8-12. The study's primary window (strokes 3-5) assumes the call is the push; an announcement moves the
push 1-3 strokes later. Classification is lexical, on the episode's text (no boat data): ANNOUNCE if it contains
"in two", "take a", "take (5|10|five|ten|twenty)", "ready", "give me (5|10|five|ten)", "last 10"; else EXECUTE.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

COURSE = os.environ.get("COURSE") == "1"          # the course (bends, heading) in the boat's own forecast
EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
ANN = re.compile(r"in two|take a\b|take (\d+|five|ten|twenty|three)|\bready\b|give me (\d+|five|ten)|last 10|last ten", re.I)


def main():
    sids = sys.argv[1:] or ["race_06", "race_07"]
    D = pd.read_csv(os.path.join(EXP, "episodes.csv"))
    for sid in sids:
        s = CP.prepare(CP.load_session(os.path.join(ROOT, "data", "local", "coxing", "sessions", sid)), course=COURSE)
        pw = D[(D.session == sid) & (D.target == "power") & (D.stage == "dec")]
        ann = pw.text.str.contains(ANN)
        print("%s: %d power calls, %d announcements" % (sid, len(pw), int(ann.sum())))
        for lab, sel in (("announce", ann), ("execute", ~ann)):
            js = pw.loc[sel, "j"].to_numpy()
            if len(js) < 3:
                continue
            curve = [np.nanmean(CP.gather([s], [js], "speed", h, h)) for h in range(1, 16)]
            st = lambda jj: np.array([np.nanmean(CP.gather([s], jj, "speed", 3, 5)), np.nanmean(CP.gather([s], jj, "speed", 8, 12))])
            og, mg, sg, pg = CP.permutation_test(st, [s], [js], 2000, 1)
            print("  %-8s n=%3d  curve (mm/s, strokes 1-15): %s" % (lab, len(js), " ".join("%+.0f" % (1000 * c) for c in curve)))
            print("           strokes 3-5 %+.0f (null sd %.0f, p %.2f) | strokes 8-12 %+.0f (null sd %.0f, p %.2f)"
                  % (1000 * (og[0] - mg[0]), 1000 * sg[0], pg[0], 1000 * (og[1] - mg[1]), 1000 * sg[1], pg[1]))


if __name__ == "__main__":
    main()
