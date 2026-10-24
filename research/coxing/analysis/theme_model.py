"""Learned theme-intensity model: one-vs-rest probability per theme, CV-validated.

Hand-written lexicons separated technical well (AUC 0.92) and motivational badly
(0.60), because words like "go", "more" and "yeah" occur in every theme and a human
writing a word list cannot weight them. Learning the weights from the labels fixes
exactly that, and keeps the output continuous -- a probability per theme per window --
so no argmax is taken and the ill-posed single-label decision never arises.

Validation is stratified 5-fold, scored by AUC because the quantity we need downstream
is a ranking of how strongly a window carries a theme, not a hard assignment.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from labels import labels

THEMES = ["technical", "motivational", "tactical"]


def build(c=4.0):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=4000, C=c, class_weight="balanced"))


if __name__ == "__main__":
    d = json.load(open("to_label.json"))
    X = np.array([x[2] for x in d])
    y = np.array(labels())
    src = np.array([x[0] for x in d])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    print("one-vs-rest theme intensity, 5-fold CV (AUC)")
    print("%-14s %8s %9s %9s %9s" % ("theme", "n pos", "lexicon", "learned", "best C"))
    from lexicon import counts
    LEXC = np.array([[counts(t)[k] for k in THEMES] for t in X], dtype=float)
    keep = {}
    for j, th in enumerate(THEMES):
        yb = (y == th).astype(int)
        lex_auc = roc_auc_score(yb, LEXC[:, j])
        best = None
        for c in (0.5, 1, 2, 4, 8, 16):
            p = cross_val_predict(build(c), X, yb, cv=cv, method="predict_proba")[:, 1]
            a = roc_auc_score(yb, p)
            if best is None or a > best[0]:
                best = (a, c, p)
        keep[th] = best[2]
        print("%-14s %8d %9.3f %9.3f %9s" % (th, yb.sum(), lex_auc, best[0], best[1]))

    print()
    print("per corpus, learned model (AUC):")
    for th in THEMES:
        yb = (y == th).astype(int)
        row = []
        for s in ("masters", "elite"):
            m = src == s
            if len(np.unique(yb[m])) < 2:
                row.append("   n/a")
            else:
                row.append("%6.3f" % roc_auc_score(yb[m], keep[th][m]))
        print("   %-14s masters %s   elite %s" % (th, row[0], row[1]))

    np.savez("theme_cv.npz", **{th: keep[th] for th in THEMES})
    print()
    print("A theme whose AUC clears about 0.80 is usable as a continuous regressor;")
    print("below that the downstream impulse response is fitting noise about that theme.")
