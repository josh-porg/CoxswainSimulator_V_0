"""Stratified k-fold cross-validation of automatic coding against hand labels.

Task: predict the dominant Nugent theme of an eight-word window. The reference standard
is the analyst's hand coding of 178 windows (labels.py). Models are compared on the
same folds; the keyword rules used in the earlier analysis are the baseline.

Held-out elite recordings are NOT used for model selection -- only the labelled pool is
touched during CV, and the elite/masters split is reported separately so that a model
that only works on one corpus is visible as such.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from labels import labels

TACT = re.compile(r"\b(2[0-3]\d|1[89]\d)\b|\bin two\b|\btake (a )?(10|ten|five|5)\b|"
                  r"\b\d+\s*m\b|metre|meter|\blengths?\b|passing|pass a|yield|clear|"
                  r"stern deck|bow|lane|course|finish line|requalif|handicap|"
                  r"starboard|port\b|corner|turn\b|halfway|quarter|split|stroke rate|"
                  r"\b2[0-9]\b|per stroke|boat ahead|these other|walk on|seat", re.I)
TECH = re.compile(r"catch|finish(es)?\b|legs?\b|bod(y|ies)|swing|suspend|hang|handle|"
                  r"blade|square|release|sequenc|reach|ratio|length|long\b|press|connect|"
                  r"shoulder|sit up|chest|breathe|rhythm|together|complete|direct|"
                  r"sharp|attack|front end|hook|send|puddle|stretcher|toes|heels|hands|"
                  r"tidy|loose|prep\b|drive|flat|low\b|clean", re.I)
MOTI = re.compile(r"\bgood\b|\byes\b|\bnice\b|excellent|great|that's it|you've got|"
                  r"dig deep|commit|capable|grit|power\b|strong|believe|boat ?mates|"
                  r"teammates|hurt|100%|110|doing it|proud|remember|promise|courage|"
                  r"find it in you|show (them|us|me)|we can|we have|ladies|guys|boys|"
                  r"come on|let's go|yeah", re.I)


def rules_predict(texts):
    out = []
    for s in texts:
        if TACT.search(s):
            out.append("tactical")
        elif TECH.search(s):
            out.append("technical")
        elif MOTI.search(s):
            out.append("motivational")
        else:
            out.append("other")
    return np.array(out)


d = json.load(open("to_label.json"))
X = np.array([x[2] for x in d])
y = np.array(labels())
src = np.array(["masters" if x[0] == "masters" else "elite" for x in d])

MODELS = [
    ("majority class", make_pipeline(TfidfVectorizer(), DummyClassifier(strategy="most_frequent"))),
    ("MultinomialNB, words", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), MultinomialNB(alpha=.3))),
    ("ComplementNB, words", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), ComplementNB(alpha=.3))),
    ("LogReg, word 1-2gram", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                           LogisticRegression(max_iter=2000, C=5, class_weight="balanced"))),
    ("LogReg, char 3-5gram", make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
                                           LogisticRegression(max_iter=3000, C=5, class_weight="balanced"))),
    ("LinearSVC, word", make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                      LinearSVC(C=1, class_weight="balanced"))),
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("5-fold stratified CV on %d hand-labelled windows" % len(y))
print("%-24s %9s %9s %9s %9s" % ("model", "accuracy", "macro F1", "F1 moti", "F1 tech"))
print("-" * 66)
rp = rules_predict(X)
print("%-24s %9.3f %9.3f %9.3f %9.3f"
      % ("keyword rules (base)", (rp == y).mean(), f1_score(y, rp, average="macro", zero_division=0),
         f1_score(y, rp, labels=["motivational"], average="macro", zero_division=0),
         f1_score(y, rp, labels=["technical"], average="macro", zero_division=0)))
best = None
preds = {}
for name, mdl in MODELS:
    p = cross_val_predict(mdl, X, y, cv=cv)
    preds[name] = p
    acc, mf1 = (p == y).mean(), f1_score(y, p, average="macro", zero_division=0)
    print("%-24s %9.3f %9.3f %9.3f %9.3f"
          % (name, acc, mf1,
             f1_score(y, p, labels=["motivational"], average="macro", zero_division=0),
             f1_score(y, p, labels=["technical"], average="macro", zero_division=0)))
    if name != "majority class" and (best is None or mf1 > best[1]):
        best = (name, mf1)

print()
print("best by macro F1: %s (%.3f)" % best)
p = preds[best[0]]
print()
print(classification_report(y, p, zero_division=0, digits=3))
print("confusion (rows = hand label, cols = predicted):")
order = ["technical", "motivational", "tactical", "other"]
cm = confusion_matrix(y, p, labels=order)
print("%-14s %s" % ("", " ".join("%12s" % o[:12] for o in order)))
for o, row in zip(order, cm):
    print("%-14s %s" % (o, " ".join("%12d" % v for v in row)))
print()
for s in ("masters", "elite"):
    m = src == s
    print("  accuracy on %-8s %.3f  (n=%d)" % (s, (p[m] == y[m]).mean(), m.sum()))
