import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import io
import sys

EDITS = [
    (r"""\noindent One effect survived within-race validation---catch calls, recovered
independently by keyword and by language-model labelling---but it does not survive a
model in which boat speed is also predicted from its own history, and the additional
races could not resolve it in either direction. Treated as one coupled process, with a
causal transformer and a linear autoregression fitted to 38 recordings, neither stream
improved held-out prediction of the other, although both models recover coupling planted
in synthetic data of the same shape. Coding""",
     r"""\noindent One effect survived within-race validation---catch calls---but not a model
that also predicts speed from its own past. Modelled as one coupled process over 38
recordings, neither calls nor boat improved held-out prediction of the other, though a
causal transformer and a linear model both recover planted coupling. Coding"""),
    (r"""speech. A keyword lexicon transferred across races at a fifth of coverage, and no
optical method recovered boat speed from video""",
     r"""speech. No optical method recovered boat speed from video"""),
    (r"""survive that separation. The same is likely of any call made by feel, which on the
coxswain's own account includes most technical calls. Future corpora should be""",
     r"""survive that separation. Future corpora should be"""),
    (r"""The transformer does not outperform the linear reference on these data. It is included
because it is the form that scales: it can represent multi-step dependencies,
interactions between call types and per-recording context as data accumulate, and a new
recording enters as a transcript and a boat log on a shared clock, with no change to the
code.""",
     r"""The transformer does not outperform the linear reference on these data. It is retained
because it scales to multi-step dependencies, interactions between call types and
per-recording context as data accumulate; a new recording enters as a transcript and a
boat log on a shared clock."""),
    (r"""The coupling reported for the races is therefore absent at a size the
pipeline can see, not unmeasurable; effects of the size in Table~\ref{tab:power} lie well
below that.""",
     r"""Coupling in the races is absent at a size the pipeline can see; effects of the
size in Table~\ref{tab:power} lie well below it."""),
]
for p in ("paper.tex", "paper_sage.tex"):
    s = io.open(p, encoding="utf-8").read()
    for old, new in EDITS:
        if s.count(old) != 1:
            sys.exit("%s: %d matches for %r" % (p, s.count(old), old[:60]))
        s = s.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print(p, "ok")
