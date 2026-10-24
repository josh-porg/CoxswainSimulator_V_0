"""Theme intensity by lexicon counting, replacing single-label classification.

Cross-validation showed single-label coding of eight-word windows reaching only 0.52
macro F1, with technical/tactical confusions dominating. The cause is that the task is
ill-posed: a window genuinely carries more than one theme, so forcing an argmax discards
information and invents disagreement.

Counting is well-posed and is what the downstream model actually needs. For each theme
we count matching tokens in the speech attached to each stroke; a stroke can carry
several themes at once, which is what really happens. No decision is made, so there is
nothing to be wrong about -- the only question is whether the lexicons separate the
themes, which is testable against the hand labels.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import re

LEX = {
    "technical": [
        r"catch", r"finish", r"leg", r"bod(y|ies)", r"swing", r"suspend", r"hang",
        r"handle", r"blade", r"square", r"release", r"sequenc", r"reach", r"ratio",
        r"length", r"\blong\b", r"press", r"connect", r"shoulder", r"sit up", r"chest",
        r"breathe", r"rhythm", r"complete", r"direct", r"sharp", r"attack",
        r"front end", r"hook", r"send", r"puddle", r"stretcher", r"toes", r"heels",
        r"\bhands\b", r"tidy", r"loose", r"\bdrive\b", r"\bflat\b", r"\blow\b",
        r"clean", r"squeeze", r"\bslide\b", r"\bknees\b", r"\brock\b", r"\bpush\b",
        r"\bpull\b", r"\btall\b", r"\bset\b",
    ],
    "motivational": [
        r"\bgood\b", r"\byes\b", r"\bnice\b", r"excellent", r"great", r"that.s it",
        r"you.ve got", r"dig deep", r"commit", r"capable", r"grit", r"strong",
        r"believe", r"boat ?mates", r"teammates", r"\bhurt\b", r"100", r"110",
        r"doing it", r"proud", r"remember", r"promise", r"courage", r"\bfight\b",
        r"\bgo\b", r"come on", r"let.s go", r"\byeah\b", r"\bboys\b", r"\bgirls\b",
        r"\bladies\b", r"\bguys\b", r"no regrets", r"well done", r"good work",
        r"\bhungry\b", r"\bmore\b", r"\bwalk\b", r"\bbreak(ing)?\b",
    ],
    "tactical": [
        r"\b1[89]\d\b", r"\b2[0-3]\d\b", r"\bin two\b", r"take (a )?(10|ten|five|5)",
        r"\b\d+\s*m\b", r"metre", r"meter", r"\blengths?\b", r"passing", r"\byield\b",
        r"\bclear\b", r"stern deck", r"\bbow\b", r"\blane\b", r"\bcourse\b",
        r"finish line", r"\bhalfway\b", r"\bquarter\b", r"\bsplit\b", r"\brate\b",
        r"\b2[4-9]\b", r"\b3[0-9]\b", r"per stroke", r"\bseat\b", r"\bstarboard\b",
        r"\bport\b", r"\bcorner\b", r"\bturn\b", r"\bstrokes\b", r"ready now",
        r"this one", r"\bstep\b", r"\btier\b", r"\bsecond(s)?\b", r"\bminute\b",
    ],
}
RX = {k: [re.compile(p, re.I) for p in v] for k, v in LEX.items()}


def counts(text):
    return {k: sum(len(r.findall(text)) for r in RX[k]) for k in RX}


if __name__ == "__main__":
    import json
    import numpy as np
    from labels import labels

    d = json.load(open("to_label.json"))
    y = np.array(labels())
    C = np.array([[counts(x[2])[k] for k in ("technical", "motivational", "tactical")] for x in d])
    names = ["technical", "motivational", "tactical"]

    print("lexicon hits per 8-word window, by hand label")
    print("%-14s %5s %11s %14s %11s" % ("hand label", "n", "tech hits", "moti hits", "tact hits"))
    for lab in names + ["other"]:
        m = y == lab
        if m.sum() == 0:
            continue
        print("%-14s %5d %11.2f %14.2f %11.2f"
              % (lab, m.sum(), C[m, 0].mean(), C[m, 1].mean(), C[m, 2].mean()))

    print()
    print("does each lexicon pick out its own class? (AUC, one-vs-rest)")
    for j, lab in enumerate(names):
        s = C[:, j].astype(float)
        pos, neg = s[y == lab], s[y != lab]
        # rank-based AUC with ties handled
        allv = np.concatenate([pos, neg])
        r = allv.argsort().argsort().astype(float)
        # average ranks for ties
        for v in np.unique(allv):
            idx = allv == v
            r[idx] = r[idx].mean()
        auc = (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
        print("   %-14s AUC %.3f   (n=%d vs %d)" % (lab, auc, len(pos), len(neg)))
