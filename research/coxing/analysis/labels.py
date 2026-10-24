"""Hand-coded reference labels for 178 eight-word windows.

Coded by the analyst against the definitions and exemplars in Nugent et al. (2025):
  T technical    - instruction or feedback on body movement / stroke mechanics
  M motivational - praise or encouragement, including shared identity and focus cues
  C tactical     - crew changes and their preparation, position, other crews, boat
                   metrics (split, rate), steering
  O other        - not a call to the crew (commentary, announcer, applause, noise)

A window may contain more than one theme; it is coded by the dominant one, which is
what a share measure requires. Single coder, not blinded to the hypothesis - the
central limitation of this reference standard and stated as such in the paper.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)

CODES = (
    # masters, windows 0-69 (70)
    "CCTMTTCCTTCTCTCTTTCTMMTCTTTTTCTMTMCMTTTTMCTTCCCMMCCTCCCCMMMTMCTMMMMTCM"
    # elite, windows 70-177 (108), in blocks of ten
    "OTTTTTTCCC"
    "TTOTTCCTOT"
    "CMCMCTCCCT"
    "TMTMCMOOTT"
    "TTMCMCCCMT"
    "TCTCCMCTTT"
    "TTCCMTTCCO"
    "CCTCCMMCMT"
    "CMTMCMTMTC"
    "OOTCCTTCCC"
    "CCCCMTMT"
)

LABEL = {"T": "technical", "M": "motivational", "C": "tactical", "O": "other"}


def labels():
    return [LABEL[ch] for ch in CODES]


if __name__ == "__main__":
    import collections
    import json
    d = json.load(open("to_label.json"))
    L = labels()
    assert len(L) == len(d), "have %d labels for %d items" % (len(L), len(d))
    print("labelled %d windows" % len(L))
    for src in ("masters", "elite"):
        sel = [L[i] for i, x in enumerate(d) if (x[0] == "masters") == (src == "masters")]
        c = collections.Counter(sel)
        n = len(sel)
        print("  %-8s n=%3d  tech %4.1f%%  moti %4.1f%%  tact %4.1f%%  other %4.1f%%"
              % (src, n, 100 * c["technical"] / n, 100 * c["motivational"] / n,
                 100 * c["tactical"] / n, 100 * c["other"] / n))
    c = collections.Counter(L)
    n = len(L)
    print("  %-8s n=%3d  tech %4.1f%%  moti %4.1f%%  tact %4.1f%%  other %4.1f%%"
          % ("all", n, 100 * c["technical"] / n, 100 * c["motivational"] / n,
             100 * c["tactical"] / n, 100 * c["other"] / n))
