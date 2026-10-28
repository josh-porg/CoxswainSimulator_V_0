"""What 5k could a rower reach at a given aerobic ceiling? A rough chain from VO2max to 5k erg time.

    .venv/Scripts/python.exe research/training/ceiling.py [--t5k 1337.7] [--mass 73.5]

Chain, each link stated so the estimate can be audited and replaced:

1. 5k power from time: Concept2, P = 2.80 / (split/500)^3.
2. 5k split -> 2k split by Paul's law, a coaching rule of thumb (not a peer-reviewed law): the split slows by
   about 5 s/500 m per doubling of distance, so 2k = 5k - 5*log2(5000/2000) = 5k - 6.6 s.
3. 2k power taken as the maximal power of an incremental test (an approximation: the last 2-min stage of a
   7 x 2-min test is likely a little higher than 2k average power, so VO2max here is if anything understated).
4. VO2max from maximal power in female rowers: VO2max (mL/min) = 9.58 * MPO (W) + 958, r = 0.94, SEE 162 mL/min
   (4.6 %), 20 club-to-Olympic women, cross-validated on 10 (Mazza et al. 2023, IJSPP 18:861).

The same chain run backwards turns a VO2max level into a 5k. Reference levels for women rowers: Polish national
squads, U23 average 3.5-3.8 L/min ("high" above 3.8), a 2010 World Championship bronze double 4.45 L/min
(Klusiewicz et al. 2014, J Hum Kinet 44:121). Because both directions use one chain, its biases mostly cancel
when comparing her to a reference level; the absolute VO2max printed for her is rough (+/- ~10 %).
"""
from __future__ import annotations

import argparse
import math

PAUL_S_PER_DOUBLING = 5.0
SLOPE, INTERCEPT = 9.58, 958.0          # Mazza et al. 2023: mL/min per W, mL/min


def watts_from_split(split500):
    return 2.80 / (split500 / 500.0) ** 3


def split_from_watts(w):
    return 500.0 * (2.80 / w) ** (1.0 / 3.0)


def shift(split500, d_from, d_to):
    return split500 + PAUL_S_PER_DOUBLING * math.log2(d_to / d_from)


def vo2max_from_5k(t5k):
    """L/min."""
    p2 = watts_from_split(shift(t5k / 10.0, 5000, 2000))
    return (SLOPE * p2 + INTERCEPT) / 1000.0


def t5k_from_vo2max(vo2_l):
    p2 = (vo2_l * 1000.0 - INTERCEPT) / SLOPE
    return 10.0 * shift(split_from_watts(p2), 2000, 5000)


def mmss(t):
    m, s = divmod(t, 60.0)
    return "%d:%04.1f" % (m, s)


LEVELS = (("trained club woman (assumed)", 3.2),
          ("Polish U23 squad, average", 3.65),
          ("Polish U23 squad, high", 3.8),
          ("World Championship bronze double, 2010", 4.45))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5k", type=float, default=1337.7)
    ap.add_argument("--mass", type=float, default=73.5)
    ap.add_argument("--target", type=float, default=1080.0)
    a = ap.parse_args()
    now = vo2max_from_5k(a.t5k)
    need = vo2max_from_5k(a.target)
    print("now: 5k %s -> VO2max ~%.2f L/min (%.0f mL/kg/min)" % (mmss(a.t5k), now, 1000 * now / a.mass))
    print("target: 5k %s needs ~%.2f L/min (%.0f mL/kg/min), +%.0f%%" % (mmss(a.target), need, 1000 * need / a.mass,
                                                                    100 * (need / now - 1)))
    for name, v in LEVELS:
        print("  %-42s %.2f L/min -> 5k %s  (+%.0f%% on her estimate)" % (name, v, mmss(t5k_from_vo2max(v)),
                                                                          100 * (v / now - 1)))


if __name__ == "__main__":
    main()
