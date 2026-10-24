"""Head of the Charles 2026: where the squad's eight and four should finish, and the odds of requalifying.

    python research/crew/feeders/charles_update_2026.py

New since the 18 September reports:
  * the 2026 entries are published (RegattaCentral 10096): 31 Women's Grand Master Eights [50+],
    26 Women's Veteran Fours [60+, 70+]; returning crews are bowed in their 2025 finishing order;
  * Tail of the Lake was raced on 4 October 2026 (HereNow 21486), and its 2025 edition (21380)
    calibrates it against the 2025 Charles through the crews that raced both;
  * the four's own CoxBox at Tail of the Lake measures its split on the water.

Feeder scan: Textile River 2026 (HereNow 21478) has no crew entered in either event at the right
age; Quinsigamond Snake 2026 (21477) was collegiate only; Pink Ribbon Row 2026 (CrewTimer r16136)
has the squad's four and eight and Pocock's eight, but none of the other entered 60+ fours. So 2026 form is measured for the Pacific Northwest crews only, and every other crew is
carried by its 2025 Charles result.

Requalification: the top 50% of finishers (floor), as in the earlier reports.
"""
from __future__ import annotations

import csv
import os
import re
import contextlib
import io

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "crew", "feeders"))   # its data and outputs (gitignored)
ARCH = os.path.join(HERE, "..", "..", "scratchpad_archive_2026-09-27", "crew")
RNG = np.random.default_rng(20261006)
N_SIM = 40000
HOCR_D = 4800.0


def mmss(s):
    return "%d:%04.1f" % (int(s // 60), s % 60)


def secs(t):
    t = t.strip()
    if not re.match(r"^\d+:\d+(\.\d+)?$", t):
        return None
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def load_rows(path):
    out = []
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("|")
        if len(f) < 8:
            continue
        ev, place, bow, club, stroke, raw, adj, adjusted = f[:8]
        age = re.search(r"Age: (\d+)", adj)
        out.append(dict(event=ev, club=club.strip(), stroke=stroke.strip("() ").split(",")[0].strip().lower(),
                        raw=secs(raw), age=int(age.group(1)) if age else None,
                        penalised=bool(re.search(r"SB|FTY", adj))))
    return out


def field(path):
    return [(r["club"], float(r["time_s"])) for r in
            csv.DictReader(x for x in open(path, encoding="utf-8") if not x.startswith("#"))]


def club_key(c):
    c = c.lower().replace("'", "").replace("rowing association", "association")
    c = re.sub(r"\b(composite|rowing club|rowing|crew|inc\.?|club)\b", "", c)
    c = re.sub(r"\s+[a-z]$", "", c.strip())          # "Lake Union B" -> "lake union"
    return re.sub(r"\s+", " ", c).strip()


# --- 1. Tail of the Lake, 2026 against 2025: the day and the course ---------------------------
t25 = load_rows(os.path.join(HERE, "totl_2025_rows.txt"))
t26 = load_rows(os.path.join(HERE, "totl_2026_rows.txt"))
pairs = []
for a in t25:
    for b in t26:
        if (a["event"] == b["event"] and club_key(a["club"]) == club_key(b["club"]) and a["stroke"] == b["stroke"]
                and a["raw"] and b["raw"] and not a["penalised"] and not b["penalised"]
                and a["age"] and b["age"] and abs(a["age"] - b["age"]) <= 3):      # the same crew, not a new one
            pairs.append((a["event"], a["club"], a["stroke"], a["raw"], b["raw"], b["raw"] / a["raw"]))
ratios = np.array([p[5] for p in pairs])
boot = np.array([np.median(RNG.choice(ratios, ratios.size)) for _ in range(4000)])
IDX, IDX_SE = float(np.median(ratios)), float(boot.std())
print("TAIL OF THE LAKE, 2026 against 2025: the same crews (same event, club and stroke)")
for p in pairs:
    print("   %-20s %-26s %-14s %s -> %s  x%.3f" % (p[0], p[1], p[2], mmss(p[3]), mmss(p[4]), p[5]))
print("   %d crews: median %.3f (bootstrap se %.3f), spread %.3f-%.3f -> 2026 was %.1f%% quicker"
      % (ratios.size, IDX, IDX_SE, ratios.min(), ratios.max(), 100 * (1 - IDX)))
print("   (a crew's own year-on-year change is inside each ratio; the median is the day)")
# the same people in doubles and singles (HereNow 21380 and 21486, read 2026-10-06): (boat, 2025, 2026)
SMALL = [("2x", "R. Colven/M. Biery", 975.1, 932.9), ("2x", "K. Jenkins/C. Jenkins", 894.3, 876.7),
         ("2x", "L. Christiansen/M. McKinney", 1071.4, 999.8), ("2x", "J. Dykema/L. Doermann", 1017.2, 980.3),
         ("2x", "S. Foster/H. Powell", 1279.9, 1173.0),
         ("1x", "Cody Jenkins", 1022.9, 952.2), ("1x", "Eric Hawn", 1178.4, 1028.6), ("1x", "Hank Koerner", 1324.1, 1103.3),
         ("1x", "Andres Hermosilla", 1290.3, 1070.4), ("1x", "Olyvia Peterson", 1264.1, 1205.7),
         ("1x", "Irena Baker", 1473.4, 1226.7), ("1x", "Stephanie Geoffrion", 1325.9, 1080.2)]
print("   the same people in small boats:")
for b, who, a, c in SMALL:
    print("   %-3s %-30s %s -> %s  x%.3f" % (b, who, mmss(a), mmss(c), c / a))
# rough water costs a slow boat more: the index as a line in the 2025 time, fitted on the crew boats
# and doubles (singles shown, not fitted: a different boat class in a chop)
X = np.array([p[3] for p in pairs] + [a for b, _w, a, _c in SMALL if b == "2x"])
Y = np.array([p[5] for p in pairs] + [c / a for b, _w, a, c in SMALL if b == "2x"])
A_ = np.vstack([np.ones_like(X), (X - 1000.0) / 100.0]).T
coef, res, *_ = np.linalg.lstsq(A_, Y, rcond=None)
resid_sd = float(np.sqrt(np.sum((Y - A_ @ coef) ** 2) / (len(Y) - 2)))
cov = resid_sd ** 2 * np.linalg.inv(A_.T @ A_)


def index_at(t2025):
    x = np.array([1.0, (t2025 - 1000.0) / 100.0])
    return float(x @ coef), float(np.sqrt(x @ cov @ x))


print("   index = %.3f %+.4f per 100 s of 2025 time (%d crews, residual sd %.3f): %.3f at 16:00, %.3f at 20:00"
      % (coef[0], coef[1], len(Y), resid_sd, index_at(960)[0], index_at(1200)[0]))

# --- 2. Tail of the Lake 2025 -> the Charles 2025 ---------------------------------------------
four25 = dict(field(os.path.join(ARCH, "field_four_2025.csv")))
eight25 = dict(field(os.path.join(ARCH, "field_2025.csv")))
FOUR_PAIRS = [   # TOTL 2025 Womens Masters 4+ crew (age 60+) -> HOCR 2025 Veteran Fours entry
    ("Martha's Moms", "robins", "Marthas Moms"), ("Montlake", "bradshaw", "Montlake Rowing"),
    ("Lake Union B", "berner-hays", "Lake Union"), ("Conibear", "lockett", "Conibear"),
    ("Conibear", "goodlett", "Conibear B")]
conv4 = []
print("\nTAIL OF THE LAKE 2025 -> CHARLES 2025, Veteran Fours (the Pacific Northwest crews in both)")
for club, stroke, hocr in FOUR_PAIRS:
    t = next(r["raw"] for r in t25 if r["event"] == "Womens Masters 4+" and r["club"] == club and r["stroke"] == stroke)
    conv4.append(four25[hocr] / t)
    print("   %-14s %s -> %-16s %s   x%.4f" % (club, mmss(t), hocr, mmss(four25[hocr]), conv4[-1]))
conv4 = np.array(conv4)
C4, C4_SD = float(conv4.mean()), float(conv4.std(ddof=1))
print("   mean x%.4f, sd %.4f (%.1f%%)" % (C4, C4_SD, 100 * C4_SD / C4))
# the squad's own 50+ eight is the one eight in both: 2025 TOTL (Kelly, avg 55) -> 18:27.2
t_e25 = next(r["raw"] for r in t25 if r["event"] == "Womens Masters 8+" and r["stroke"] == "kelly")
C8 = eight25["Sammamish Rowing Association"] / t_e25
print("   eight: the squad's 50+ eight, TOTL %s -> Charles %s, x%.4f (one crew; its spread taken as the fours')"
      % (mmss(t_e25), mmss(eight25["Sammamish Rowing Association"]), C8))


def from_totl(t26_raw, conv, conv_sd):
    """A 2026 Tail of the Lake time on the 2025 Charles scale: back to 2025's day (the index for a
    boat this slow, solved for the 2025-equivalent time), then across."""
    t25 = t26_raw / IDX
    for _ in range(20):
        ix, ix_se = index_at(t25)
        t25 = t26_raw / ix
    mean = t25 * conv
    sd = mean * np.hypot(ix_se / ix, conv_sd / conv)
    return mean, sd


# --- 3. the squad's four: three measurements ---------------------------------------------------
R, R_SD, RIEGEL = 0.9778, 0.0107, 1.06          # same-boat flat-water -> Charles (feeder.py), Riegel


def from_split(split, dist):
    comb = R * (HOCR_D / dist) ** (RIEGEL - 1.0)
    t = split / 500.0 * comb * HOCR_D
    return t, t * R_SD / R


four_t26 = next(r["raw"] for r in t26 if r["event"] == "Womens Masters 4+" and r["stroke"] == "mayers")
cb_dist, cb_time = 3834.0, 18 * 60 + 12.0       # the four's CoxBox at Tail of the Lake (session 12:37)
cb_split = cb_time / cb_dist * 500.0
routes = [("Tail of the Lake, against the crews that raced both", *from_totl(four_t26, C4, C4_SD)),
          ("Tail of the Lake, CoxBox split %s over %.0f m" % (mmss(cb_split), cb_dist), *from_split(cb_split, cb_dist)),
          ("Pink Ribbon Row, 2:24.0 average over 4000 m", *from_split(144.0, 4000.0)),
          ("Pink Ribbon Row, 2:22.0 sustained", *from_split(142.0, 4000.0))]
print("\nTHE FOUR: what each measurement says it will row at the Charles")
for name, m, s in routes:
    print("   %-58s %s  (+-%.0f s)" % (name, mmss(m), s))
OWN_SD = 12.0                                                     # race-day, as in requal_curve.py
# two calibrations, not three measurements: the CoxBox and Pink Ribbon Row routes share R
CROSS = (routes[0][1], float(np.hypot(routes[0][2], OWN_SD)))
SPLIT_M = float(np.mean([routes[1][1], routes[2][1]]))
SPLIT = (SPLIT_M, float(np.hypot(routes[1][2], OWN_SD)))
print("   calibration A, crews that raced both:     %s, sd %.0f s" % (mmss(CROSS[0]), CROSS[1]))
print("   calibration B, the four's split x R:      %s, sd %.0f s" % (mmss(SPLIT[0]), SPLIT[1]))
print("   they differ by %.0f s (%.1f%%): how much faster the Charles is than Lake Union, per metre"
      % (SPLIT[0] - CROSS[0], 100 * (SPLIT[0] / CROSS[0] - 1)))
FOUR_MEAN, FOUR_SD = SPLIT                                        # reported under each, and as a 50/50 mix
# a check on calibration B's ratio for a four: Pink Ribbon Row 2025 (CrewTimer r14976, old course)
# -> Charles 2025, Martha's Moms' 60+ four (avg 66 at both feeders), the one such four that raced both
mm_prr, mm_hocr = 22 * 60 + 12.13, four25["Marthas Moms"]
print("   check: Martha's Moms 60+ four, Pink Ribbon Row 2025 %s -> Charles %s, ratio %.3f (the eights' R: %.3f)"
      % (mmss(mm_prr), mmss(mm_hocr), mm_hocr / mm_prr, R))
print("   so the flat-water-to-Charles ratio behind calibration B holds for a 60+ four: B is the better-founded")
print("   the 18 Sept report forecast 21:40")

# --- 4. the squad's eight -----------------------------------------------------------------------
eight_t26 = next(r["raw"] for r in t26 if r["event"] == "Womens Masters 8+" and r["stroke"] == "frederick")
e_totl, e_totl_sd = from_totl(eight_t26, C8, C4_SD)
E_PRIOR, E_PRIOR_SD = 1117.1, 16.9          # the 18 Sept erg-ratio forecast and its backtest RMSE
we = np.array([1 / e_totl_sd ** 2, 1 / E_PRIOR_SD ** 2])
EIGHT_MEAN = float((we * np.array([e_totl, E_PRIOR])).sum() / we.sum())
EIGHT_SD = float(np.sqrt(1 / we.sum() + 5.7 ** 2))
print("\nTHE EIGHT (taken as the Frederick eight, avg 56: the squad's fastest at Tail of the Lake)")
print("   Tail of the Lake %s -> Charles %s (+-%.0f s)" % (mmss(eight_t26), mmss(e_totl), e_totl_sd))
print("   18 Sept forecast (erg ratio) %s (+-%.0f s)" % (mmss(E_PRIOR), E_PRIOR_SD))
print("   combined %s, sd %.0f s incl. 5.7 s race-day" % (mmss(EIGHT_MEAN), EIGHT_SD))

# --- 5. the 2026 fields ------------------------------------------------------------------------
YEAR_SD = 0.013     # the Charles itself, year to year, as a fraction (winner-normalised spread)
CREW_SD = 0.024     # one crew's own change from one autumn to the next (2-2.7%, matched clubs)


def entries(path):
    return [r for r in csv.DictReader(x for x in open(path, encoding="utf-8") if not x.startswith("#"))]


ALIAS = {"Martha's Moms": "Marthas Moms", "Riverfront Recapture": "Riverfront Recapture",
         "BMA Boat Club": "BMA", "FILO Rowing": "FILO"}


OLDER = {}


def simulate(name, entry_path, field25, ours, ours_mean, ours_sd, measured):
    ent = entries(entry_path)
    n = len(ent)
    nq = n // 2
    last = np.array(sorted(field25.values()))
    clubs = [e["club"] for e in ent]
    kind, base, sd = [], [], []
    for c in clubs:
        if c == ours:
            kind.append("ours"); base.append(ours_mean); sd.append(ours_sd / ours_mean)
        elif c in measured:
            m, s = measured[c]
            kind.append("Charles %d" % OLDER.get(c, (0, 0, 0))[2] if c in OLDER else "TOTL 2026")
            base.append(m); sd.append(np.hypot(s / m, 0.012))
        elif ALIAS.get(c, c) in field25:
            kind.append("2025 Charles"); base.append(field25[ALIAS.get(c, c)]); sd.append(CREW_SD)
        else:
            kind.append("new"); base.append(np.nan); sd.append(CREW_SD)
    base, sd = np.array(base), np.array(sd)
    new = np.isnan(base)
    places = np.zeros((N_SIM, n), int)
    for k in range(N_SIM):
        day = np.exp(RNG.normal(0, YEAR_SD))                       # the river, for everyone
        t = base.copy()
        t[new] = RNG.choice(last, new.sum())                        # an unknown crew: anyone from last year
        t = t * day * np.exp(RNG.normal(0, sd))
        t[kind.index("ours")] = ours_mean * day * np.exp(RNG.normal(0, sd[kind.index("ours")]))
        places[k] = np.argsort(np.argsort(t)) + 1
    mine = places[:, kind.index("ours")]
    print("\n%s: %d entries, requalify = top %d" % (name, n, nq))
    print("   %-4s %-42s %-13s %-10s %s" % ("bow", "crew", "evidence", "time", "median place"))
    med = np.median(places, axis=0)
    for i in np.argsort(med):
        print("   %-4s %-42s %-13s %-10s %4.0f   P(top %d) %3.0f%%"
              % (ent[i]["bow"], clubs[i][:42], kind[i], "-" if new[i] else mmss(base[i]), med[i],
                 nq, 100 * np.mean(places[:, i] <= nq)))
    out = dict(median=float(np.median(mine)), p10=float(np.percentile(mine, 10)), p90=float(np.percentile(mine, 90)),
               requal=float(np.mean(mine <= nq)), top10=float(np.mean(mine <= 10)), top5=float(np.mean(mine <= 5)),
               medal=float(np.mean(mine <= 3)))
    print("   THE SQUAD: median place %d (80%%: %d-%d), requalify %.0f%%, top 10 %.0f%%, top 5 %.1f%%, medal %.1f%%"
          % (out["median"], out["p10"], out["p90"], 100 * out["requal"], 100 * out["top10"], 100 * out["top5"],
             100 * out["medal"]))
    return out, places, kind, clubs


# Pacific Northwest crews with a 2026 time: the entry's crew, by club and age, at Tail of the Lake
four_pnw = {"Conibear": ("Conibear", "goodlett"), "Mt. Baker": ("Mt. Baker B", "arvidson")}
measured4 = {}
for entry, (club, stroke) in four_pnw.items():
    t = next(r["raw"] for r in t26 if r["event"] == "Womens Masters 4+" and r["club"] == club and r["stroke"] == stroke)
    measured4[entry] = from_totl(t, C4, C4_SD)
print("\nPacific Northwest fours with a 2026 time (age 60+ at Tail of the Lake):")
for k, (m, s) in measured4.items():
    print("   %-12s -> Charles %s (+-%.0f s)" % (k, mmss(m), s))
print("   Martha's Moms raced Tail of the Lake as a 70+ crew in 23:53.5, which is not the crew that")
print("   placed 13th in 2025; Green Lake's crew was 59 on average. Both are carried by 2025 or as new.")

# --- 5b. older Charles results for crews with no 2025 time (added 2026-10-15) ---------------------
# A crew's most recent Charles time before 2025, put on 2025's scale by the ratio of the two years'
# field medians (the river and the field of that year), with its uncertainty growing as a random
# walk in the years since: CREW_SD x sqrt(2026 - year).
def older_times(rows_by_year, entry_clubs, known, match):
    med = {y: float(np.median([t for _, t in r])) for y, r in rows_by_year.items()}
    out = {}
    for c in entry_clubs:
        if c in known or ALIAS.get(c, c) in known or c == "Sammamish Rowing Association":   # ours has its own forecast
            continue
        for y in sorted(rows_by_year, reverse=True):
            hit = [t for club, t in rows_by_year[y] if match(c, club)]
            if hit:
                t25 = min(hit) * med[2025] / med[y]
                out[c] = (t25, t25 * CREW_SD * np.sqrt(2026 - y), y)
                break
    return out


def same_club(entry, club):
    a, b = club_key(entry), club_key(club)
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))


v4 = {}
for r in csv.DictReader(open(os.path.join(HERE, "..", "..", "..", "hocr_wvet4_results.csv"))):
    if r["band"] in ("60-69", ""):
        v4.setdefault(int(r["year"]), []).append((r["club"], float(r["seconds"])))
v8 = {y: field(os.path.join(ARCH, "field_%d.csv" % y)) for y in (2022, 2023, 2024, 2025)}
OLDER4 = older_times(v4, [e["club"] for e in entries(os.path.join(HERE, "hocr2026_veteran_fours_entries.csv"))],
                     set(four25) | set(measured4), same_club)
print("\nOlder Charles results for fours with no 2025 time (on 2025's scale):")
for c, (t, sd, y) in OLDER4.items():
    print("   %-28s %d -> %s (+-%.0f s)" % (c[:28], y, mmss(t), sd))
measured4.update({c: (t, sd) for c, (t, sd, y) in OLDER4.items()})
OLDER.update(OLDER4)

four_A, *_ = simulate("WOMEN'S VETERAN FOURS [60+, 70+], calibration A (crews that raced both)",
                      os.path.join(HERE, "hocr2026_veteran_fours_entries.csv"), four25,
                      "Sammamish Rowing Association", CROSS[0], CROSS[1], measured4)
with contextlib.redirect_stdout(io.StringIO()):
    four_B, *_ = simulate("", os.path.join(HERE, "hocr2026_veteran_fours_entries.csv"), four25,
                          "Sammamish Rowing Association", SPLIT[0], SPLIT[1], measured4)
print("   calibration B (the four's split x R): median place %d (80%%: %d-%d), requalify %.0f%%"
      % (four_B["median"], four_B["p10"], four_B["p90"], 100 * four_B["requal"]))
print("   50/50 between the two calibrations: requalify %.0f%%" % (50 * (four_A["requal"] + four_B["requal"])))

pocock = next(r["raw"] for r in t26 if r["event"] == "Womens Masters 8+" and r["club"] == "Pocock")
measured8 = {"Pocock Rowing Club": from_totl(pocock, C8, C4_SD)}
print("\nPocock's eight (avg 50) at Tail of the Lake %s -> Charles %s"
      % (mmss(pocock), mmss(measured8["Pocock Rowing Club"][0])))
OLDER8 = older_times(v8, [e["club"] for e in entries(os.path.join(HERE, "hocr2026_grand_master_eights_entries.csv"))],
                     set(eight25) | set(measured8), same_club)
print("\nOlder Charles results for eights with no 2025 time (on 2025's scale):")
for c, (t, sd, y) in OLDER8.items():
    print("   %-28s %d -> %s (+-%.0f s)" % (c[:28], y, mmss(t), sd))
measured8.update({c: (t, sd) for c, (t, sd, y) in OLDER8.items()})
OLDER.update(OLDER8)
eight_out, *_ = simulate("WOMEN'S GRAND MASTER EIGHTS [50+]",
                         os.path.join(HERE, "hocr2026_grand_master_eights_entries.csv"), eight25,
                         "Sammamish Rowing Association", EIGHT_MEAN, EIGHT_SD, measured8)

# --- 6. what the four needs --------------------------------------------------------------------
print("\nTHE FOUR: requalification against its own time (the 2026 field, simulated)")
for target in (1260, 1290, 1300, 1310, 1320, 1335, 1350):
    with contextlib.redirect_stdout(io.StringIO()):
        o, *_ = simulate("", os.path.join(HERE, "hocr2026_veteran_fours_entries.csv"), four25,
                         "Sammamish Rowing Association", float(target), OWN_SD, measured4)
    comb = R * (HOCR_D / 4000.0) ** (RIEGEL - 1.0)
    print("   Charles %s  = %s per 500 on flat water   -> requalify %3.0f%%, median place %d"
          % (mmss(target), mmss(target / HOCR_D / comb * 500.0), 100 * o["requal"], o["median"]))


# --- 7. the simulated fields, saved for the race card (docs/RACEDAY.md) --------------------------
def field_draws(entry_path, field25, ours, measured, n_draw=N_SIM):
    """Every other crew's time and the day factor, per simulated race, so a target time for our
    crew can be scored against the field: P(place <= k | our time T on the same day)."""
    ent = entries(entry_path)
    clubs = [e["club"] for e in ent]
    last = np.array(sorted(field25.values()))
    others = [c for c in clubs if c != ours]
    base, sd = [], []
    for c in others:
        if c in measured:
            m, s = measured[c]; base.append(m); sd.append(np.hypot(s / m, 0.012))
        elif ALIAS.get(c, c) in field25:
            base.append(field25[ALIAS.get(c, c)]); sd.append(CREW_SD)
        else:
            base.append(np.nan); sd.append(CREW_SD)
    base, sd = np.array(base), np.array(sd)
    new = np.isnan(base)
    rng = np.random.default_rng(20261009)
    day = np.exp(rng.normal(0, YEAR_SD, n_draw))
    t = np.tile(base, (n_draw, 1))
    t[:, new] = rng.choice(last, (n_draw, int(new.sum())))
    t = t * day[:, None] * np.exp(rng.normal(0, 1, t.shape) * sd[None, :])
    return dict(others=t.astype(np.float32), day=day, n=len(clubs), nq=len(clubs) // 2)


def last_place(field25, ours):
    times = sorted(field25.values())
    return times.index(field25[ours]) + 1 if ours in field25 else None


OUT_RD = os.path.join(HERE, "..", "..", "raceday")
os.makedirs(OUT_RD, exist_ok=True)
for key, path, f25, own_mean, own_sd, meas in (
        ("four", "hocr2026_veteran_fours_entries.csv", four25, CROSS[0], CROSS[1], measured4),
        ("eight", "hocr2026_grand_master_eights_entries.csv", eight25, EIGHT_MEAN, EIGHT_SD, measured8)):
    d = field_draws(os.path.join(HERE, path), f25, "Sammamish Rowing Association", meas)
    lp = last_place(f25, "Sammamish Rowing Association")
    np.savez_compressed(os.path.join(OUT_RD, "field_%s.npz" % key), others=d["others"], day=d["day"],
                        n=d["n"], nq=d["nq"], own_mean=own_mean, own_sd=own_sd,
                        last_place=-1 if lp is None else lp, last_field=len(f25))
    print("saved field_%s.npz: %d entries, requalify top %d, own forecast %s +- %.0f s, 2025 place %s of %d"
          % (key, d["n"], d["nq"], mmss(own_mean), own_sd, lp, len(f25)))
