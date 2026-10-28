"""A research-backed supplementary training plan for a club rower, regenerated after every test.

    .venv/Scripts/python.exe research/training/plan_generator.py --profile data/local/training/<athlete>.json
                                                             [--weeks 16] [--out data/local/training/plans]

The athlete trains three times a week with the team (on the water or the erg); this plans everything else: the
low-intensity aerobic volume the team week lacks, strength training, testing, deloads and tapers, around the team
sessions, on a four-year horizon. Inputs are a private profile (tests, body mass, the team week, available days and
time, the target); outputs are the athlete's zones, a projection of her trajectory from the squad's own records with
the probability of the target, the macrocycle to the target date, and the first weeks in detail, each rule tagged
with the evidence behind it (docs in research/training/README.md; the review in the athlete's plan document).

Rerun after each test: the zones, the projection and the plan all move with the newest result.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------------------------------
# evidence keys used in the output; the full review is in the plan document and research/training/README.md
EVIDENCE = {
    "TID": "Most training below the first threshold (~75-80% of time): Seiler 2010; Fiskerstrand & Seiler 2004; "
           "Treff et al. 2017 (polarized not superior to pyramidal when low-intensity share is equal); Oliveira et al. "
           "2024 meta-analysis (small VO2peak advantage for polarized, SMD 0.24); elite rowers 14-20 h/wk, mostly "
           "pyramidal, no distribution proven superior (systematic review of elite rowers).",
    "VOLUME": "Volume is the long-term lever: Norwegian international rowers raised low-intensity volume over three "
              "decades while cutting high-intensity work (Fiskerstrand & Seiler 2004).",
    "STRENGTH": "Heavy strength training alongside endurance training improves economy and performance (Ronnestad & "
                "Mujika 2014); in rowers small effects on leg strength (SMD 0.42) and rowing performance (SMD 0.32) "
                "regardless of type or level (Thiele et al. 2020); heavy loads raise strength and power more than "
                "strength-endurance work in female rowers (Thiele et al. 2020, Front Physiol).",
    "CONCURRENT": "The AMPK-mTOR 'switch' comes from rodent muscle stimulation; in humans prior endurance work does not "
                  "blunt mTORC1 signalling after lifting, and adding lifting raises PGC-1a expression (Coffey & Hawley "
                  "2017 review). Over 43 trials, concurrent training leaves maximal strength (SMD -0.06) and muscle "
                  "size (-0.01) intact but blunts explosive strength (-0.28), mainly in the same session; with 3 h or "
                  "more between them the effect was not significant (Schumann et al. 2022). Interference appears in "
                  "trained lifters and within 20 min (Petre et al. 2021); 0 h between them cut strength gains against 6 "
                  "or 24 h, and VO2peak rose most with 24 h (Robineau et al. 2016). Lifting first gives more strength "
                  "(Murlasits et al. 2018). So: lift and row on separate days where possible, otherwise lift first and "
                  "row 6 h or more later; keep the power work of the build phase off rowing days.",
    "PERIOD": "Periodize, but no model is proven superior: block periodization of intervals shows a small, "
              "low-quality advantage (Molmen et al. 2019); a load-matched trial found none (Almquist et al. 2022).",
    "DELOAD": "3:1 load-to-recovery mesocycles are convention, not trial-tested; progression caps such as 10% per "
              "week and the acute:chronic ratio have no demonstrated causal link to injury (Impellizzeri et al. 2020): "
              "progress gradually and let fatigue markers decide.",
    "TAPER": "Taper 8-14 days, cut volume 41-60%, keep intensity and frequency (Bosquet et al. 2007; PLOS ONE 2023 "
             "meta-analysis).",
    "HIIT": "Short blocks of intervals at ~90% of VO2peak velocity improve 2k erg power in trained rowers (Driller et "
            "al. 2009, n = 10); the team week already holds two hard sessions.",
    "HRV": "HRV-guided adjustment gives a small VO2max benefit, larger in amateurs and women (Granero-Gallegos et al. "
           "2020); vagal markers improve, resting HR not (Manresa-Rocamora et al. 2021).",
    "CYCLE": "No phase-based programming: phase effects on performance are trivial and low quality (McNulty et al. "
             "2020), and an umbrella review finds no phase effect on strength or training adaptations (Colenso-Semple "
             "et al. 2023; a narrative review argues for follicular-weighted lifting, Kissow et al. 2022, on small "
             "trials). Symptoms are common and cost training (6812 women: fatigue 86%, cramps 84%; Bruinvels et al. "
             "2021), so a symptom day changes that day. Cycles over 35 days or none for 3 months: see a doctor (REDs). "
             "Oral contraceptives: probably trivial performance effect, no general advice (Elliott-Sale et al. 2020). "
             "The site's optional cycle tracker applies this in the browser only; nothing is stored in the plan.",
    "IRON": "Iron deficiency affects ~15-35% of female athletes; exercise raises hepcidin and cuts absorption for "
            "hours (Sim et al. 2019): ferritin test once or twice a year, supplement only on a result.",
    "HEAT": "Core temperature runs ~0.3-0.7 C higher in the luteal phase (Baker et al. 2020): on warm days drink more "
            "and pace by watts, not heart rate.",
    "REDS": "Low energy availability harms health and performance in female athletes: eat to the training (2023 IOC "
            "REDs consensus, Mountjoy et al.).",
    "PROTEIN": "Gains from protein plateau near 1.6 g/kg/day (Morton et al. 2018), spread over meals.",
    "RIB": "Rib stress injury risk factors are unproven (D'Ailly et al. 2017); amenorrhoea, low bone density, "
           "technique and sudden load changes recur in case reports: progress rowing volume steadily.",
    "CARBS": "Daily carbohydrate scaled to the training load, roughly 3-5 g/kg on light days and 5-7 g/kg on ~1 h/day "
             "of moderate training (Burke et al. 2011, J Sports Sci; figures approximate, from the paper's guideline "
             "table as commonly reproduced - confirm with a sports dietitian).",
    "SLEEP":"Athletes often sleep under 7 h; plan sleep to individual need rather than one target (Walsh et al. 2021 "
             "expert consensus).",
    "SEX": "Women gain lower-body strength and muscle at the same relative rate as men, and more upper-body strength "
           "(Roberts et al. 2020).",
    "SQUAD": "Projection from the squad's own longitudinal 5k records (research/training/squad_progression.py).",
}

# the team's power zones (its "Power Zone Training" tab): % of FTP; FTP is the 5k's watts x 0.935 in the team sheet
ZONES = [("Z1 recovery", 0.00, 0.55), ("Z2 endurance (UT2)", 0.56, 0.75), ("Z3 high UT", 0.76, 0.90),
         ("Sweet spot", 0.85, 0.95), ("Z4 threshold (AT)", 0.91, 1.05), ("Z5 VO2 (TR)", 1.06, 1.20),
         ("Z6 anaerobic", 1.21, 1.50), ("Z7 neuromuscular", 1.50, 9.99)]
FTP_FROM_5K = 0.935


def watts(split500):
    return 2.80 / (split500 / 500.0) ** 3


def split(w):
    return 500.0 * (2.80 / max(w, 1e-9)) ** (1.0 / 3.0)


def mmss(seconds, tenths=True):
    m, s = divmod(round(float(seconds), 1 if tenths else 0), 60.0)       # round first: never "21:60"
    return ("%d:%04.1f" if tenths else "%d:%02.0f") % (int(m), s)


def time5k_from_watts(w):
    return 10.0 * split(w)


# ---------------------------------------------------------------------------------------------------------------
def load_profile(path):
    p = json.load(open(path, encoding="utf-8"))
    for t in p["tests"]:
        t["date"] = dt.date.fromisoformat(t["date"])
    p["tests"].sort(key=lambda t: t["date"])
    p["start"] = dt.date.fromisoformat(p["start"])
    p["target"]["by"] = dt.date.fromisoformat(p["target"]["by"])
    return p


def latest_5k(p):
    five = [t for t in p["tests"] if int(t["distance_m"]) == 5000]
    t = five[-1]
    return t, watts(t["time_s"] / 10.0)


def zones(p):
    t, w5 = latest_5k(p)
    ftp = FTP_FROM_5K * w5
    out = []
    for name, lo, hi in ZONES:
        wl, wh = lo * ftp, min(hi, 2.5) * ftp
        out.append(dict(zone=name, watts=(round(wl), round(wh)),
                        split=(mmss(split(wh)) if wh > 0 else "-", mmss(split(wl)) if wl > 0 else "-")))
    return dict(test=t, watts_5k=w5, ftp=ftp, zones=out)


# ---------------------------------------------------------------------------------------------------------------
def projection(p, squad_json, seasons=None, n=20000, seed=7):
    """Monte Carlo of the 5k from now to the target date: each year a relative change drawn from the squad's women's
    year-on-year record at the matching training age (under-40s for training ages under 3.5, all ages after, where
    young data run out)."""
    rng = np.random.default_rng(seed)
    d = json.load(open(squad_json))
    pairs = [q for q in d["pairs"] if q["sex"] == p.get("sex", "F")]
    young = [q for q in pairs if q["age0"] is not None and q["age0"] < 40]

    def band(train_age):
        src = young if train_age < 3.5 else pairs
        lo, hi = (0, 1.5) if train_age < 1.5 else ((1.5, 3.5) if train_age < 3.5 else (3.5, 99))
        g = np.array([q["rel_per_year"] for q in src if lo <= q["train_age"] < hi])
        return g if g.size >= 5 else np.array([q["rel_per_year"] for q in pairs if lo <= q["train_age"] < hi])

    t, w5 = latest_5k(p)
    first = p["tests"][0]["date"]
    age_now = (t["date"] - first).days / 365.25
    years = (p["target"]["by"] - t["date"]).days / 365.25
    whole = int(np.floor(years))
    frac = years - whole
    w = np.full(n, w5)
    path = []
    for y in range(whole + (1 if frac > 0.05 else 0)):
        span = 1.0 if y < whole else frac
        g = band(age_now + y)
        draw = rng.choice(g, size=n, replace=True)
        w = w * (1.0 + draw) ** span
        path.append(dict(date=(t["date"] + dt.timedelta(days=round(365.25 * (y + span)))).isoformat(),
                         train_age=round(age_now + y + span, 2),
                         p10=time5k_from_watts(np.percentile(w, 90)), p50=time5k_from_watts(np.median(w)),
                         p90=time5k_from_watts(np.percentile(w, 10))))
    target_w = watts(p["target"]["time_s"] / 10.0)
    need = (target_w / w5) ** (1.0 / years) - 1.0
    ladder = []                                                     # fixed rates sustained every year to the target date
    for label, rate in (("squad median, years 2-3", float(np.median(band(2.0)))),
                        ("squad 90th percentile, years 2-3", float(np.percentile(band(2.0), 90))),
                        ("squad best single year, years 2-3", float(np.max(band(2.0))))):
        wf = w5 * (1.0 + rate) ** years
        ladder.append(dict(label=label, rate=rate, watts=wf, time=time5k_from_watts(wf)))
    return dict(start=dict(date=t["date"].isoformat(), time=t["time_s"], watts=w5), target_watts=target_w,
                years=years, required_rate=need, p_target=float(np.mean(w >= target_w)), path=path, ladder=ladder,
                squad_best_rate_years_2_3=float(max(q["rel_per_year"] for q in pairs if 1.5 <= q["train_age"] < 3.5)))


# ---------------------------------------------------------------------------------------------------------------
def phase_on(date, p):
    """The training phase on a date, from the team's season: three weeks' transition after the key fall (head) race,
    base to the end of February, build through April, spring racing to the key spring race, summer base to August,
    fall racing to the key fall race."""
    fm, fd = p.get("fall_key_race_md", (10, 17))
    sm, sd = p.get("spring_key_race_md", (6, 1))
    y = date.year
    fall = dt.date(y, fm, fd)
    prev_fall = dt.date(y - 1, fm, fd)
    if fall < date <= fall + dt.timedelta(days=21) or prev_fall < date <= prev_fall + dt.timedelta(days=21):
        return "transition"
    if date > fall or date < dt.date(y, 3, 1):
        return "base"
    if date < dt.date(y, 5, 1):
        return "build"
    if date <= dt.date(y, sm, sd):
        return "spring-race"
    if date < dt.date(y, 9, 1):
        return "summer-base"
    return "fall-race"


def days_to_next_key(date, p):
    out = []
    for md in (tuple(p.get("fall_key_race_md", (10, 17))), tuple(p.get("spring_key_race_md", (6, 1)))):
        for year in (date.year, date.year + 1):
            d = dt.date(year, md[0], md[1])
            if d >= date:
                out.append((d - date).days)
                break
    return min(out)


STRENGTH = {   # Ronnestad & Mujika 2014; Thiele et al. 2020: heavy loads, few reps, through the year
    "transition": ("Easing back in", "2 sets of 12-15 reps, light (about 60% of your max); technique and full range", 1),
    "base": ("Strength", "3-4 sets of 6-8 reps at 75-85% of your max, 2-3 min rest; later in base, 4 sets of 4-6 at about 85%", 2),
    "build": ("Strength into power", "3-4 sets of 3-5 reps at 85-90% of your max, plus 3 sets of 5 fast lifts (power cleans or jumps)", 2),
    "spring-race": ("Holding strength", "2-3 sets of 3-5 reps at about 85% of your max; short and sharp", 1),
    "summer-base": ("Strength", "3-4 sets of 6-8 reps at 75-85% of your max", 2),
    "fall-race": ("Holding strength", "2-3 sets of 3-5 reps at about 85% of your max; short and sharp", 1),
}
EXERCISES = ("back or front squat (or leg press)", "Romanian deadlift or trap-bar deadlift", "bench pull or seated row",
             "hip thrust", "split squat", "pull-up or lat pull-down", "trunk: dead bug, side plank, Pallof press")


def build_week(monday, p, year_index, week_in_block, base_minutes):
    """One week: the team sessions as given, and the athlete's own sessions placed around them."""
    phase = phase_on(monday + dt.timedelta(days=3), p)
    team = {s["day"]: s for s in p["team_week"]}
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hard_team = [s["day"] for s in p["team_week"] if s["zone"] in ("Z4", "Z5", "Z6")]
    deload = week_in_block == 4
    to_key = days_to_next_key(monday, p)
    taper = to_key <= 13 and phase != "transition"                # the race week itself can fall in the next phase
    scale = 0.6 if deload else 1.0
    if taper:
        scale = 0.5                                              # 41-60% volume cut, intensity and frequency kept
    if phase == "transition":
        scale = 0.5
    aerobic = base_minutes * scale
    n_strength = STRENGTH[phase][2] if not taper else 1
    sessions = {d: [] for d in days}
    for d, s in team.items():
        sessions[d].append(dict(kind="team", what=s["what"], zone=s["zone"], minutes=s["minutes"],
                                work_minutes=s.get("work_minutes", s["minutes"])))
    free = [d for d in days if d not in team and d in p["available_days"]]
    # strength: not the day before a hard team session where avoidable; lift on team-free days or after an easy team day
    order = sorted(free, key=lambda d: (days[(days.index(d) + 1) % 7] in hard_team, d not in ("Tue", "Thu", "Sat")))
    lift_days = sorted(order[:n_strength], key=days.index)
    for d in lift_days:
        sessions[d].append(dict(kind="strength", what="%s: %s" % (STRENGTH[phase][0], STRENGTH[phase][1]),
                                zone="-", minutes=45 if not deload else 30))
    # low-intensity aerobic volume: one long session on the weekend, the rest split over the free days left
    long_day = next((d for d in ("Sun", "Sat") if d in free), free[-1] if free else None)
    # the minutes placed add up to the week's target (to 5 min), except where the per-session caps run out of days
    r5 = lambda x: int(5 * round(x / 5.0))
    cap_long, cap_short = p.get("long_session_cap_min", 90), 60
    others = sorted((d for d in free if d != long_day), key=lambda d: (d in lift_days, days.index(d)))
    remaining = r5(aerobic)
    long_m = 0
    if long_day and remaining >= 20:
        long_m = remaining if remaining < 60 else min(max(r5(0.5 * remaining), remaining - cap_short * len(others)),
                                                       cap_long)
        remaining -= long_m
    n_sessions = min(len(others), -(-remaining // cap_short)) if remaining >= 20 else 0
    if n_sessions and remaining / n_sessions < 30 and long_day:      # too little for a short session: lengthen the long one
        add = min(remaining, cap_long - long_m)
        long_m, remaining = long_m + add, remaining - add
        n_sessions = min(len(others), -(-remaining // cap_short)) if remaining >= 30 else 0
    if long_m:
        sessions[long_day].append(dict(kind="aerobic", what="Long and steady: erg, water, bike, run or ski",
                                       zone="Z2", minutes=long_m))
    for k, d in enumerate(others[:n_sessions]):
        m = min(r5(remaining / (n_sessions - k)), cap_short)
        remaining -= m
        sessions[d].append(dict(kind="aerobic", what="Steady and easy at rate 18-20, focus on technique (or bike, run, ski)",
                                zone="Z2", minutes=m))
    # a lift and an aerobic session on one day: lift first, row at least 6 h later [CONCURRENT]
    for d in lift_days:
        for s in sessions[d]:
            if s["kind"] == "aerobic":
                s["what"] += " - at least 6 h after the lift (lift in the morning, row in the evening)"
    # zone accounting for the week
    zmin = {"low": 0.0, "mid": 0.0, "high": 0.0}
    for d in days:
        for s in sessions[d]:
            if s["kind"] == "strength":
                continue
            z = s["zone"]
            band = "low" if z in ("Z1", "Z2") else ("mid" if z in ("Z3", "Z4", "SS") else "high")
            work = s["minutes"] if s["kind"] != "team" else s.get("work_minutes", s["minutes"])
            zmin[band] += work
            if s["kind"] == "team":
                zmin["low"] += s["minutes"] - work                  # warm-up, cool-down, rests: easy
    total = sum(zmin.values())
    tag = " (taper)" if taper else (" (deload)" if deload and phase != "transition" else "")
    return dict(monday=monday.isoformat(), phase=phase + tag,
                sessions=sessions, minutes=total, share_low=zmin["low"] / total if total else 0.0)


def plan(p, weeks):
    out = []
    monday = p["start"] - dt.timedelta(days=p["start"].weekday())
    base0 = p["own_aerobic_minutes_start"]
    cap = p["own_aerobic_minutes_cap"]
    grow = p.get("load_week_growth", 0.05)
    minutes = base0
    block = 0
    last_phase = None
    for i in range(weeks):
        mon = monday + dt.timedelta(weeks=i)
        ph = phase_on(mon + dt.timedelta(days=3), p)
        if ph != last_phase:
            block = 0
            last_phase = ph
        block += 1
        week_in_block = (block - 1) % 4 + 1
        year_index = (mon - p["start"]).days // 365
        w = build_week(mon, p, year_index, week_in_block, minutes)
        out.append(w)
        if week_in_block < 4 and ph not in ("transition",):
            minutes = min(minutes * (1.0 + grow), cap)
    return out


def macrocycle(p):
    rows = []
    year = p["start"].year
    last = None
    d = p["start"]
    while d <= p["target"]["by"]:
        ph = phase_on(d, p)
        if ph != last:
            rows.append(dict(start=d.isoformat(), phase=ph))
            last = ph
        d += dt.timedelta(days=7)
    return rows


def test_schedule(p):
    dates = []
    for y in range(p["start"].year, p["target"]["by"].year + 1):
        for (m, dd), what in (((12, 10), "5k (team winter test)"), ((4, 15), "5k (team spring test) + optional 2k"),
                               ((9, 25), "5k (team fall test)")):
            d = dt.date(y, m, dd)
            if p["start"] <= d <= p["target"]["by"]:
                dates.append((d.isoformat(), what))
    return sorted(dates)


def milestones(proj, tests):
    """The 5k each test should show if the squad's 90th-percentile rate (on track) or median rate (floor) holds from
    the latest test. Two tests running below the floor is the signal to change the plan."""
    rates = {r["label"]: r["rate"] for r in proj["ladder"]}
    on, floor = rates["squad 90th percentile, years 2-3"], rates["squad median, years 2-3"]
    d0, w0 = dt.date.fromisoformat(proj["start"]["date"]), proj["start"]["watts"]
    out = []
    for d, _ in tests:
        y = (dt.date.fromisoformat(d) - d0).days / 365.25
        out.append(dict(date=d, on_track=time5k_from_watts(w0 * (1 + on) ** y),
                        floor=time5k_from_watts(w0 * (1 + floor) ** y)))
    return out


def rules(p, grams=True):
    """The adjustment and health rules as (section title, [rule]) pairs: one source for the markdown report and the
    website. grams=False leaves out figures that would reveal body mass (for a published page). Written to the
    athlete, in the second person: the page and the report are hers."""
    adjust = [
        "Check in each morning: resting heart rate or HRV if you have it, and how ready you feel out of 10. Two low "
        "mornings in a row? Make today's own session very easy (Z1) or rest. The team session is as you feel [HRV].",
        "Your own easy minutes grow about %.0f%% a week, from %d toward %d. Every fourth week is lighter (about 60%%) "
        "so the work can sink in [DELOAD]." % (100 * p.get("load_week_growth", 0.05), p["own_aerobic_minutes_start"],
                                              p["own_aerobic_minutes_cap"]),
        "In the two weeks before a key race your own volume halves; the team's intensity stays. You should feel "
        "fresh, not flat [TAPER].",
        "Lifting and rowing on the same day: lift first, row at least 6 hours later [CONCURRENT].",
        "Missed a session or a week? Pick up where you are; don't make it up. One miss doesn't undo anything.",
        "One slow test is noise. Two tests in a row below the floor: keep your volume steady, look at sleep, food and "
        "illness first, then we change the plan.",
        "Your zones, milestones and plan update by themselves after every test.",
    ]
    mass = p.get("body_mass_kg")
    if mass and grams:
        health = ["Protein: about %.0f g a day (1.6 g/kg), over 4 meals, one after lifting [PROTEIN]." % (1.6 * mass),
                  "Carbohydrate to match the day: roughly %.0f-%.0f g on light days, %.0f-%.0f g on days with two "
                  "sessions or the long one (approximate). Eat soon after hard or long sessions [CARBS] [REDS]."
                  % (3 * mass, 5 * mass, 5 * mass, 7 * mass)]
    else:
        health = ["Protein: about 1.6 g per kg of body weight a day, over 4 meals, one after lifting [PROTEIN].",
                  "Carbohydrate to match the day: roughly 3-5 g/kg on light days, 5-7 g/kg on days with two "
                  "sessions or the long one (approximate). Eat soon after hard or long sessions [CARBS] [REDS]."]
    health += [
        "Eat to the training. Missed or irregular periods, colds that keep coming back, tiredness that won't lift, "
        "falling results or bone pain are signs you're under-fuelled: hold your volume and see a doctor or sports "
        "dietitian [REDS].",
        "Sleep as much as you need; waking without an alarm on rest days is a good sign. Protect the nights before "
        "team sessions [SLEEP].",
        "Pain at the side of your chest that gets worse with rowing or deep breaths: stop the rowing volume and get "
        "it checked [RIB].",
        "Your cycle doesn't change the plan, but how you feel does. On a bad symptom day, make your own session "
        "20-30 minutes easy, or rest. The tracker on the plan page does this for you, privately [CYCLE].",
        "A ferritin (iron) blood test once or twice a year; iron supplements only if it comes back low [IRON].",
        "Warm days after ovulation: drink more and pace by watts, not heart rate [HEAT].",
    ]
    return [("How the plan adjusts", adjust), ("Fuel, sleep and health", health)]


# ---------------------------------------------------------------------------------------------------------------
def report(p, z, proj, weeks, macro, tests):
    L = []
    t = z["test"]
    L.append("# Supplementary training plan (generated %s)\n" % dt.date.today().isoformat())
    L.append("Latest 5k: %s on %s (%.0f W). FTP (team convention, 5k watts x %.3f): %.0f W.\n"
             % (mmss(t["time_s"]), t["date"].isoformat(), z["watts_5k"], FTP_FROM_5K, z["ftp"]))
    L.append("## Zones\n\n| zone | watts | split /500m |\n|---|---|---|")
    for r in z["zones"]:
        L.append("| %s | %d-%d | %s-%s |" % (r["zone"], r["watts"][0], r["watts"][1], r["split"][0], r["split"][1]))
    L.append("\n## The target, against the squad's record\n")
    L.append("Target %s for 5k by %s = %.0f W: %.0f%% more power than now; %.1f%% a year for %.1f years. The squad's best "
             "single year at training ages 2-3 is %+.1f%%. Probability of the target on the squad's record: %.1f%%."
             % (mmss(p["target"]["time_s"]), p["target"]["by"].isoformat(), proj["target_watts"],
                100 * (proj["target_watts"] / proj["start"]["watts"] - 1), 100 * proj["required_rate"], proj["years"],
                100 * proj["squad_best_rate_years_2_3"], 100 * proj["p_target"]))
    L.append("\n| by | training age | 10th pct (fast) | median | 90th pct (slow) |\n|---|---|---|---|---|")
    for r in proj["path"]:
        L.append("| %s | %.1f | %s | %s | %s |" % (r["date"], r["train_age"], mmss(r["p10"]), mmss(r["p50"]), mmss(r["p90"])))
    L.append("\nIf one rate held every year to the target date:\n\n| sustained yearly gain | rate | 5k at the target date |\n|---|---|---|")
    for r in proj["ladder"]:
        L.append("| %s | %+.1f%% | %s |" % (r["label"], 100 * r["rate"], mmss(r["time"])))
    L.append("| needed for the target | %+.1f%% | %s |" % (100 * proj["required_rate"], mmss(p["target"]["time_s"])))
    L.append("\n## Macrocycle\n\n| from | phase |\n|---|---|")
    for r in macro:
        L.append("| %s | %s |" % (r["start"], r["phase"]))
    L.append("\n## Tests and milestones\n\nOn track = the squad's 90th-percentile yearly rate held from the latest test; "
             "floor = the squad median. Two tests running below the floor: change the plan.\n")
    L.append("| test | what | on track | floor |\n|---|---|---|---|")
    for (d, w), m in zip(tests, milestones(proj, tests)):
        L.append("| %s | %s | %s | %s |" % (d, w, mmss(m["on_track"]), mmss(m["floor"])))
    L.append("\n## Strength\n\nExercises (pick 4-5 a session; legs and hinge first, pulls, then trunk): %s. Lift on "
             "days without rowing where possible; on a shared day lift first and row at least 6 h later (the fast "
             "power work of the build phase is the most sensitive to interference); 20-40 g protein after "
             "lifting; leave 2 reps in reserve; add load when the top set feels like "
             "3+ in reserve [STRENGTH] [CONCURRENT].\n" % "; ".join(EXERCISES))
    L.append("| phase | focus | prescription | sessions/wk |\n|---|---|---|---|")
    for ph, (focus, rx, n) in STRENGTH.items():
        L.append("| %s | %s | %s | %d |" % (ph, focus, rx, n))
    L.append("\n## The first %d weeks\n" % len(weeks))
    for w in weeks:
        L.append("### Week of %s: %s (%.0f min, %.0f%% low intensity)\n" % (w["monday"], w["phase"], w["minutes"],
                                                                            100 * w["share_low"]))
        for d, ss in w["sessions"].items():
            if ss:
                L.append("- **%s**: " % d + "; ".join("%s %s%s" % (s["what"], ("%d min" % s["minutes"]),
                                                                    (" @ " + s["zone"]) if s["zone"] != "-" else "")
                                                      for s in ss))
        L.append("")
    for title, lines in rules(p):
        L.append("## %s\n" % title)
        L.extend("- " + x for x in lines)
        L.append("")
    L.append("## Evidence\n")
    for k, v in EVIDENCE.items():
        L.append("- **[%s]** %s" % (k, v))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--weeks", type=int, default=16)
    ap.add_argument("--squad", default=os.path.join(ROOT, "data", "local", "training", "squad_progression.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "local", "training", "plans"))
    a = ap.parse_args()
    p = load_profile(a.profile)
    z = zones(p)
    proj = projection(p, a.squad)
    weeks = plan(p, a.weeks)
    macro = macrocycle(p)
    tests = test_schedule(p)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, "%s_%s" % (p["id"], dt.date.today().isoformat()))
    open(stem + ".md", "w", encoding="utf-8").write(report(p, z, proj, weeks, macro, tests))
    with open(stem + "_weeks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["week_of", "phase", "day", "kind", "session", "zone", "minutes"])
        for wk in weeks:
            for d, ss in wk["sessions"].items():
                for s in ss:
                    w.writerow([wk["monday"], wk["phase"], d, s["kind"], s["what"], s["zone"], s["minutes"]])
    json.dump(dict(zones=z, projection=proj, macro=macro, tests=tests), open(stem + ".json", "w"), indent=1, default=str)
    print("written %s.md / _weeks.csv / .json" % os.path.relpath(stem, ROOT))
    print("latest 5k %s (%.0f W), FTP %.0f W; target %s by %s: needs %.1f%%/yr; P(target) on the squad's record %.1f%%"
          % (mmss(z["test"]["time_s"]), z["watts_5k"], z["ftp"], mmss(p["target"]["time_s"]), p["target"]["by"],
             100 * proj["required_rate"], 100 * proj["p_target"]))
    for r in proj["path"]:
        print("  %s  median %s  (10th %s, 90th %s)" % (r["date"], mmss(r["p50"]), mmss(r["p10"]), mmss(r["p90"])))
    for r in proj["ladder"]:
        print("  sustained %-36s %+5.1f%%/yr -> %s" % (r["label"], 100 * r["rate"], mmss(r["time"])))
    for wk in weeks[:6]:
        print("  week %s %-22s %4.0f min, %3.0f%% low" % (wk["monday"], wk["phase"], wk["minutes"], 100 * wk["share_low"]))


if __name__ == "__main__":
    main()
