# Supplementary training plans for club rowers

Two scripts. One measures how this squad's rowers actually improve. The other turns that, and the
training-science literature, into a plan around a rower's team sessions. Everything about an individual
(names, tests, body mass, the generated plans) stays in `data/local/training/`. Rowers appear here
only as anonymous ids.

## 1. The squad yardstick — `squad_progression.py`

```bash
.venv/Scripts/python.exe research/training/squad_progression.py [--xlsx data/local/erg/Erg_and_Workouts_<date>.xlsx]
```

The script reads the team erg sheet's "Erg Test POST HERE" tab: one row per rower, and one column per
testing season, with the season label above the 5k column. Each 5k becomes power through Concept2's
relation:

    P = 2.80 / (t / 5000)^3        (t the 5k time in s; t/5000 the pace in s/m)

From these it builds year-on-year pairs: tests 0.75–1.5 years apart, giving the relative power change per
year. Each pair is tagged with training age (years since the rower's first recorded test) and age.

Result (sheet as of October 2026: 71 rowers, 319 pairs, seasons since 2013), for women under 40:

| training age | pairs | median | 90th pct | best |
|---|---|---|---|---|
| 0–1.5 y | 26 | +7.7 %/y | +23 % | +31 % |
| 1.5–3.5 y | 21 | +3.9 %/y | +8.4 % | +12.0 % |
| 3.5 y on | 1 | — | — | — |

Two limitations. Training age counts from the first recorded test, which for adult-onset club rowers is
close to when they started. And the sheet has almost no young women past year 3: the year-4+ record
comes from masters rowers, who plateau near 0 %/y. For a rower under 30, that band is a poor guide.

## 2. The plan — `plan_generator.py`

```bash
.venv/Scripts/python.exe research/training/plan_generator.py --profile data/local/training/<athlete>.json --weeks 16
```

The input is a private JSON profile. It holds:

- the tests;
- sex, birth year and body mass;
- the team week (day, zone, total and work minutes);
- the free days;
- the starting and maximum minutes of the rower's own aerobic work;
- the key spring and fall race dates;
- the target.

The output, written to `data/local/training/plans/<id>_<date>.md`, `_weeks.csv` and `.json`, contains:

| section | how it is made |
|---|---|
| Zones | team convention: FTP = 5k watts × 0.935; zones as % of FTP (Z1–Z7, sweet spot) |
| Projection | Monte Carlo over the squad's yearly changes at the matching training age, plus fixed-rate paths (squad median, 90th percentile, best year) and the rate the target needs |
| Milestones | the 5k each team test should show on the 90th-percentile path (on track) and the median path (floor) |
| Macrocycle | phases from the club calendar: 3-week transition after the fall key race, base to February, build to April, spring racing, summer base, fall racing |
| Weeks | team sessions as given. Her own sessions: two Z2 sessions (one long), two strength sessions (one while racing). Own aerobic minutes grow by `load_week_growth` per loading week to the cap. Every 4th week is a deload (×0.6). A 13-day taper (×0.5) before each key race. Transition ×0.5. Weekly low-intensity share reported. |
| Strength | heavy-load prescription by phase and the exercise list |
| Rules | HRV/readiness, milestone checks, fuelling (protein 1.6 g/kg; carbohydrate by day, approximate), REDs warning signs, rib pain, cycle tracking |

Each rule carries an evidence tag (`[TID]`, `[STRENGTH]`, …), defined in `EVIDENCE` at the top of the
script.

## 3. The website — `site_build.py` and `site/index.html`

```bash
ATHLETE_NAME=... SITE_PASSPHRASE=... ERG_SHEET_ID=... python research/training/site_build.py --config config.json --out _site
```

A GitHub Action in a separate public repository runs this every day. It:

1. downloads the link-viewable team erg sheet as xlsx;
2. recomputes the squad yardstick;
3. reads the athlete's row: tests, sex, birth year, body mass;
4. merges that with a non-identifying `config.json` (team week, free days, volume settings, key races, start,
   target);
5. regenerates the whole plan, from its fixed start date to the target date;
6. publishes the result to GitHub Pages.

So a new test on the sheet moves the zones, milestones and projection the next morning, without
anyone touching the code.

Privacy works like the race-card site (`scripts/raceday_encrypt.py`):

- **What is public:** the page shell, which holds no data.
- **What is encrypted:** everything about the athlete, in `data.enc` (AES-256-GCM, with the key derived by
  PBKDF2-SHA256 over 250,000 iterations from a passphrase). The browser decrypts it.
- **What stays secret:** the sheet id, the athlete's row label and the passphrase are repository secrets.
- **What never enters the bundle:** names and body mass. The fuelling rules are given per kg.

## 4. The ceiling — `ceiling.py`

This script maps a 5k time to a VO2max and back, through four links:

1. Concept2 power;
2. Paul's law (a coaching rule of thumb, 5 s/500 m per doubling of distance);
3. maximal power taken as 2k power;
4. the female-rower equation VO2max (mL/min) = 9.58 × MPO + 958 (r = 0.94, SEE 4.6 %; Mazza et al. 2023, IJSPP).

It then compares the result with Polish national-squad reference values (Klusiewicz et al. 2014). For a
22:17.7 5k it gives about 2.6 L/min. 18:00 needs about 4.2 L/min (+61 %), near a 2010 World
Championship bronze double (4.45 L/min). The full-time-athlete estimate in the textbook (centred on
19:00, range 18:30–19:30) uses this chain. The absolute numbers are rough (±10 %).

## Evidence base (review of October 2026)

| topic | finding | source |
|---|---|---|
| Intensity distribution | ~75–80 % of sessions below the first threshold. Elite rowers train 14–20 h/wk, mostly pyramidal, and no distribution is proven best. Polarised was not better than pyramidal with the easy share fixed at 93 %. Polarised gives a small VO2peak advantage. | Seiler 2010 IJSPP; Zhong et al. 2025 IJSPP (systematic review); Treff et al. 2017 Front Physiol; Oliveira et al. 2024 Sports Med |
| Volume | Norwegian medallists 1970s→1990s: +20 % hours, low-lactate work 30→50 h/month, race-pace work 23→7 h/month | Fiskerstrand & Seiler 2004 SJMSS |
| Strength | In rowers: leg strength SMD 0.42, rowing performance SMD 0.32. Heavy loads beat strength-endurance in young elite female rowers. Heavy strength improves endurance economy. Women gain at the same relative rate as men. | Thiele et al. 2020 J Sports Sci; Thiele et al. 2020 Front Physiol; Rønnestad & Mujika 2014 SJMSS; Roberts et al. 2020 JSCR |
| Concurrent: outcomes | Concurrent training leaves maximal strength (SMD −0.06) and muscle size (−0.01) intact but blunts explosive strength (−0.28), mostly in the same session; not significant with ≥ 3 h between. Interference shows in trained lifters within ~20 min. 0 h vs 6 or 24 h: strength gains lower; VO2peak best with 24 h. Lifting first: +4.0 kg 1RM. | Schumann et al. 2022 Sports Med (43 trials); Petré et al. 2021 Sports Med; Robineau et al. 2016 JSCR; Murlasits et al. 2018 J Sports Sci |
| Concurrent: mechanism | AMPK–mTOR "switch" from rodent stimulation; in humans prior endurance work does not blunt mTORC1 signalling or protein synthesis after lifting, and combined sessions raise PGC-1α; no single pathway explains interference, which may grow with training history. Lifting induces PGC-1α4 (IGF-1 up, myostatin down). | Coffey & Hawley 2017 J Physiol; Apró et al. 2013; Wang et al. 2011 J Appl Physiol; Ruas et al. 2012 Cell |
| Periodisation | Block periodisation: small, low-quality advantage. A load-matched trial found no difference. | Mølmen et al. 2019 OAJSM; Almquist et al. 2022 Front Physiol |
| Progression | ACWR and 10 % rules have no shown causal link to injury | Impellizzeri et al. 2020 IJSPP |
| Intervals | 8 × 2.5 min at 90 % vVO2peak improved 2k power in trained rowers (n = 10) | Driller et al. 2009 IJSPP |
| Taper | ~2 weeks, volume −41–60 %, intensity and frequency kept | Bosquet et al. 2007 MSSE |
| HRV | HRV-guided training gives a small extra VO2max gain, larger in amateurs and women | Granero-Gallegos et al. 2020 IJERPH |
| Menstrual cycle | Trivial phase effects, low-quality evidence; individualise | McNulty et al. 2020 Sports Med |
| Energy availability | Low energy availability (REDs) harms health and performance | Mountjoy et al. 2023 BJSM (IOC consensus) |
| Protein | Gains plateau at ~1.6 g/kg/day | Morton et al. 2018 BJSM |
| Carbohydrate | ~3–5 g/kg light days, ~5–7 g/kg moderate (approximate; not checked against the primary table) | Burke et al. 2011 J Sports Sci |
| Sleep | Individual need rather than a fixed 7–9 h target | Walsh et al. 2021 BJSM |
| Rib stress | Risk factors unproven; 4–6 weeks lost | D'Ailly et al. 2017 (systematic review) |
| Peak age | Women in the 2008 Olympic rowing A/B finals averaged 27.5 years | Lithuanian study of the Beijing finals |

Most trials run 4–12 weeks, in elite or highly trained athletes, mostly men. None follows an adult-onset
club rower for years. That is why the targets come from the squad's own record and are re-set after
every test.

Tests: `tests/unit/test_training_plan.py`. It checks that key-race weeks are tapered (including a
race that falls in the next phase), that the placed minutes add up to the plan, the low-intensity
share, that no heavy lift falls the day before intervals, the deload cut, and the zone convention.
