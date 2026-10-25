# Project status

*Last reviewed 2026-10-25 (every row checked against TRACKING's Done list and the other planning documents). Update this file whenever work finishes or a priority moves;
it is the one-page answer to "where are we, and what is next".*

Rowing shell simulator: full 6-DOF rigid-body dynamics, a released trainer, and an
offline physics programme, written against the ultimate goal of a **stochastic**
optimal control problem on the Charles — uncertain stream, crew, friction and
temperature — with the deterministic path as the prerequisite. A second line of
work studies coxing itself: what a coxswain's calls do to the boat.

**Where everything lives** — the full index, with what to update when, is
[PROJECT_MANAGEMENT.md](PROJECT_MANAGEMENT.md). In short:

| document | what it holds |
|---|---|
| this file | the state of each workstream and what is next |
| [TRACKING.md](TRACKING.md) | issues: every known defect (Open), everything finished (**Done** — check it before starting anything), fixed bugs |
| *Blade and Body* ([the plan](https://claude.ai/artifact/5ZM6yag3fYTViD3fqwhgv6)) | the physics review and staged plan: phases, gates, decisions |
| *Rowing Physics Ledger* ([the ledger](https://claude.ai/artifact/XBcnRHMy2Par4PAbw8kr6n)) | every force the research model applies, with its source and status |
| [PHYSICS_PROGRAMME.md](PHYSICS_PROGRAMME.md) | phase table, **Blocked** and on what, decisions log |
| [DATA_REQUESTS.md](DATA_REQUESTS.md) | letters to authors and labs, and their replies |
| [SOURCES.md](SOURCES.md) | the evidence, numbered by section |

Test suite: **2095 passing, 0 failing** in the fast lane (`pytest -m "not slow"`, run
2026-10-25, 9 min 10 s). The full suite, with the strict xfails that pin known model errors
(drive duration against on-water pairs, race pace without controlled power), was last run
2026-09-12: 1771 passing, 14 expected failures.

---

## 1. At a glance

| workstream | state | most recent | next |
|---|---|---|---|
| **Released trainer** | v0.13 (2026-09-11). Physics profile `shipped`, **frozen**: no accuracy change reaches it without a scorecard that justifies promotion | leg-mass placement fixed for research only, shipped left on `legacy` (2026-09-26) | nothing scheduled |
| **Physics programme** (`research` profile) | phase 2 gate passed (dynamic oar, slip blade, 6-DOF hull); phase 3 tier 2 lift/drag a study; phase 4.1 closed; **phase 4.3 in progress**: rung 1 (hands on the handle) and 2a (sourced on-water drive law) built; trunk rungs 1–5 on [CR06]'s woman rebuilt; knee/ankle support moment from a sourced foot-force direction ([BU15]) and centre of pressure ([VI17]) | **2026-10-25 (§179):** rung 5 rebuilt two-way: the hull's recoil lowers the trunk's effective inertia to ~6.3 kg·m², so the record's controlled cases (3.7° bias error, 117 ms divergence) are withdrawn; two studies broken by the fitted blade's step fixed. 2026-10-24 (§177–178): the hip two ways agree at maximum handle force, not at the catch | trunk controller designed on the coupled plant, then the predictor and load estimate re-run on it; the hip at the catch (from below ~0.3 vs from above ~3.3 N·m/kg); rung 2, the body relaxed against the handle |
| **Like-for-like validation** | two athletes ([BR24] elite M1x, [CR06] W1x) and [K05]'s population | pace passes (+0.2%); hull drag verified; with [K05]'s timing and leg travel the population's boat motion is reproduced (§175) | the athletes still swing 12–20% too much (IVV 55–59% vs 49%); the late-drive unloading |
| **Charles trajectory optimisation** | deterministic receding-horizon leg stalled near 409 m (unchanged since 2026-09-13); the race-day line work now carries it | 2026-10-22: HOCR arch rules corrected, bridge points no longer pinned, a Radau-collocation racing line (`line_ocp.py`); 2026-10-21: depth priced on the wave term only for research boats | the Cambridge-side corridor (decided 2026-10-21, not built); then the stochastic problem |
| **Race-day product** | HOCR 2026 raced 17 Oct (four 21st of 26, eight 8th of 31); post-race analysis done | 2026-10-19 to 10-23: race review, club line comparison (six crews), wind vs steering, skill vs externalities, adaptation Monte Carlo, coxswains as operators; lake pages for Whatcom and Sammamish (fog model 10-22) | decide whether race cards move to the research depth term (their depth seconds are overstated, flagged 10-21); retro actions in RACEDAY |
| **Coxing research** | review "What makes a call work?" and experimental paper "What the boat answers" (12 coded sessions incl. HOCR 2026); power calls +17 mm/s at strokes 3–5; single-call reliability 0.065 | 2026-10-24: OKID/ERA call-response model agrees with the event study and the transformer: with these data the calls' effect on speed within eight strokes is below what can be identified | second human coder; seat recorders on the Charles; Stage 3 randomised calls in practice |
| **Data requests** | 14 letters 2026-09-19; Vieira 2026-10-24 | replies: Formaggia, Kleshnev (data, [BR24]), Buckeridge (referred) → McGregor (**declined 2026-10-21**); direction then sourced from [BU15] | **Kleshnev follow-up: Gmail draft ready, needs his address** (his reply went to the university account) |

---

## 2. What works

**Physics.** 6-DOF rigid-body dynamics after Formaggia et al. (2009); hull
hydrostatics from an exact mesh with a b-spline surrogate; Michell/Sretenskii wave
drag with finite depth; wind with a log profile; de Leva segment inertias on a full
joint chain. The `research` profile adds a **dynamic oar** (oar angle as a state,
slip-quadratic blade force, [CR06]'s sweep catch, 1.2 kg sculls, fitted sculling
C2) that holds pace on the full hull, and a tier 2 lift/drag blade as a study.

**Validation against a real stroke** ([BR24], SOURCES §156–159): at his measured
432 W the research model rows **4.652 m/s against his 4.641**; its hull drag matches
the drag his own recovery implies (75.4 N against 73–79 N at 4.64 m/s).

**Steering (research).** Fins on Whicker & Fehlner's eq. [1] (reflection 2.0, a named
choice), Munk factor 0.50 refitted and unchanged: full helm 3.0 deg/s, radius 167 m at 25°
and 97 m at 45°, directionally stable, and the skeg-loss slew inside the reported band
(SOURCES §169).

**Crew.** Phase-dependent balance authority, learned stroke-to-stroke trim,
coupled-oscillator synchronisation, blades-on-water contact. Leg masses placed from
the proximal joint (research).

**Numerics.** Non-dimensionalised NLP (×25 in iterations); the reference-frame bug
class closed by 31 invariance tests.

**River.** Charles channel raster, bridges, clearance field, progress field.

**Coxing tooling.** Caption de-duplication with per-word timing; cox-box display
reading from video; blocked out-of-sample scoring with matched circular-shift nulls;
the plug-and-play coupled call/boat model (`research/callmodel/`).

---

## 3. Open problems, in priority order

### 3.1 The boat's speed fluctuation is too large  *(highest)*
Research model 61% IVV against [BR24]'s 49% at matched power and speed. Decomposed
(SOURCES §158–159): ~4 points are the crew's timing — the model's body is
**ergometer** data (Caplan & Gardner), and on the water the legs drive earlier and
the seat peaks lower ([K05], [BRM]); ~1 is the pull shape; <1 is stroke averaging;
**~6–7 are the catch**: the model's blade delivers 1–45 N in the first 0.2 s where
he delivers 55–111 N, under every blade law tried (slip, lift/drag at any amplitude,
with or without added mass). Hull drag is not the cause. **Located (§161):** not the entry rate but the oar balance — with the rower's reflected
inertia off the oar, his force reproduces his catch dip (3.44 vs 3.46 m/s), though the boat
runs 3.9% fast. **Re-located (§162, 2026-09-27):** evaluated on his own oar motion and boat
speed, the slip law gives 37 of his 105 N·s of drive impulse per oar, and no constant C2 at any
centre of pressure gives his catch or his finish. The model reaches his pace only by turning its
oar faster than he does; the oar balance of §161 is how it compensates. The blade law — a blade
that loads with little normal slip — is the fix, then the body split. *Same day:* driven by his
measured blade force, the model's hull and his body swing 50.8% against his 49.1%, so hull and
body are right and the gap is the blade's force time course. The fix is a blade built from
sourced physics only ([CG07] lift/drag, [LB19] added mass, strip integration, immersion) and
validated on both athletes — not coefficients fitted to one (the research sculling C2 is
already a [CR06]-only fit). His handle-force scale is open (question to Kleshnev drafted). *Built and validated (§163):* every blade made from sourced physics
(flume or full-size coefficients, Patton or [LB19] added mass, strip integration) settles within
±3% of the two athletes' mean speed with nothing fitted, and none moves IVV more than 1.5 points.
The IVV gap is the crew's force time course; phase 4.3 (#1) carries it. *29 Sep (§165):* each blade source is now checked against a number it prints —
[CR06]'s own model on her stroke gives 3.80 m/s against their 3.83 — and the map's contradictory
combinations are refused in code. *Rung 1 of phase 4.3 (§167):* with the hands held on the handle the model's IVV is
46–55% against 49%, and speed is within ±2% at equal power on both athletes, power predicted
rather than imposed. The remaining gap is the hands' time law (the handle force peaks too early),
which rung 2's torque-driven body is for. *30 Sep (§169a–170):* the ergometer body's legs and trunk
travel 10% too far and its arms bend at the catch; [K05]'s on-water population patterns, digitised,
give a drive law that reproduces [BR24]'s measured oar angle unfitted. With it, hands on the handle and
the tier 2 blade, IVV is 54% and 50% against 49% on both athletes. *§171:* the law smoothed (its linear
interpolation had put acceleration spikes on the oar); the force curve's width now agrees with the populations
(peak/mean 1.94 / 1.71 against 1.6–1.9) but it peaks early (0.11 / 0.28 s against 0.38–0.43). Blade added mass now
runs with the hands on the handle: IVV moves 2–4 points the right way, but held constant from entry it spikes the
handle at entry (550–1200 N). *§172:* that spike is the hands' blade acceleration, not the smoothing; Grift's
entrainment would not remove it, and his realistic stroke shows a small force after the catch — the blade's
immersion is what starts from zero. *2 Oct (§173):* built from BioRow's norms (wetting when the blade turns
driving, buried 4° later); it predicts BioRow's catch-to-buried and entry timing unimposed, and the population
force curves bound the entrained water at entry below ~0.1 of Patton. Best model: immersion, no added mass.
*§174:* a population test from [K05]'s own hands, body and force: with the population's body the slip law
reproduces the population's force shape and catch dip; tier 2 fails it; the drive law now has physical
turning points. Open: the blade unloads early late in the drive (rig gearing). *§175:* with the population's
leg travel too (the model's legs travel 17% too far) the population's catch dip, acceleration and velocity
swing are reproduced; the two athletes still swing 12–20% too much (IVV 55–59% vs 49%).

### 3.2 The crew is driven by ergometer kinematics
The cause of 3.1's crew share, and the motivation for phase 4. A sourced on-water
driver now exists: [K05]'s segment travels and timings and [BR24]'s seat and trunk
curves. Re-timing the model's body onto his curves recovers 4–5 points, but only about 2 of those transfer: driven by the [CR06] athlete's on-water timing instead, it recovers ~2 (SOURCES §160). What the two on-water scullers share — a drive of 0.53–0.54 of the cycle (model 0.47) and a later body swing on the recovery — is what the driver should encode; the drive's shape is individual. *Built and run (§162):* the shared drive length and recovery swing move IVV under a point, so the transferable part is the drive curves' shape; the driver stays a research option.

### 3.3 The finish
The slip-release fix validated on [CR06]'s athlete does not transfer to [BR24]: his
push after the finish is 1.4% of peak and his recovery handle force is positive, so
the blade-out oar never turns round. The turn-round belongs with phase 4.3's hands.
Promotion stays blocked.

### 3.4 Drive time is boat-dependent
Two independent on-water single-scull sources ([K05], [BR24]) give 1.00 s at 32 spm;
[HF09]'s pairs give 0.75 s. The five strict xfails against the pairs stay; moving
every boat to the pairs figure would take the single further from him.

### 3.5 `mean_handle_power` overstates handle power
It dots the oarlock force with the handle velocity; on the ideal lever that reads
~1.46× a measured handle power for [BR24]'s rig. It converts every power scale in
the shipped trainer, so any fix is a promotion question. *2026-09-27:* `definition="handle"`
now gives true handle power, (1 − r_h/L) of the oarlock figure; the default is unchanged, so
switching a consumer is the promotion decision.

### 3.6 Receding-horizon leg does not reach 850 m; Route C does not converge
Both unchanged since mid-September (SOURCES §27–33). Route C's slide travel is short
of measured, and none of its numbers are claimable.

### 3.7 The eight is validated only by inference
[BR24] is a single; Holt's data are singles and pairs. The boat the Head of the
Charles is raced in has no like-for-like target.

---

## 4. Waiting on other people

| what | from | blocks |
|---|---|---|
| footplate centre of pressure (heel to ball) through the drive; on-water confirmation of the women's foot-force direction | direction now sourced from [BU15] (ergometer, SOURCES §177); McGregor declined 2026-10-21; Sinclair, Baca, Smith (§11, sent 2026-09-19) for on-water; Draper (LinkedIn only, not yet contacted) | phase 4.3's knee/ankle split (their sum is settled) |
| origin/frame confirmation of the trunk channel; handle-force point; other stroke rates; mean blade burial | Kleshnev (§2) — follow-up approved 2026-10-25, a Gmail draft waiting on his address | confirms §156; a rate sweep would test IVV against rate |
| fore-rear centre of pressure through the drive (raw) | Vieira (§11e), sent 2026-10-24 | separates the four women; figures already digitised |
| tabulated immersion and entrainment data, tangential force traces | Grift (§9) — thesis figures read; exact numbers wanted | the immersion refit's precision; phase 3's gate |
| the others in DATA_REQUESTS | sent 2026-09-19, no reply yet | see each section |
| more synchronised cox audio + boat logs | crews and coaches with archives | the coxing results (power: 6–24 races) |

---

## 5. Wanted, but not a priority

* **Bridge approach clearance** — piers and the HOCR arch rules are modelled (2026-10-22); clearance on the
  approach between control points is not checked.
* **Stream field** — discharge data loaded, current uniform.
* **Crew fatigue over 4.8 km** — reserve state exists, depletion model does not.
* **Steering: the rudder response ratio and the fin's reflection factor** — the research eight
  now runs Whicker & Fehlner's fin law (SOURCES §169). Refitting the Munk factor returns the
  literature 0.50 at every reflection, and full helm gives 3.0 deg/s with a 97–99 m radius.
  But 25° buys 3.8–4.8× what 5° does, where a coxswain reports ~3×. Needs a measured
  rudder-step heading trace from a coxed eight.
* **Visualisation of a whole leg.**
* **Route C as a frozen rower model** — blocked on 3.6.
