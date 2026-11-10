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
| **Physics programme** (`research` profile) | phase 2 gate passed (dynamic oar, slip blade, 6-DOF hull); phase 3 tier 2 lift/drag a study; phase 4.1 closed; **phase 4.3 in progress**: rung 1 (hands on the handle), 2a (sourced on-water drive law) and **rung 2 (the hands' time law as an output)** built; trunk rungs 1–5 on [CR06]'s woman rebuilt and reproduced; knee/ankle support moment from a sourced foot-force direction ([BU15]) and centre of pressure ([VI17]) | **2026-11-09 (§228): a validation ladder, 21 of 23 passing.** A ladder for the flexible-oar rower (docs/VALIDATION_LADDER.md), bands fixed before the run: shaft modes, numerics, a full-size tank trial, the single sculler's handle force, both athletes' speed, power, oar timing and force shape, and the scorecard all pass. Boat-speed fluctuation fails for both athletes (+4.4 / +4.1 points against ±3): the flexible oar loads too slowly in the first ~0.15 s after the catch, the open "catch build-up" item. On the way, the tank trial showed the blade carries about a tenth of the textbook water (as the population data had said), and two bugs were fixed: the athletes' own body still carried the population's trunk timing, and a bent blade's wetting was read on the wrong angle. **2026-11-08 (§227): the rigid oar's match was an artefact; the flexible shaft fixed.** Against the stronger athlete's measured handle force, the rigid oar carrying the blade's water spikes at the catch and pushes back mid-drive — the water locked to the hands acts like ~67 kg at the handle — and that is where its low speed fluctuation and low power came from. The flexible shaft had two bugs: the blade's release read the handle side of the shaft rather than the bent blade (a driving blade was pulled out early), and the blade force pointed along the straight oar. Fixed in one place; his force curve now has his shape (centre at 0.49 of the drive, as measured) and his power to 3 %, and the fluctuation gap is 3.4 / 3.7 points (was 5.9 / 4.2). Split hand-over and second-half release are now rung 2's defaults. **2026-11-08 (§226): the stall fixed; the gap measured.** The stronger athlete's stall was the blade being declared out right after the catch whenever the stroke slowed; the release now acts only in the second half of the arc, and his stroke completes and converges. The best match to both athletes is now the rigid oar with the blade's water and the split muscle hand-over (speeds within 1 %, fluctuation within 1–2 points); the flexible shaft is 4–6 points high, an open question about how much of the blade's water a flexible oar passes to the boat. **2026-11-08 (§225): the knee fixed; the stall narrowed.** The foot force's vertical now comes from the body's own balance and the measured seat load, which brings the single sculler's knee within 13–22 % of the published model and reconciles two conflicting sources. The stronger athlete's stall is not the dynamics (his own stroke round-trips exactly) but the borrowed population force at the catch; fixing it needs muscles that respond to the load. **2026-11-08 (§224): vibration analysis; a dynamic shaft.** The coupling between the shaft and the rest of the model is cleared (it contracts at every instant), and the swinging mode is damped at every frozen instant, so the culprit is the massless hinge: in the water the blade carries 13 kg of water, making the shaft's first bending mode about 1 Hz and overdamped. A measured study of 71 sculls confirms the shaft stiffness to 0.3 % and gives 3.4 % damping. The new dynamic shaft (blade and water inertia, solved with the hull) converges with the time step and puts the single sculler at 4.20 m/s; the stronger athlete stalls. Bending stays off. **2026-11-08 (§223): the stronger athlete's bending stroke is unstable.** Its handle force swings every ~0.25 s and the swing grows; halving the time step does not remove it, so rung 2 with a bending shaft is unstable at that force (and marginal for the single sculler). Bending stays off for rung 2. The body's hip moments match Hase et al. 2002 within 7–14 % per unit handle force; the knee has the right shape but too much early extension. **2026-11-07 (§222): the on-water force–velocity measured.** Nordez et al. 2025 (from the authors): on-water V0 4.53 m/s, a hold of about 0.6–0.7. With it and no free parameter, bending passes the sweep boats and [CR06]; [BR24] still rings, so bending stays an option. Hase et al. 2002 gives joint moments that pass the statics check, the next target for the body. **2026-11-06 (§221): bending passes with measured shafts.** A measured sweep shaft (Prinsen & Wilkins 1986, EI 8000 N·m²) puts the sweep boats in Kleshnev's bands with bending on (hold 4; the four also at hold 1). The muscles' own force–velocity supplies a hold of about 0.6–0.7 (reflexes perhaps 0.3 more); the athletes need 1–2. Enabling bending is now the user's decision. **2026-11-06 (§220): a crew that holds its own pace, and stiffer sweep shafts.** With the crew holding its own rhythm, the scull bending runs come within 0–0.6 points of the rigid oar's IVV for both athletes; the rigid runs are unchanged. Sweep shafts are ~2.6× stiffer than sculls (Concept2's spec); the bench law was too soft. The sweep boats improve but stay ~0.05 below their blade-efficiency bands. Bending stays off. **2026-11-06 (§219): bending passes wherever the crew's motion is set.** With her own body timing and her oar motion imposed, the measured scull shaft brings [CR06]'s handle force from 20.2 % to 12.5 % error and matches her late drive and her speed; [BR24]'s imposed-motion run improves too. Where the model solves the crew's motion (rung 2), she is level with the rigid oar but his stroke clock rings: the crew needs a sourced impedance about its planned motion. The open problem is the crew, not the shaft. **2026-11-05 (§218): why bending broke the boat.** The bending shaft rang on the crew's reflected inertia because the crew's pull had no force–velocity; a sourced handle force–velocity (an option) damps it without moving the rigid results. With her oar motion imposed, bending brings [CR06]'s handle force closer to the measured curve (the first in-boat pass) but holds too much load late in the drive, so IVV and the sweep boats' blade efficiency still fail: bending stays off while the late drive is worked on. **2026-11-03 (§217): oar bending made a research default and withdrawn the same day.** The scull's stiffness had been misread (~9× too stiff); with the measured shaft rung 2 moves away from both athletes (IVV 53.4 / 60.7 %) and the sweep law's blades lag 20–38° into the finish, so the research profile rows rigid again and bending waits for a dynamic shaft model. The scorecard's blade efficiency is now Kleshnev's energy measure on each boat type's band: the eight 0.805, the four 0.777, both pass. **2026-11-03 (§216):** the muscle check now runs on rung 2's power-matched stroke: the elbow is feasible, the hip and knee fail only at the catch angles the strength data do not reach; the elbow's EMG timing needs co-contraction (both elbow groups and the shoulder). Oar bending in rung 2: small, mixed between the athletes (a default is the user's call). **2026-11-02 (§215):** the trunk's predictor over its 100 ms delay cuts its wander 22–40 % (to 13° / 10°); holding the seat still after the finish (as every source has it) worsens IVV; a bug in the muscle check's arm fixed; the straight-arm body's muscle check waits for a power-matched stroke. **2026-11-02 (§214): no open on-water seat trace exists**; the published leg landmarks (McBride & Elliott; Kleshnev 2007; BioRow 2026) cannot separate the candidate leg timings, since the two athletes span them more widely. BioRow's curves are commercial (DATA_REQUESTS §14). **2026-11-02 (§213): the trunk clock is the research default** (scorecard: every target passes; the eight 6.00 m/s, swing 37.1 %). A population leg clock ([K05], timed through the handle speed) fits the athletes' seats better than the keyframes but moves their IVV +1.7 / −0.7 points: an option, not a default. **2026-11-02 (§212): population travel tested.** It improves the hand (1.9° / 4.3°; finish elbow 101° / 94°) but under-predicts rung 2's IVV by 3.5–4.3 points: the athletes' legs travel 8–16 % more than the population's. With her own travel [CR06] lands within 1.1 points; [BR24] also needs his leg timing. Off by default. **2026-11-02 (§211): the trunk on the population's on-water clock; a modular body.** Pitto 2024's upper-trunk timing fits both athletes better than the ergometer keyframes; with it, rung 2 (sourced throughout) gives IVV 50.8 % / 51.5 % against 53.4 / 52.1 before (measured 49.1 / 49.4), and the body-generated hand with the stretch is within 2.1° / 3.3° of the measured oar. New pieces plug in through `BodyOptions` (segment clocks, reach allowance) and `HandChain` (named reach terms). **2026-11-02 (§210): the "1.12 arm lengths" is a stretch.** The scapula retracts through the drive (Urbanczyk 2021), so not girdle posture; rebuilt from their own channels, both athletes' upper-back-to-hand chains stretch in proportion to the handle force, ~11–14 cm at each one's peak force (spring 0.7° rms; each athlete's stretch predicts the other's whole drive to 2.5° / 1.5°). An option, off (athlete-measured); a source or a third athlete is needed. **2026-11-01 (§209): the arms on their own clock.** Pitto et al. 2024's population elbow (12 scullers on water) gives a held 10.4° and a minimum-jerk draw from 46.5 % of the drive (1.3° rms); the hand is generated as shoulder + arm around the whole cycle. The hand-over spike is gone (peak 1174 / 1039 °/s² against the law's 1220 / 1164; was 6300). On the athletes' own timing the finish elbow lands in the population band and the late drive matches to 2°; what remains is the girdle (~10° mid-drive) and, on the keyframed body, the legs' and trunk's sequencing. **2026-11-01 (§208): body build begun.** Sourced joint muscles (Anderson 2007, Frey-Law 2012) and the trunk's intrinsic and reflex layers (Moorhouse & Granata 2007) built. The catch posture is fixed by coupling the hands to the shoulders while the arms are straight. That exposed that the keyframed body moves the shoulders too early, and that the measured grips reach 1.12 arm lengths mid-drive (a shoulder girdle the model lacks). The trunk's lower layers halve §206's wander (to 13–17°) but cannot hold it alone. Strength data stop short of the catch's angles. **2026-11-01 (§207): neuromuscular control literature review** (docs/NEUROMUSCULAR_CONTROL_REVIEW.md): the field models self-control as layers. At the bottom is the muscle's own zero-delay impedance; next, spinal reflexes at 20–45 ms (trunk 40 ms); on top, an efference-copy estimator corrected by delayed sensing. §206's τ'' collapse is a neutral delay loop; the trunk wander lacks the zero-delay layer and the 40 ms reflex; sequencing should emerge from segment dynamics. A gated build order is proposed. **2026-10-31 (§201–202): rung 2 passes S2.5's gate on both athletes with the sourced cross-flow normal blade** (IVV +2.1 / +3.3 points, speed −1.2 / +2.6 %, nothing fitted); the tangential load was a construction artefact ([CG07] measured points, [H10]). **Promoted to the research profile** (scorecard: every target passes on both boats; blade efficiency 0.793 / 0.765 in Kleshnev's band). **Rower (§203–204):** rung 2's catch turns dynamic at burial (entry spike gone); the trunk in the loop needs an internal model, and with one reproduces rung 2, so the remaining IVV gap is not trunk control. **(§205–206):** with each athlete's own leg and back timing, rung 2 reproduces their IVV within ~1 point: the gap is body sequencing; next a multi-clock body. **(§199–200):** [H10]'s on-water parallel force traced: ~⅒ of the quasi-steady law mid-drive and out of phase; at its level the review blade is within 3 % of both athletes' speed at their power, nothing fitted. **(§198, corrects §196):** the simulated blade path matches the athletes' measured scull paths; the review blade's tangential load is right on the sweep path and ~2× [H10]'s sculler figure on scull paths (sliding at 40–55°), which is the closed-loop deficit. Next: [H10]'s parallel force by phase. **2026-10-27, blade literature review** (docs/BLADE_HYDRODYNAMICS_REVIEW.md, §183–189): two published measured blade loads on known motions ([GTW21] half-scale, [RLBQV19] full-size with a bending oar) extracted and every project law tested: the research profile's slip law has the wrong shape; tier 2 over-predicts the catch and has the tangential sign wrong; the project's own fin law (Whicker & Fehlner's cross-flow form with the fins' Polhamus/Helmbold slope, §190; prior-work check: the slope and form already existed for the fins) fits both, its face-on coefficient (~2 half-scale, ~1.3 full scale) being the free-surface regime; a possible validation paper (review §4a); oar bending (their bench law as a quasi-static hinge) improves every law; [BR24]'s late-drive gap is his recorded motion, not missing blade physics. **2026-10-26:** rung 5 reproduced with gains designed on the coupled plant (§182; the 2026-10-25 withdrawal reversed): the hull's recoil lowers the trunk's inertia to 6.4 kg·m², so a correctly designed loop's delay limit is 114 ms, inside human trunk latency, and needs an internal model. Rung 2 (§181): the population's muscles give [CR06] IVV 51.7% vs 49.4% (hands 2× closer), [BR24] 54.7% vs 49.1%. Young's recovery profiles (§180): the fast pivot costs up to 1.9% | **the multi-clock body in the order of review §12** (muscle groups → intrinsic layer → reflexes → modules on a neural phase oscillator → delay-aware estimator), awaiting go-ahead; the review's blade and oar bending into the simulator (research options), then rung 2 on them; the free-surface cap; the catch's build-up ([vV22], [G19]); [CR06]'s blade load as the next athlete test; the catch's turning point in rung 2; the hip at the catch; sprint 3 (the stochastic Charles line) |
| **Like-for-like validation** | two athletes ([BR24] elite M1x, [CR06] W1x) and [K05]'s population | pace passes (+0.2%); hull drag verified; with [K05]'s timing and leg travel the population's boat motion is reproduced (§175) | the athletes still swing 12–20% too much (IVV 55–59% vs 49%); the late-drive unloading |
| **Charles trajectory optimisation** | deterministic receding-horizon leg stalled near 409 m (unchanged since 2026-09-13); the race-day line work now carries it | 2026-10-22: HOCR arch rules corrected, bridge points no longer pinned, a Radau-collocation racing line (`line_ocp.py`); 2026-10-21: depth priced on the wave term only for research boats | the Cambridge-side corridor (decided 2026-10-21, not built); then the stochastic problem |
| **Race-day product** | HOCR 2026 raced 17 Oct (four 21st of 26, eight 8th of 31); post-race analysis done | 2026-10-19 to 10-23: race review, club line comparison (six crews), wind vs steering, skill vs externalities, adaptation Monte Carlo, coxswains as operators; lake pages for Whatcom and Sammamish (fog model 10-22) | depth: race cards, reviews and site on the accurate (research) depth term since 10-22, pacing and line scripts since 10-25; retro actions in RACEDAY |
| **Coxing research** | review "What makes a call work?" and experimental paper "What the boat answers" (13 coded sessions, two coxswains, incl. HOCR 2026); power calls raise rate (+0.11 spm, p 0.005), speed at strokes 3–5 +11 mm/s; single-call reliability ~0.02 | **2026-10-31 (§197):** a push's build-up is the boat's inertia (simulated τ ~7 s; strokes 3–5 see ~60% of a sustained gain), calling policy differs by coxswain, crews answer rate calls holding power. 2026-10-30: second coxswain (race_07) added: rate and channel results held, speed results weakened. 2026-10-24: OKID/ERA call-response model agrees with the event study and the transformer: with these data the calls' effect on speed within eight strokes is below what can be identified | second human coder; seat recorders on the Charles; Stage 3 randomised calls in practice |
| **Data requests** | 14 letters 2026-09-19; Vieira 2026-10-24 | replies: Formaggia, Kleshnev (data, [BR24]), Buckeridge (referred) → McGregor (**declined 2026-10-21**); direction then sourced from [BU15] | Kleshnev follow-up sent 2026-10-25 (BioRow's published address); awaiting reply |

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
| origin/frame confirmation of the trunk channel; handle-force point; other stroke rates; mean blade burial | Kleshnev (§2) — follow-up sent 2026-10-25 | confirms §156; a rate sweep would test IVV against rate |
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
