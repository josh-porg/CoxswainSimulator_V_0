# Project status

*Last reviewed 2026-09-27. Update this file whenever work finishes or a priority moves;
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

Test suite: **1875 passing, 0 failing** in the fast lane (`pytest -m "not slow"`, 7 min 47 s,
run 2026-09-27). The full suite, with the strict xfails that pin known model errors
(drive duration against on-water pairs, race pace without controlled power), was last run
2026-09-12: 1771 passing, 14 expected failures.

---

## 1. At a glance

| workstream | state | most recent | next |
|---|---|---|---|
| **Released trainer** | v0.13 (2026-09-11). Physics profile `shipped`, **frozen**: no accuracy change reaches it without a scorecard that justifies promotion | leg-mass placement fixed for research only, shipped left on `legacy` (2026-09-26) | nothing scheduled |
| **Physics programme** (`research` profile) | phase 2 gate passed (dynamic oar, slip blade, 6-DOF hull); phase 3 tier 2 lift/drag wired as a study; phase 4.1 closed, **4.3 next** | [BR24] like-for-like runs (SOURCES §156–159) | sprint 1: unsteady blade load against his measured blade (#5), then 4.3's hands on the handle (#1) |
| **Like-for-like validation** | first athlete where rig, rate, power and boat response are one person's ([BR24], elite M1x); [CR06] traces rebuilt into `data/literature` | pace passes (+0.2%); IVV 61% vs 49% decomposed; hull drag verified; on-water timing transfers ~2 points between athletes | the blade at his kinematics: a slip law gives 35% of his drive impulse (§162) |
| **Charles trajectory optimisation** | deterministic receding-horizon leg stalled near 409 m at the station-450 pinch; stochastic machinery solves per block | not revisited since 2026-09-13 (research wave drag wired into the optimisers) | resume after the physics settles |
| **Coxing research** | foundations paper frozen 2026-09-23 for IJSSC; working copy revised with a coupled-process section; call/boat transformer pipeline built and validated on synthetic data | first run: no coupling either way on 3 races + 35 transcripts; the catch-call effect is explained by the boat's own history | more synchronised races (the pipeline takes them as folders) |
| **Data requests** | 14 letters sent 2026-09-19 | replies: Formaggia (§4), Kleshnev (§2, data received), Buckeridge (§11c, referred to McGregor — draft ready); Grift (§9) partly answered by his open thesis | send the Kleshnev and McGregor drafts |

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
that loads with little normal slip, i.e. an unsteady load — is the fix, then the body split.

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
| foot-force direction (women) | McGregor, via Buckeridge (§11c) — draft ready to send | phase 4.3 knee and ankle |
| origin/frame confirmation of the trunk channel; other stroke rates | Kleshnev (§2) — draft ready to send | confirms §156; a rate sweep would test IVV against rate |
| tabulated immersion and entrainment data, tangential force traces | Grift (§9) — thesis figures read; exact numbers wanted | the immersion refit's precision; phase 3's gate |
| the others in DATA_REQUESTS | sent 2026-09-19, no reply yet | see each section |
| more synchronised cox audio + boat logs | crews and coaches with archives | the coxing results (power: 6–24 races) |

---

## 5. Wanted, but not a priority

* **Bridge piers as constraints** — bridges are landmarks only; clearance under the
  arches is not yet a constraint.
* **Stream field** — discharge data loaded, current uniform.
* **Crew fatigue over 4.8 km** — reserve state exists, depletion model does not.
* **Steering study conclusions** — machinery, no written answer.
* **Visualisation of a whole leg.**
* **Route C as a frozen rower model** — blocked on 3.6.
