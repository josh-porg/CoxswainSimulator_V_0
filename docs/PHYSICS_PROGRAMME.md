# The offline physics programme

Fixing two defects in the physics **offline**, while the released trainer
stays exactly as it is. This file is the project record: what phase we are
in, what the gate for leaving it is, and what is outstanding.

- **Bugs and open modelling questions** live in [TRACKING.md](TRACKING.md).
- **Where every empirical number comes from** lives in [SOURCES.md](SOURCES.md).
- **This file** is the plan and its state. One line changes per completed
  task; nothing here is a duplicate of the other two.

Started 2026-09-12.

---

## Why

Two defects, both known, both recorded:

1. **The blade carries no velocity term.** `oar_force(t, timing, side)` takes
   stroke phase and side, so a crew makes the same force at 2 m/s as at 6.
   The steady balance then forces `η ∝ v` — a line through the origin, which
   is measured and now scored.
2. **The crew kinematics come from a stationary ergometer**, and the
   stochastic augmentation on top of them was chosen rather than measured.
   Ritchie (2008) measured that an ergometer cannot reproduce the recovery,
   which is where the hull is fastest and its drag highest.

Neither fix may reach the shipped trainer until it has been justified
against the validation scorecard.

## How the split is held

`coxswain/physics.py` resolves physics **by name**. Nothing constructs it
implicitly.

| profile | blade | rower | who runs it |
|---|---|---|---|
| `shipped` | tier 0, prescribed | prescribed, erg-fitted | the game, the online report |
| `research` | tier 1, dynamic oar → 2 | prescribed now; forward-dynamic, torque-driven next | the validation scorecard, and the offline report (the four only — see Blocked) |
| `learned` | tier 2 | a trained policy | training and study runs |

**`research` is PARTIAL, and says so.** Until 2026-09-12 it resolved to the
efficiency-only wiring, which collapses the boat, and its summary said
UNSTABLE. It now makes the oar angle a dynamic state on the full 6-DOF hull,
which held phase 2's gate. What is still unfinished is named in its own
summary: the crew is prescribed from ergometer data and does not follow the
dynamic oar, only synchronised crews run, and only the validation scorecard
is ported. **The prescribed oar block refuses a boat carrying this stamp**,
so nothing -- the report included -- can silently simulate the shipped oar
under the research label.

`shipped` is **frozen, not current**. The freeze is held by a source scan
over everything the trainer can execute, not by convention — a behaviour
test cannot cover a path no test exercises, which is exactly how v0.12
shipped its crash.

## The blade tiers

| tier | model | state |
|---|---|---|
| 0 | prescribed force profile, a function of stroke phase | **shipped** |
| 1 | [CR06] Model 1 — normal load from blade slip, oar angle a dynamic state | **on the full hull, in `research`** — crew still prescribed |
| 2 | [CR06] Model 2 family — lift and drag resolved against angle of attack | **wired into the dynamic oar** (`blade_law="liftdrag"`) and measured on the full hull, on provisional coefficients [CG06a]; not yet a profile |
| 3 | (rower, not blade) transformer policy outputting joint torques | planned |

---

## Phases

| # | phase | gate | status |
|---|---|---|---|
| 0 | Scaffolding: profiles, validation battery, freeze guards | scorecard reproduces every number the report already claims | **done** |
| 1 | ~~Tier 1 blade as an efficiency factor~~ | **failed its gate — merged into phase 2** | **closed** |
| 2 | Tier 1 blade: slip-quadratic **force**, oar angle a dynamic state | η against v — is the line through the origin gone, and does net propulsive impulse survive at race pace? | **gate passed on the full 6-DOF hull; `research` repointed, PARTIAL** |
| 3 | Tier 2 blade: lift and drag on angle of attack | reproduces the sign and timing of Grift's measured tangential force | **wired and measured; the gate needs Grift's traces** |
| 4 | Forward-dynamic rower (Rongère's recursion, driven by joint torques — the torque drive is ours, not theirs) | predicted CoM excursion lands in the measured band **without being fitted to it** | **4.1 closed on a finding; 4.3 next, segment inertias in** |
| 5 | Tier 3 infrastructure: vectorised env, delay channels, BC dataset | throughput, measured, against the 10⁶–10⁷ steps training needs | planned |
| 6 | Tier 3 training: BC then constrained PPO | the five acceptance tests, none of them trained on | planned |
| 7 | Uncertainty, and the coupled-oscillator replication | — | planned |
| — | Promotion to `shipped` | a scorecard that justifies it | deferred, explicitly |

**A phase can fail its gate, and one already has.** Phase 1 was scheduled
as "switch the blade model on" and phase 2 as "then make the oar angle a
state". Phase 1 turns out to have no viable operating point at all — the
boat collapses from 3.92 m/s to 0.63 — because the efficiency factor is
the *destabilising* half of the blade physics and the restoring half lives
in the force model. The two are not separable. Recorded rather than
quietly re-planned; see [TRACKING.md](TRACKING.md).

### Phase 0 — done

- [x] Profile registry; `shipped` frozen, `research` and `learned` declared
- [x] `resolve()` and `resolve(None)` both give `shipped`
- [x] Blade coefficient follows the rig (84.5 sweep / 58.7 sculling); outboard from the boat, not [CR06]'s 2.28 m
- [x] Trainer physics pinned in one place (`viz/menu.build_boat`)
- [x] Source scan failing if the trainer names a non-frozen profile, with a planted-offence test so the guard can fail
- [x] Validation battery: targets carry provenance; `pending` carries its reason; above-tier targets are `n/a`, never `pass`
- [x] Baseline recorded for `shipped` on the eight and the four
- [x] One definition of the efficiency measurement, shared by the scorecard and the test that found the defect
- [x] `make_report.py --physics`, defaulting to `shipped`, with the profile **stamped on the page** so a figure cannot silently change meaning between runs
- [x] The `research` profile advertises that it is currently unstable, so nobody resolves it and quotes a speed

**Gate met. Getting there turned up four defects**, all now in
[TRACKING.md](TRACKING.md), none of which any test was catching:

1. **Handle power moved +53%** at `a053540` and nothing noticed. It is the
   unit that converts watts into `power_scales`, so the boat is driven 35%
   softer for the same crew watts, and **every speed the report published
   is stale**. The scorecard found it on its first run.
2. **The suite was not green.** 26 failures at HEAD, reported as none. 14
   CasADi tests had been unable to run since 26 August; 12 more in
   `test_paper_validation.py`.
3. **The optimiser and the simulator disagreed about the rudder.** Behind
   those 14: `hydro_casadi` used the small-angle limit of the simulator's
   lift curve, dropping both the stall term and the cross-flow term — up to
   6.5% low, and unbounded past stall, which an optimiser will exploit.
   Fixed; they now agree to 0.0.
4. **The drive is 18–28% too long**, because `drive_fraction` was refitted
   to ergometer data and no longer matches on-water pairs. Defect two of
   this programme, in the most measurable quantity in the stroke.

That is the argument for phase 0 having been worth a phase, and it is
recorded here because the argument was made before the evidence existed.

### Phase 1 — closed, gate failed

- [x] Construct the blade model on the `research` profile and measure what moves
- [x] Establish whether the efficiency-only wiring has an operating point — **it does not**
- [x] Rule out the sweep shape as the cause (flatness 0.00 / 0.30 / 0.60 all collapse)
- [x] Identify the mechanism and pin it (`tests/test_blade_tier1.py`)

The boat collapses to 0.63 m/s from 3.92. Efficiency is evaluated at the
instantaneous surge, whose minimum coincides with peak oar force; thrust is
cut to 63%, the boat slows, the dip deepens. Positive feedback with no
restoring term, because the restoring term is in the force model.

**It also corrected the record.** `SOURCES.md` §7 says switching the blade
model on costs "5.10 to 4.28 m/s". That was measured at 70 s, and 70 s is
not convergence — from 3.4 m/s it reads 1.88 at 70 s and 0.63 at 250 s.

### Phase 2 — gate passed on the full hull; report ported (the four), tier 2 built alongside

Tier 1 properly: the slip-quadratic **force** from [CR06] Model 1, with the
oar angle as a dynamic state per seat. The rower drives the handle, the
blade resists, the angle follows.

- [x] Torque balance about the pin, as a standalone unit (`coxswain/crew/oardynamics.py`), tested in isolation before anything is wired in
- [x] The comparison that justifies the change, measured rather than asserted
- [x] Inertia **derived** from de Leva masses and the joint chain, not fitted — and it varies sevenfold across the drive, so the balance carries the `½(dI/dφ)φ̇²` term
- [x] Drive fraction predicted, unfitted — and on the oar alone, at a fixed 4.85 m/s under a constant pull, it lands on the on-water measurement. **On the full model it does not**: no single power gives both on-water drive time and published pace (below)
- [x] Couple it to hull surge on a reduced model, and answer the gate (`coxswain/sim/oarloop.py`)
- [x] A seam in the shipped simulator (`RowingSimulator._oar_loads`), cut as a verbatim move and held **bit-identical** by the golden trajectory
- [x] `DynamicOarSimulator` on the full 6-DOF hull: one oar angle per seat, blade load applied at the blade, slip against the water-relative **oarlock** velocity so a turning boat's two sides differ (`coxswain/sim/dynamic_oar.py`)
- [x] Handle power is closed-form — work per drive is `peak × ∫shape dφ` whatever the speed — and the integrated measurement agrees to 1%
- [x] The gate on the full hull, with the crew's surge swing present — **passes, but weaker** (below)
- [x] `research` repointed at the dynamic oar; UNSTABLE retired for PARTIAL, which names what is left
- [x] The prescribed oar block refuses a dynamic-oar boat, so no consumer can run the shipped oar under a research label
- [x] The scorecard settles dynamic-oar boats at **stated watts**, not power scales -- the only scale-to-watts conversion in the project is the questionable `mean_handle_power`
- [x] The report refuses `--physics research` at argument parsing, before any expensive stage
- [x] `scorecard.run("research")` end to end: both defect targets pass on the eight and the four, where `shipped` fails both
- [x] Score `blade_efficiency_level` on the dynamic oar, **measured on the run** — and it **fails**: 0.586 on the eight and 0.590 on the four at race pace, against 0.754–0.816
- [x] Drive the dynamic oar the way every consumer drives a simulator: `DynamicOarSimulator.run()` on the base class's contract, and `simulator_for(boat)` choosing the simulator from the boat's stamp, with a dynamic-oar boat refused unless it states `handle_watts`
- [x] `fit_reduced_model` asks the factory, so steering authority is fitted with the physics the stamp names
- [x] Report ported: `--physics research` runs. The four is driven at its own erg watts; the **masters eight's simulated parts are left off the page** — no sourced wattage — while its lines are still priced
- [x] The report's door now refuses only physics that does not exist (`learned`), and its caveats that stated the shipped defect as fact are functions of the profile
- [x] **Run end to end** (`--physics research --quick --steer-leg 150`, 229 s, clean log): page stamped research, the eight's gap stated, no shipped-only claim on it. The four steers through the dynamic oar — reactive 1.96 m rms, predictive 1.10 m rms with **0 / 285** solver fallbacks — and the quasi-steady evaluator's optimism on the four reads **+5%** (3.63 m/s priced against 3.47 settled), against **+35%** recorded under `shipped`. One quick run, not yet a sweep
- [x] Blade-path figure in the inertial frame, key events labelled (`coxswain/viz/bladepath.py`): drawn from the dynamic run, on the offline report under its own tab, and drawing BOTH load components when the simulator runs tier 2. Under the TIER 1 blade it showed the second flow reversal belongs to the prescribed schedule, not to rowing. **Under tier 2 that does not hold**: the eight at rate 28 and 380 W shows three slip reversals, one of them a re-anchor near the finish, with the blade anchored 11% of the drive. Tier 2 normal load peaks at 2231 N against 720 N for tier 1 at the same power, and the tangential load at 188 N, 8% of the normal. So whether a real blade re-anchors near the finish is a question for measured traces, not a settled finding of either tier
- [ ] Immersion curve — **blocked on data**, see below
- [ ] Entrainment term in place of a constant added mass — **blocked on the same data**

**Gate:** does the η-against-v line stop passing through the origin?
Measured on the baseline it crosses zero at 2.0% of mean speed on the eight
and 0.07% on the four. And does net propulsive impulse survive at race
pace, where the prescribed-angle substitution gave ≈ 0?

#### The unit works, measured before wiring

`OarDynamics` integrates `I·φ̈ = τ_handle(t) + l·F_n(φ, φ̇, v)` from the
catch to the finish angle. Catalogue eight at rate 28, constant 900 N·m
pull, propulsive impulse `∫F_n cos φ dt` over the drive:

| v (m/s) | prescribed angle | **dynamic angle** |
|---|---|---|
| 2.80 | +459.3 N·s | **+256.1 N·s** |
| 4.85 | **+0.2 N·s** | **+162.5 N·s** |
| 6.00 | **−171.2 N·s** | **+126.7 N·s** |

At racing speed the prescribed model delivers **nothing**, and above it the
blade **brakes the boat** — `v·cos φ` overwhelms `l·φ̇` through mid-drive,
slip changes sign, and the oar is dragged through the water on a timetable
regardless of whether there is anything to push against.

Let the angle respond and the impulse is positive everywhere and **falls as
the boat speeds up** (256 → 163 → 127 N·s). That falling thrust is the
restoring term the efficiency-only wiring lacked, and the reason that
wiring ran away while this one should not.

Drive duration, now an output, comes out at 0.970 / 0.720 / 0.634 s across
the same three speeds for the same pull.

One thing the unit made obvious that the prescribed model could not: **at
the catch the water helps the rower.** With the sweep rate zero the only
relative motion is the boat carrying the blade, the water resists *that*,
and the resulting torque drives the oar towards the finish. It is the
anchored part of the drive — slip positive — and it is why the blade-path
figure shows the blade running backwards through the water only later in
the stroke. A blade is not purely a brake, and a test written here on the
assumption that it was asserted the wrong sign and failed.

#### Two things that become outputs

Worth stating before the work starts, because they are the reason it is
worth doing and they are also how it gets validated:

- **`flatness` stops being a parameter.** The shape of the sweep becomes
  whatever the torque balance produces. `SOURCES.md` §7 says the default of
  0.0 is "probably too peaked" and that 0.30 reproduces Kleshnev, but keeps
  0.0 because the change "should be made on the strength of a measured
  oar-angle trace rather than on one indirect constraint". Make the angle
  dynamic and nobody has to choose.
- **Drive duration stops being a formula.** With the angle dynamic you may
  prescribe *when* the rower extracts or *at what angle*, not both. Real
  rowers extract at a body position, so the finish angle is the natural
  input and the drive duration becomes a **prediction** — testable directly
  against [HF09]'s on-water pairs, which is exactly the measurement the
  current ergometer-fitted formula misses by 18–28%.

#### The gate: passed, on a reduced model

`coxswain/sim/oarloop.py` couples the oar balance to hull surge — the
smallest model that can answer the question. Two equations: the hull
accelerating under blade thrust minus drag, and the oar under handle
torque minus blade resistance.

Scored the way the baseline was — *where the fitted η-against-v line
reaches zero, as a multiple of mean speed*:

| | zero crossing |
|---|---|
| baseline, prescribed force | **0.020** — through the origin |
| the floor the target sets | 0.15 |
| **reduced dynamic model** | **8.9** |

η is now nearly flat, 0.571 → 0.604 across 3.00 → 5.24 m/s, **and at a
level a blade efficiency could actually have**. The baseline had it
rising 0.24 → 0.52 over the same range.

| peak τ | speed | W per rower | η |
|---|---|---|---|
| 200 N·m | 3.00 | 79 | 0.571 |
| 450 | 4.04 | 178 | 0.597 |
| 900 | 5.24 | 356 | 0.604 |

*Re-measured 2026-09-13 after the blade-centre correction* — the blade force now acts at [CR06]'s `l`, 2.30 m on the eight, not the oar's tip at 2.56 m (TRACKING, Fixed): the zero crossing is **6.4** (was 8.9), still about forty times the
floor, and η runs 0.528 → 0.569 across 2.92 → 5.13 m/s (200 N·m: 2.92 m/s,
0.528; 450: 3.95, 0.557; 900: 5.13, 0.569).

**And it reaches published race pace at a power a rower could produce.**
Driven at one stated 380 W per rower, the eight at rate 32 settles at
5.33 m/s (published band 5.0–5.6) and the four at 4.76 (band 4.5–5.1).
*Re-measured after the blade-centre correction:* 5.21 and 4.66, both still
in band.
The prescribed model needed 720 W and 795 W to get there and overshot
every band anyway — which is why that regression test is a strict xfail.
The boat-class ordering falls out too: at equal power per rower, eight >
four > double > single, imposed nowhere.

**What this does not show.** The crew's mass does not move in the reduced
model, so there is no intracycle surge swing — and the swing is what
destroyed the efficiency-only wiring. A pass here is necessary and not
sufficient. It settles the shape of η(v); it cannot settle whether the
coupled model stays stable once the crew is moving again. That is what the
full wiring is for.

**Two bugs the level check caught**, both of which the shape check would
have passed straight through:

- The rig's **gearing was applied to the blade force** as well as to the
  handle force, by analogy with `hull_load`. For the hull-plus-crew system
  the only external horizontal forces are the blade force and the drag —
  the pull on the handle, its reaction, and the stretcher force are all
  internal. The lever sets how hard the rower must pull for a given blade
  force, not how much of it reaches the boat. Charging it twice cost a
  factor of 3.2 and put η at 0.18, which no blade has.
- A **sculler was charged for one oar and credited with two**, which
  flattered the single by a factor of two in power and put it 28% above
  its published race pace.

#### The gate on the full hull: holds, and weaker than the reduced model promised

`DynamicOarSimulator` puts the oar physics on the full 6-DOF simulator, with
the prescribed crew surging on its own clock. That is the thing the reduced
model said it could not show.

**It does not collapse.** On the eight at rate 28 — the regime where the
efficiency-only wiring fell to 0.63 m/s — runs started at 3.4 and at 6.5 m/s
meet at a racing speed, with the swing present. The restoring term the force
model supplies is what the efficiency factor lacked.

**The gate passes, by twelve times the floor.**

| | η zero crossing | η/v spread |
|---|---|---|
| baseline, prescribed force | 0.020 × mean speed | 2.7% (flat) |
| floor the target sets | 0.15 | — |
| reduced model, no crew swing | 8.9 | — |
| **full 6-DOF hull** | **1.81** | **39%** |

| peak τ | speed | W per rower | η | surge swing |
|---|---|---|---|---|
| 200 N·m | 2.97 | 79 | 0.557 | 78% |
| 450 | 4.21 | 178 | 0.665 | 58% |
| 900 | 5.48 | 355 | 0.691 | 46% |

The crossing is 1.81 on the full hull, not the reduced model's 8.9: η still
rises with speed, 0.557 → 0.691, where the reduced model's was nearly flat.
The swing is back, and it is largest exactly where the boat is slowest.

*Re-measured 2026-09-13 after the blade-centre correction* — the blade force now acts at [CR06]'s `l`, 2.30 m on the eight, not the oar's tip at 2.56 m (TRACKING, Fixed): the full-hull crossing is **1.57** (was 1.81) and the η/v spread 37% (was
39%) — still ten times the floor. 200 N·m: 2.90 m/s, η 0.518, swing 80%;
450: 4.13, 0.633, 59%; 900: 5.39, 0.660, 47%.

**The obvious explanation is wrong, and was measured before it could be
written down.** The guess was that the swing starves the blade at low speed,
as it did the efficiency-only wiring. Blade efficiency weighted by force, at
instantaneous against mean speed:

| peak torque | swing | instantaneous / mean |
|---|---|---|
| 200 N m | 77% | **1.089** |
| 450 | 58% | 0.923 |
| 900 | 46% | 0.873 |

The reverse of the prediction: the swing slightly *helps* the blade when the
boat is slow and costs it more when fast. So the blade channel is not why
eta is low at low speed. The next candidate is the drag channel -- drag power
is steeply nonlinear in speed, so a large swing spends power that the
numerator R(v_mean) v_mean never counts (Hofmijster's velocity efficiency).

**Measured, and it is half the story.** Drag power averaged over a settled
stroke, against drag power at the mean speed:

| peak torque | swing | ⟨R(v)·v⟩ / R(v̄)·v̄ | η at mean speed | η, charged for the swing |
|---|---|---|---|---|
| 200 N·m | 77% | **1.164** | 0.558 | 0.650 |
| 450 | 58% | 1.074 | 0.665 | 0.715 |
| 900 | 46% | 1.062 | 0.692 | 0.734 |

The swing wastes 16% of the drag power at 3 m/s and 6% at 5.5 m/s. Charge
for it and the rise in η across the range halves, from 24% to 13%. What is
left is the blade itself: force-weighted blade efficiency at mean speed rises
0.563 → 0.671 over the same range, because a slower boat lets the same pull
slip more. That is not a defect — SOURCES §7 already records blade efficiency
rising with boat speed as a real term in the power budget — and it is exactly
the speed dependence the prescribed model could not have.

**So the residual is physics, not a phase 2 bug.** Half is the blade slipping
more on a slower boat. Half is power lost to the surge swing, which is real,
but whose size rides on the crew's prescribed motion — defect two, and phase
4's job. Pinned by `test_the_residual_rise_in_eta_is_half_swing_and_half_blade`.

#### The scorecard, on the same harness as the baseline

`scorecard.run("research")` — the first time the research physics has been
scored by the battery that recorded the defect. Dynamic-oar boats settle at
stated watts per rower (80–360), not power scales.

| target | band | `shipped` | `research` eight | `research` four | tier 2 eight | tier 2 four |
|---|---|---|---|---|---|---|
| η zero crossing | ≥ 0.15 | 0.020 **fail** | 1.462 pass | 1.592 pass | 2.214 pass | 2.370 pass |
| η/v spread | ≥ 0.08 | 0.027 **fail** | 0.380 pass | 0.389 pass | 0.429 pass | 0.440 pass |
| surge swing | 30–60% | — | 46.1% pass | 49.6% pass | 42.0% pass | 45.5% pass |
| blade efficiency level | 0.754–0.816 | n/a | 0.586 **fail** | 0.590 **fail** | 0.664 **fail** | 0.671 **fail** |

*Tier 2 columns:* `scorecard.run("research", blade_law="liftdrag")`, stamped
`research+liftdrag` on every score. A study, not a profile — the coefficients
are provisional until their primary source is read. On the battery that
recorded the defect, tier 2 moves every target the right way: both defect
targets pass by a wider margin, the surge swing comes down, and the efficiency
level rises from 0.59 to 0.67 — still failing the band, which matches the
bespoke full-hull measurement to the third decimal. Lift is part of the
efficiency gap, not all of it.

*Re-measured 2026-09-13 after the blade-centre correction* — the blade force
now acts at [CR06]'s `l`, 2.30 m on the eight, not the oar's tip (TRACKING,
Fixed). Same battery, same boats, rate 28:

| target | band | tier 1 eight | tier 1 four | tier 2 eight | tier 2 four |
|---|---|---|---|---|---|
| η zero crossing | ≥ 0.15 | 1.222 pass | 1.300 pass | 2.115 pass | 2.368 pass |
| η/v spread | ≥ 0.08 | 0.356 pass | 0.363 pass | 0.425 pass | 0.440 pass |
| surge swing | 30–60% | 47.2% pass | 50.4% pass | 42.9% pass | 46.4% pass |
| blade efficiency level | 0.754–0.816 | 0.559 **fail** | 0.566 **fail** | 0.631 **fail** | 0.639 **fail** |

Every pass and fail is unchanged. A blade closer to the pin sweeps more slowly
for the same slip, so the defect targets pass by a little less and the
efficiency level falls about 0.03 further below the band, on both tiers.

Both defect targets that `shipped` fails, `research` passes, on both boats.
(The crossing reads 1.46 here against 1.81 from the torque sweep above: the
operating points differ, and the scorecard's is the canonical number.)

**The level target fails, and it is measured, not inferred.** A dynamic-oar
boat carries no `blade_model` and no sweep to read a level from, so it is now
measured on the run itself: `1 − |slip|/|blade speed|` from the blade model's
own definition, weighted by the blade force the water actually applied, at
the integrated oar angle and rate and the oarlock's instantaneous water
speed — so the crew's surge swing is inside it.

| boat | watts per rower | speed | measured on the run | the schedule-based level at that speed |
|---|---|---|---|---|
| eight | 80 | 2.96 | 0.615 | 0.491 |
| eight | 360 | 5.51 | **0.586** | 0.811 |
| four | 80 | 2.58 | 0.636 | 0.431 |
| four | 360 | 4.78 | **0.590** | 0.738 |

Two things, and they pull in different directions:

- **The defect's signature is gone from the level too.** Evaluated on the
  prescribed schedule, the level rises with speed — 0.49 to 0.81 on the eight —
  which is η ∝ v restated at the blade. Measured on the dynamic run it is
  flat, and falls slightly with power.
- **But it sits a quarter below Kleshnev's 0.785 ± 0.031.** A real miss, kept
  on the page and pinned in `test_research_passes_the_defect_targets_shipped_fails`.

It also sets up a tension recorded in [TRACKING.md](TRACKING.md): the eight
and four reach published race pace at 380 W per rower **with** blades a
quarter less efficient than measured ones. Either something upstream is
generous, or 380 W is high. **It is not the efficiency definition:** measured
energetically on the same runs — propulsive power over the power put into the
blade — the level is 0.622 on the eight and 0.624 on the four at race pace,
about +0.035 on the instantaneous figure and still well below 0.754. The gap
is physics; the next candidate is the missing lift, which is tier 2.

**Published race pace at 380 W per rower, full hull:**

| boat | speed | band | |
|---|---|---|---|
| eight, rate 32 | 5.557 | 5.0–5.6 | in |
| four, rate 32 | 4.902 | 4.5–5.1 | in |
| single, rate 30 | 4.22 | 4.1–4.7 | in |

**The single used to miss** — 3.993 m/s, 2.6% below its band — and it was
recorded as a miss rather than tuned away. The cause turned out to be a bug,
not the power: a sculler's two oars were each balanced against the rower's
whole reflected inertia, so the body was counted once per oar and 2.2% of the
handle work over a drive went unaccounted for. With one seat balance —
`(I_crew + n I_oar) φ̈ = −n τ + Σ blade` — the energy books close and the
single settles at 4.22, inside its band. Sweep seats are bit-identical, so the
eight and four above did not move. (The reduced model's 4.05 was measured
before the same fix and has not been re-measured.)

#### An unfitted prediction — true of the oar alone, not of the full model

**Corrected 2026-09-13.** This section used to be headed "an unfitted
prediction that lands on the water" and called it the best result so far. The
measurement it rests on is real, but it was taken on the oar balance **alone**,
at a **fixed** 4.85 m/s, under a **constant** pull. On the full 6-DOF hull, at a
stated power and the measured front-loaded pull, it does not hold.

What was measured, and still stands as a statement about the unit: eight at
rate 28, boat held at 4.85 m/s, constant handle torque swept across a 4.4x
range of power —

| | drive fraction |
|---|---|
| the oar alone, 183–808 W per rower | 0.319 – 0.406 |
| measured on the water, [HF09] pairs, 20.6–31.5 spm | 0.296 – 0.395 |
| the ergometer-fitted formula | 0.378 – 0.465 |

What the full model does, settled at a stated power with the measured pull:

| boat | rate | W per rower | speed | drive fraction | on the water [HF09] | erg formula |
|---|---|---|---|---|---|---|
| eight | 28 | 180 | 4.23 | 0.534 | 0.362 | 0.445 |
| eight | 28 | 380 | 5.62 | **0.400** | 0.362 | 0.445 |
| eight | 32 | 380 | 5.56 | **0.467** | 0.395 | 0.468 |
| eight | 32 | 500 | 6.16 | 0.420 | 0.395 | 0.468 |
| eight | 32 | 650 | **6.78** | **0.377** | 0.395 | 0.468 |
| four | 32 | 380 | 4.90 | 0.510 | 0.395 | 0.468 |

At rate 28 and a race power the eight's drive sits between the water and the
erg formula. At rate 32 the drive *time* barely changes (0.856 s to 0.875 s),
so its fraction climbs to the erg formula's value. On the water crews shorten
the drive as rate rises, and they do it by pulling harder — and the model does
that too: at 650 W the eight's drive fraction reaches the on-water 0.377. **But
it is then doing 6.78 m/s**, far past the published 5.0–5.6 for an eight at
rate 32.

So **no single power reproduces both the on-water drive time and the published
race pace.** At the power that gives the pace the drive is 18% long; at the
power that gives the drive time the boat is 20% fast. That is physics, not a
missing input. Tracked in [TRACKING.md](TRACKING.md).

**It is probably not the blade-efficiency gap**, which this section first said
it very likely was. Tried on the oar balance alone — eight at rate 28, boat held
at 4.85 m/s, the torque 380 W needs — with a *draft* tier 2 blade on the
provisional [CG06a] coefficients: energetic efficiency rises from 0.624 to
0.735, most of the way to Kleshnev's floor, and the drive fraction does not move
at all (0.399 both ways; 0.361 and 0.359 at 5.5 m/s). More grip closes the
efficiency gap and leaves the drive time where it was, so the two gaps look
like separate causes. Oar alone and a draft blade, so this is evidence, not a
result, until it is measured on the full hull.

Two qualifications on the comparison itself: [HF09] measured coxless **pairs**,
so an eight and a four are not like-for-like against them; and there is no
sourced on-water drive fraction for a single.

#### Tier 2 on the full hull — half the efficiency gap, and the boats get too fast

`DynamicOarSimulator(..., blade_law="liftdrag")` runs the tier 2 blade: lift
and drag on the angle of attack, both load components applied to the hull, the
normal one turning the oar. The default law is still tier 1, arithmetic
untouched. Same boats, same stated 380 W per rower, 14 strokes:

| boat | blade | speed (m/s) | drive fraction | measured blade efficiency | surge swing |
|---|---|---|---|---|---|
| eight, rate 28 | tier 1 | 5.62 | 0.400 | 0.586 | 45% |
| eight, rate 28 | **tier 2** | **6.00** | **0.379** | **0.663** | 41% |
| four, rate 32 | tier 1 | 4.90 | 0.510 | 0.592 | 54% |
| four, rate 32 | **tier 2** | **5.27** | **0.486** | **0.669** | 50% |

Three things, measured rather than expected:

- **It closes a little under half the efficiency gap.** On the scored
  definition the level rises from 0.59 to 0.67, against a floor of 0.754 — so
  it still fails, by less. (The oar-alone probe's 0.74 was the *energetic*
  definition, which reads higher; that offset was measured before.)
- **The drive shortens by about 5%, and only because the boat is faster.** On
  the oar alone, at a fixed speed, tier 2 did not move the drive at all. On the
  full hull the boat speeds up and the drive follows. The four at rate 32 is
  still 0.486 against 0.395 on the water, so most of that gap remains, and
  "separate causes" mostly holds — with that qualification.
- **A better blade makes the boats too fast at 380 W.** The four settles at
  5.27 m/s, above its published 4.5–5.1. That turns the tension recorded
  earlier — boats reaching published pace with blades a quarter less efficient
  than real ones — from a quirk of a leaky blade into a real question upstream:
  either something in the drag is generous, or 380 W per rower is high for
  those published paces. That is the published-race-power data item already
  listed as blocked, now with a sharper reason to want it.

On provisional coefficients from a secondary source, and not yet a profile:
`research` still runs tier 1, and nothing here moves until the primary is read
and the scorecard runs on tier 2.

#### The inertia question, answered — derived, not fitted

The open decision was whether to lump an effective inertia (cheap, but a
new fitted parameter) or wait for the forward-dynamic rower. **Neither was
necessary.** The generalised inertia for the coordinate φ is
`Σᵢ mᵢ |∂xᵢ/∂φ|²`, and every term is already in the model: de Leva segment
masses, and segment velocities from the joint chain. `reflected_inertia()`
computes it.

It is **not a constant**: about 93 kg·m² early in the drive falling to 13
at the finish, because the legs move a great deal of mass per radian of oar
and the arms very little. A handle speeding up through the second half of
the drive is that, not a change of effort.

So the balance carries the term a varying inertia requires —
`I(φ)·φ̈ + ½(dI/dφ)·φ̇² = τ` — and dropping it would be a first-order error,
not a refinement.

**Two honest limits.** The profile *diverges at both ends*: the prescribed
crew motion does not stop when the prescribed oar sweep does, so `vᵢ/φ̇`
blows up. That is an inconsistency in the current kinematics, not a
property of rowing, and it means the reduction to one coordinate is valid
only over the interior of the drive — samples below a quarter of the peak
sweep rate are dropped and the ends clamped. And the profile *inherits the
ergometer*, since the joint angles behind it are the erg-fitted ones. When
the rower becomes forward-dynamic this stops being computed and becomes a
consequence of the multibody chain.

#### The superseded design question: what inertia?

`I·φ̈ = τ_handle − τ_blade` needs an inertia, and **the oar's own is far too
small.** Measured on the catalogue eight: `inertia_about_lock = 4.44 kg m²`,
and mid-drive slip of 2.77 m/s gives a blade force of 648 N at 2.56 m
outboard — 1660 N·m, so **φ̈ = 374 rad/s²** on the oar's inertia alone. The
sweep it has to produce peaks at about 11 rad/s², thirty times less: the
oar would slam through the drive in a fraction of the time.

The missing inertia is the rower. Reflecting crew mass through the 1.14 m
inboard:

| reflected crew mass | I | φ̈ at 1660 N·m |
|---|---|---|
| oar alone | 4.4 kg m² | 374 rad/s² |
| 40 kg | 56.4 | 29 |
| 60 kg | 82.4 | 20 |
| 80 kg | 108.4 | 15 |

So the effective inertia has to be ~25× the oar's, and essentially all of
it is body. **The oar angle's dynamics are the rower's dynamics.**

From that, the choice looked like this, and it looked like a real one:

1. **Lumped effective inertia** — oar plus the crew's reflected mass, as one
   number per seat. Cheap, keeps the prescribed crew kinematics, and gets
   the feedback loop closed. But the effective inertia is a new fitted
   parameter, which is the kind of thing this programme exists to remove.
2. **Wait for the forward-dynamic rower (phase 4)** and let the inertia come
   out of the multibody chain, where it is geometry rather than a fit.

**Both were wrong**, because the quantity is computable from things already
in the model — see the section above. The instinct that a default inertia
would become a parameter nobody remembered fitting was right; the remedy,
making it a required argument and forcing a choice, was not. Kept on the
record because a wrong framing that survived a day of work is worth being
able to recognise again.

---

### Phase 3 — tier 2 built, wired, drawn and scored; its gate is blocked on data

Everything phase 3 can do without Grift's traces is done: the lift-drag blade
as a tested unit, wired into the dynamic oar behind `blade_law`, both load
components on the blade-path figure, and the validation battery run on it as a
study. Its gate — does tier 2 reproduce the sign and timing of the measured
tangential force — needs the time-resolved traces of Grift et al. (2021), and
the coefficients themselves need their primary source. Both are in Blocked,
below. Tier 2 stays a study, not a profile, until they arrive.

### Phase 4 — the forward-dynamic rower (4.1 closed on a finding; 4.3 next)

**Why this is next.** The two gaps phase 2 and 3 left open both point at the
crew, not the blade. The drive runs long at a race power and tier 2 does not
shorten it; and the reflected inertia the dynamic oar leans on diverges at the
catch and the finish, because the prescribed body does not stop when the oar
does. Both are symptoms of a crew whose motion is prescribed in time, from an
ergometer, while its oar is not.

**The formalism** stays as decided — recursive Newton–Euler on a
free-floating base, tree-structured with kinematic loops, the boat, oars and
crew topology exactly — driven by joint torques, not muscles.

*Corrected on reading the paper* ([RK11], SOURCES.md). This page used to
credit the torque-driven part to Rongère, Khalil & Kobus (2011). It is not
theirs. Their model is **inverse dynamics**: every active joint, the oar angle
included, follows a prescribed periodic B-spline, and the boat's motion and
the joint torques are what the recursion returns — a clock crew, as ours was.
Their one kinematic loop (legs, buttock, sliding seat) is closed by projection
through a pseudo-inverse of the loop Jacobian, not by constraint forces; the
rower has no arms, and the oars are driven independently of the body. And
they conclude what 4.1 found: a total inverse-dynamics approach "is not so
realistic", because the oar motion and force cannot be controlled correctly,
and "a better approach may be an hybrid approach composed of inverse and
direct dynamics". Step 4.3 is that hybrid. The recursion and the loop
bookkeeping are theirs to follow; the torque drive and the handle held by a
force are not in the paper and have to be built and validated here.

**The steps, in the order each can be validated before the next is built:**

- [ ] **4.1 — The hands follow the oar.** Before any torque-driven chain, make
  the prescribed crew kinematically consistent with the dynamic oar.

  *The baseline, measured first* (settled stroke, 380 W): the clock crew's
  hands sat up to **0.19 m** off the dynamic handle on the eight and **0.74 m**
  on the single, whose oar ran 48° behind the body mid-drive. The arms are
  0.7 m long, so the plan's "rest of the body solving to reach it" could not
  be an arm solve: the whole body has to follow.

  *Built* (`coxswain/crew/follow.py`, `DynamicOarSimulator(crew="follows")`, a
  study): through the drive the body's stroke time is the prescribed time at
  which the sweep had the current angle, read from the stroke table — which is
  therefore kept, not bypassed — with velocities and accelerations by the chain
  rule. The oar's inertia is built from exactly those velocities, so the body
  has one kinetic energy whether the hull or the oar is asked; before this the
  oar assumed a following body and the hull felt a clock one. The recovery is
  retimed from the dynamic finish to the next catch.

  *The criterion first written here was wrong.* The reflected inertia cannot
  stop diverging in 4.1: it diverges because the prescribed sweep's rate is
  zero at its ends, and that is 4.2's to remove. The criteria are now: hands on
  the handle (holds, under 2 mm), one kinetic energy (holds, to 1e-9), and
  **the crew hands the hull no momentum of its own — which fails.**

  *The gate failure.* The hull feels the crew only through Σm·a. On the eight
  at 380 W that integrates to **+361 N·s per stroke** against a real change in
  crew momentum of +108; the clock crew closes to 0.005. Nearly all of it is at
  the finish (−241 N·s), where the dynamic oar is stopped dead at the finish
  angle and a following body with it; −34 at the catch, where the retimed
  recovery arrives moving and the drive starts from rest; the rest inside the
  rate floor. The first settled run — eight 5.62 → 5.06 m/s, drive fraction
  0.400 → 0.424 — measured that defect, not the physics, and predicted the
  opposite of what I had stated beforehand. It is void. Pinned as a strict
  xfail in `tests/test_crew_follows.py`; TRACKING has the fix options.

  *Fix (b), and what it showed.* The velocity jumps are now handed to the
  hull as impulses through the system mass matrix, conserving the momentum
  of hull plus crew (unit-tested exactly). Measured with RK4's own weights,
  that leaves **+148 N·s a stroke on the eight** and +13.5 on the single,
  sitting in the first step off the catch and the last steps before the
  finish — inside the rate floor, where the pose runs through the stroke
  table at the true clock rate and the velocity is scaled by the capped one.
  Using the true clock rate in the acceleration was tried and overflowed (it
  reaches 7×10¹⁴ at the catch). The settled runs with the impulses — eight
  5.36 m/s against the clock crew's 5.62, single 4.20 against 4.22 — still
  carry that leak and are not results.

  *A requirement 4.3 inherits from [CR06]'s release rule:* the oar must be
  handed into the drive **already moving**. Wired in as a study, the rule
  froze the oar at the catch, because today the drive is started by the water
  loading a blade parked at rest under a zero pull (TRACKING). The rower that
  carries the oar through the recovery is what removes that.

  *Qualified on reading [CR06] in full:* the direction of slaving decides
  it. [CR06] slaves the **oar to the body** — smooth prescribed leg, back and
  arm motions, the oar angle given by the hand-on-handle relation, arms free —
  and that is consistent by construction: no jumps, momentum exact, and the
  oar never stops. What fails is the direction 4.1 took, a body slaved to an
  oar whose angle is *integrated* under a torque. [CR06]'s price is that the
  oar is not predicted; the coordination is fitted.

  **Closed on a finding, not a pass.** A body slaved *kinematically* to the
  oar cannot hold its hands on the handle and conserve momentum at once,
  because the ergometer-fitted body is still moving where the sweep's rate is
  zero — the same inconsistency `reflected_inertia` already documents. The
  way out is to enforce hand-on-handle by a constraint force rather than by
  construction: the kinematic loop of Rongère's formulation, i.e. step 4.3.
  4.2 (the sweep as an output) needs that too, so it follows 4.3 rather than
  preceding it. The following crew stays as a study, marked as not
  momentum-consistent.
- [ ] **4.2 — Sweep shape becomes an output.** With the hands on the dynamic
  handle, `OarAngleSweep.flatness` has nothing left to set, and is deleted from
  this path. *Validated by:* the sweep that results, against a measured oar-angle
  trace when one is obtained.
- [ ] **4.3 — Joint torques drive the chain.** *Groundwork so far:* every
  segment now carries its principal moments of inertia from de Leva's own
  Table 4 radii of gyration, lumped segments by the parallel-axis theorem —
  the chain had masses and lengths but no rotational inertia. Recursive
  Newton–Euler on the
  free-floating base, with a stroke-tracking controller fitted to the erg data
  as ONE operating point, not as the model. *Validated by:* reproducing that one
  operating point without being tuned past it.
- [ ] **4.4 — Synchronisation and roll controllers.** Hand-written, per the
  plan: the classical baseline tier 3 has to beat. The coupled-oscillator model
  is kept as the null hypothesis.

**Gate:** does the predicted crew centre-of-mass excursion land in the measured
band without being fitted to it? That is the test the prescribed crew cannot
take, because for it the excursion is an input.

**What phase 4 is expected to move, stated before it is measured,** so the
measurement can contradict it: the drive fraction at race power (0.467 on the
eight at rate 32, against 0.395 on the water), and the part of the efficiency
gap tier 2 left (0.664 against a floor of 0.754).

## Blocked, and on what

Letters asking for the three datasets below are drafted in
[DATA_REQUESTS.md](DATA_REQUESTS.md) §9 (Grift et al., TU Delft — the
immersion curve and the tangential force traces) and §10 (Hill & Fahrig —
the boat speeds that go with their drive durations).

| what | blocked on | why it cannot be guessed |
|---|---|---|
| Immersion curve refit | Grift et al. (2019), JFM 866 — the full C_D against depth figure | Three points from the abstract (1.10 at the surface, 1.60 at ~20 mm, 1.30 deeper) are enough to show ours is *qualitatively* wrong — monotone where the measurement has an optimum — and not enough to fit. Three second-hand numbers are not a basis for physics. |
| Entrainment term | the same paper | They show a single added-mass coefficient does not capture prolonged acceleration and define an entrainment rate instead. The rate is in the paper, not in the abstract. |
| Tier 2 coefficients — **located, provisional** | Caplan & Gardner's own paper, to verify the constants | Found via a secondary source, [CG06a] in SOURCES: `C_L = A_l sin 2α`, `C_D = A_d sin²α`, with A_l = 1.25 and A_d = 2.07 for the Big Blade. The primary is paywalled. The shape is corroborated and the scale agrees with [CR06] in order, so tier 2 can be built on them — labelled as resting on a secondary source until the primary is checked. |
| Tier 2 validation | Grift et al. (2021), JFM 918 — time-resolved force traces | The only source found that gives the tangential component, which the model has never had. |
| Coordination replication | the forward-dynamic crew (phase 4) | Nothing to run it against yet. |
| The masters eight in the research report | a sourced handle power per rower for a masters eight | The dynamic oar is driven at stated watts. `MASTERS_POWER` is a force scale, and the only scale-to-watts conversion in the project is the questionable `mean_handle_power`. Coaching material gives ranges (roughly 100–200 W for social masters), not measurements, so the eight's steering run and settled speed are left off the research page rather than invented. Its lines are still priced, because the route evaluator never runs the simulator. |
| Validating against published race pace | a published **race power** per boat class | The regression tests compare settled speed against published pace while driving each boat at `power_scales = 1.0` — which is 540 W per rower for an eight at 24 spm, 720 at 32, 854 at 38, 795 for the four and **1124 for a single at 30**, against roughly 330 W a crew can hold for six minutes. Every boat beats race pace, which is the only thing that could have happened. Scale 1.0 is a force scale, not a wattage, and it does not even mean the same thing between boats. Fixing the comparison means stating the power first, and that needs a source. |
| Putting the drive fraction right | a decision, not data | [HF09] on-water pairs and Telfer's ergometer rowers disagree by ~0.08 of the cycle and both are right about their own conditions. Choosing the on-water number moves the time base of every calibration in the project, so it belongs in `research` behind the scorecard. |

---

## Decisions taken

| date | decision | why |
|---|---|---|
| 2026-09-12 | Torque level, not muscle level | Hill-type muscles buy fidelity we cannot validate without EMG; gross efficiency is known to be 0.20 and rate-independent, so the metabolic layer is a scalar applied afterwards |
| 2026-09-12 | Tier 1 before tier 2 | Tier 1 fixes the defect; tier 2's value is decided by phase 1's gate |
| 2026-09-12 | Phase 0 is worth a whole phase | The baseline gets set before anyone has a stake in beating it. It found two bugs on its first run, which settles the argument |
| 2026-09-12 | Coupled-oscillator replication is in scope | Promoted to a tier 3 acceptance test, so it sits on the critical path rather than beside it |
| 2026-09-12 | The 1.7× fluctuation gap was a data problem | It rests on [IVV25] alone, whose extremes are skewed opposite to a real boat-speed curve. Against Holt the model is +10–31%, an ordinary discrepancy |
| 2026-09-12 | Policy runs slower than the physics (20–50 Hz) | Voluntary motor bandwidth is a few Hz, not a hundred; it is both cheaper and more faithful, and it gives the delay requirement structurally |
| 2026-09-12 | Roll is a budgeted constraint, not a weighted penalty | A weighted sum silently picks a Pareto point nobody chose |
| 2026-09-12 | Phases 1 and 2 merged: tier 1 **is** the slip force with a dynamic oar angle | The efficiency factor alone is the destabilising half of the blade physics and has no operating point. Measured, not argued |
| 2026-09-12 | Settle measurements run to 250 s, not 70 | 70 s is not convergence for this system, and two recorded figures were unconverged transients quoted as equilibria |
| 2026-09-12 | Tier 3 tokenises on **stroke phase**, not time — 32 tokens per cycle, rate as a conditioning feature | Rate-invariant by construction; a time token covers a different amount of stroke at 20 spm than at 36, so the policy would see a shifted distribution for no physical reason |
| 2026-09-12 | Sensory delays are **per channel**, not one number: proprioceptive ~30–50 ms, vestibular ~50–100, auditory ~160, visual ~180–200 | They are different pathways with different latencies, and they map onto the mechanical-versus-sensory coupling already scoped in `PLAN_SYNCHRONISATION_AND_BLADES.md`. Values are textbook and need a citation before they reach the report |
| 2026-09-12 | The behaviour-cloning prior is **annealed out**, and constrains only what the erg genuinely establishes — joint limits, legs–trunk–arms sequencing, torque envelopes — not the torque trajectory | The trajectory is the part the fixed stretcher contaminates. A permanent L2 pull toward it is a pull toward the defect; PPO's KL term is a different regulariser toward a different centre |
| 2026-09-12 | PPO is the baseline; **short-horizon analytic gradients (SHAC family)** run alongside it | The CasADi path already gives analytic derivatives, which most RL problems lack. Also: one shared policy with a seat embedding and a centralised critic (MAPPO), and GRU and temporal-convolution baselines so the transformer has to earn its place |
| 2026-09-12 | Reward is **distance over a fixed number of stroke cycles**, under the existing `WPrimeBalance` energy budget | Instantaneous speed invites transient exploits; without an energy budget a policy produces unbounded power |
| 2026-09-12 | Broken regression tests are xfail-strict with the reason, never loosened or deleted | Four of today's findings were sitting in failing tests. A loosened test would have hidden all four; a deleted one would have lost the evidence |

## Open questions

- The neural-network rowing paper neither of us can place. Proceeding as
  though the application is genuinely open; rowing NN work found is all
  estimation, and generative muscle control has never been pointed at an oar.
- `mean_handle_power` dots the **oarlock** force with the **handle**
  velocity; under the ideal lever those are not a conjugate pair. It sits
  underneath every power number in the project.
