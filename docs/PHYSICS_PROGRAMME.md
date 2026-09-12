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
| `research` | tier 1 → 2 | forward-dynamic, torque-driven | the offline report |
| `learned` | tier 2 | a trained policy | training and study runs |

**`research` is currently unstable and says so.** Tier 1 resolves to the
efficiency-only wiring, which collapses the boat (see phase 1 below). It is
deliberately left resolving to tier 1 rather than quietly dropped to tier 0,
because the collapse is the finding and a profile that silently behaved like
`shipped` would hide it — but it must not look safe, so the summary string
says "UNSTABLE" and a test asserts that it does. **Do not quote a speed from
`research` until phase 2 lands the slip force.**

`shipped` is **frozen, not current**. The freeze is held by a source scan
over everything the trainer can execute, not by convention — a behaviour
test cannot cover a path no test exercises, which is exactly how v0.12
shipped its crash.

## The blade tiers

| tier | model | state |
|---|---|---|
| 0 | prescribed force profile, a function of stroke phase | **shipped** |
| 1 | [CR06] Model 1 — normal load from blade slip, oar angle a dynamic state | in progress |
| 2 | [CR06] Model 2 family — lift and drag resolved against angle of attack | planned |
| 3 | (rower, not blade) transformer policy outputting joint torques | planned |

---

## Phases

| # | phase | gate | status |
|---|---|---|---|
| 0 | Scaffolding: profiles, validation battery, freeze guards | scorecard reproduces every number the report already claims | **done** |
| 1 | ~~Tier 1 blade as an efficiency factor~~ | **failed its gate — merged into phase 2** | **closed** |
| 2 | Tier 1 blade: slip-quadratic **force**, oar angle a dynamic state | η against v — is the line through the origin gone, and does net propulsive impulse survive at race pace? | **next** |
| 3 | Tier 2 blade: lift and drag on angle of attack | reproduces the sign and timing of Grift's measured tangential force | planned |
| 4 | Forward-dynamic rower (Rongère's formalism, torque-driven) | predicted CoM excursion lands in the measured band **without being fitted to it** | planned |
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

### Phase 2 — next

Tier 1 properly: the slip-quadratic **force** from [CR06] Model 1, with the
oar angle as a dynamic state per seat. The rower drives the handle, the
blade resists, the angle follows.

- [x] Torque balance about the pin, as a standalone unit (`coxswain/crew/oardynamics.py`), tested in isolation before anything is wired in
- [x] The comparison that justifies the change, measured rather than asserted
- [ ] Wire it into the simulator: two states per seat
- [ ] `StrokeTable` bypassed on the `research` profile (it assumes the chain depends on stroke time and nothing else)
- [ ] Hands follow the dynamic angle, so the crew kinematics solve online
- [ ] `flatness` deleted as a free parameter — it becomes an output
- [ ] Blade-path figure in the inertial frame at three speeds, key events labelled
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

#### The open design question: what inertia?

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

So there are two ways to do phase 2, and they are a real choice:

1. **Lumped effective inertia** — oar plus the crew's reflected mass, as one
   number per seat. Cheap, keeps the prescribed crew kinematics, and gets
   the feedback loop closed. But the effective inertia is a new fitted
   parameter, which is the kind of thing this programme exists to remove.
2. **Wait for the forward-dynamic rower (phase 4)** and let the inertia come
   out of the multibody chain, where it is geometry rather than a fit.

Option 1 is the smaller step and can be validated against option 2 later;
option 2 is the honest one but reorders the programme. **Not decided.**

---

## Blocked, and on what

Letters asking for the three datasets below are drafted in
[DATA_REQUESTS.md](DATA_REQUESTS.md) §9 (Grift et al., TU Delft — the
immersion curve and the tangential force traces) and §10 (Hill & Fahrig —
the boat speeds that go with their drive durations).

| what | blocked on | why it cannot be guessed |
|---|---|---|
| Immersion curve refit | Grift et al. (2019), JFM 866 — the full C_D against depth figure | Three points from the abstract (1.10 at the surface, 1.60 at ~20 mm, 1.30 deeper) are enough to show ours is *qualitatively* wrong — monotone where the measurement has an optimum — and not enough to fit. Three second-hand numbers are not a basis for physics. |
| Entrainment term | the same paper | They show a single added-mass coefficient does not capture prolonged acceleration and define an entrainment rate instead. The rate is in the paper, not in the abstract. |
| Tier 2 coefficients | Caplan & Gardner (2007) C_L, C_D against sweep angle | Needs digitising. |
| Tier 2 validation | Grift et al. (2021), JFM 918 — time-resolved force traces | The only source found that gives the tangential component, which the model has never had. |
| Coordination replication | the forward-dynamic crew (phase 4) | Nothing to run it against yet. |
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
| 2026-09-12 | Broken regression tests are xfail-strict with the reason, never loosened or deleted | Four of today's findings were sitting in failing tests. A loosened test would have hidden all four; a deleted one would have lost the evidence |

## Open questions

- The neural-network rowing paper neither of us can place. Proceeding as
  though the application is genuinely open; rowing NN work found is all
  estimation, and generative muscle control has never been pointed at an oar.
- Phase- or time-based tokenisation for tier 3. Leaning phase: rate-invariant
  by construction. Cheap now, expensive after training starts.
- PPO only, or PPO plus short-horizon analytic gradients? We have a
  differentiable simulator, which most RL problems do not.
- `mean_handle_power` dots the **oarlock** force with the **handle**
  velocity; under the ideal lever those are not a conjugate pair. It sits
  underneath every power number in the project.
