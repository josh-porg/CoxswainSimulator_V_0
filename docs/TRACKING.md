# Bugs, gaps and things being tracked

What is known to be wrong, known to be missing, or known to be a
placeholder. Kept separate from `SOURCES.md`, which records what was
built and why; this file records what has *not* been settled.

Anything fixed moves to the bottom with the evidence that fixed it, so
the file is also a record of what kind of thing goes wrong here.

---

## Open — correctness

### The shipped route and pacing optimisers charge depth to the whole resistance
**Impact: high for published pacing targets and optimised lines on shallow
water; frozen, so not changed.**

`RouteEvaluator._build_speed_table` solves `F(v, h) v³ = v_ref³` and
`CoursePacing.speed_for_power` multiplies the whole deep resistance by
`F(v, h)`. The simulator's `hull_resistance`, and `shallow.py`'s own docstring,
apply the factor to the wave term only: Day et al. note the viscous terms are
"less likely to be sensitive to water depth". So the optimisers and the
simulator disagree about the same boat on the same water.

Measured on the eight at the power for 5.2 m/s in deep water, same resistance
throughout (m/s; A shipped, B factor on the wave term, C research):

| depth | A: factor × total | B: factor × wave | C: Sretenskii | A → B | B → C |
|---|---|---|---|---|---|
| 1.5 m | 3.752 | 5.103 | 5.149 | +1.351 | +0.046 |
| 2.0 m | 4.193 | 5.006 | 5.141 | +0.813 | +0.135 |
| 2.5 m | 4.538 | 4.981 | 5.115 | +0.444 | +0.134 |
| 3.0 m | 4.762 | 5.106 | 5.108 | +0.344 | +0.001 |
| 4.0 m | 5.006 | 5.178 | 5.175 | +0.172 | −0.003 |
| 5.0 m | 5.111 | 5.192 | 5.191 | +0.080 | −0.001 |

At 3.0 m the shipped pacing model has the eight 0.34 m/s slower than the
simulator's own treatment would; at 2.0 m, 0.81 m/s. `scripts/course_pacing.py`
sets the published targets through it.

*Not changed:* the game and the report are frozen on `shipped`. Research boats
carry a depth-aware wave table, which takes both optimisers past this branch
(shallow-water item). Fixing it for `shipped` would move published targets and
needs a decision.

### The shipped Michell table is too coarse below 4 m/s
**Impact: low -- race speed is unaffected; warm-up, paddling and low-power
pieces are not.**

`MichellWave.tabulate()` samples 64 speeds from 0.5 to 8 m/s (0.119 m/s apart)
and interpolates linearly. Against the direct integral on the eight's
production grid (641 × 81), worst and mean error by speed band:

| samples | 2–3 m/s | 3–4 m/s | 4–5 m/s | 5–6 m/s | 6–7.5 m/s |
|---|---|---|---|---|---|
| 64 (shipped) | 14.3%, 2.7% | 1.7%, 0.5% | 0.5%, 0.1% | 0.1%, 0.03% | 0.03%, 0.01% |
| 151 | 4.3%, 0.7% | 0.3%, 0.1% | 0.09%, 0.02% | 0.02%, 0.01% | 0.01%, 0.00% |
| 301 | 1.05%, 0.19% | 0.08%, 0.02% | 0.02%, 0.01% | 0.01%, 0.00% | 0.00%, 0.00% |

The eight's Michell humps fall between samples at low speed. Uniform and
trapezoid weights behave the same. `FiniteDepthWaveTable`, which research
boats carry, samples 301 (about 10 s once per hull in place of 2 s);
`shipped` keeps 64.

### Michell's sum reads about 3% high, and the shipped trainer uses it
**Impact: low on speed, and a decision for the shipped default.**

Found checking `coxswain/hydro/michell.py` against a printed curve:
Lazauskas (2009) [LV09] Fig. 7.1, the inviscid wave resistance of the Wigley
hull (L/B 10, L/T 16, `S = 0.1487 L²`), which is also the 1979 Workshop's
value of 0.89 at `Fr = 0.2`. The formula is right: the thesis's eq. 5.17,
with `λ = sec θ`, reduces exactly to the `4ρg²/(πU²)` form the module uses.
The quadrature is not. `resistance()` puts `dx dz` on every grid point,
ends included, so the waterline row, where the depth decay is 1, and the bow
and stern stations count at full weight rather than half. It is first order.

*Measured, 1000 C_W on the Wigley* (figure read by eye, ±0.05):

| Fr | figure | uniform, default 81×41 | uniform, production 641×81 | trapezoid, default 81×41 |
|---|---|---|---|---|
| 0.20 | 0.89 | 0.939 | 0.919 (+3.3%) | 0.865 |
| 0.30 (hump) | 2.14 | 2.352 | 2.215 (+3.5%) | 2.128 |
| 0.345 (hollow) | 1.24 | 1.377 | 1.283 (+3.5%) | 1.229 |
| 0.50 (peak) | 4.52 | 4.880 | 4.646 (+2.8%) | 4.513 |
| 1.00 | 1.82 | 2.034 | 1.903 (+4.5%) | 1.835 |

Trapezoid weights converge by 321×81 to within −0.6% and +0.9% of the figure.

*What it costs the boats.* `Boat` builds every hull's wave table on the
production grid, 641 stations × 81 levels. On the rate-32 eight the shipped sum
overstates wave drag by **3.0–3.4%** between 4.23 and 6.0 m/s. Wave drag is
8–11% of the total there, so at the same power the boat is **0.09–0.13% slow**,
about 0.4 s over 2000 m.

*Done.* `MichellWave(quadrature="trapezoid")`, off by default; `"uniform"`
keeps the shipped sum bit for bit, pinned by a test. Tests pin the trapezoid
sum to Fig. 7.1 at five Froude numbers, and the uniform excess at the peak
(`tests/test_michell_quadrature.py`).

*Decided, 2026-09-13: the research model takes it.* The aim is the research
model's accuracy, and the game is frozen. So `MichellWave` keeps `"uniform"` as
its default, which leaves `shipped` and the Holt comparison exactly where they
were, and the `research` and `learned` profiles use trapezoid weights through
`wave="sretenskii"` (see the shallow-water item).

### No model of a blade squared and immersed before the catch
**Impact: medium.** The model has two clean regimes — a feathered skim
on the recovery, and a normal drive — and names the third explicitly as
out of scope ("a squared blade catching is a crab, a different
regime"). The transition is not represented: a blade that squares while
still in the water must grip, and that involuntary early catch is
approximated by scaling the drive impulse from the *instantaneous* heel
rather than latching the heel at square-up.

Cannot be latched without state, because `breakdown()` is stateless for
RK4. Would need the same `on_step` treatment the crew timing got.

### The crew do not lean to correct the boat
**Impact: medium.** `skeleton(t)` and `segment_state(t)` take only time
— neither has any input for the boat's heel. So the crew never shifts
mass laterally in response to being unset, in the physics or the
picture.

They *do* swing across the boat as part of the stroke (a sweep rower is
wound round toward the handle; measured lateral travel is up to 0.50 m),
and that is consistent between the drawn joints and the modelled mass.
What is missing is the *reflex*: `trunk_lean_authority` counts leaning
as an available balance moment, and `PhaseAuthority` includes it in the
recovery limit, but no actual lateral mass movement happens. The moment
arrives entirely through handle heights at the riggers.

Correcting it means giving the kinematics a lean input driven by the
balance command, which changes the crew mass field and therefore the
trim — not a drawing change.

### Crew timing is wired in the trainer but not in `run()`
**Impact: medium.** `CoupledCrew` advances through `FixedStepLoop.on_step`,
which only the trainer uses. Every analysis script calling
`RowingSimulator.run` still gets fixed phase offsets, so published
figures do not include the resync dynamics.

---

### The blade does not know the boat's speed
**Impact: high — it is the cause of the item below.**

`oar_force(t, timing, side)` and `Boat.oar_forces_at(times, sides)` take
stroke phase and side. There is no velocity argument anywhere in the
force path, so a crew makes the same propulsive force at 2 m/s as at 6.
A real blade's force falls as the hull catches up with it — that is what
slip is — so the model's error grows with distance from wherever it was
calibrated, in both directions.

With the force independent of speed, the steady balance `R(v)·v = η·P`
forces `η = (R(v)/P)·v`, and `R` is set by the force rather than by the
speed, so **η comes out proportional to v**. The catalogue was calibrated
where an eight races, so the line passes through one plausible-looking
point. That is why it was never caught.

Measured by `coxswain.validation` on 2026-09-12. Both boats at rate 28,
four power levels each, as `speed (m/s) / η`. *(An earlier table here,
0.366 rising to 0.766, was taken before `a053540` and is deleted rather
than kept, because a known-wrong table on the page is something somebody
will quote.)*

| scale | eight | four |
|---|---|---|
| 0.25 | 2.81 / 0.239 | 2.39 / 0.197 |
| 0.45 | 3.92 / 0.342 | 3.34 / 0.277 |
| 0.70 | 5.02 / 0.432 | 4.30 / 0.352 |
| 0.95 | 5.93 / 0.517 | 5.07 / 0.420 |

η/v is 0.0849–0.0872 on the eight (spread 2.7%) and 0.0819–0.0830 on the
four (1.3%). Fitting η against v and asking **where the line reaches zero
efficiency** gives 0.090 m/s on the eight and 0.003 on the four — 2.0%
and 0.07% of each boat's mean speed. The line goes through the origin on
both hulls. The defect is unchanged and now scored; the *level* moved,
for the reason in the next item.

It explains the split the report now measures: the quasi-steady
evaluator runs **+7% optimistic against the eight and +35% against the
four**, because the four races at 2.70 m/s in the simulator and 3.63 on
the river.

*Next:* a velocity term in the blade force — force from the blade's
motion relative to the water rather than from stroke phase alone. That
invalidates every calibration built on the present path (Holt
validation, the golden trajectory, the published targets), so it is a
decision rather than a patch. `tests/test_blade_velocity.py` pins the
mechanism and carries a strict xfail for the invariant that should hold,
so the fix announces itself.

### The blade model cannot be wired in as an efficiency alone
**Impact: high — it is what "tier 1" was scheduled to be.**

`Boat.blade_model` scales the transmitted force by the instantaneous slip
efficiency instead of the oar's fixed 0.78. It looks like a cheap first
step toward a real blade. **It has no viable operating point.** On the
eight at rate 28, scale 0.45, where the prescribed model settles at
3.92 m/s:

| duration | started at 3.4 | started at 6.5 |
|---|---|---|
| 70 s | 1.88 m/s | 2.17 m/s |
| 250 s | **0.63 m/s** | **0.64 m/s** |

At 70 s it has not converged — which is why `SOURCES.md` §7's recorded
"5.10 to 4.28 m/s" is a transient, not an equilibrium. Run it out and the
boat collapses to a crawl from either direction. Flattening the oar sweep
does not rescue it: 0.63 m/s at flatness 0.00, 0.66 at 0.30, 1.60 at 0.60.

**The mechanism.** Efficiency is evaluated at the instantaneous hull surge,
which swings 56% peak-to-peak with its minimum near 40% of the drive — the
same place the oar force peaks. Over a settled cycle at a mean 3.904 m/s,
force-weighted blade efficiency is **0.628** evaluated at the mean speed
and **0.494** at the instantaneous one, against the **0.780** it replaces.
At peak oar force the hull is doing 2.76 m/s and the efficiency is 0.40.

So thrust is cut to 63%, the boat slows, the surge dip deepens, efficiency
falls again. That is positive feedback, and it runs away because the
wiring supplies **only the destabilising half of the physics**: a real
blade whose slip rises also makes *less force*, and that restoring term
lives in the force model, not in the efficiency factor.

*Consequence for the plan:* tier 1 is not "the blade model, switched on".
It is the slip-quadratic **force**, which needs the oar angle as a dynamic
state. The two phases are not separable and have been merged. Pinned by
`tests/test_blade_tier1.py`.

### The dynamic oar's blade is a quarter less efficient than a real one
**Impact: high — it is the one scorecard target `research` fails.**

Measured on the run (2026-09-13), force-weighted over the drive at the
integrated oar states and the oarlock's instantaneous water speed:

| boat | watts per rower | speed | measured | band [K07] |
|---|---|---|---|---|
| eight | 360 | 5.51 m/s | **0.586** | 0.754–0.816 |
| four | 360 | 4.78 m/s | **0.590** | 0.754–0.816 |

The shape is right and the level is not. Evaluated on the old prescribed
schedule the level rose with speed (0.49 → 0.81 on the eight), which is η ∝ v
at the blade; measured on the dynamic run it is flat and falls slightly with
power. So the defect is gone, and what is left is a level gap.

**An open tension, and tier 2 sharpened it.** The eight and four reach
published race pace at 380 W per rower *with* these blades. A blade a quarter
less efficient than a real one should not do that, so something else is
generous, 380 W is high, or the comparison is not like-for-like. **With the tier
2 blade the boats overshoot:** the four settles at 5.27 m/s against a published
4.5–5.1. So it is not a quirk of a leaky blade — either the drag is generous or
380 W is high for those paces, which is the published-race-power data item.

It has been seen before, independently. SOURCES §47 put the tier 1 slip blade
on the prescribed sweep, in one degree of freedom with nothing fitted, and
predicted 4.10 m/s against 3.82 measured on the water: 7% high. That was a
different test (a prescribed sweep, not the dynamic oar), so it corroborates
the direction rather than the size, and it does not say whether the drag or
the power is the generous one.

Candidates, in the order they are being checked:

1. ~~**Definition.**~~ **Checked and ruled out** (2026-09-13). The scored
   figure is the instantaneous `1 − |slip|/|blade speed|`, force-weighted;
   Kleshnev's may be energetic — propulsive power over the power put into the
   blade. Both measured on the same settled runs:

   | run | instantaneous | energetic |
   |---|---|---|
   | eight, 360 W, 5.51 m/s | 0.586 | 0.622 |
   | four, 360 W, 4.84 m/s | 0.590 | 0.624 |
   | eight, 80 W, 2.96 m/s | 0.615 | 0.690 |

   The definition is worth about +0.035 at race pace — a sixth of the gap —
   and leaves the level well below 0.754 either way. So the gap is physics,
   and the next candidate is the first one that is.
2. **No lift.** [CR06] Model 1 is a pure normal force. Sliasas & Tullis find
   propulsion lift-dominated in the first half of the drive, exactly where
   this model's blade is least efficient. That is tier 2. **Measured
   (2026-09-13): it closes a little under half the gap.** With lift and drag
   on the angle of attack, on the provisional [CG06a] coefficients, the level
   is 0.663 on the eight and 0.669 on the four at 380 W, against 0.586 and
   0.592 — still below 0.754, so this is part of the cause and not all of it.
3. **The pull shape.** Kleshnev's handle-force curve against drive progress,
   applied here as a function of oar angle.
4. **The reflected inertia**, derived from the ergometer kinematics and
   clamped at both ends where it diverges.

Pinned by `test_research_passes_the_defect_targets_shipped_fails`, which
asserts the fail and the value, so a fix announces itself.

### The research report has no masters eight to simulate
**Impact: medium — half the fleet is missing from one page.**

The offline report runs `--physics research`, and the dynamic oar drives a
boat at a stated handle power per rower. The four has one: its own erg
watts. The masters eight does not. `MASTERS_POWER = 0.658` is a force scale,
and turning a scale into watts means `mean_handle_power` — which dots the
oarlock force with the handle velocity, not a conjugate pair under the ideal
lever, and already an open question. Coaching material gives ranges, not
on-water measurements.

So under research the eight's steering run and settled speed are **left off
the page, and the page says why**. `reference_eight` and `quasi_steady_gap`
refuse a dynamic-oar profile rather than guess. The eight's racing lines are
still priced, because the route evaluator is quasi-steady and never runs the
simulator.

*Next:* a sourced masters-eight handle power — an instrumented masters crew
or a published power meter study — and the eight goes back on the page.
Pinned by `test_the_masters_eight_is_refused_under_the_dynamic_oar`.

### The steering table's caption is transcribed, and wrong on a research page
**Impact: low — a caption, but it contradicts the table beside it.**

`build_report`'s "Steering the real boat" caption quotes numbers as prose:
"six seconds is 29 m for the eight and 16 m for the four", "12.88 m rms",
"holds the line to 0.80 m, better than the eight". They were true of one
shipped run and are not recomputed. On a `--physics research` page there is
no eight in the table at all, and the four's predictive run reads 1.10 m rms,
not 0.80. It is the same failure the rest of the page was rebuilt to avoid:
nothing on it should be transcribed.

*Next:* compute the look-ahead distances and the rms figures from the run's
own rows, or drop the numbers from the caption.

### The full model cannot match on-water drive time and published pace at once
**Impact: high — it undoes a result the programme had called its best.**

On the full 6-DOF hull, at a stated power and the measured front-loaded pull,
the eight's drive fraction at rate 32 is 0.467 at 380 W (published pace, 5.56
m/s) against 0.395 measured on the water. Pulling harder shortens the drive —
0.420 at 500 W, 0.377 at 650 W — but at 650 W the boat is doing 6.78 m/s, far
past the published 5.0–5.6. **No single power gives both.** The four at rate 32
and 380 W is worse, 0.510.

The earlier claim that the unfitted drive fraction "lands on the water" was
measured on the oar balance alone, at a fixed 4.85 m/s under a constant pull.
It is true of that unit and false of the full model, and has been corrected in
PHYSICS_PROGRAMME rather than left standing.

This entry first called it most likely the same physics as the
blade-efficiency gap above. **That now looks wrong.** On the oar balance alone —
eight at rate 28, 4.85 m/s, the torque 380 W needs — a draft tier 2 blade on the
provisional [CG06a] coefficients raises energetic efficiency from 0.624 to 0.735
and leaves the drive fraction exactly where it was, 0.399. More grip closes the
efficiency gap without shortening the drive, so these look like two causes.
[HF09] measured pairs, so the comparison is not like-for-like for an eight or a
four.

**Confirmed on the full hull, with a qualification** (2026-09-13). Tier 2
shortens the drive by about 5% — the eight from 0.400 to 0.379 at rate 28, the
four from 0.510 to 0.486 at rate 32 — but only because the boat is faster; at a
fixed speed it did not move the drive at all. The four is still 0.486 against
0.395 on the water, so most of the gap remains. And tier 2 makes the four too
fast at 380 W (5.27 m/s against a published 4.5–5.1), which is recorded in the
blade-efficiency item above.

*Next:* the drive time needs its own cause — candidates are the pull shape, which is an erg-fitted
curve applied by angle, and the reflected inertia clamped at the catch, which
each added about 0.03 of the stroke on the oar alone.

### Blade added mass is large, and the catch decides what it does
**Impact: medium now, high later — its size is established; its effect on speed is not readable yet.**

Built as a study, `DynamicOarSimulator(blade_added_mass="patton")`, off by
default. A blade accelerating normal to its face drags water with it:
Patton's aspect-ratio-2 plate in unbounded fluid, quoted by [G19],
`m_h = 0.84 (πρ/4) l_a l_b²` — **21.4 kg** on a sweep Big Blade (0.52 × 0.25 m),
**13.1 kg** on a scull (0.43 × 0.215 m). An upper bound: no free surface, no
entrainment growth, a rectangular blade. Because the blade's normal
acceleration contains the hull's own acceleration at the blade, hull and oars
are solved together — the generalised mass matrix augmented with the oar
angles, still symmetric positive definite. Entry and exit take design choice
(ii): added mass on while the blade is in, with no impulse at the switch, and
the momentum that leaves out measured.

*Measured at 380 W* (blade-centre baseline → with added mass):

| boat, rate | speed (m/s) | drive fraction | blade efficiency |
|---|---|---|---|
| eight, 28 | 5.530 → 5.612 | 0.376 → 0.426 | 0.559 → 0.449 |
| eight, 32 | 5.461 → 5.518 | 0.440 → 0.503 | 0.558 → 0.428 |
| coxed four, 32 | 4.829 → 4.879 | 0.481 → 0.550 | 0.566 → 0.462 |
| single, 30 | 4.171 → 4.096 | 0.525 → 0.616 | 0.614 → 0.553 |

**The speeds are not results.** The largest added-mass force is at the catch,
t = 0: +426 N on the eight, against the parked blade's water-driven load at
that instant. And the momentum the no-impulse choice leaves out is large:
**63 N·s per blade at entry** on the eight (≈ 510 N·s across the crew every
stroke) and −25 N·s at exit; 56 and −23 on the four, 21 and −13 on the single.
That is the order of a blade's whole drive impulse, so the sweep boats running
*faster* with a much less efficient blade is most likely the hull not paying
for the water set moving at entry.

*What stands:* the force is hundreds of newtons, comparable to drag; the
heavier oar-plus-water sweeps more slowly, lengthening the drive by 0.05–0.09
of the stroke; and the entry momentum is sized.

*Blocked on the same thing as [CR06]'s release rule:* a physical catch. A real
blade enters with its normal velocity near zero ([CR06] eq. 16), so almost no
water momentum is created at entry; ours enters parked in a moving boat at
`w_n ≈ 3 m/s`. It needs the rower carrying the oar into the drive (phase 4.3),
or `m_h` ramped in with immersion, which needs a vertical oar-angle trace.

*On the sweep catch, 2026-09-14.* The block was the catch, and the sweep catch
removes it: the blade enters at zero normal velocity, so the water it sets
moving at entry is gone by construction. Added mass is now allowed with
`catch="sweep"`; an oar still in the air carries an identity row with the
sweep's own acceleration and no added mass. Measured at equal power (380 W
including entry work, torque rescaled three times, 12 strokes, last 4), sweep
catch without and with Patton added mass:

| boat, rate | speed, sweep → + added mass | drive fraction | blade efficiency | surge swing | momentum left out per blade, entry / exit |
|---|---|---|---|---|---|
| eight, 28 | 5.866 → **5.969** m/s (+1.8%) | 0.344 → 0.347 | 0.712 → **0.754** | 40.6 → 39.2% | -1.1 / -24.5 N·s |
| eight, 32 | 5.871 → **5.977** m/s (+1.8%) | 0.393 → 0.397 | 0.727 → **0.769** | 44.9 → 43.5% | -2.5 / -23.4 N·s |
| coxed four, 32 | 5.153 → **5.258** m/s (+2.0%) | 0.423 → 0.427 | 0.699 → **0.743** | 48.4 → 46.8% | -0.6 / -22.7 N·s |
| single, 30 | 4.379 → **4.425** m/s (+1.1%) | 0.456 → 0.463 | 0.680 → **0.726** | 59.4 → 58.0% | -1.1 / -12.1 N·s |

**Entry momentum left out falls to 2.5 N·s per blade at most**, against 63 on the
parked catch -- the one-step lag between the entry test and the step. **Exit
momentum left out is still 12–24 N·s per blade**: release is at the finish
angle with the blade still driving, so the water set moving there drops out
with no impulse. With the entry artefact gone the speeds are readable within
that bound: added mass makes the boats 1.1–2.0% faster and raises blade
efficiency to 0.726–0.769 -- inside Kleshnev's 0.754–0.816 on eight, 28, eight, 32. Still an upper bound on `m_h` (unbounded fluid, no
entrainment growth) and a study, off by default and not in any profile.

*Still open.* The exit: it needs the oar to turn round after release, which is
the recovery defect blocked on a measured deceleration law.

### Shallow water: three chosen numbers price most of the Charles, and the quasi-steady use is the bigger error
**Impact: high for the Charles — it is the regime the course is rowed in.**

Wave resistance is Michell's integral for **deep** water, multiplied by a
depth factor (`coxswain/hydro/shallow.py`): Schlichting's matched-speed
construction up to depth Froude 0.92, then a **chosen** smooth blend to a
**chosen** cap of 3.0 at `Fr_h = 1`, then a **chosen** relaxation back to 1 by
`Fr_h = 1.6`. None of the three is measured for a shell; the module says so.

*Measured on the surveyed channel* (2016–17 isobaths, 63,637 rowable cells of
36 m²; depth median 3.80 m, quartiles 2.60 and 5.01 m, 5th percentile 1.64 m):

| boat speed | median `Fr_h` | Schlichting (0.5–0.92) | chosen blend to cap (0.92–1.0) | chosen relaxation (1.0–1.6) |
|---|---|---|---|---|
| 4.0 m/s | 0.66 | 81.8% | 4.3% | 4.9% |
| 5.0 m/s | 0.82 | 65.1% | 10.9% | 23.8% |
| 5.5 m/s | 0.90 | 52.9% | 10.3% | 36.7% |
| 6.0 m/s | 0.98 | 37.4% | 14.6% | 45.5% |

So at race speed 35–60% of the rowable water is priced by the three chosen
numbers rather than by any calculation. **A finite-depth Michell integral is
worth doing**: it replaces all three with the hull's offsets and the depth.

*But it is not the largest error there.* Day et al. (2011) [D11], full text read:
an unsteady inviscid thin-ship code for any depth (Doctors, Day & Clelland
2010, J. Ship Res. 54(2)) matched a towing tank on an oscillating Wigley hull
at `Fr_h = 1.0` for low oscillation frequency, `τ = Uω/g ≤ 0.3`, while the
**quasi-steady** approach — steady wave resistance at the instantaneous speed,
which is what this model does — was "extremely poor", dramatically
underestimating the peaks. At `τ > 0.7` measured resistance exceeded even the
unsteady prediction, attributed to unsteady viscous effects (transition
triggered at peak deceleration, confirmed with hot films on a real single).
A boat at 5 m/s rating about 32 has `τ ≈ 1.7`, past both.

*In order of what it removes:* (1) steady finite-depth Michell — removes the
three chosen numbers; (2) unsteady wave resistance with memory — what [D11]
validated at low frequency; (3) unsteady viscous resistance — no validated
model yet, flagged. The ledger records the same.

*Sourcing for (1), 2026-09-13.* Two sources read in full (SOURCES §6).
Lazauskas's thesis [LV09] prints the whole **deep-water** thin-ship chain:
Havelock source, source strength `2U Y_ξ`, hull transform with depth weight
`e^{kζ}`, free-wave spectrum, and `R_W = (π/2)ρU²∫|A|²cos³θ dθ`. That is the
integral `michell.py` already implements in Tuck's form, so it gives an
independent check. Li & Ellingsen (2016) [LE16] prints the **finite-depth**
wave pieces for a pressure source: the dispersion relation, the per-angle
root that replaces `k₀ sec²θ`, the group-velocity denominator, the Heaviside
cut-offs and the critical speed `√(gh)`. **Missing: the finite-depth Havelock
source**, which sets the hull's depth weight and the spectrum's prefactor
once a bed is present. Candidates: Wehausen & Laitone (1960), Srettensky
(1936), Scragg & Nelson (1993). **Not implemented.** That first check is done:
the formula agrees analytically and the shipped quadrature reads about 3%
high ("Michell's sum reads about 3% high", above).

*Sourced and built, 2026-09-13* — supersedes "Missing" above. Wehausen &
Laitone (1960) [WL60] print Sretenskii's (1937) finite-depth thin-ship
integral, eq. (20.69), with an erratum to its coefficient. Built as
`FiniteDepthMichell` (`coxswain/hydro/finite_depth_michell.py`), now the research
profiles' wave model. It passes the three checks: the deep limit reproduces `MichellWave`
to 1.00000; at `Fr_h ≤ 0.5` it is within 0.14% of deep water; and resistance
peaks at `Fr_h` 0.98–0.99 at all four Charles depth quantiles.

*What it says about `shallow.py`:* below `Fr_h ≈ 0.9` Schlichting's
construction agrees to a few percent. **The three chosen numbers overstate
wave drag by up to about 2×:** the cap of 3.0 where the integral gives 1.4–2.3,
and a relaxation still at 2–3 above critical where the integral is already
1.0–1.4. On the eight at 2.60 m and 5.50 m/s that is 82.7 N of wave drag against
39.7 N, about 43 N of a total near 345 N. In 1.64 m of water depth moves the
humps and hollows, which no multiplying factor can represent (SOURCES §6).

*Wired into the research model, 2026-09-13.* `research` and `learned` carry
`wave="sretenskii"`; `shipped` is untouched. The boat gets a depth-aware table
(within 0.64% of the direct integral between its depth rows) and
`hull_resistance` uses it instead of the factor. On the Charles at constant
power, quasi-steady, the eight's time lost to depth over 4,826 m:

| deep-water speed | chosen factor | Sretenskii | research − shipped |
|---|---|---|---|
| 4.5 m/s | +6.8 s | +1.1 s | −7.0 s |
| 5.0 m/s | +14.5 s | +13.7 s | −1.8 s |
| 5.5 m/s | +28.3 s | +13.6 s | −15.5 s |
| 6.0 m/s | +38.1 s | +5.5 s | −33.4 s |

The validation scorecard (deep water, so the trapezoid correction only) moves by
under 1% everywhere, speed per watt +0.10 to +0.13%, and no target changes
status (SOURCES §6).

*The optimisers too, 2026-09-13.* A boat whose wave table knows depth now
takes it everywhere depth enters:

- `RouteEvaluator` solves the boat's own power balance
  `(R_deep(v) + W_h(v) − W_deep(v)) v = R_deep(v_ref) v_ref` for its depth-to-speed
  table, keeping the never-faster-in-shallower-water guard;
- `CoursePacing(wave_table=...)` adds `W_h − W_deep` to the deep resistance
  instead of multiplying it by the factor; the pacing scripts now pass
  `boat.wave_table`, which a shipped boat's plain table leaves without effect;
- the CasADi models (`SixDofModel`, `StrokeResolvedModel`) get a
  `WaveSurface`: the table sampled onto cubic B-splines, one in speed and log
  depth below `Fr_h` 0.8 and one in `R/U²` against `Fr_h` and log depth through
  critical, blended. Worst error against the table 1.00% (0.58 N at `Fr_h`
  1.00 in 1.5 m); a speed-only surface had missed the peak by 17–26%. In deep
  water this replaces the constant wave coefficient with Michell's integral.

*What moved, and why.* Pacing speed for the eight at the power that holds
5.2 m/s in deep water, split into the two changes (m/s):

| depth | A: factor × total | B: factor × wave | C: Sretenskii | A → B | B → C |
|---|---|---|---|---|---|
| 1.5 m | 3.752 | 5.103 | 5.149 | +1.351 | +0.046 |
| 2.0 m | 4.193 | 5.006 | 5.141 | +0.813 | +0.135 |
| 2.5 m | 4.538 | 4.981 | 5.115 | +0.444 | +0.134 |
| 3.0 m | 4.762 | 5.106 | 5.108 | +0.344 | +0.001 |
| 4.0 m | 5.006 | 5.178 | 5.175 | +0.172 | −0.003 |
| 5.0 m | 5.111 | 5.192 | 5.191 | +0.080 | −0.001 |

**Most of it was not the wave model.** The shipped optimisers scale the whole
resistance by the shallow factor, where the simulator scales only the wave term
(A → B); Sretenskii then adds at most 0.16 m/s at these speeds (B → C), and its
large effect is supercritical. See "The shipped route and pacing optimisers
charge depth to the whole resistance".

**Still open.** Steady resistance at the instantaneous speed, which [D11] found is
the larger error near critical. The first research table for a hull takes about
two minutes to build on the production grid (124 s for a 5.2 m/s route table),
cached for the process after that.

### The dynamic oar's drive is started by the water, not the rower
**Impact: now low — resolved for `research` and `learned`, which catch by the sweep since 2026-09-13; it remains only in the rest catch, which no profile uses.**
The item is kept open for that default and for the stuck-oar finding pinned
in `tests/test_release_rule.py`. The catch's *remaining* miss against measured
slips is a different defect: the body on the oar balance (the Holt singles
item under validation gaps).

Found by wiring [CR06]'s release rule in as a study (`release="slip"`: the
blade carries load only while its normal velocity is driving, so it never
brakes). Every settled run collapsed — the eight to 1.69 m/s, the single to
0.84 — with the drive never finishing.

The cause, traced on one stroke of the eight at 380 W:

- the oar is **reset to rest at the catch**, and the rower's pull shape is
  **exactly zero there** (it reaches 0.41 an eighth of the way through the
  sweep and peaks at 0.995);
- so under the default rule **the drive is started by the water**: the boat
  carries the parked blade, the non-driving slip loads it towards the finish
  at **−829 N with the handle torque at 0.0**, and the rower's torque is
  still only 21 N·m at 0.06 s. The single: −185 N at the catch;
- under [CR06]'s rule that load is gone and nothing moves the oar: it sits at
  56° (eight) or 65° (single) for the whole stroke and the boat coasts.

The same measurement showed the default rule **braking at the finish**:
+75 N on the eight and +109 N on the single as the stroke ends, with 35 and
43 drive samples carrying a braking load.

The switch itself is correct and unit-tested — a released blade loads
neither the hull nor the oar, and a driving blade is loaded exactly as
before — and the stuck oar is pinned as a finding in
`tests/test_release_rule.py`. It is not usable until the oar enters the
drive already moving, as a real oar does and as [CR06]'s prescribed body
guarantees. That needs the rower to carry it through the recovery (phase
4.3); a non-zero pull at the catch would be an unsourced number.

*Addressed as a study, 2026-09-13: `catch="sweep"`.* [CR06] section 2.5 starts
the drive when the blade's normal velocity through the water is zero (their
eq. 16), and their oar follows the body until then. The sweep catch does the
same with the prescribed sweep: the oar is put exactly on it after every step
with its blade out, and is handed to the torque drive on the step eq. 16
holds, carrying the sweep's angle and rate. No number is chosen. The rest
catch stays the default and the stuck-oar finding stays pinned for it.

Two things found before the numbers could be read:

- **The release rule never bites.** With the blade entering already moving, 0
  blade-in samples had non-driving slip on a settled stroke of any boat, so
  `release="slip"` and `"angle"` give identical runs. The finish braking the
  rule was built to remove came from the parked catch.
- **The sweep carries energy the torque never did.** At entry the oar and the
  rower's reflected inertia hold `½ I φ̇²`: 71 J a seat on the eight at rate
  28, 33 W a rower, 8.9% of the handle power (11.1% at rate 32, 8.3% on the
  four, 3.6% on the single). It is the rower's work, as [CR06]'s body supplies
  it, and `run_strokes` now counts it (`StrokeRecord.entry_work`). The first
  comparison left it out, and read the eight at 6.01 m/s: an unequal-power
  number, withdrawn.

*At equal power*, peak torque rescaled until handle power including entry work
is 380 W, 12 strokes, last 4:

| boat, rate | speed, rest → sweep | drive fraction | blade efficiency | surge swing | entry work | entry |
|---|---|---|---|---|---|---|
| eight, 28 | 5.533 → **5.866** m/s (+6.0%) | 0.376 → 0.344 | 0.559 → **0.712** | 46.4 → 40.6% | 31.9 W | 6.0° past catch, −1.29 rad/s |
| eight, 32 | 5.464 → **5.871** (+7.4%) | 0.440 → 0.393 | 0.558 → **0.727** | 51.4 → 44.9% | 38.8 W | 4.9°, −1.27 rad/s |
| coxed four, 32 | 4.831 → **5.152** (+6.6%) | 0.480 → 0.423 | 0.566 → **0.699** | 54.5 → 48.4% | 30.2 W | 3.6°, −1.10 rad/s |
| single, 30 | 4.172 → **4.379** (+5.0%) | 0.525 → 0.456 | 0.614 → **0.680** | 64.7 → 59.4% | 13.5 W | 2.1°, −0.91 rad/s |

The parked blade's braking at the catch was costing 5–7% of boat speed and
most of the blade-efficiency shortfall: the level target (0.754–0.816,
Kleshnev) still fails, but 0.56 has become 0.70–0.73. Surge swing falls and
stays in its band.

*The validation scorecard on it, 2026-09-13.* The scorecard's own operating
points (80–360 W, 16 strokes), its `Settled` records and drag pricing, and its
own `measure()`; only the settle differs: `catch="sweep"`, and peak torque
rescaled three times per point so handle power including entry work is the
stated wattage. Research profile, tier 1, rate 28, rest → sweep:

| target | eight | coxed four | status |
|---|---|---|---|
| blade-efficiency zero crossing (× mean speed) | 1.22 → **12.26** | 1.29 → **14.58** | pass |
| spread of η/v | 0.356 → 0.523 | 0.362 → 0.534 | pass |
| speed per watt (m/s per 100 W) | 0.18846 → 0.20018 (+6.2%) | 0.32912 → 0.34506 (+4.8%) | pass |
| surge swing, fastest point | 47.2 → 41.3% | 50.3 → 44.9% | pass (30–60) |
| blade efficiency level, fastest point | 0.559 → **0.714** | 0.566 → **0.681** | fail (0.754–0.816) |

No target changes status. The signature of the blade defect this programme
opened on -- efficiency proportional to speed, a line through the origin -- is
gone: the fitted line now reaches zero at twelve times the mean speed. Blade
efficiency falls gently with power instead, on the eight from **0.770 at 80 W,
inside Kleshnev's band**, to 0.714 at 360 W, where the level target is scored.

*Power matching, 2026-09-13.* `PhysicsProfile.catch` names a profile's catch,
`"rest"` for every profile. `DynamicOarSimulator.torque_for_power` turns a
wattage into peak torque for either rule:

- **rest catch:** the closed form `peak_torque_for_power`, exactly; the pull is
  a function of angle over a fixed arc, so its work does not depend on the run;
- **sweep catch:** matched on a straight settle -- start from the closed form,
  12 strokes, rescale by `watts / measured handle power` three times -- and
  cached per boat configuration (name, timing, mass, power scales, rig
  geometry, stamp, wave model, depth, watts, blade law), because a trajectory
  fit builds many simulators on one boat. On the single at 300 W the matched
  torque settles within 1.5% of the stated watts, and it is below the closed
  form: the rower pays for the energy carried in.

`simulator_for` and the scorecard's `settle_dynamic` both take the profile's
catch and this torque, so a sweep-catch profile builds its crews at matched
power (`tests/test_torque_for_power.py`). With every profile on `"rest"`,
nothing that runs a profile has changed.

*Switched, 2026-09-13: `research` and `learned` catch by the sweep.* Every
research crew is now built at matched power. Scorecard, research profile, rate
28, the scorecard's own settles (rest → sweep):

| target | tier 1 eight | tier 1 four | tier 2 eight | tier 2 four |
|---|---|---|---|---|
| zero crossing (× mean speed) | 1.22 → 9.51 | 1.29 → 15.75 | 2.10 → 6.12 | 2.35 → 6.66 |
| speed per watt | 0.18846 → 0.20008 | 0.32912 → 0.34510 | 0.20160 → 0.21055 | 0.35175 → 0.36354 |
| surge swing | 47.2 → 41.3% | 50.3 → 44.9% | 42.8 → 38.9% | 46.4 → 42.3% |
| blade efficiency level | 0.559 → **0.714** fail | 0.566 → **0.681** fail | 0.630 → **0.826** fail (above) | 0.639 → **0.795 pass** |

The first pass of the blade-efficiency level target on any boat: tier 2 on the
four, inside Kleshnev's 0.754–0.816. Tier 2 on the eight overshoots the band,
tier 1 still falls short on both, and tier 2 rests on provisional coefficients.

*What moved with it.* The report's research four at its own roster watts
(131.2 W a rower) settles at 3.638 m/s against 3.448 under the rest catch
(+5.5%), swing 52.4 → 47.4%. **At the stated 380 W the eight and the four now
run faster than their published race-pace bands** -- eight 5.87 m/s at rate 32
(band 5.0–5.6), four 5.15 (band 4.5–5.1) -- where the single, 4.38, stays in its
4.1–4.7. That is the recorded tension sharpened, not a new defect: 380 W a rower
is a stated operating point with no source, and a more efficient blade needs
less of it for the same pace. It stays open until race power per boat class is
sourced. Tests that pinned the rest-catch torque or blade-efficiency level were
re-pinned with dated notes, not loosened.

*Race power, sourced in part, 2026-09-14.* Kleshnev's measured power–rate
regressions [K00] put elite men sweep at **about 358 W at the handle at rate
32** -- total power 418 W, less the 16.8% that goes through the footstretcher,
which `handle_watts` does not count. So 380 W is close to elite men's sweep
power, not club power, and the club race-pace bands the eight and four now
exceed were the mismatch. BioRow's modelled targets [BR26], scaled to 380 W by
the cube-root law, give the eight 5.87 m/s if their power is handle power and
6.19 if it is total, against the research model's 5.87; a coxless four 5.49
against the model's coxed four at 5.15; a single 4.51 against 4.38. Their rates
(38–40) and masses are not the model's (30–32), so this is a range check, not a
match, and neither source is a measured race. Holt et al.'s measured races
[H20] are the like-for-like check for singles and pairs.

*A measured target, found 2026-09-13, and the model misses it.* [H20]
Table 1 gives catch and finish slip, the gate angle rowed while gate force is
below 196 N at the catch and 98 N at the finish (Coker 2010; SOURCES). Values:
M1x 7.7 / 14.1°, W1x 9.7 / 18.1°, M2- 3.7 / 8.5°, W2- 5.6 / 8.5°. The model's
gate force was taken per oarlock as handle force (|τ| / inboard) plus blade
force, over a settled stroke at Holt's conditions (`holt_slips.py`). The
research profile, pairs on the double:

| class | catch slip, normal / axis force (Holt) | finish slip, normal / axis (Holt) | arc (Holt) | peak gate force, normal |
|---|---|---|---|---|
| M1x | **17.0 / 22.1°** (7.7°) | **20.3 / 21.3°** (14.1°) | 110° (105.4°) | 458 N |
| W1x | **22.5 / 27.9°** (9.7°) | **24.7 / 25.6°** (18.1°) | 110° (106.0°) | 319 N |
| M2- | **17.5 / 22.0°** (3.7°) | **20.2 / 21.3°** (8.5°) | 110° (82.0°) | 468 N |
| W2- | **23.1 / 28.5°** (5.6°) | **25.2 / 25.2°** (8.5°) | 110° (80.4°) | 318 N |

Which force the thresholds apply to is unsettled: normal to the shaft
(likely, via Kleshnev's convention) or along the boat's axis (Holt's power
description). Under either, **the model's catch slip is 2–4× Holt's on every
class and its finish slip 1.4–3×**, as a share of the arc too. Force builds
too slowly after the catch and fades too early before the finish. The blade
enters 1.4–2.1° past the catch, so the slip is not the entry angle. It is how
slowly load comes on after entry. Two candidates, not yet separated:

- The fitted pull shape u^1.4852 (1 − u)^2.2278. It is zero at both ends
  and was fitted to two Kleshnev points in the middle of the drive, not to
  its ends.
- The blade entering at zero normal velocity, and so at zero load, by
  [CR06]'s rule.

The pairs row a sculling arc (110° against 82°), so their slips are not like
for like. The singles are.

*Separated on the singles (`holt_slips_split.py`): it is the pull shape.*
Force normal to the shaft, per scull:

| | handle force first > 196 N | blade force alone | pull shape at Holt's catch slip | shape reaches half peak | peak handle / blade / sum |
|---|---|---|---|---|---|
| M1x | 20.9° past the catch | never (peak 154 N) | **0.20** of peak at 7.7° | 16.3° | 305 / 154 / 458 N |
| W1x | 32.7° | never (peak 106 N) | **0.27** of peak at 9.7° | 16.3° | 214 / 106 / 319 N |

The blade carries about a third of the pin load throughout, as the lever
requires, so it cannot pass the threshold alone and does not delay the sum.
The pull shape u^1.4852 (1 − u)^2.2278 is what is slow: at Holt's measured
catch slip it gives a fifth to a quarter of peak pull. The peak gate force
(458 N against Holt's 497) and the peak position (40% of the drive against
Holt's −20.1° peak-force angle, 39.8% of the arc) both agree. **The fit is
right in the middle and wrong at both ends**, which is where its two Kleshnev
points gave it nothing.

*Can the two exponents simply be refit to Holt? No* (no simulation; the
shape evaluated on Holt's own arcs). The slip thresholds were taken as a
share of Holt's peak gate force: per gate for singles, and pairs' forces
halved on the reading that they are summed over the two rowers, which is
flagged.

| class | current shape's slips on Holt's arc (Holt) | refit: peak position + catch slip pinned | finish slip that refit predicts (Holt) |
|---|---|---|---|
| M1x | 12.7 / 18.9° (7.7 / 14.1°) | a = 0.894, b = 1.357 | 11.3° (14.1°) |
| W1x | 16.5 / 22.2° (9.7 / 18.1°) | a = 1.030, b = 2.248 | 27.1° (18.1°) |
| M2- | 10.1 / 14.9° (3.7 / 8.5°) | a = 0.588, b = 0.818 | 3.7° (8.5°) |
| W2- | 13.4 / 17.5° (5.6 / 8.5°) | a = 0.573, b = 0.988 | 8.4° (8.5°) |

1. **The current shape is too slow at both ends on every class, even on
   Holt's own arcs.** This is independent of the simulator.
2. **A two-exponent refit is not consistent.** Pinning the peak and the
   catch leaves the finish off by 0.1–9°, and the exponents fall below 1, an
   infinitely steep catch.

§68 already names what the family lacks: **curve width**, the degree of
freedom Warmenhoven et al.'s review ties to elite mean-to-peak ratios. It
also records that skilled pairs **balance by timing asymmetry**, stroke
peaking earlier and bow loading the finish. A single shared shape cannot do
that, which fits the steered sweep pair's leftover 5–6° sideslip above.

*The time-domain descriptors agree* (`holt_force_curve.py`; singles at
Holt's conditions; gate force per scull, normal to the shaft; drive from the
catch to the oar's finish):

| | M1x model (Holt) | W1x model (Holt) |
|---|---|---|
| catch to peak force | **0.54 s** (0.43) | **0.62 s** (0.39) |
| rate of force development, catch to peak | 842 N/s (960) | 516 N/s (760) |
| peak / mean force | **2.29** (1.90) | **2.23** (1.87) |
| catch to minimum boat speed | 0.24 s (0.14) | 0.26 s (0.12) |
| mean / peak gate force | 199 / 458 N (261 / 497) | 144 / 319 N (199 / 371) |
| drive duration | 0.96 s | 1.07 s |

At equal power the model pulls **24–28% less mean force over a longer
drive**. The drive is 0.555 and 0.585 of the stroke, where Holt report no drive
time and TRACKING's drive-fraction check already fails. Its curve is **too
peaked** (peak/mean 2.2–2.3 against 1.9) and **reaches its peak too late**.
Holt's peak/mean ratio is exactly the curve-width measure §68 says this shape
family cannot vary. Widening the Beta curve is the same two-exponent refit
above, and that failed, so the family is the limit. It needs end slopes set
independently of its peak and width, and a measured curve to fit them.

The pull shape and the drive's length are coupled: a pull that comes on
sooner would also shorten the drive. So the drive-fraction item and this one
should be refit together, not one after the other.

*A measured curve was already in hand: [CR06] Fig. 3* (SOURCES). Its
handle force perpendicular to the oar, against the measured oar angle, is a
women's single's full drive with both ends, at rate 30.9. Each shape family
was fitted to it by least squares, then evaluated on Holt's arcs at Holt's
thresholds, a second and independent dataset:

| shape | rms error to [CR06] | peak u | slips on Holt's arcs, catch / finish: M1x · W1x · M2- · W2- |
|---|---|---|---|
| `DRIVE_SHAPE` (1.4852, 2.2278) | 0.082 | 0.400 | 12.7/18.9 · 16.5/22.2 · 10.1/14.9 · 13.4/17.5 |
| Beta refit (1.122, 1.635) | **0.045** | 0.407 | **10.1/13.8 · 14.0/16.9 · 8.1/10.9 · 11.5/13.5** |
| Beta ending at u = e | 0.045 (e = 0.995) | 0.408 | same as the Beta refit |
| u^a (1 − u^c)^b, four parameters | 0.041 | 0.423 | 9.5/14.8 · 13.9/17.8 · 7.6/11.7 · 11.5/14.1 |
| Holt, measured | — | 0.31–0.42 | 7.7/14.1 · 9.7/18.1 · 3.7/8.5 · 5.6/8.5 |

The measured [CR06] curve itself, on a 105° arc, gives 8.4 / 12.6° at the
M1x thresholds, within 1.5° of Holt. It gives 14.2 / 15.2° at the W1x
thresholds, a slower catch than Holt's women by 4.5°. That is athlete
scatter between two measurements, and both sit well inside the model's miss.

- **The refit moves every class 2–6° toward Holt** and halves the error to
  the measured curve.
- **The richer families add nothing** that justifies extra parameters. The
  Beta family was not the limit; its fit to two mid-drive points was.
- **The pairs' remaining catch miss** (8.1 against 3.7°) is on a sweep arc
  the scull curve does not describe.

This supersedes "the family is the limit" above.

*Built as a study and measured (`holt_cr06_shape.py`).* `OarForceProfile(shape="cr06")`
takes `CR06_DRIVE_SHAPE = (1.1218, 1.6350)`; the `"kleshnev"` default is
unchanged bit for bit, and an unknown shape name is refused. Research profile
at Holt's conditions, torque matched at equal power under each shape (a
first run was invalid; see Fixed):

| class | speed error | catch slip (Holt) | finish slip (Holt) | peak / mean (Holt) | catch to peak (Holt) | blade eff. |
|---|---|---|---|---|---|---|
| M1x | −8.5 → −8.2% | 17.0 → 16.2° (7.7) | **20.3 → 16.2°** (14.1) | **2.29 → 2.03** (1.90) | 0.54 → 0.55 s (0.43) | 0.710 → 0.714 |
| W1x | −11.5 → −11.3% | 22.5 → 22.9° (9.7) | **24.7 → 21.0°** (18.1) | **2.23 → 1.97** (1.87) | 0.62 → 0.62 s (0.39) | 0.721 → 0.725 |
| M2- | +0.0 → +0.3% | 17.5 → 16.3° (3.7) | **20.2 → 16.1°** (8.5) | **2.31 → 2.05** (1.88) | 0.49 → 0.48 s (0.36) | 0.740 → 0.745 |
| W2- | −0.4 → −0.3% | 23.1 → 23.8° (5.6) | **25.2 → 20.5°** (8.5) | **2.23 → 1.98** (1.89) | 0.55 → 0.55 s (0.36) | 0.751 → 0.755 |

- **The finish and the curve's width are fixed or much closer.** Finish slip
  is down 3.7–4.7°, and peak/mean is now within 0.1–0.2 of Holt.
- **Speed and swing barely move** (≤ 0.3 points, ≤ 2% swing), so the shape
  was not what made the singles slow.
- **The catch slip and the time to peak force do not move** (±1°, ±0.01 s).
  That contradicts the shape-only evaluation, which predicted a 2–3° catch
  gain. So at the catch it is **not only the shape**. The dynamics absorb the
  faster rise.
  - Early in the drive the gate force is almost all handle force.
  - The blade builds load from zero slip (∝ slip²).
  - The oar's balance carries the body's reflected inertia, about 93 kg·m²
    at the catch.

  Both hold blade load far below the quasi-static lever share, where
  handle + blade = handle × (1 + r_h / ℓ). *Next diagnosis:* the torque split
  over the first 20° of the drive, into I θ̈, blade moment and handle.

The `"cr06"` shape stays a **study**: it improves two of Holt's descriptors
and harms none, but it rests on one athlete and leaves the catch
unexplained, so no profile adopts it yet.

*The catch, decomposed (`holt_catch_balance.py`, M1x at 334 W).* The seat
balance was split over the first 30° of the settled drive. Residual under
9 N·m throughout.

| past the catch | n·τ | blade moment | I_seat·θ̈ | ½ I′ θ̇² | I_seat | oar rate |
|---|---|---|---|---|---|---|
| 2.8° | −25.8 N·m | 1.4 | −68.1 | 34.9 | 62.5 kg·m² | 50 °/s |
| 7.7° | −107.6 | 29.8 | −104.6 | 26.8 | 56.9 | 57 °/s |
| 12.8° | −203.0 | 80.3 | −173.2 | 50.5 | 51.6 | 69 °/s |
| 20.3° | −336.6 | 192.3 | −241.2 | 98.7 | 41.9 | 93 °/s |
| 30.6° | −471.9 | 404.1 | −192.3 | 123.5 | 30.6 | 127 °/s |

Over the first 10° the blade takes only 5–35% of the rower's torque. The
rest accelerates the oar and the body's reflected inertia, 54–62 kg·m²
there. The blade enters at zero slip with the oar sweeping at about 50 °/s,
and its load grows with slip squared. The `"cr06"` shape nearly doubles the
torque at 2.8° (−47.9 N·m), but most of the extra also goes into
acceleration: blade moment 1.7 against 1.4. That is why its catch slip does
not move.

**It also shows the slip measure above was wrong.** It took gate force as
|τ| / inboard + |F_n|, but τ is the rower's muscle torque. The part that
accelerates the body never reaches the handle. The force at the handle is
what the oar's own balance needs, (|ℓ F_n| + I_oar |θ̈|) / r_h. On these
numbers the M1x then passes 196 N only about 22–23° past the catch, not at
17°. **The slip tables above understate the model's catch miss.**

*Corrected (`holt_slips_oarside.py`, `holt_force_curve_oarside.py`), and three
claims above withdrawn.* Gate force per scull is now what the oar carries,
(|ℓ F_n| + I_oar |θ̈|) / r_h + |F_n|. Holt's conditions, equal power, pairs on
the double's sculls (not comparable to Holt's sweep pairs):

| class | catch slip, default / cr06 (Holt) | finish slip, default / cr06 (Holt) | peak gate, default / cr06 (Holt) |
|---|---|---|---|
| M1x | **21.5** / 20.6° (7.7) | **15.4** / 10.5° (14.1) | 477 / 432 N (497) |
| W1x | **26.0** / 26.4° (9.7) | **21.0** / 16.5° (18.1) | 330 / 299 N (371) |
| M2- | 22.0 / 21.3° (3.7) | 14.8 / 8.8° (8.5) | 497 / 451 N (484) |
| W2- | 26.4 / 27.2° (5.6) | 21.1 / 16.6° (8.5) | 333 / 302 N (347) |

| singles, oar-side | catch to peak | rate of force development | peak / mean | mean gate |
|---|---|---|---|---|
| M1x default / cr06 (Holt) | 0.56 / 0.55 s (0.43) | 858 / 786 N/s (960) | 2.47 / 2.19 (1.90) | 194 / 197 N (261) |
| W1x default / cr06 (Holt) | 0.63 / 0.62 s (0.39) | 529 / 483 N/s (760) | 2.34 / 2.08 (1.87) | 141 / 144 N (199) |

- **Withdrawn: "force fades too early before the finish."** With the default
  shape the singles' finish slip is **within 1.3–2.9° of Holt**. The torque
  measure had overstated it, because near the finish the decelerating body
  hands its energy to the oar.
- **Withdrawn: "the cr06 shape fixes the finish."** On the corrected measure
  it overshoots, finishing 1.6–3.6° short of Holt. It brings peak/mean closer
  (2.47 → 2.19 against 1.90) but lowers the peak gate force (477 → 432 N
  against 497). It is a trade-off, not an improvement, and stays an unadopted
  study.
- **Stands, and larger: the catch.** Catch slip is 21.5–26° against Holt's
  7.7–9.7° on the singles, a miss of 13–16°. Neither shape moves it.
- **Not the inertia clamp** (`catch_rate_floor.py`). Moving `RATE_FLOOR` from
  0.25 to 0.60 cuts the catch inertia from 62.0 to 49.5 kg·m² and moves the
  M1x catch slip only from 21.5 to 21.9°.
- **Not the oar's speed.** [CR06]'s measured oar is 10° past the catch at
  0.160 s, averaging about 62 °/s, and 30° past at 0.368 s. The model sweeps
  50–69 °/s over the first 13° and 127 °/s at 30°, comparable.
- **It is load at a given angle and speed.** [CR06] measures 209 N of handle
  force at 7.5°, about 104 N per scull and 156 N at the gate. The model
  carries about 30 N at the gate there. With the boat at about 3.1 m/s and the
  oar at 1.05 rad/s at 60°, the quasi-steady blade's slip is only about
  0.3 m/s, worth a few newtons of C₂ slip².
- **Candidate: blade added mass,** absent from the research profile. Patton's
  estimate is about 13 kg per scull blade, and at the catch's large normal
  acceleration that is hundreds of newtons.

*Measured, and a sign error in the correction above
(`holt_catch_added_mass.py`).* The gate force was taken from the balances,
which need no blade model and include any added-mass force. With signs,
H = I_crew θ̈ + n τ + ½ I′ θ̇² on the hands and blade torque = H + n I_oar θ̈,
and θ̈ from the simulator's own derivative. A first run used `np.gradient`,
which spikes at the sweep-to-torque hand-over; that run was stopped. Singles
at Holt's conditions, default shape, power matched with the same simulator in
the loop:

| | catch slip (Holt) | finish slip (Holt) | gate at 5° / 10° | peak gate (Holt) | speed error | blade eff. |
|---|---|---|---|---|---|---|
| M1x, no added mass | 21.5° (7.7) | **19.3°** (14.1) | 12 / 51 N | 478 N (497) | −8.5% | 0.710 |
| M1x, Patton added mass | 20.6° | 20.0° | 59 / 83 N | 452 N | −7.1% | 0.753 |
| W1x, no added mass | 26.0° (9.7) | **23.8°** (18.1) | 19 / 51 N | 331 N (371) | −11.5% | 0.721 |
| W1x, Patton added mass | 25.6° | 23.9° | 48 / 56 N | 317 N | −10.7% | 0.768 |

[CR06]'s measured athlete: roughly 110 / 190 N per scull at 5° / 10°.

- **Added mass is not the catch.** It lifts the load at 5° about fourfold
  but moves the catch slip by under 1°. At equal power it adds 0.8–1.4
  points of speed and 0.04–0.05 of blade efficiency. That is a real effect,
  but not this one.
- **The oar-side formula above had a sign error.** It added I_oar |θ̈|
  unconditionally. Near the finish the oar decelerates and its inertia
  helps the blade, so the handle carries less. The catch agrees between the
  two methods (21.5, 26.0°); the finish does not.
  - **The singles' finish slip is 19.3 and 23.8°, 5–6° long against Holt.**
    The withdrawal of "force fades too early" above was itself wrong; the
    original finding stands in substance.
  - The `"cr06"` finish and the oar-side force-timing values used the same
    formula. They were re-measured with the signed balance
    (`holt_force_signed.py`, one consistent measure; these rows supersede
    every earlier slip and force-timing table in this item):

    | single | catch / finish slip (Holt) | gate at 5° / 10° | catch to peak (Holt) | RFD (Holt) | peak / mean (Holt) | mean / peak gate (Holt) | drive |
    |---|---|---|---|---|---|---|---|
    | M1x default | 21.5 / 19.3° (7.7 / 14.1) | 12 / 51 N | 0.56 s (0.43) | 860 N/s (960) | 2.54 (1.90) | 188 / 478 N (261 / 497) | 0.96 s |
    | M1x cr06 | 20.6 / **15.2°** | 17 / 67 N | 0.55 s | 787 N/s | **2.26** | 192 / 433 N | 0.95 s |
    | W1x default | 26.0 / 23.7° (9.7 / 18.1) | 19 / 51 N | 0.63 s (0.39) | 530 N/s (760) | 2.42 (1.87) | 137 / 331 N (199 / 371) | 1.07 s |
    | W1x cr06 | 26.4 / **20.1°** | 23 / 64 N | 0.61 s | 488 N/s | **2.14** | 140 / 299 N | 1.06 s |

    **The cr06 claim stands on the correct measure.** It brings the finish
    to within 1.1–2.0° of Holt, 3.6–4.1° shorter than the default, and
    peak/mean closer. It lowers peak gate force by 9–10%, moves speed by
    ≤ 0.3 points, and leaves the catch (13–16° long) and time to peak
    (0.12–0.24 s late) untouched. A trade-off; it stays an unadopted study.
- **The leading explanation is structural.** Every force curve used is
  *measured handle or gate force*: Kleshnev's points, [CR06]'s F_hand and
  Holt's descriptors. The model applies the curve as the rower's **muscle
  torque** τ on a balance that also carries the body's reflected inertia.
  - Early in the drive most of τ accelerates the body, so the handle force
    lags the measured curve: a long catch.
  - Late in the drive the decelerating body hands energy back, so force
    lingers: a long finish.
  - Applied as handle force, the measured [CR06] curve gives 8.4° at Holt's
    M1x thresholds (7.7°), from the shape-only evaluation above.
  - Changing it means changing how the dynamic oar and the body share a
    balance, which is phase 4.3's job. Recorded as the hypothesis to test
    there, not changed here.

*The hypothesis probed (`catch_handle_force_probe.py`): about half the catch,
none of the finish, and a third of the singles' speed gap.*
`OarDynamics.from_boat` takes a scalar inertia as a study override. Set to the
oar's own second moment, it removes the body from the oar balance, so the pull
acts as handle torque, what the measured curves record. The clock crew still
moves the hull, and power is matched with this build in the loop. Signed
balance with exact acceleration, Holt's conditions:

| single | catch / finish slip (Holt) | gate at 5° / 10° | catch to peak (Holt) | peak / mean (Holt) | peak gate (Holt) | speed error | oar rate at 10° |
|---|---|---|---|---|---|---|---|
| M1x, body on oar, default | 21.5 / 19.3° (7.7 / 14.1) | 12 / 51 N | 0.56 s (0.43) | 2.54 (1.90) | 478 N (497) | −8.5% | 62 °/s |
| M1x, oar only, default | **14.7** / 20.0° | 48 / 128 N | 0.49 s | 2.17 | 465 N | **−4.9%** | 76 °/s |
| M1x, oar only, cr06 | **13.3 / 15.9°** | 75 / 155 N | 0.49 s | **1.91** | 420 N | **−4.6%** | 80 °/s |
| W1x, body on oar, default | 26.0 / 23.8° (9.7 / 18.1) | 19 / 51 N | 0.63 s (0.39) | 2.42 (1.87) | 331 N (371) | −11.5% | 55 °/s |
| W1x, oar only, default | **19.9** / 24.5° | 35 / 90 N | 0.56 s | 2.15 | 329 N | **−8.2%** | 64 °/s |
| W1x, oar only, cr06 | **19.8 / 20.6°** | 53 / 110 N | 0.56 s | **1.89** | 296 N | **−7.9%** | 67 °/s |

- **The body on the oar costs 6–8° of catch slip.** It is still 5.6–10°
  long without it, so the rest is the blade's entry.
- **The finish does not depend on it at all** (19.3 → 20.0°). The finish is
  the pull shape's, and cr06 fixes it in either build.
- **With the body off, cr06's peak/mean equals Holt's,** 1.91 against 1.90
  and 1.89 against 1.87.
- **The singles' speed gap closes by 3.3–3.9 points** at Holt's power as
  printed, with no change to hull, blade or water.

**A power-accounting defect behind that.** In the standard build the rower is
charged τ θ̇, which includes the energy put into the body through the oar. At
the finish the model holds the oar dead, and that energy is not returned to
the water. Holt's Peach power is *handle* power, which excludes it. So at "334
W" the standard build delivers less to the handle than Holt's rower did, most
of all in a single, where the body is the largest share of the moving mass.
*Sized (`finish_energy_loss.py`), and one claim withdrawn.* Over the settled
last stroke, the oar-plus-reflected-body kinetic energy on the last driving
step, ½ I_seat θ̇², is what the held finish discards. Research profile,
default shape, Holt's conditions, pairs on the double:

| class | discarded per rower per stroke | as power | share of handle power | I_seat, oar rate at the finish |
|---|---|---|---|---|
| M1x | 45.8 J | 26.5 W | **7.9%** | 17.6 kg·m², 131 °/s |
| W1x | 34.0 J | 18.6 W | **8.3%** | 16.0 kg·m², 118 °/s |
| M2- | 57.8 J | 36.7 W | **9.7%** | 17.5 kg·m², 147 °/s |
| W2- | 41.8 J | 24.5 W | **10.2%** | 16.0 kg·m², 131 °/s |

- **Withdrawn: "the first candidate for the singles-versus-pairs spread that
  lies in the model."** The loss is 8–10% of handle power on every class, and
  slightly larger on the higher-rating pairs. It is a general gap between the
  model's rower power and Holt's handle power, not a singles effect. The
  spread stays parked on the measurement side.
- **It is most of what the probe gained.** An 8% power sink is worth about
  2.7% of speed at the drag law's exponent of about 3, against the probe's
  3.3–3.9 points.
- **Not all of it is a defect.** A real rower also spends work stopping the
  body and oar at the finish. What is a defect is comparing the model's τ θ̇
  power, which includes it, with Holt's handle power, which does not. Phase 4.3's
  chain decides where that energy goes.

Caveats: the probe's seat balance carries one oar's inertia, not two, a small
error. And an oar with no body on it is lighter than a real one. This is a
diagnosis for phase 4.3, not a candidate change.

*Time-resolved measurements from world-class scullers ([LE26]; digitised
2026-09-14).* Legge et al. (2026) publish group-mean curves over the cycle for
13 men (4.91 m/s, 36.7 spm) and 12 women (4.40 m/s, 34.1 spm). The curves are
gate force per gate, stretcher force and boat acceleration. They were digitised
from the raster figures (`legge26/digitise_fig3.py`, `digitise_fig45.py`):
gridline calibration residual ≤ 0.9 px, and repeated panels of the same mean
agree to ≤ 11 N and ≤ 0.1 m/s². Following their Fig. 2, the catch is taken at
minimum boat acceleration: 46.5% of the cycle for the men, 46.0% for the women.

| | men | women |
|---|---|---|
| boat acceleration: minimum / zero after the catch / second peak | −14.2 m/s² / 53.2% / 4.9 m/s² at 79% | −11.0 m/s² / 52.8% / 3.9 m/s² at 78% |
| **gate force at the catch**, both gates summed (Fig. 3 is summed; SOURCES [LE26]) | **203 N** (about 102 per gate) | **154 N** (about 77 per gate) |
| gate force 20% / 10% / 5% of the cycle before the catch | −40 / 47 / 131 N | −24 / 32 / 105 N |
| time from the catch to 196 N | 0 s | 0.044 s |
| catch to peak gate force | 0.379 s (1194 N) | 0.406 s (916 N) |
| drive, catch to gate force back at zero | 0.486 of the cycle, 0.794 s | 0.490, 0.862 s |
| peak / mean gate force over that drive | 1.61 | 1.68 |
| **stretcher force at the catch** | **426 N** | **335 N** |
| stretcher leads gate force to 300 N by | 0.110 s | 0.113 s |
| stretcher and gate forces converge | by about 52% of the cycle | by about 54% |

- **The gate is loaded before the drive.** Gate force rises from about −40 N a
  fifth of a cycle before the catch to 150–200 N *at* the turning point. So the
  model's gate force, zero until the blade enters, cannot be compared with a
  196 N threshold without that pre-load. By [H20]'s rule these scullers' catch
  slip is 0 to a few degrees. Part of the model's catch "miss" is what the
  instrument counts. What the pre-load is (the blade touching the water before
  the turning point, the hands already loading the handle, the oar's own
  deceleration) is not separable from these curves. The oar's inertia alone is
  worth about 30–40 N per scull.
- **The body is driven through the feet into the catch.** At the catch the
  stretcher carries 184–229 N more than the gate, and the difference closes by
  mid-drive. That excess is what accelerates the rower's body, and it does not
  pass through the handle. It is direct support for the phase 4.3 hypothesis
  that the dynamic oar wrongly charges the body's acceleration to the handle
  torque.
- **The measured drive is 0.49 of the cycle.** At similar rates the model's is
  0.555–0.585 (TRACKING's drive-fraction item), and its catch to peak is
  0.56–0.63 s against 0.38–0.41 measured.
- *Model at [LE26]'s speeds* (`legge26/model_vs_legge.py`,
  `compare_shapes.py`). No power is published, so torque was matched to the
  measured speed. Research profile, the rower's mass and stature set to each
  cohort's, 20 °C water from [ITTC11]. Per-gate force comes from the signed
  balances; [LE26]'s summed curve is halved. One time base from the catch:

  | | power needed | peak per gate | at the catch | catch to peak | shape rms | min boat accel. | accel. zero after catch | accel. rms |
  |---|---|---|---|---|---|---|---|---|
  | men, [LE26] | ([K00] 489 W) | 597 N | 101 N | 0.379 s | — | −14.2 m/s² | 0.069 of cycle | — |
  | men, model default | 523 W | 715 N | 0 | 0.481 s | 0.366 | ~~−11.2~~ **−9.9** | **0.143** | 3.0 m/s² |
  | men, model cr06 | 519 W | 641 N | 0 | 0.476 s | 0.335 | ~~−11.2~~ −9.9 | 0.142 | 3.0 m/s² |
  | women, [LE26] | ([K00] 302 W) | 458 N | 77 N | 0.406 s | — | −11.0 | 0.069 | — |
  | women, model default | 363 W | 522 N | 0 | 0.524 s | 0.345 | ~~−9.0~~ **−7.8** | **0.141** | 2.4 m/s² |
  | women, model cr06 | 360 W | 468 N | 0 | 0.519 s | 0.317 | ~~−9.0~~ −7.8 | 0.139 | 2.3 m/s² |

  *Corrected 2026-09-14* (`model_vs_legge_airmask.py`, same build, true
  per-sample blade air mask). The struck values were evaluated with the
  blade forced into the water through the sweep catch's pre-entry phase. The
  same computation on the original CSV reproduces them (−11.2 / −9.0,
  0.143 / 0.141, rms 2.95 / 2.33). On the corrected CSV only the minimum
  moves: the zero after the catch is unchanged and rms is 2.97 / 2.35 m/s².
  Force, power, peak and timing are identical.

  - **The slow load-on shows up in the hull itself.** The measured boat is
    back to accelerating 0.069 of the cycle after the catch, with a sharp
    first peak of about 4 m/s² at 0.09. The model's stays decelerating twice
    as long, to 0.14, has no first peak, and its catch dip is ~~20%~~ **29–30%**
    shallower (corrected for the air-mask artefact; the late return and the
    missing first peak are unchanged).
  - **The force comes on late and runs long.** The measured force reaches
    about 450 N (men) per gate by 0.10 of the cycle. The model's is near zero
    until the blade enters at about 0.04, is about 200 N at 0.13, and peaks
    0.10–0.12 s late and 7–20% high. It is still pulling near the finish,
    where the measured boat's acceleration has fallen to zero. Normalised
    shape error over the drive is 0.32–0.37 of peak, most of it timing. The
    cr06 shape trims the peak and the error slightly, but it does not move
    the start.
  - **Three independent measures now agree on the same defect:** [H20]'s
    slips, [LE26]'s force curve, and [LE26]'s boat acceleration. Together with
    the stretcher lead, it points at phase 4.3.
  - **Recovery.** The model's recovery acceleration also differs: a dip at
    −0.33 of the cycle and a peak at −0.20, against [LE26]'s single broad peak
    at −0.26. That is the prescribed, ergometer-derived crew timing, TRACKING's
    kinematics defect.
  - **Power.** At these speeds the model needs 519–523 W (men) and 360–363 W
    (women), against [K00]'s handle power at the same rates of 489 and 302 W:
    7% and 20% more.
  - Small steps at the catch and the finish in the model's curves are the
    oar's reset and held finish, not digitisation.

  *The two candidate fixes against the same curves (`legge26/probe_vs_legge.py`).*
  Three builds at [LE26]'s speeds, torque matched to speed:
  - **standard**, the research model as it is;
  - **oar only**, the body's reflected inertia off the oar balance, so the
    pull acts as handle force and the clock crew carries the body on the hull;
  - **oar + mass**, oar only plus Patton blade added mass.

  | | accel. zero after catch | first peak | gate force at 0.05 / 0.10 cycle | catch to peak | shape rms | power |
  |---|---|---|---|---|---|---|
  | men, [LE26] | 0.069 | 4.14 m/s² | 208 / 416 N | 0.379 s | — | — |
  | men, standard | 0.143 | 0.76 | 5 / 35 N | 0.481 s | 0.366 | 523 W |
  | men, oar only | 0.137 | 1.59 | 22 / **145 N** | 0.443 s | **0.208** | 436 W |
  | men, oar + mass | 0.138 | 1.32 | 29 / 115 N | 0.494 s | 0.280 | 455 W |
  | women, [LE26] | 0.069 | 3.49 | 157 / 315 N | 0.406 s | — | — |
  | women, standard | 0.141 | 0.79 | 6 / 37 N | 0.524 s | 0.345 | 363 W |
  | women, oar only | 0.135 | 1.38 | 17 / **109 N** | 0.482 s | **0.222** | 335 W |
  | women, oar + mass | 0.137 | 1.14 | 23 / 86 N | 0.544 s | 0.299 | 320 W |

  *Checked 2026-09-14 against the air-mask artefact* (`probe_vs_legge_airmask.py`,
  2.7 kg oar as recorded). Every number in this table is unchanged. Only the
  catch minimum, not tabulated here, moves: standard and oar only
  −11.5 → −9.9 m/s² (men) and −9.3 → −7.8 (women); oar + mass
  −10.0 → −9.9 and −7.9 → −7.8.

  - **Taking the body off the oar fixes much of the force curve.** Shape
    error falls about 40%, the force at 0.10 of the cycle quadruples, and the
    peak comes 0.04 s earlier. The power needed for the same speed falls
    8–17%, which is the body energy the standard build charges and discards.
  - **Added mass makes the force curve worse**, later and flatter.
  - **Neither moves the boat's acceleration.** It still returns above zero at
    0.135–0.138 of the cycle against a measured 0.069, with a first peak a
    third to two fifths of the measured one. So the hull's slow catch is *not*
    mainly the oar balance. Near the catch the hull feels the crew's own
    reversal most, and that is prescribed from ergometer kinematics
    (TRACKING's kinematics defect). The mismatched recovery shape points the
    same way.

  *Checked, and it is the crew (`legge26/catch_accel_split.py`).* The
  model's surge acceleration around the catch was split into contributions from
  the simulator's own `ForceBreakdown`. Each force along the velocity was divided
  by the effective surge mass, so the parts sum to the total. Standard build,
  men (women the same pattern), m/s²:

  | fraction of cycle from catch | crew reaction | blade | drag | model | [LE26] |
  |---|---|---|---|---|---|
  | −0.10 | −4.5 | 0.0 | −1.0 | −5.5 | −4.3 |
  | −0.05 | −8.3 | 0.0 | −0.8 | −9.1 | −8.2 |
  | 0.00 | −9.2 | 0.0 | −0.7 | −9.8 | **−14.2** |
  | 0.05 | −8.8 | 0.0 | −0.5 | −9.3 | −6.0 |
  | 0.07 | −7.7 | 0.02 | −0.4 | −8.1 | **+0.4** |
  | 0.10 | −4.9 | 0.09 | −0.4 | −5.2 | **+4.0** |
  | 0.20 | +3.7 | 1.3 | −0.3 | +4.7 | +2.4 |

  - **The approach to the catch is right.** To 0.05 of the cycle before it,
    the crew term tracks the measured deceleration within about 1 m/s².
  - **The reversal is wrong.** Measured, the boat dips to −14.2 m/s² at the
    catch and swings to +4.0 within 0.10 of the cycle, about 0.16 s. The
    model's crew term holds a broad −9 m/s² plateau through the catch, is
    still −4.9 at 0.10, and crosses zero only at 0.14 (women 0.141).
  - **The blade is negligible there.** It is under 0.5 m/s² until 0.14 of the
    cycle, so no blade or handle-force change can make the measured swing: the
    crew term is ten times larger.
  - **The measured catch is sharper, not later.** Model and measurement both
    have their minimum at the catch; the ergometer-fitted body reverses too
    gently. So the hull's catch is the kinematics defect ("The drive is
    18–28% too long, and the cause is the ergometer"), not the oar. It sets a
    second target for phase 4.3, the body's reversal, alongside the handle-force
    split.
  - The split is undefined for an instant where the total acceleration
    crosses zero (men, 0.14), because the effective mass blows up; nothing
    else is affected.

  *The body itself, against a measured on-water body (`cr06/traces_fig3_body.py`,
  `model_body_vs_cr06.py`).* [CR06] Fig. 3 carries measured leg displacement
  (seat relative to foot) and back displacement (shoulder relative to hip) for
  their women's single, T = 1.94 s. Both were read exactly from the figure's
  vector paths (grid residual ≤ 4 × 10⁻⁴ s, 2 × 10⁻⁴ m). The model's chain,
  catalogue single at the same period, gives hip − ankle and shoulder − hip,
  aligned at the leg minimum (the catch):

  | | [CR06], measured | model |
  |---|---|---|
  | leg acceleration at the catch | **12.0–15.4 m/s²**, over a 30× range of spline smoothing; down to about 4 by 0.05 of the cycle | **6.9 m/s²**, analytic; flat through 0.05, 5.9 at 0.08 |
  | leg velocity, 0 to half its drive peak | 0.075–0.078 s | 0.102 s |
  | leg displacement at 0.05 / 0.10 / 0.15 of the cycle (no smoothing) | 0.039 / 0.122 / 0.229 m | 0.033 / 0.128 / 0.262 m |
  | back: 25% / 50% / 75% of its drive range reached at | 0.39 / 0.585 / 0.662 s | **0.287 / 0.459** / 0.637 s |
  | back range over the drive | 0.398 m | **0.516 m** |

  - **The legs reverse 1.7–2.2× more sharply than the model's.** The measured
    body has a short acceleration spike at the catch; the model has a broad,
    gentle plateau. That is the same pattern as the hull: a measured
    −14.2 m/s² dip against the model's −9 m/s² plateau. The leg travel itself
    agrees to millimetres; it is the shape of the reversal that is wrong.
  - **The trunk opens too early and too far.** It is 0.10–0.13 s early at 25%
    and 50% of its swing and travels 30% further. The measured rower swings
    the back in the late drive; the ergometer-fitted chain swings it through
    mid-drive. That is a sequencing defect in the prescribed kinematics, and it
    shapes the model's mid-to-late drive surge too.
  - This is one athlete, a women's single, but consistent in size and
    sharpness with [LE26]'s hull dips for 25 world-class scullers. With the
    split above, it closes the chain from hull to body: the model hull's slow
    catch is the prescribed body's gentle leg reversal.

  *Can the existing sequencing warps retime it? The trunk, yes; the catch, no
  (`cr06/sequencing_grid.py`).* `SegmentSequencing` (shank, thigh, trunk) is the
  existing within-phase warp. It is `SYNCHRONOUS` (zero) everywhere, and SOURCES
  §30's calibrated value was never adopted. The rower was rebuilt with legs
  (shank = thigh) from 0 to 0.25 and trunk from 0 to −0.25, kinematics only,
  and scored against [CR06]'s measured body:

  | legs | trunk | leg accel. at catch | 0 → half leg velocity | back 50% | leg shape rms | back shape rms |
  |---|---|---|---|---|---|---|
  | [CR06] | | 14.1 m/s² | 0.077 s | 0.585 s | — | — |
  | 0 | 0 (default) | 6.9 | 0.103 s | 0.460 s | 0.053 | 0.138 |
  | 0 | −0.15 | 6.9 | 0.103 s | **0.583 s** | 0.053 | **0.068** |
  | 0.05 | −0.15 | 6.4 | 0.092 s | 0.587 s | 0.094 | 0.070 |
  | 0.10 | −0.15 | 5.7 | 0.089 s | 0.593 s | 0.135 | 0.075 |
  | 0.15 | −0.15 | 5.0 | 0.087 s | 0.597 s | 0.180 | 0.078 |

  - **A trunk lag of −0.15 fixes the trunk's timing.** Its 50% point lands at
    0.583 s against 0.585 s, and the back shape error halves, 0.138 → 0.068.
  - **A leg lead makes the catch gentler, not sharper.** Leg acceleration at
    the catch falls from 6.9 to 4.7 m/s² as the warp grows, and the leg shape
    worsens. The warp moves leg motion earlier in the drive without
    sharpening the reversal itself.
  - **18 of 36 combinations are unreachable.** Every leg lead without enough
    trunk lag leaves the hands short of the handle in the early drive
    (stroke phase 0.13–0.17).
  - **The gentle leg reversal is structural, not a timing setting.** The body
    is Caplan & Gardner's four common keyframes fitted with a few Fourier
    harmonics. A smooth low-harmonic fit through four instants cannot make a
    14 m/s² spike that is gone by 0.05 of the cycle. It can be retimed, not
    sharpened.
  - So phase 4.3's hull target belongs to the body's representation: a
    measured on-water joint trajectory, or a body driven by joint torques
    whose catch reversal comes out of the stretcher and blade loads. No warp
    of the present kinematics reaches it.
  - *Hull test of the trunk retiming at [LE26]'s speeds* (legs 0, trunk
    −0.15; `legge26/retimed_body_vs_legge.py`):

    | | accel. zero after catch | first peak | accel. rms | gate shape rms | power |
    |---|---|---|---|---|---|
    | men, [LE26] | 0.069 | 4.14 m/s² | — | — | — |
    | men, standard | 0.143 | 0.76 | ~~3.01~~ 3.04 | 0.366 | 523 W |
    | men, trunk retimed | 0.146 | 0.37 | ~~2.87~~ **2.91** | 0.344 | 534 W |
    | men, oar only | 0.137 | 1.59 | ~~3.00~~ 3.03 | 0.208 | 436 W |
    | men, retimed + oar only | 0.139 | 1.13 | ~~2.91~~ 2.94 | **0.203** | 432 W |
    | women, [LE26] | 0.069 | 3.49 | — | — | — |
    | women, standard | 0.141 | 0.79 | ~~2.38~~ 2.40 | 0.345 | 363 W |
    | women, trunk retimed | 0.144 | 0.46 | ~~2.22~~ **2.24** | 0.326 | 367 W |
    | women, oar only | 0.135 | 1.38 | ~~2.33~~ 2.35 | 0.222 | 335 W |
    | women, retimed + oar only | 0.137 | 0.98 | ~~2.18~~ **2.20** | **0.218** | 330 W |

    *Corrected 2026-09-14* (`retimed_body_vs_legge_airmask.py 0 -0.15`, 2.7 kg
    oar as recorded, true blade air mask). Only accel. rms moves (struck
    values were evaluated with the blade forced into the water before
    entry). The catch minima become −9.9 / −10.3 m/s² men and −7.8 / −8.2
    women, standard / retimed.

    - **Trunk timing trims the hull error ~~5–7%~~ 4–7% and moves no landmark.** The
      catch zero stays at 0.14–0.15 of the cycle, and the first peak shrinks.
    - **It trades one mismatch for another.** It flattens the recovery hump
      the synchronous body had in the wrong place. But it adds a late-drive
      surge of about 5 m/s² near 0.45 of the cycle, where the measured boat is
      near zero: the trunk now swings into the finish in one go.
    - **The two fixes are complementary.** Body off the oar fixes the force
      curve; trunk retiming trims the hull. Together they give the best of
      both columns, and the catch is still twice as slow as measured.
    - This confirms the grid: the timing warps reach the trunk, and nothing
      but a changed body representation reaches the catch. Not adopted: a
      trunk warp calibrated to one athlete is a fitted number, and it moves
      the error rather than removing it.

  *The decisive test: the body driven on a MEASURED leg time-law
  (`legge26/warped_body_vs_legge.py`).*
  - **Method.** The model's own body (geometry and masses kept) was driven
    along a phase map T(t). T was chosen so its leg displacement follows
    [CR06]'s measured on-water leg curve as a fraction of the cycle:
    x = x(T), v = v(T) T′, a = a(T) T′² + v(T) T″. The simulator's
    `crew_field` was replaced on the instance, which reaches both the mass
    matrix and the force breakdown. The body is off the oar balance, the
    torque is matched to [LE26]'s speed, and the variants are with and without
    the trunk lag.
  - **A first run is void for magnitudes, and its timing is withdrawn.** It
    warped raw digitised points with a lightly smoothed map (λ 2 × 10⁻⁷).
    T″ turned every wiggle into ±15–25 m/s² of hull acceleration and a
    −32 m/s² catch spike. Its "zero after the catch at 0.070–0.072" (against
    [LE26]'s 0.069) was noise crossing zero, not the catch. Its outputs are
    kept as `*_noisy_lam2e-7.*`.
  - **The clean run's smoothing was chosen by kinematics alone**, before any
    hull result: λ = 10⁻⁴ gives warped leg acceleration at the catch 19.4
    and 15.5 m/s², against [CR06]'s 14.1 scaled to [LE26]'s periods, 19.9 and
    17.1. Roughness falls 30-fold.

  | | accel. zero after catch | first peak | catch dip | accel. rms | gate shape rms |
  |---|---|---|---|---|---|
  | men, [LE26] | 0.069 | +4.14 | −14.2 | — | — |
  | men, oar only (smooth body) | 0.137 | +1.59 | ~~−11.5~~ −9.9 | ~~3.00~~ 3.03 | 0.208 |
  | men, leg-warped | **0.165** | **−1.35** | ~~−19.1~~ **−17.8** | ~~3.18~~ 3.17 | 0.202 |
  | men, leg-warped + trunk | 0.168 | −1.40 | ~~−19.3~~ −18.6 | ~~3.63~~ 3.62 | 0.200 |
  | women, [LE26] | 0.069 | +3.49 | −11.0 | — | — |
  | women, oar only | 0.135 | +1.38 | ~~−9.3~~ −7.8 | ~~2.33~~ 2.35 | 0.222 |
  | women, leg-warped | **0.162** | **−0.90** | ~~−14.7~~ **−14.1** | ~~2.65~~ 2.64 | 0.217 |
  | women, leg-warped + trunk | 0.165 | −0.97 | −14.8 | 2.99 | 0.213 |

  *Corrected 2026-09-14* (`warped_body_vs_legge_airmask.py`, 2.7 kg oar as
  recorded, true blade air mask). Struck values were evaluated with the blade
  forced into the water before entry. The zero after the catch, the first
  peak and the gate shape are unchanged.

  - **The measured leg reversal fixes the dip's sharpness and overshoots its
    depth.** The broad smooth-body plateau (now −9.9 / −7.8 m/s² at its
    lowest) becomes a narrow spike at the catch, like [LE26]'s, but
    ~~34%~~ **25–28%** too deep.
  - **It does not fix the return; it delays it.** After the dip the boat
    stalls near −3 m/s² from 0.05 to about 0.12 of the cycle, crosses zero at
    0.162–0.168 (the smooth body: 0.14) and never has a positive first peak.
    The measured boat is at +4 m/s² by 0.09.
  - **The return needs blade propulsion.** Once the legs have reversed, the
    body accelerating toward the bow pushes the hull back. The hull turns
    positive only when blade force exceeds that. [LE26]'s gate force per side
    is about 208 N at 0.05 and 416 N at 0.10 of the cycle; the model's, with
    the body off the oar, is 22 and 145 N.
  - **Correction to the decomposition above.** Its bullet "no blade or
    handle-force change can make the measured swing" compared the crew term
    with the *model's* too-weak blade. The measured swing needs **both**: the
    sharp body reversal makes the dip, and a fast blade load makes the return
    and the first peak.
  - **One athlete's recovery does not transfer.** [CR06]'s recovery,
    compressed to [LE26]'s period, puts a +8 to +11 m/s² surge near −0.3 of the
    cycle and a finish spike near 0.5 that [LE26] does not have. So the rms
    error is worse even where the catch dip is right.
  - *Next test:* the warped body together with a handle force following
    [LE26]'s measured time curve from the catch. The two measured inputs
    together should reproduce the dip, the return and the first peak if
    nothing else is missing.

  *Both measured inputs (`legge26/measured_inputs_vs_legge.py`).* The handle
  torque was imposed as a function of time, following [LE26]'s per-gate force
  curve from the catch. Handle force = gate / (1 + r_h/ℓ), clipped at zero,
  scaled to reach [LE26]'s speed. It was run with the smooth body and with the
  body on [CR06]'s measured leg time-law, oar-only balance in both:

  | | accel. zero after catch | first peak | catch dip | accel. rms | gate shape rms | force scale |
  |---|---|---|---|---|---|---|
  | men, [LE26] | 0.069 | +4.14 | −14.2 | — | — | — |
  | men, measured force | **0.131** | **+2.18** | ~~−11.5~~ −9.9 | ~~2.85~~ **2.87** | ~~0.070~~ **0.063** | 0.87 |
  | men, measured force + body | 0.160 | −0.81 | ~~−19.1~~ −17.9 | ~~3.13~~ 3.18 | ~~0.094~~ 0.061 | 0.86 |
  | women, [LE26] | 0.069 | +3.49 | −11.0 | — | — | — |
  | women, measured force | **0.129** | **+1.97** | ~~−9.3~~ −7.8 | ~~2.22~~ **2.24** | **0.059** | 0.92 |
  | women, measured force + body | 0.155 | −0.34 | ~~−14.8~~ −14.2 | ~~2.48~~ 2.47 | 0.057 | 0.91 |

  *Corrected 2026-09-14* (`measured_inputs_vs_legge_airmask27.py`, the 2.7 kg
  oar these rows used, at T/160 with the true blade air mask). The struck
  values came from one run at the default step (T/80), evaluated with the
  blade forced into the water before entry.
  - **Catch minimum:** moves by the air mask alone.
  - **Gate shape error:** moves with the step, since gate force comes from
    in-water samples.
  - **Unchanged:** zero after the catch, first peak and speed-matched power
    (within 1%).

  - **The measured force time-law is the best build so far.** First peak
    +2.0–2.2 m/s² (against 1.4–1.6 for oar only) and acceleration rms
    2.2–2.9. It reaches [LE26]'s speed with 87–92% of their force.
  - **Part of the return is lost at the blade's entry.** The sweep catch holds
    the blade out until about 0.04 of the cycle, where the force steps in,
    whereas [LE26]'s gates carry about 100 N at the catch itself.
  - **Adding the measured leg time-law makes it worse.** The sharp reversal
    pushes the hull back harder after the catch than the measured blade load
    offsets. The dip is ~~34%~~ **26–29%** too deep, and there is no positive
    first peak.
  - **If the two inputs were consistent, the hull could not miss**: blade,
    drag and crew momentum are all this model has. The candidate: the model
    body's moving mass is too great. Its legs agree with [CR06] to
    millimetres, but its trunk travels 30% further (0.516 against 0.398 m;
    ratio 0.77), about the size of the ~~34%~~ 26–29% overshoot.
  - **Tested and rejected** (`measured_inputs_vs_legge.py 0.77`). The head,
    upper and mid trunk and arms had their velocity and acceleration relative
    to the lower trunk scaled by 0.77, with the legs untouched:

    | | accel. zero after catch | first peak | catch dip | accel. rms |
    |---|---|---|---|---|
    | men, force + body | 0.160 → **0.157** | −0.81 → −0.57 | −17.9 → **−17.6** | 3.18 → 2.78 |
    | men, force only | 0.131 → 0.129 | +2.18 → +2.43 | −9.9 → −9.7 | 2.87 → 2.78 |
    | women, force + body | 0.155 → 0.152 | −0.34 → −0.15 | −14.2 → −13.9 | 2.47 → 2.15 |
    | women, force only | 0.129 → 0.127 | +1.97 → +2.17 | −7.8 → −7.6 | 2.24 → 2.16 |

    *Corrected 2026-09-14* (`measured_inputs_vs_legge_airmask27.py` with and
    without `0.77`: 2.7 kg oar, T/160, true blade air mask). The first
    version, −19.1 → −18.8 and 3.13 → 2.74 men, was at the default step with
    the blade forced into the water before entry. The conclusions below
    hold as written: the dip moves 0.3 m/s², and rms falls 12–13%. Any
    number quoted below from the first version is superseded by this
    table.

    - **The dip and the return barely move.** The catch reversal is carried
      by the legs, hips and lower trunk, not the upper body. The rms error
      falls 12–13%, mostly from a smaller recovery surge: real but secondary.
    - **The likelier cause is mixing athletes.** The body law is one women's
      single at 30.9 spm ([CR06]), and the force and hull are 25 elite
      scullers at 34–37 ([LE26]). Scaling one athlete's leg reversal by
      (1.94/P)² to a faster rate is an assumption, and the probable source of
      the over-deep dip.
  - *Next: a self-consistent test.* [CR06] Fig. 3 carries one athlete's body,
    handle force and boat velocity from the same stroke. Driving the model
    with her body and force time-laws at her own rate, and scoring the
    predicted boat *velocity* against her measured one, removes both the
    cross-athlete assumption and differentiation noise.
  - **Result: the model predicts her boat** (`cr06/self_consistent_cr06.py`).
    Setup:
    - **Rig:** [CR06] Table 1 (75 kg rower, 19.7 kg boat, 1.2 kg scull,
      s 0.83, ℓ 1.805 m) with her measured arc (60.49 / −44.35°).
    - **Rate:** T = 1.94 s. The model's own drive-fraction law gives 0.462
      against her 0.461, so it is left alone.
    - **Handle torque:** her measured `F_hand(t)` × s. Fig. 3's force is
      both hands summed (SOURCES, [CR06]: power 260 against 251 W of drag,
      and the elite-women magnitude from [LE26]), so each oar takes half.
    - **Stature:** 1.787 m. [CR06] gives none, but at that stature the
      model's leg travel is 0.582 m against her measured 0.581.
    - **Nothing fitted:** force scale 1, oar-only balance, half the default
      step.

    | build | mean speed | error | v min / max | velocity rms | power |
    |---|---|---|---|---|---|
    | [CR06] measured | 4.191 | — | 3.06 / 5.13 | — | 260 W implied |
    | summed, smooth body | 4.087 | −2.5% | 2.59 / 4.87 | 0.203 | 275 W |
    | **summed, her measured leg** | **4.101** | **−2.2%** | 2.79 / 5.21 | **0.143** | 277 W |
    | per oar, her measured leg | 5.674 | +35.4% | 4.13 / 6.72 | 0.307 | 530 W |

    - **Speed:** predicted to 2.2% from her force and body alone. This
      validates the hull, blade and oar dynamics together at one operating
      point. The summed reading was chosen on the power and magnitude checks
      before this run finished; per oar, the model refuses it independently
      (+35%).
    - **Oar angle is predicted too.** It is a dynamic state, driven by the
      imposed torque against blade slip. It follows her measured angle
      within a few degrees through the drive and reaches the finish at
      0.455 of the cycle, against her release at 0.461. Per oar, the drive
      is over by 0.36.
    - **Her leg time-law is worth 30% of the velocity-trace error** (0.203 to
      0.143 m/s), mostly on the catch dip and the recovery peak.
    - **The catch dip is still too deep**, 2.79 against 3.06 m/s, with every
      input from one athlete, one stroke, at her own rate. So
      mixing athletes was not the main cause of the [LE26] overshoot. What is
      left: the trunk (model back travel 0.513 against her 0.398 m, and not
      on her time-law), the catalog hull shape, and the catch transition.
    - Two small ripples in mid-recovery (0.50, 0.63) come from the leg warp,
      not from the data.
    - **Settled, and the dip's timing is right.** The last eight strokes agree
      to 1 mm/s (4.1015 to 4.1005). The minimum falls at 0.147 of the cycle
      against her 0.143. Only its depth is wrong.
    - **Her back travel, one-athlete version** (`self_consistent_back_cr06.py`;
      upper body relative to the lower trunk ×0.776 = 0.398 / 0.513 m):
      4.103 m/s (−2.1%). The recovery peak comes to 5.10 against her 5.13,
      and the velocity rms falls a further 20%, 0.143 to **0.115 m/s**. The
      catch minimum moves 2.79 to 2.82 against 3.06. So as with [LE26], the
      trunk's excess travel shapes the recovery, not the catch.
    - *Next:* the body on her clock. The leg warp put her leg minimum at the
      model's time (0.993 of the cycle), not hers (0.004), so the body
      reversed about 21 ms before the force. And the trunk is not on her
      back time-law: it comes forward 32 ms late and opens 62 ms early
      (`alignment_cr06.py`).
    - **Leg on her clock** (`self_consistent_body_cr06.py`): 4.105 m/s
      (−2.1%), velocity rms 0.143 → **0.118 m/s**. The alignment alone is
      worth 17%. The minimum is still 2.81 m/s, now at 0.155.
    - **Ruled out: hull added mass.** Strip-theory surge added mass on this
      hull is 0.64 kg, which is [CR06]'s own 0.0065 of displacement.
    - **Partly: blade entry.** Under the sweep catch the blade enters at
      0.084 s (58.2°, 0.044 of the cycle). Her force before that, 5.5 of
      282 N·s over the drive (2%), is never applied, and it falls exactly
      where the dip forms. Over about 97 kg of boat, rower and oars that is
      worth roughly 0.06 m/s, about a fifth of the dip's excess
      (`catch_entry_cr06.py`). **Bounded:** the rest catch, which loads her
      force from t = 0, lifts the minimum from 2.81 to 2.87 m/s (the
      0.06 estimated) and the speed to 4.154 (−0.9%). But it ends the drive
      at 0.430 of the cycle against her release at 0.461, so it is a bound,
      not a fix (`rest_catch_bound_cr06.py`).
    - **The trunk's timing is the largest remaining lever.** With the leg on
      her clock and the upper body following her measured back
      displacement in time (the model's travel kept): 4.111 m/s (−1.9%),
      catch minimum **2.92 at 0.143** (hers 3.06 at 0.143), recovery peak
      5.15 against 5.13. Velocity rms **halves**, 0.118 → **0.074 m/s**,
      and is 0.203 → 0.074 from the smooth body. Measured from the first
      measured-leg build (minimum 2.79, 0.27 m/s too deep), her clock and her
      trunk timing remove 0.13 (0.02 and 0.11). Blade entry is worth about
      0.06 more; about 0.08 is left.
    - **Her whole measured body: the hull follows her boat.** Leg on her
      clock, trunk on her back time-law *and* her back travel (×0.776):
      4.109 m/s (−1.9%), minimum 2.93 at 0.143, recovery peak 5.09 against
      5.13, velocity rms **0.035 m/s**.

      | body | velocity rms | catch minimum |
      |---|---|---|
      | smooth catalog body | 0.203 | 2.59 |
      | + her leg time-law | 0.143 | 2.79 |
      | + on her clock | 0.118 | 2.81 |
      | + her trunk timing | 0.074 | 2.92 |
      | + her trunk travel | **0.035** | 2.93 |
      | [CR06] measured | — | 3.06 |

      - **The recovery is solved.** Five-sixths of the velocity-trace error
        was the body, and the three measured body features remove it: leg
        time-law, trunk timing, trunk travel. Travel shapes the recovery
        peak, and timing shapes the catch.
      - **What remains is the catch: 0.13 m/s too deep, 0.05–0.20 of the
        cycle, with the right timing.** The blade entry bound covers about
        0.06 of it. The rest is where 4.3's stretcher-driven catch and early
        blade load must act.
      - **For 4.3, the catalog body is a representation problem, not a data
        problem.** Driven on one athlete's measured leg and back, the model
        reproduces her hull through the stroke. Its hull, blade and oar
        dynamics are not the gap.
  - **Every rower is built from de Leva's MALE table** (found 2026-09-14).
    `RowerAnthropometry(sex="male")` is the default, and nothing in `coxswain/`
    passes a sex: not `catalog`, not `viz/menu.py:449` (the game's lineups),
    not the roster path. That includes the Women's Veteran 60+ four the
    project is built around, the women's squad boats, the [LE26] women and
    [CR06]'s woman. The roster knows its squad is "women" (it uses that for
    assumed stature), but `Rower`/`Lineup` carry no sex.
    - **Measured on [CR06]'s woman** (`cr06/female_segments_cr06.py`,
      T/320). With the female table, stature re-anchored to her leg
      travel is 1.758 m (was 1.787). Back travel is 0.516 m, ratio 0.771
      (was 0.775). Each thigh is 11.1 kg, and the thigh's centre of mass
      sits nearer the hip.
    - **Smooth body:** 4.088 m/s against 4.087, velocity rms 0.198
      against 0.203, catch minimum 2.60 against 2.59. The kinematics
      barely move, because stature is anchored to the measured leg
      travel.
    - **Whole measured body:** 4.112 m/s (−1.9%) against 4.109. Catch
      minimum 2.94 at 0.145 against 2.93, recovery peak 5.05 against 5.09,
      velocity rms **0.032** against 0.035 m/s. Settled to 0.1 mm/s.
    - **So the one-athlete conclusions stand, slightly better on the right
      table.** The catch dip is unchanged, so the male table is not part of
      the remaining catch gap. It matters for anything that reads segment
      masses directly, which is what step 4.3's joint torques will do, so
      her runs use the female table from here on.
  - **Step 4.3, first rung: the stretcher force** (2026-09-14). The body's
    momentum balance along the boat, with the seat frictionless:
    stretcher force = rower mass × absolute centre-of-mass acceleration +
    handle pull along the axis.
    - **From her data through [CR06]'s own body model**
      (`cr06/stretcher_from_her_data_cr06.py`; [CR06] eq. 1, and eq. 5 with
      r = 0.4, i.e. all 75 kg at the seat plus 0.4 × back; smoothing weight
      3×10⁻⁶ to 3×10⁻⁵). *Correction 2026-09-14: first recorded as "no
      model". Eq. 5 is a model, and it overstates the acceleration of her
      centre of mass (below).*
      - **Catch landmark:** [LE26]'s landmark, minimum boat acceleration, is
        −8.2 to −8.8 m/s² at +0.009 to +0.021 s after her maximum oar angle.
      - **At the landmark:** stretcher **408–448 N**, summed gate force
        34–36 N. Her absolute acceleration is +5.2 to +5.7 m/s² (boat −8.8,
        legs +12.0, back 0.4 × 4.5).
      - **Lead:** the stretcher reaches 300 N 0.02 s *before* the landmark,
        the gate force at 0.21 s after it, a lead of **0.23 s**.
      - **Peaks:** stretcher 675–726 N against gate 792 N, a ratio of
        0.85–0.92.
    - **That is [LE26]'s structure.** Their world-class women have 335 N
      stretcher against 154 N gate at the catch, stretcher leading (men:
      0.11 s), and a peak ratio of 0.94. One athlete agrees with a
      different cohort on all three, so the target is real, not a quirk of
      either dataset.
    - **The model fails it.** On her validated stroke
      (`cr06/stretcher_from_balance_cr06.py`: female table, whole measured
      body, T/320):
      - **Catch (t = 0):** stretcher **33 N** against 34 N gate, ratio 0.98.
      - **Timing:** the stretcher reaches 300 N 0.017 s *after* the gate.
      - **Peak:** ratio 0.89, which does agree.

      The model's rower has almost no absolute acceleration at the catch,
      where hers has +5 m/s². The hull velocity trace still matches hers to
      0.032 m/s rms.
    - **Split, part 1: the body model** (`cr06/catch_relative_acc_cr06.py`,
      body field only).
      - **Relative acceleration:** at the landmark, the 12-segment body's
        centre of mass accelerates at **12.40 m/s²** relative to the hull,
        against 14.20 from eq. 5 on her data. It lies below eq. 5 at every
        instant from −0.10 to +0.10 s.
      - **Contributions:** legs 4.03, trunk and head 7.36, arms 1.00.
      - **Not timing:** the model's lower trunk tracks her leg acceleration
        to within 0.5 m/s².
      - **Cause:** mass distribution. The shanks, part of each thigh and the
        arms do not move with the seat, which eq. 5's point mass assumes.
    - **That worth:** about 135 N of the gap. The segmental body is the more
      faithful centre of mass, so her stretcher at the catch is better
      estimated as 75 × (−8.55 + 12.40) + 24 ≈ **310 N**. That is still
      ~9× the gate force, and close to [LE26]'s *directly measured* 335 N
      for world-class women. **The target stands.**
    - **Part 2: the model's 33 N was an evaluation artefact** (2026-09-14).
      The script took hull acceleration from `sim.derivative` after setting
      `sim._in_air` all-False. That puts the blade in the water through the
      sweep catch's pre-entry phase (0–0.042 of the cycle, 53 samples),
      adding braking the simulation never applied. The books caught it:
      ∫M a dt = −3.96 N·s over a stroke whose velocity returns to its start.

      | hull acceleration | minimum | books over the stroke | stretcher at the catch |
      |---|---|---|---|
      | blade forced into the water (wrong) | −12.49 at 0.000 | −3.964 N·s | 33 N |
      | true per-sample air mask | −10.46 at 0.002 | 0.003 N·s | **190 N** |
      | gradient of the simulated velocity | −10.46 at 0.002 | 0.000 N·s | 190 N |

      - The two correct methods agree to 0.05 m/s² everywhere. The forced
        one is off by up to 2.07 m/s² at the catch.
      - **Corrected result:** the model's stretcher at the catch is **190 N**,
        against her ~310 N (segmental body) and [LE26]'s measured 335 N.
        Gate force is 34 N, so the stretcher leads. The ~120 N shortfall is
        the model's hull braking 1.9 m/s² harder than hers (−10.46 against
        −8.55 m/s²): the same fact as the 0.13 m/s-too-deep dip. It is
        external force, not body representation, since her 310 N already
        uses the segmental body.
  - **Step 4.3, second rung: the hip moment, top-down from the handle**
    (2026-09-14, `cr06/hip_moment_cr06.py`, `hip_moment_terms_cr06.py`).
    - **Why only the hip.** In the model's plane the seat's vertical load and
      the stretcher's vertical force share one equation. Knee and ankle
      moments are indeterminate until a foot-force direction for her is
      sourced ([BU13] ch. 8 has only men sweeping on an ergometer). The free
      body above the hip — trunk, head, arms — needs no foot data.
    - **A rigid body on her time-laws**, replacing the travel-scaled field,
      which was not a rigid chain:
      - **Hip:** from her leg channel.
      - **Trunk angle:** from her back channel, then remapped rigidly to her
        back travel: x = L cos a, x′ = x_c + K(x − x_c), a′ = arccos(x′/L),
        with L = 0.570 m and K = 0.771.
      - **Trunk masses** at fixed fractions along the link.
      - **Arms** as point masses carried by the shoulder.
    - **Checks, all passed before the moment was read:**
      1. **Kinematics round trip:** with both channels on the leg warp and
         K = 1, the field reproduces the model's own to 2×10⁻¹⁶ m,
         9×10⁻¹⁶ m/s and 7×10⁻¹⁵ m/s².
      2. **Her hull still follows:** 4.115 m/s (−1.8%), velocity rms
         **0.030 m/s**, slightly better than the non-rigid body's 0.032.
         Without the travel scaling it is 0.063.
      3. **Books close:** linear momentum −0.001 N·s over the stroke. About
         the moving hip, the analytic moment equals dH/dt + v_hip × P to
         0.4 N·m against a 93 N·m scale (rms 0.02). The stretcher at the
         catch is 186 N, matching rung 1's corrected 190 N.
    - **Result, extension moment of both hips** (per kg of 75 kg; per side
      is half):

      | instant | K = 0.771 (her travel) | K = 1 | [BU13] hip, ergometer | [BU13] L5/S1 |
      |---|---|---|---|---|
      | catch (min hull accel., 0.002) | 206 N·m, **2.74** N·m/kg | 2.95 | 2.93–4.43 | 7.82–11.87 |
      | maximum handle force (0.262) | 258 N·m, **3.45** | 3.54 | 5.36–5.52 | 11.92–12.17 |
      | peak in the drive (0.243) | 267 N·m, **3.56** | 3.67 | 5.80–5.93 | 12.99–13.67 |

      Flexion minimum −0.80 N·m/kg at 0.573, in the recovery.
    - **What it is made of** (terms sum exactly to the totals):

      | instant | handle | gravity | inertia (points + spin) |
      |---|---|---|---|
      | catch, trunk 127° | +6 N·m (23 N × 0.240 m) | +108 N·m | +89 + 2.7 N·m |
      | maximum handle force, trunk 110° | +125 N·m (549 N × 0.228 m) | +70 N·m | +61 + 2.5 N·m |

      - **The catch** is carried by the upper body's weight and the
        reversal of its motion, not by the handle.
      - **At maximum handle force** the handle is about half the moment.
        Its lever is only the hand's height above the hip, 0.23 m.
    - **Against [BU13]: pattern agrees, magnitude does not, and theirs cannot
      be the target.**
      - **Pattern:** catch below maximum handle force, and peak just before
        it.
      - **Side:** [BU13]'s tabulated hip is almost certainly one side. The
        right hip is doubled only as the pelvis input (§7.2.3.1, Fig. 7.2).
        So per side the model is about 3× lower: 1.72 against 5.4 N·m/kg at
        maximum handle force.
      - **The magnitude is out of reach.** 5.4 N·m/kg per side is ~405 N·m.
        At ~10 N/kg of handle force (their Fig. 7.7 range; ch. 8's men
        10.6), the handle gives ~86 N·m per side with this lever. The other
        ~320 N·m per side would have to come from gravity and inertia.
        This 44 kg upper body supplies ~130 N·m for *both* hips from
        weight at any trunk angle. Their L5/S1 values imply a 1.2–1.4 m
        effective lever on the handle force.
      - Bottom-up inverse dynamics accumulates error from foot centre of
        pressure, seat-force placement and finite differences. Their own
        text quotes 6–232% on peak moments, from gait.
      - **Use of [BU13] for 4.3:** pattern and order of magnitude only.
    - **Status of 4.3's rungs.**
      - **Stretcher force:** target sourced (~310 N at the catch from her
        data on a segmental body; [LE26] measures 335 N). The model gives
        190 N, short because its hull brakes 1.9 m/s² harder at the catch.
      - **Hip moment:** internally validated, statics-bounded, no
        quantitative external target yet.
      - **Knee and ankle:** blocked on a sourced foot-force direction.
        *Searched 2026-09-14* (SOURCES, "Foot-force direction on the
        water"). Sinclair et al. (2009) and Baca et al. (2007) measured
        on-water stretcher force in 3-D or in components, but for men only,
        and published no vertical values. [BU13] ch. 8 is men sweeping on an
        ergometer. Still blocked; a data request is the route.
      - **Next rung, determinate without foot data:** a forward-dynamic hip.
        Drive the trunk with the rung-2 hip torque, legs still prescribed.
        It must reproduce her trunk motion and hull to integration precision
        before any perturbation is read.
  - **Step 4.3, third rung: the trunk driven forward by the hip torque**
    (2026-09-14, `cr06/forward_hip_cr06.py`).
    - **Method.** The trunk angle becomes a state. Everything else stays as
      in rung 2: hip motion, hull acceleration, handle force, and the arms'
      offset from the shoulder. The moment balance is linear in the trunk's
      angular acceleration, which is solved exactly from two evaluations.
      Effective inertia: −8.8 to −8.9 kg·m² over the stroke, and the probe
      recovers the prescribed acceleration exactly. One-way coupling: the
      hull does not feel the forward trunk yet.
    - **A script bug, caught before any result.** The saved series carried
      a sample at phase 1.0, the same instant as phase 0. The periodic
      spline's wrap point landed 2×10⁻¹⁵ s from it, and the spline swung to
      10⁹–10¹² just past the seam. The first run blew up on it. That sample
      is now dropped, with an assertion on knot spacing.
    - **Round trip, passed:**

      | run | max trunk-angle error over the stroke |
      |---|---|
      | reconstructed moment against the saved rung-2 moment | 0.00 N·m apart |
      | A: reference torque, RK4 1000 and 2000 steps | **0.0028°** at both |
      | B: the saved rung-2 torque | **0.0147°** |

      A's error does not change with the step, so it is not integration.
      It sits at the end of the cycle, consistent with the smoothed warp
      being very slightly non-periodic at its seam.
    - **Open loop, unstable.** Run A started 0.1° off ends **72° off** after
      one stroke, a growth of 720×. That is an average e-folding time of
      about ln(720) / 1.94 s ≈ **0.29 s**: an inverted pendulum, 44 kg of
      upper body leaning over the hip on an effective inertia of 8.8 kg·m².
    - **So 4.3's torque-driven trunk needs feedback, and this sets the
      requirement.** Any stroke-tracking controller must correct an error
      that e-folds in about 0.3 s. The plan's quoted proprioceptive and
      vestibular delays, 30–100 ms, are shorter than that, but they are not
      yet sourced. *Next rung:* the same trunk with a tracking controller
      around the rung-2 torque as feedforward. Checks: the feedback torque
      it needs stays small against the feedforward, and it survives a
      sourced sensory delay.
  - **Step 4.3, fourth rung: a delayed feedback controller on the trunk**
    (2026-09-14, `cr06/tracking_hip_cr06.py`).
    - **Setup.** Hip torque = the rung-2 feedforward + a delayed PD on the
      trunk-angle error, M_fb = s(−Kp e(t−δ) − Kd ė(t−δ)).
      - *Slope:* s is the balance's own d(moment)/dα, −8.67 to −9.15
        kg·m² over the stroke.
      - *Gains, from the plant, not tuned:* Kp = ω_n² + 1/τ² with
        τ = 0.29 s (rung 3), and Kd = 2ζω_n with ζ = 0.7.
      - *Delays swept* over the sourced windows (SOURCES, "Sensory and
        reflex delays"): 0, 45, 75, 100 and 117 ms.
      - *Disturbances:* (1) a 0.1° starting offset; (2) the saved rung-2
        torque plus a 10 N·m bias, a chosen size.
    - **Sanity checks, passed.**
      - At ω_n = 20 rad/s with no delay, the offset decays to 0.00000°
        using 6.4 N·m of feedback.
      - With no feedback it grows to 72.1°, as in rung 3.
    - **Results** (max / end tracking error; peak feedback torque and its
      share of the 267 N·m feedforward):

      | ω_n | delay | 0.1° offset | torque + 10 N·m bias | held |
      |---|---|---|---|---|
      | 5 rad/s | 0–75 ms | 0.10 / 0.000°, 0.6 N·m | 3.8–3.9 / 3.3–3.5°, 22–23 N·m (8–9%) | yes |
      | 5 rad/s | 100 ms | 0.10 / 0.002° | 4.0 / 3.2°, 25 N·m (9.5%) | yes |
      | 5 rad/s | 117 ms | 0.10 / 0.013° | 4.6 / 3.0°, 30 N·m (11%) | yes |
      | 10 rad/s | 0–45 ms | 0.10 / 0.000°, 1.7 N·m | 0.7 / 0.7°, 13–19 N·m (5–7%) | yes |
      | 10 rad/s | 75 ms | 0.13 / 0.13° (not decaying) | 1.5 / 1.5°, 40 N·m (15%) | at the edge |
      | 10 rad/s | 100 ms | 8.0 / 6.3°, 235 N·m | 56 / 41°, 1640 N·m | **no** |
      | 10 rad/s | 117 ms | 60°, 1010 N·m | 91°, diverged | **no** |

    - **What it settles.**
      - **A feedback loop delayed as much as sourced trunk-muscle responses
        (90–117 ms) can hold her trunk, but only slowly.** At 5 rad/s it
        holds at every delay, and correcting a 0.1° offset costs 0.6 N·m.
        A constant 10 N·m bias, though, leaves a 3–4.6° steady error.
      - **Faster feedback buys accuracy but not delay tolerance.** At
        10 rad/s the bias error falls to 0.7° up to 45 ms of delay. It
        reaches the edge at 75 ms and fails at 100 ms: the classic limit
        near bandwidth × delay ≈ 1. Anticipation's ~17 ms [MS16] does not
        move it off the edge.
      - **So feedforward must carry the stroke.** At human latencies,
        feedback can make only slow corrections to the torque-driven trunk.
        For 4.3's stroke-tracking controller and tier 3's learned rower,
        the torque pattern itself has to be learned or prescribed almost
        exactly. The plan's per-channel delay design is now a measured
        constraint, not an assumption.
    - **Caveats.**
      - *One-way coupling:* the hull does not yet feel the controlled trunk.
      - *Controller form:* ζ is fixed at 0.7 and the controller is linear
        PD, so a richer one (integral action, prediction) could shift the
        bias error. The bias size is chosen.
      - *Latencies:* for trunk muscles in general, not rowers, and the
        upper-limb reflex windows only order the short delays.
  - **Step 4.3, fifth rung: the controlled trunk inside the simulation,
    two-way coupled** (2026-09-14, `cr06/coupled_trunk_cr06.py`).
    - **Method.** Trunk angle and rate are appended to the simulator's
      state. A study subclass overrides three methods: `derivative`,
      `augmented_initial_state`, and `_catch`, which otherwise zeroes
      everything after the oar angles at each catch.
      - *The loop:* the hull feels the trunk through the crew field, and the
        trunk feels the hull through its moment balance. Both are affine in
        the trunk's angular acceleration.
      - *Per stage:* each RK4 stage evaluates the simulator's own derivative
        at α = 0 and 1 and solves the hip-moment equation exactly.
      - *Torque:* the rung-2 feedforward plus undelayed PD at 5 rad/s,
        gains from the plant.
      - *Run:* her rig and force, 12 strokes at T/320.
    - **Two script slips, fixed before any result.** The loaded study
      scripts read their own command line, so the case name was parsed as
      rung 2's travel scale. Earlier, the rung-3 periodic seam (above).
    - **Results** (last stroke):

      | case | speed | velocity rms vs her | catch min | trunk error max / end | feedback torque |
      |---|---|---|---|---|---|
      | undisturbed | 4.1173 m/s | 0.0297 m/s | 2.98 | 0.0017 / −0.0010° | 0.0 N·m |
      | 0.1° start offset | 4.1173 | 0.0297 | 2.98 | 0.0017 / −0.0010° (decayed) | 0.0 N·m |
      | 10 N·m bias | 4.1183 | 0.0299 | 2.99 | **3.71 / −3.43°** | 22.2 N·m (8.3%) |
      | rung 2, prescribed trunk | 4.115 | 0.030 | 2.98 | — | — |
      | rung 4, one-way, bias, no delay | — | — | — | 3.79 / 3.46° | 22.5 N·m |

    - **Validated.** Undisturbed, the coupled system reproduces rung 2's
      hull to 2 mm/s, and her trunk motion to 0.002° with no feedback
      torque. The trunk equation, its feedforward and the coupling agree.
    - **Coupling changes almost nothing without delay.** Under the bias,
      trunk error and feedback torque match the one-way rung 4 within 0.1°
      and 0.3 N·m. A trunk held 3.4° off its path moves the hull by only
      1 mm/s. ~~So for this stroke, rung 4's delay findings carry over to
      the coupled system.~~ **Withdrawn — see below: with a 117 ms delay
      the coupled system diverges.**
    - **With delay, it does not carry over** (`coupled_trunk_delayed_cr06.py
      bias 0.117`). Same 5 rad/s PD, fed the trunk error 117 ms old. Over
      12 strokes the trunk error reaches **45.6°** (−31.4° at the end), with
      485 N·m of feedback (182% of the feedforward). Hull velocity rms
      against her data rises to 0.64 m/s, and stroke speeds swing from
      4.07 to 4.21 m/s. Rung 4, one-way over one stroke, had 4.59° / 30.4 N·m.
    - **Linear check** (`delayed_pd_linear.py`). The trunk's measured unstable
      mode, e″ = λ²e − Kp e(t−δ) − Kd ė(t−δ) with λ = 1/0.29 s, integrated
      for 40 s:
      - **5 rad/s:** stable at every delay swept. Growth −3.5 /s at 0 ms,
        −1.75 at 100 ms, −0.70 at 117 ms. **Boundary ≈ 132 ms.**
      - **10 rad/s:** stable to 45 ms (−6.8 /s); **unstable at 75 ms**
        (+0.45 /s); diverges within 3.5 s at 100–117 ms. **Boundary ≈ 72 ms.**
      - **Consequences:**
        - *Rung 4 correction:* its "10 rad/s at 75 ms: at the edge" is in
          fact slowly unstable. A one-stroke horizon hid growth of +0.45 /s.
        - *5 rad/s at 117 ms is linearly stable one-way*, but with only
          ~15 ms of margin.
        - *So the coupled divergence is not the one-way loop's.* Either
          the two-way coupling eats that margin, or the delayed coupled
          script is wrong.
    - **Discriminated: not the script; the limit is between 100 and 117 ms.**
      The coupled delayed script (5 rad/s, 10 N·m bias, 12 strokes):

      | delay | coupled: max / end error, feedback | one-way rung 4 (one stroke) | velocity rms vs her |
      |---|---|---|---|
      | 45 ms | 3.80 / 3.35°, 23.0 N·m | 3.87 / 3.38°, 23.0 N·m | 0.0303 m/s |
      | 75 ms | 3.88 / 3.21°, 23.7 N·m | 3.91 / 3.28°, 23.3 N·m | 0.0312 |
      | 100 ms | 4.17 / 2.89°, 27.9 N·m | 4.01 / 3.18°, 25.2 N·m | 0.0331 |
      | 117 ms | **45.6 / −31.4°, 485 N·m (diverged)** | 4.59 / 2.99°, 30.4 N·m | 0.64 |

      - **The delay buffer is right.** 45 ms holds and matches the one-way
        loop.
      - **The long-run boundary of the real system is between 100 and
        117 ms**, below the constant-coefficient linear estimate of ~132 ms.
        Two things the linear model leaves out could take that margin: the
        two-way coupling, and a plant whose slope and unstable rate vary
        through the stroke (slope −8.67 to −9.15 kg·m²). These runs do not
        separate them. The margin shows before the edge: feedback torque and
        velocity rms rise with delay, so 100 ms may itself be marginal over
        longer runs.
      - **Against the sourced trunk latencies** (unanticipated 103–117 ms,
        ~17 ms shorter anticipated [MS16]), the gentlest useful loop sits on
        its stability edge. It holds at anticipated latencies and fails at
        the slow end of unanticipated ones. **So rung 4's conclusion
        strengthens:** at human latencies feedback can be only a slow trim
        on a feedforward stroke, and even a slow trim needs anticipation to
        be safely stable. For 4.3's controller and tier 3 that makes the
        feedforward torque pattern the thing that must be right, and
        prediction (anticipating the stroke's own disturbances) a
        requirement rather than a refinement.
    - **Prediction, checked on the linear trunk mode** (2026-09-14,
      `smith_predictor_linear.py`, `state_predictor_linear.py`; e″ = λ²e + u,
      λ = 1/0.29 s, PD gains as above, 40 s runs, boundary by bisection):

      | controller | 5 rad/s: 117 ms / 200 ms / boundary | 10 rad/s: 117 ms / 200 ms / boundary |
      |---|---|---|
      | plain delayed PD | −0.70 /s / diverges / **132 ms** | diverges / diverges / **71 ms** |
      | plain Smith predictor, exact model | diverges within ~3 s at every delay 100–200 ms | — |
      | state predictor, exact model | −3.50 / −3.51 /s / **> 600 ms** | −7.02 / −7.02 /s / > 600 ms |
      | state predictor, model λ −20% | −3.70 / −1.82 /s / 343 ms | −7.64 / −6.12 /s / 472 ms |
      | state predictor, model λ +20% | −3.25 / −2.54 /s / 415 ms | −6.34 / −4.57 /s / 471 ms |

      (Negative rates are the decay of the trunk error.)
      - **The plain Smith predictor fails even with an exact model.** That
        is the textbook result for an unstable plant: its free-running
        internal model drifts on the unstable mode.
      - **The finite-horizon state predictor works.** x̂ = A^L x(t−δ) +
        Σ A^(L−1−i) B u over the delay window, with exact ZOH matrices. With
        an exact model it returns the *undelayed* decay rates at 117 and
        200 ms. With the model's λ 20% wrong either way it still holds to
        343–472 ms.
      - **Trap caught on the way.** The window sum was first updated
        recursively, through the unstable model. Its round-off grew like
        e^(λt) and made the exact-model predictor "diverge" at +3.3 /s in
        40 s runs. It is now summed directly from precomputed powers; the
        debug trace shows prediction error ≤ 8×10⁻⁹ once the window fills.
      - **So the conclusion sharpens.** Human latencies stop limiting the
        trunk controller once it has even a rough internal model of the
        trunk, run forward over the delay with its own recent commands.
        That is what the plan's tier-3 policy, given its own action history
        across a delay-aware context window, is meant to supply. Without
        such a model, the gentlest useful feedback sits on its stability
        edge at sourced trunk latencies.
      - *Caveat and next rung:* this is the linear, constant-coefficient
        mode. The next rung is the same predictor inside the nonlinear
        coupled simulation at 117 ms, the case that diverged, with a local
        linear model (slope and λ) as the internal model.
    - **The predictor in the nonlinear coupled simulation**
      (`coupled_trunk_predictor_cr06.py bias 0.117`). The local linear
      model (λ = 1/0.29 s) runs forward over the window. The controller's
      own commands sit on a 1 ms grid, held between samples and summed
      directly.

      | 117 ms, 10 N·m bias | max / end trunk error | feedback torque | velocity rms vs her | speeds |
      |---|---|---|---|---|
      | plain delayed PD | 45.6 / −31.4° (diverged) | 485 N·m | 0.64 m/s | swinging 4.07–4.21 |
      | **state predictor** | **12.3 / −11.1° (stable)** | 78 N·m (29%) | 0.040 m/s | settling, 4.1252 → 4.1218 |
      | undelayed PD | 3.71 / −3.43° | 22.2 N·m | 0.030 m/s | settled |

      - **Stability restored:** prediction stops the divergence on the real
        stroke.
      - **Accuracy not restored:** the steady offset is 3.2× the undelayed
        loop's, using 3.5× the feedback torque.
    - **Mechanism, confirmed linearly** (`predictor_bias_linear.py`): the
      same bias as a constant 1.13 rad/s² on the plant, absent from the
      model.

      | controller | steady trunk error |
      |---|---|
      | plain PD, no delay | 2.59° (analytic −d/(Kp − λ²) = 2.59°) |
      | predictor, 117 ms, bias unmodelled | **5.43°** |
      | predictor, 117 ms, with a disturbance estimate | **2.59°** |

      - An unmodelled constant load makes the predictor misread the
        delayed state's drift, so it settles at a larger offset. A
        disturbance estimate removes that: here, a slow observer on the
        mismatch between the delayed measured and modelled acceleration.
      - The coupled run's larger ratio (3.2× against 2.1×) fits the real
        trunk's instability rate and moment slope varying through the
        stroke, which the constant model does not follow. The runs do not
        prove that part.
      - **What 4.3 takes from this:**
        1. Feedforward carries the stroke.
        2. Without an internal model, feedback at human latencies sits on
           its stability edge.
        3. An internal model run forward over the delay restores
           stability.
        4. To keep accuracy that model must also *estimate unseen loads*,
           not only replay its own commands.

        For tier 3, the learned rower's context must be able to infer
        disturbances, not only compensate for delay.
      - ~~*Not yet run:* the predictor with a disturbance estimate inside
        the coupled simulation.~~ Run below.
    - **Predictor + disturbance estimate in the coupled simulation**
      (`coupled_trunk_predictor_obs_cr06.py bias 0.117`).
      - *Observer:* slow, K_OBS = 2 /s. It compares the delayed measured
        trunk acceleration (5 ms difference of the logged rate) with the
        model's λ²e(t−δ) + u(t−δ) + d̂.
      - *Prediction:* d̂ is added to the commands the model is driven with
        over the window.

      | 117 ms, 10 N·m bias, last stroke | max / end trunk error | feedback torque | velocity rms vs her | speeds |
      |---|---|---|---|---|
      | plain delayed PD | 45.6 / −31.4° (diverged) | 485 N·m | 0.64 m/s | swinging |
      | predictor only | 12.3 / −11.1° | 78 N·m | 0.040 | settling |
      | **predictor + disturbance estimate** | **4.44 / −3.61°** | **27.6 N·m (10%)** | **0.0314** | settling, 4.1216 → 4.1184 |
      | undelayed PD (reference) | 3.71 / −3.43° | 22.2 N·m | 0.030 | settled |

      - **Accuracy restored at the slow end of sourced trunk latencies.** The
        steady offset is within 0.2° of the undelayed loop's, for 24% more
        feedback torque. The hull's velocity trace stays at her boat's.
      - **The estimate settles at d̂ = −1.49 rad/s²**, against the bias's
        1.13 in magnitude. The sign is right for the model's convention,
        where a positive torque bias is a negative acceleration.
      - The extra 32% is model mismatch the estimate also absorbs: the trunk's
        instability rate and moment slope vary through the stroke, and the
        internal model is constant. So d̂ estimates everything the model does
        not know, not the bias alone. That is the property wanted.
    - **The trunk-controller thread of 4.3, as validated rung by rung**
      (her stroke, female table, two-way coupled):
      1. **Feedforward** carries the stroke. The rung-2 torque alone
         reproduces her trunk and hull.
      2. **Plain feedback** at sourced human latencies sits on its stability
         edge. The long-run boundary is 100–117 ms against 103–117 ms
         latencies.
      3. **Prediction over the delay** restores stability, but leaves a
         3.2× offset under an unseen load.
      4. **Disturbance estimation** restores accuracy to within 0.2° of an
         undelayed loop.

      For tier 3: the learned rower's context must both predict over its
      own latency and infer unseen loads. The plan's action-history window
      gives it the means; the acceptance tests should include a constant
      unmodelled load at a sourced delay.
    - *Open:* the legs. Knee and ankle are blocked on a foot-force
      direction; the data request is drafted (DATA_REQUESTS §11). Also a
      stroke-varying internal model, which should shrink d̂ toward the bias
      itself.
    - **Caveat.** The speed is still easing by ~0.7 mm/s per stroke at
      stroke 12: the run starts at the mean speed, and its first stroke
      overshoots to 4.40 m/s. The rung-2 comparison holds to 2 mm/s
      regardless.
    - *Next:* confirm one delayed case inside the coupled loop (5 rad/s,
      117 ms, bias). Then the legs, which stay blocked at the knee and
      ankle until a foot-force direction is sourced.
    - **The same artefact reaches five earlier [LE26] scripts.**
      `model_vs_legge`, `probe_vs_legge`, `retimed_body_vs_legge`,
      `warped_body_vs_legge` and `measured_inputs_vs_legge` built their hull
      acceleration the same way.
      - **Void until rerun:** every catch-window acceleration they recorded,
        above and in PHYSICS_PROGRAMME §4.3 — catch dip depths, the zero
        crossing after the catch, first peaks, rms.
      - **Being rerun** as `*_airmask.py` copies with the true mask. The
        originals and their outputs are kept.
      - **Unaffected:** `catch_accel_split.py`, which already set the true
        mask per sample (its "crew reaction dominates the catch" split
        stands). Also the Holt slip and gate-force scripts and the
        energy-book and step checks, which evaluate in-water samples only.
        And every [CR06] velocity comparison, which reads integrated states.
    - The shipped game is frozen, so any correction is research-side, and it
      needs a sex carried from the lineup or roster.
  - **Found while building it: the scull weighs as much as a sweep oar.**
    `SCULLING_OAR` (coxswain/boats/rig.py) sets no mass, so it inherits the
    `Oar` default of 2.7 kg, which is documented as a *composite sweep oar*
    (SOURCES, roll authority). [CR06] Table 1 measured 1.2 kg for a scull.
    About the lock that is 2.71 against 1.24 kg m² (uniform rod; [CR06]'s
    own I_G + m d² gives 1.233). Every scull run so far, including the
    [LE26] catch comparisons with the oar-only balance, carried twice the
    oar inertia. It matters most where the rower's torque is balanced
    against oar inertia alone.
    - **Fixed in the research profile (2026-09-14).** `PhysicsProfile.scull_mass`,
      1.2 kg on `research` and `learned`, `None` on `shipped`. `apply()` sets
      it on sculling rigs only, mass alone, geometry untouched
      (`tests/test_research_scull_mass.py`). The shipped game keeps 2.7 kg
      for its recovery roll authority. On the full research balance, where
      the reflected crew inertia sits on the oar, a 1.2 kg scull runs at the
      default step: 3.868 against 3.850 m/s at equal torque (+0.5%). The
      finer step below is needed only by the oar-only probe.
    - **Correction (2026-09-14): half the step is NOT enough for the
      oar-only probe, and mean speed was the wrong convergence test.**
      Rerunning [LE26]'s men with the 1.2 kg scull at T/160 reported 13% less
      power (423 against 484 W) at the same torque law, scale and speed. The
      gap is all in the water: entry work, torque in the air and torque past
      the finish are the same for both oars. The oar's energy books show why
      (`legge26/scull_mass_energy_books*.py`; per oar, over the in-water
      drive):

      | cohort, oar | ∫ I φ̈ φ̇ | ΔKE | φ̈ range, rad/s² | handle work |
      |---|---|---|---|---|
      | men, 2.7 kg | 6.6 J | 6.4 J | −66 to +8 | 394 J |
      | men, 1.2 kg | **370 J** | 2.6 J | **−459** to +9 | 346 J |
      | women, 2.7 kg | 5.3 J | 5.1 J | −50 to +6 | 300 J |
      | women, 1.2 kg | **−18 J** | 2.2 J | −113 to **+58** | 303 J |

      - **The light oar's books do not close at T/160**, badly under the
        men's load and mildly under the women's. Its oar mode is not
        resolved, so the men's power drop is integration error, not physics.
      - **The mean speed hid it.** It converged in the [CR06] check
        (0.6 mm/s), but speed averages over the drive.
      - **The committed research change stands.** On the full research
        balance (reflected crew inertia on the oar), a 1.2 kg scull at
        T/80, T/160 and T/320 gives:
        - speed: 3.8682, 3.8688 and 3.8691 m/s;
        - power: 267.91, 267.91 and 267.90 W;
        - φ̈ in the water: a smooth −5.5 to +4.1 rad/s² with one sign
          change at every step.

        It is resolved at the default step (`cr06/full_balance_step_check.py`).
        The problem is confined to the oar-only probe, where a 1.2 kg·m² oar
        faces the blade slip alone.
      - **The probe converges at T/320.** Men, 1.2 kg:
        - at T/320, books 2.9 J against ΔKE 2.8 J and handle work 397.2 J;
        - at T/640, books 2.8 J against 2.8 J and handle work 397.1 J;
        - the −147 rad/s² entry transient is real and resolved.

        The 2.7 kg control at T/640 does 394.7 J, against 394.3 J at T/160,
        with books 6.4 against 6.4 J. So the heavy oar was already resolved
        at T/160, and every earlier 2.7 kg result stands. At equal resolution
        the lighter scull does 0.6% *more* handle work (397.1 J), not 13%
        less. The [LE26] comparison is being rerun at T/320.
      - **[CR06] one-athlete test, T/160** (summed force, smooth body): books
        3.2 J against ΔKE 1.8 J, with φ̈ only to −79 rad/s² at her lower
        load. Speed 4.0866 and power 275.2 W match the recorded run.
        **Confirmed at T/640:** books 1.8 J against 1.8 J, speed 4.0887 m/s
        (+0.05%), power 275.5 W (+0.1%), finish at 0.455. Her lower load
        leaves T/160 within 0.1%, so the one-athlete results stand as
        recorded (`cr06/cr06_energy_books_step.py`).
      - The [LE26] rows with the 1.2 kg scull at T/160 are void. **Rerun at
        T/320** (`measured_inputs_vs_legge.py`, `_scull1.2` outputs):

        | build | 2.7 kg, T/160 | 1.2 kg, T/320 |
        |---|---|---|
        | men, measured force | 481 W · zero 0.131 · min −9.9 · rms 2.87 · shape 0.063 | 479 W · 0.131 · −9.9 · 2.87 · 0.062 |
        | men, force + body | 477 W · 0.160 · −17.9 · 3.18 · 0.061 | 475 W · 0.160 · −17.9 · 3.19 · 0.060 |
        | women, measured force | 341 W · 0.129 · −7.8 · 2.24 · 0.059 | 340 W · 0.129 · −7.8 · 2.24 · 0.058 |
        | women, force + body | 339 W · 0.155 · −14.2 · 2.47 · 0.057 | 339 W · 0.155 · −14.2 · 2.48 · 0.057 |

        *Air-mask and step correction, 2026-09-14.* Both columns are now
        resolved runs with the true blade air mask.
        - **1.2 kg column:** `measured_inputs_vs_legge_airmask.py`.
        - **2.7 kg column:** `measured_inputs_vs_legge_airmask27.py`. The
          column first shown here was the original run: default step (T/80,
          not T/160 as labelled) with the blade forced into the water before
          entry. Its values were min −11.5 / −19.1 / −9.3 / −14.8, rms
          2.85 / 3.13 / 2.22 / 2.48 and shape 0.070 / 0.094 / 0.058 / 0.058.

        - **The oar's mass changes nothing measurable here.** Catch minimum,
          return to positive acceleration, rms and gate shape all agree
          within 0.01, and power within 1%. Every [LE26] catch conclusion
          above stands.
        - **Power stays within 1%**, confirming the 13% drop as integration
          error.
        - ~~**The gate force notices.** The men's gate-force shape error falls
          11% with measured force and 36% with the measured body
          (0.094 → 0.060).~~ **Withdrawn.** The fall came from comparing a
          default-step 2.7 kg run with a finer-step 1.2 kg one. At resolved
          steps the shape errors are 0.063 against 0.062 and 0.061 against
          0.060.
    - **Consequence for the probes.** Every study that builds its single
      through `research.apply` and then puts the oar-only balance on it
      (the [LE26] measured-force and measured-body scripts, the catch
      probe) now gets the 1.2 kg scull. Rerun at the default step it will
      diverge; pass `dt = estimate_step(T) / 2`. Their recorded results
      carried the 2.7 kg oar. Rerunning the [LE26] catch comparison with the
      measured scull is open.
    - **Cache key fixed with it.** `_match_key` recorded oar geometry but not
      mass, and `total_mass` leaves the oars out. A 1.2 kg boat would have
      been handed a 2.7 kg boat's matched torque
      (`test_a_different_oar_mass_is_a_different_cache_entry`).
    - *History.* The shipped game uses this oar for recovery roll
      authority, so a correction belongs in the research profile. And at the
      default fixed step (T/80, capped at 0.5/80 s) a 1.2 kg oar under the
      oar-only balance diverges in the first stroke. A catalog-geometry oar
      at 1.2 kg fails the same way, and the [CR06] geometry at 2.7 kg runs,
      so mass alone is the cause. **It is the step, not the physics:** at
      half the default step it runs, and halving again changes the mean
      speed by 0.6 mm/s (4.3214 against 4.3219 m/s over four strokes). A
      lighter oar raises the blade-slip mode's rate beyond what RK4 at T/80
      resolves. A research-profile oar-mass correction therefore has to bring
      its own step.
  - *Diagnostic trap, recorded so it is not repeated.* Chaining
    `run_strokes(1, surge_speed=v)` with v the previous stroke's *mean*
    speed ratchets the boat up: each restart sets the catch speed to the
    mean, above the true catch speed, re-injecting momentum every stroke
    (4.19 → 5.58 m/s in ten). Continuous multi-stroke runs don't have this
    problem.
  - The men's oar-only run shows two notches in gate force and acceleration,
    at 0.24 and 0.37 of the cycle, most likely the light oar chattering near
    release. Its shape error is somewhat pessimistic for that.

*Still open.* The entry angle, 2–6° past the catch, has no measured target
(the digitised [CR06] Fig. 3 carries release markers only). Refused with the
following crew and with blade added mass, both of which were waiting on
exactly this. The recovery still holds the oar at the finish. The first settle
on a new boat configuration costs about 36 strokes.

### The following crew hands the hull momentum it never had
**Impact: high — it voids every number phase 4.1's crew mode produces.**

`DynamicOarSimulator(crew="follows")` slaves the body to the oar angle
(PHYSICS_PROGRAMME, phase 4.1). The hull feels the crew only through Σm·a, so
Σm·a integrated over a stroke must equal the change in the crew's momentum.
Measured on the last of six strokes at 380 W:

| boat | crew | ∫Σm·aₓ dt | Δ(Σm·vₓ) | worst single step |
|---|---|---|---|---|
| eight | clock | +0.00 N·s | 0.00 | 0.005 |
| eight | follows | **+361** | +108 | **−241 at the finish**, −34 at the catch |
| single | clock | −0.00 | 0.00 | 0.001 |
| single | follows | **+36** | +14 | **−26 at the finish** |

Three places the acceleration is not the derivative of the velocity:

1. **The finish.** The dynamic oar reaches the finish angle still sweeping
   and is held there; a body slaved to it stops with it, from a velocity the
   acceleration never took away. This is the bulk.
2. **The catch.** The retimed recovery arrives at the prescribed catch pose
   still moving (the erg-fitted body is not at rest there); the drive starts
   from rest with the oar.
3. **The rate floor.** Near both ends the velocity scale is capped so it does
   not diverge; there the pose and the velocity part company.

The settled run it produced (eight 5.06 m/s against the clock crew's 5.62;
drive fraction 0.424 against 0.400) is that defect. It is not a result.

**The finish is a defect of `research` too, not only of this study.**
Measured on settled strokes at 380 W, the dynamic oar reaches the finish
angle still sweeping at 74% of its peak rate on the eight and 69% on the
single with the clock crew (87% and 80% following), and is held there. The
kinetic energy dropped is **4.7% of the stroke's handle work** with the clock
crew, 3.0–3.1% following. A real oar-angle trace turns round at the finish;
ours stops dead at nearly full speed, because the pull shape is clipped at
zero and nothing but the blade slows the handle. The clock crew does not hand
the hull false momentum for it, but the energy is gone, which bears on every
speed-per-watt number `research` reports. Needs a source for the handle's
deceleration into the finish before it is changed.

*Fix options, in the order I'd try them:* (a) end the dynamic drive with the
oar at rest — a rower decelerates the handle into the finish, and the torque
shape does not; that removes the finish jump at its cause, and it is a pull
shape change, so it moves `research` too and must be scored; (b) where a jump
remains, apply its momentum to the hull as an impulse, which conserves the
system's momentum and books the lost energy honestly; (c) replace the capped
floor with the sweep of phase 4.2, whose ends are not singular.
`tests/test_crew_follows.py` holds the eight to 0.5 N·s as a strict xfail, so
the fix announces itself.

**Update — fix (b) built; the item is closed as a finding** (2026-09-13).
The finish and catch jumps are handed to the hull as impulses through the
system mass matrix (conservation unit-tested to 1e-9). What remains,
measured with RK4's own weights rather than a trapezoid on the samples (which
overstated it at +182): **+148 N·s a stroke on the eight, +13.5 on the
single**, inside the rate floor at both ends. Putting the true clock rate into
the acceleration was tried and overflowed — 7×10¹⁴ at the catch. The strict
xfail now accepts only an `AssertionError`: the first attempt diverged and a
bare xfail counted the crash as the expected failure.

Conclusion: a body slaved kinematically to the oar cannot keep its hands on
the handle *and* conserve momentum, because the erg-fitted body is still
moving where the sweep rate is zero. That needs a constraint force — phase
4.3 — and is not fixable inside 4.1.

**Update — a measured deceleration, and a sourced release rule** (2026-09-13).
Reading [CR06] in full changed both halves of this item:

1. *The release rule.* [CR06] takes the blade out when its normal velocity
   returns to zero — the blade force is exactly zero there — and never stops
   the oar: it keeps moving with the hands. Ours takes the blade out at a fixed
   `finish_angle` and holds the oar dead. And because the tier 1 blade force
   is `−sign(slip) C₂ slip²` while the blade is in, ours can *brake* in the
   late drive — the "re-anchor" on the blade-path figure — which [CR06]'s
   rule excludes by construction.
2. *A measured deceleration.* [CR06] Fig. 3 is vector, and its measured oar
   angle was extracted exactly (SOURCES.md, [CR06]). At the release time the
   measured oar is still sweeping at 80 °/s, 53% of its 150 °/s peak, and
   turns round 5.3° and 0.119 s later: a mean deceleration of 11.7 rad/s².
   Ours arrives at 69% of peak on the single and stops in one step. So the
   defect is the dead stop, not the speed at release. One athlete, and the
   release time is the model's — a target to check against, not a constant
   to fit.

What would use it, not yet done: the release rule is a candidate for the
dynamic oar on its own (it needs no new number); the recovery needs something
to decelerate the oar, which is the rower, so it belongs with 4.3.

**Fix (a) is blocked on data, and the `research` finish defect stays open.**
Searched for a source on the handle's deceleration into the finish: [FE17]
and [N-FISA] (SOURCES.md) confirm a measured oar turns round at the finish —
angular velocity zero there by definition — but neither gives a deceleration
law, and [FE17]'s 117 °/s peak was at 17–18 spm with no power reported, so it
is context and not a target. Needs a measured oar-angle trace, already on the
Blocked list.

**Update — the turn-round is reproduced from her own handle force** (2026-09-14,
`cr06/recovery_oar_from_her_force.py`, `finish_sweep_vs_her.py`,
`finish_dynamic_oar.py`). [CR06] Fig. 3 is that measured trace, and it
carries her handle force through release.
1. **Newton's law on the blade-out oar, fed only her data, turns it round
   as she does.**
   - *Equation:* I_lock θ̈ = −(F⊥ s + m_O d a_hull cos θ), with F⊥ half her
     summed handle force, s = 0.83 m, and a_hull from her velocity trace.
   - *Oar:* [CR06]'s 1.2 kg scull, I_lock = 1.233 kg·m².
   - *Start:* her release state (0.894 s, −39.10°, −80.3 °/s).
   - *Result:* the oar turns round at **0.999 s, −44.03°**, against her
     measured **1.025 s, −44.34°**. Angle rms is 0.67° over the first
     0.3 s. The hull term is worth 0.05°.
   - *With the catalog's 2.7 kg oar* it turns at 1.300 s and −53.8°, so the
     finish also confirms the research profile's 1.2 kg scull.
   - *Not closed beyond the turn:* integrated to the catch, the oar reaches
     4–10° against her 60°. Her recorded recovery handle force is too
     small to swing it back. The likeliest cause is the measurement: shaft
     strain gauges calibrated for static handle loads may not read the
     recovery's inertial loads. That part is the hands' and arms' in 4.3
     either way.
2. **The prescribed sweep cannot take the oar at release.** It turns round
   at 0.897 s, the model's drive end. At her release instant it is already
   at rest (−44.35°, 2 °/s) while she still sweeps at 80 °/s. Its recovery
   is 9.4° rms from hers, so a handover would only move the dead stop.
3. **What the dynamic oar does now, on her rig** (her force, whole measured
   body, T/320).
   - *At the finish:* it reaches the finish angle (−44.35°, her turning
     angle) at **0.870 s, still sweeping at 114 °/s** (71% of its 160 °/s
     peak), and is frozen.
   - *Energy:* 4.9 J a stroke across both oars, ~2.5 W, 0.9% of 277 W.
   - *Against her:* she is at −39.1° and 80 °/s at 0.894 s, and turns at
     1.025 s.
   - *So two defects are tangled.* The finish angle is her *turning*
     angle, and the oar arrives there early and fast. Unfrozen, it would
     overshoot her turn-round by ~9° (114 °/s stopping at her ~13 rad/s²).
- **The fix path, now sourced for one athlete:**
  1. Release by [CR06]'s rule: blade out when its normal velocity returns
     to zero. It exists as `release="slip"`.
  2. After release, keep the oar dynamic with the blade out, under the
     rower's handle torque, including the negative recovery force now
     clipped to zero.
  3. Leave the recovery past the turn-round to 4.3's hands.
  - *To validate before any code change:* on her stroke, the oar must turn
    round near 1.025 s and −44.3°, with no energy dropped at a hold.

**Building the fix as a study: two attempts that did not engage, and why**
(2026-09-14, `cr06/finish_fix_cr06.py`). The study subclasses the dynamic oar
on her stroke (female table, whole measured body, T/320, 8 strokes). It
releases the blade when its normal velocity returns to zero once the drive is
under way, keeps the oar dynamic with the blade out after release, and holds
it at turn-round. No committed code is touched.
1. **Attempt 1: the fixed finish angle intercepts the release.** No release
   fired in the drive. The oar reached the finish angle (−44.35°, her turning
   angle) at 0.870 s, still sweeping at 114 °/s, with its blade still
   *driving* (normal velocity negative). The existing code froze it there
   before [CR06]'s condition could be met. The "releases" that did log came at
   1.52–1.55 s in the recovery, an artefact of the frozen oar's stored rate
   against the hull.
2. **Attempt 2: finish angle out of reach, torque clipped at zero until
   release.** The dynamic oars' finish angle was set to −80° with
   `dataclasses.replace`; the boat's prescribed sweep, pre-entry path and hand
   tracks were untouched.
   - *Result:* **no release and no turn-round in 8 strokes.** The oar swept on
     past her turning angle with the blade in, up to 21.3° from her (12.2° rms
     over 0.80–1.10 s). Hull 4.145 m/s, velocity rms 0.0320 m/s.
   - *Why:* once her pull ends near 0.87 s, the only load on the torque-driven
     oar is blade drag, ∝ slip². The slip then decays toward zero like 1/t
     but never crosses it, so the release condition is approached and never
     met. [CR06]'s own model does not meet this, because its oar is slaved to
     the moving hands, which carry it through zero slip.
   - *So the push matters:* her measured handle force goes negative near the
     finish (−40 N at her release marker). That push is what can drive the
     slip through zero, and release the blade there before it brakes the
     boat.
3. **Attempt 3: her measured handle force unclipped throughout — the fix
   engages** (`finish_fix_cr06_unclipped.py`). Same slip release, blade-out
   phase and hold.

   | last stroke | release | turn-round | rate at the hold |
   |---|---|---|---|
   | model, attempt 3 | 0.909 s, −48.7°, −91.5 °/s | **1.023–1.025 s**, −54.0° | **0.0–0.8 °/s** |
   | her measured | 0.894 s, −39.1°, −80 °/s | **1.025 s**, −44.34° | 0 |
   | unfixed model | frozen at 0.870 s, −44.35° | none | 114 °/s |

   - **Both events fire on every stroke** (8 releases, 8 turn-rounds).
     - At 8 strokes: hull 4.138 m/s (−1.3%), velocity rms 0.0318 m/s.
     - That was not settled. At 16 strokes, where the last three agree to
       0.3 mm/s: **4.1175 m/s (−1.8%)**, rms 0.0322 m/s, power 281.4 W. The
       angles moved by under 0.3°.
     - Either way the hull is essentially unchanged.
   - **The finish dynamics match hers.** Release is 0.015 s after hers and
     turn-round lands on her 1.025 s. The oar travels 5.3° from release to
     turn-round against her 5.2°, and stops at essentially zero rate: no
     energy dropped at the hold, where the unfixed model dropped 4.9 J a
     stroke. Her push on the handle is what carries the slip through zero.
   - **The angles sit 9.7° past hers at turn-round, and most of that is
     built in the drive.** Model minus her, rerun with a readout
     (`finish_fix_cr06_attempt3b.out`):

     | stroke time | 0.30 s | 0.50 s | 0.70 s | 0.80 s | 0.85 s | 0.894 s (her release) | 0.95 s | 1.025 s (turn) |
     |---|---|---|---|---|---|---|---|---|
     | model − her | −0.2° | −3.1° | −5.9° | −6.8° | −7.3° | −8.1° | −9.4° | −9.7° |

     - The lead grows from mid-drive, and 8.1° of it is there when she
       releases. The fix's phase adds 1.6° more: the model releases 0.015 s
       later and at 91.5 °/s against her 80 °/s.
     - The oar angle rms against her over 0.80–1.10 s is 9.0°, which is this
       offset. The mid-to-late drive lead is its own open item.
   - *Power* reads 282.6 W against 277 W, and the two are not like for like.
     `handle_power` adds |τ·ω| at every blade-in sample short of the finish
     angle. With the finish put out of reach, it now counts the push against
     the oar between release and turn-round, which the unfixed run skipped,
     and her unclipped negative force.
- **Status of the finish defect.**
  - *The mechanism is validated on one athlete:* slip release, a dynamic
    blade-out oar under the rower's handle force including its push, and a
    hold at turn-round. It removes the dead stop and the energy it discarded,
    and times release and turn-round to her within 0.015 s.
  - *Promotion into `dynamic_oar` is blocked.* It needs a negative handle
    torque near the finish, and the research pull shape (a Beta profile) is
    clipped at zero. The push is one athlete's measurement, so a sourced
    recovery-push law for the pull shape comes first.
  - *2026-09-26, second athlete, and it does not transfer (SOURCES §158).* On [BR24] the
    slip release fires every stroke but the blade-out oar never turns round: his push is
    −1.4% of peak (hers −9%) and his recovery handle force is +8–15 N. The turn-round is not
    in the handle-force channel; it belongs with 4.3's hands. Promotion stays blocked, now
    for that reason rather than for want of a second push.
  - *Separately open:* the late-drive angle lead, and the recovery past
    turn-round (4.3's hands).

**The drive-angle lead: her own traces imply [CR06]'s fitted blade
coefficient, not the nominal one the model uses** (2026-09-14,
`cr06/implied_c2_from_her_data.py`, no simulator).
- *Why look at C₂.* The model's oar leads hers from mid-drive (above). A blade
  that slips more sweeps further for the same boat travel. The model uses
  [CR06]'s *computed* scull value, C₂ = ½ρC₀A₀ = 58.7 N·s²/m². [CR06] p. 208
  says the value that best fits these data is "about 2.4 times the nominal",
  and with it their oar-angle error halves.
- *The check, her data only.* It reads the model's own seat balance
  backwards, per oar with the blade in:

  $$F_n = \frac{I_{lock}\,\ddot\theta + \tfrac12 F_{hand}\, s + m\, d\, a_{hull}\cos\theta}{\ell}, \qquad \text{slip} = \ell\dot\theta + v\cos\theta, \qquad C_2 = \frac{F_n}{\text{slip}^2}$$

  The rig is [CR06] Table 1: s 0.83 m, ℓ 1.805 m, I_lock 1.233 kg·m², oar
  1.2 kg with its centre 0.595 m outboard. The traces are periodic Fourier
  fits to 49–51 points.

  | smoothing | velocity ×0.98 | ×1.00 | ×1.02 |
  |---|---|---|---|
  | 8 harmonics | 145 (2.47×) | 168 (2.87×) | 194 (3.30×) |
  | 10 | 143 (2.44×) | 168 (2.87×) | 192 (3.27×) |
  | 12 | 138 (2.34×) | 163 (2.77×) | 186 (3.17×) |
  | 15 | 139 (2.37×) | 167 (2.85×) | 200 (3.41×) |

  Each cell is the median implied C₂ over 0.15–0.73…0.79 s (the window's end
  moves with the settings), where slip < −0.4 m/s
  and F_n > 20 N.
- **Result: 2.3–3.4× nominal under every smoothing and a ±2% velocity
  scale, never near 1×.**
  - At mid-drive, 0.40 and 0.50 s, the value is 146 and 143, which is
    [CR06]'s 2.4× (140.9).
  - Her slip there is −0.88 and −0.94 m/s. The nominal blade needs −1.38 and
    −1.47 m/s to carry the same force.
- *Limits.*
  - Slip is the difference of two ~4.5 m/s terms, which is why a 2% velocity
    scale moves the answer ±15%.
  - The value is unsteady where slip nears zero: 317 at 0.20 s, 1118 at
    0.80 s.
  - One athlete, and a quasi-steady blade. [CR06] attribute the excess over
    Hoerner's plate to transient added mass (Wang 2005), which C₂ lumps
    together.
- **Her stroke with C₂ × 2.4, in the simulator** (the finish-fix study,
  `finish_fix_cr06_unclipped.py`, env `C2_SCALE`).
  - *Control passes.* The ×1.0 run reproduces attempt 3b line for line, so
    the edit changes nothing but the blade.
  - *Settled at 16 strokes* (`finish_fix_cr06_c2x{1.0,2.4}_n16.out`; the
    last three strokes agree to 0.3–0.4 mm/s). The 8-stroke runs had not
    settled, falling 14–17 mm/s a stroke, but their angles were within
    0.4° of these.

    | her stroke, her force | C₂ × 1.0 (58.7) | **C₂ × 2.4 (140.9)** | [CR06] measured |
    |---|---|---|---|
    | oar angle − hers at 0.30 / 0.50 / 0.70 s | −0.2 / −2.9 / −5.7° | +2.7 / +3.9 / +4.3° | — |
    | at 0.85 / 0.894 / 1.025 s | −7.1 / −7.9 / −9.6° | +2.8 / +1.6 / −2.0° | — |
    | oar angle rms, 0.80–1.10 s (max) | 8.87° (11.27) | **2.26° (3.90)** | their own fit: < 2.1° |
    | release: time, angle, rate | 0.909 s, −48.5°, −92 °/s | 0.905 s, **−38.7°**, −108 °/s | marker 0.894 s, −39.1°, −80 °/s |
    | turn-round: time, angle | **1.025 s**, −53.9° | 1.072 s, **−46.5°** | 1.013–1.025 s, −44.3° |
    | mean deceleration, release to turn | 791 °/s² | **646 °/s²** | 672 °/s² |
    | hull speed | 4.1175 m/s (−1.8%) | 4.0901 m/s (−2.4%) | 4.191 |
    | velocity rms vs her | 0.0322 m/s | 0.0331 m/s | — |
    | power from her force | 281.4 W | **250.0 W** | 260 W implied, 251 W of hull drag |

  - **Result: [CR06]'s fitted C₂ removes the drive-angle lead.**
    - The angle error falls fourfold, to about their own fitted residual.
    - The same handle force now does 250 W against 281 W. Her force times
      her handle speed gives 260 W, and the paper's hull drag over her
      velocity takes 251 W. With the nominal blade, her force was spending
      about 30 W on extra slip.
    - The hull loses 0.7% of speed, from −1.8% to −2.4%. So the blade is
      not what props up the speed prediction.
    - Three independent reads now agree: the fitted value in the paper, her
      traces alone (2.3–3.4×, 2.4× at mid-drive), and her angle trace in
      the simulator.
  - **What it leaves.**
    - *The early drive now lags her by 2.7–4.3° from 0.30 s.* With the
      nominal blade it was 0.2° at 0.30 s. A stiffer blade slows the oar
      from the catch, so the catch is now where the remaining angle error
      comes from.
    - *The turn-round is 47 ms late and 2.2° deep.* The deceleration after
      release, 646 °/s² against her 672, is right. So is the release angle,
      −38.7° against −39.1°. The oar simply leaves the water faster.
    - *The release rate is not a like comparison.* [CR06]'s release marker
      is their model's, not measured (SOURCES). At zero slip the rate is set
      by kinematics, θ̇ = −v cosθ/ℓ, which is about 110 °/s at her release
      angle and speed. Her own kinematic slip crosses zero near 0.82 s at
      the reference smoothing, before the marker, where she is still
      sweeping faster. The turn-round timing is the measured comparison,
      and it is 47 ms late.
    - *The nominal blade's exact turn-round timing was partly luck:* it got
      there 9.6° deep, releasing late at a deep angle.
  - **Candidate, not adopted.** The fitted value would be a research-only
    `PhysicsProfile` blade option. It stays out of the shipped game, whose
    `BladeModel.sculling` stays at 58.7.
    - *Against adopting it now:* it is one athlete's fit, and it lumps
      transient added mass into a steady coefficient ([CR06] cite Wang
      2005).
    - *No double count today.* `DynamicOarSimulator(blade_added_mass=...)`
      defaults to `"none"`, and `PhysicsProfile` never sets it; these runs
      had none. The Patton model is a study, and it refuses anything but
      `release="angle"`, so it cannot run with the slip release. Adopting
      the fitted C₂ would rule out also switching on the added mass later,
      and that should be written next to the option.
    - *Next:* check C₂ × 2.4 against Holt's singles catch and finish slips
      before any profile change. It is one athlete against a population.
- **Against Holt's singles: no target gets worse by more than 0.05 s, the
  speed gap closes by 3.0–3.5 points, and blade efficiency lands in
  Kleshnev's band** (2026-09-14, `phase4/holt_singles_c2_probe.py`).
  - *Setup:* the recorded catch probe's oar-only singles (body off the oar,
    Holt's conditions at 26 °C, power matched on handle power, 16 strokes)
    with the blade's c2 scaled on every simulator built.
  - *First launch diverged in the ×1.0 control too,* before any C₂ result.
    With the research profile's 1.2 kg scull (9917c7d), the oar-only balance
    diverges at the default step, as recorded above, and the probe predated
    it. The probe now takes `OAR_MASS` and `DT_DIV`.
  - *Control passes:* 2.7 kg, default step, ×1.0 reproduces all four
    recorded probe rows digit for digit, gate forces included.
  - *The measured scull on its own (1.2 kg, step/4, ×1.0), against 2.7 kg:*
    catch slip −0.6 to −0.8°, finish ±0.3°, speed +0.2 to +0.4 points, blade
    efficiency −0.003.

    | single, 1.2 kg, step/4 | catch / finish slip (Holt) | catch to peak (Holt) | peak / mean (Holt) | peak gate (Holt) | speed error | oar rate at 10° | blade eff. (Kleshnev 0.754–0.816) |
    |---|---|---|---|---|---|---|---|
    | M1x default, C₂ × 1.0 | 14.0 / 20.1° (7.7 / 14.1) | 0.49 s (0.43) | 2.17 (1.90) | 471 N (497) | −4.7% | 77 °/s | 0.687 |
    | M1x default, **× 2.4** | 14.0 / 20.1° | 0.52 s | 2.13 | 471 N | **−1.2%** | 70 °/s | **0.781** |
    | M1x cr06, × 1.0 | 12.7 / 15.8° | 0.48 s | 1.91 | 424 N | −4.4% | 80 °/s | 0.692 |
    | M1x cr06, **× 2.4** | 12.6 / 15.7° | 0.52 s | 1.88 | 424 N | **−1.0%** | 72 °/s | **0.785** |
    | W1x default, × 1.0 | 19.3 / 24.2° (9.7 / 18.1) | 0.56 s (0.39) | 2.15 (1.87) | 332 N (371) | −7.9% | 64 °/s | 0.698 |
    | W1x default, **× 2.4** | 19.2 / 24.3° | 0.60 s | 2.11 | 332 N | **−4.8%** | 58 °/s | **0.790** |
    | W1x cr06, × 1.0 | 19.0 / 20.3° | 0.55 s | 1.89 | 299 N | −7.5% | 67 °/s | 0.702 |
    | W1x cr06, **× 2.4** | 19.0 / 20.3° | 0.60 s | **1.87** | 299 N | **−4.5%** | 60 °/s | **0.793** |

  - **Blade efficiency: 0.69–0.70 → 0.78–0.79, inside Kleshnev's
    0.785 ± 0.031 on all four rows.**
    - Kleshnev is a population measurement, independent of [CR06]. So a
      coefficient fitted to one athlete's angle trace also brings a second,
      unrelated target into its band.
    - This is the singles, oar-only probe. The scorecard's pinned fail is on
      the sweep eight and four, which a scull C₂ does not touch.
  - **Speed at Holt's power: +3.5, +3.4, +3.1 and +3.0 points.** M1x ends at
    −1.0 to −1.2%, W1x at −4.5 to −4.8%. At equal power a blade that slips
    less converts more of it.
    - This does not contradict her stroke, where the hull slowed 0.6
      points. That run was at equal *force*, and the same force then did
      31 W less work.
  - **Slips and peak force unchanged** (≤ 0.1°, same peak N). Catch and
    finish slip are thresholds on gate force, and the pull shape and body
    set them, not the blade. The 5–10° catch miss is still 4.3's.
  - **One thing worse: force peaks 0.03–0.05 s later** (0.52 against Holt's
    0.43; 0.60 against 0.39). The stiffer blade slows the oar, by 9% at 10°
    past the catch, and lengthens the drive by 0.04–0.05 s. The catch to peak
    was already 0.06–0.17 s late. Peak/mean falls 0.02–0.04: toward Holt on
    three rows, and just past Holt's on M1x cr06 (1.91 → 1.88 against 1.90).
  - **Scope.** [CR06]'s 2.4× was fitted on singles only. The sweep C₂ (84.5)
    has no fitted value, so nothing here says the sweep blade should change,
    and the eight's and four's race-pace tension is untouched.
- **Decision recorded: adopt the fitted scull C₂ in the research profile.**
  - *It is sourced:* [CR06] p. 208 for these singles.
  - *It is confirmed three ways:* her traces alone; her oar angle, fourfold
    better; Holt's singles, with speed +3 points and blade efficiency into
    Kleshnev's band.
  - *It costs:* force peaks 0.03–0.05 s later.
  - *Not frozen:* the shipped game keeps 58.7.
  - *Next:* a `PhysicsProfile` field for the scull blade coefficient, set
    only on `research`/`learned`, with tests and the fast suite. It must
    record that the fitted value lumps transient added mass and stays
    exclusive of `blade_added_mass="patton"`.

**Adopted: `PhysicsProfile.scull_c2`** (2026-09-18, `tests/test_research_scull_c2.py`).
- *The field.* `scull_c2: Optional[float]`, N·s²/m², `None` on `shipped` and
  **140.88** (2.4 × 58.7) on `research` and `learned`. Refused when
  non-positive, and refused at `blade_tier=0`, which has no blade to set it on.
- *How it reaches the blade.* `apply` stamps `boat.scull_c2` on sculling rigs
  only; `OarDynamics.from_boat` replaces the blade's `c2` when the boat carries
  it. `BladeModel.sculling()`'s own default stays 58.7, so the shipped game and
  every un-profiled caller are untouched. An explicit `from_boat(blade=...)`
  still wins, so studies keep their override.
- *Sweep is untouched* (84.5): [CR06] fitted the 2.4× on singles only, and
  `apply` never stamps a sweep rig.
- *Exclusivity enforced, not just documented.* `DynamicOarSimulator` raises if
  `blade_added_mass != "none"` on a boat carrying a fitted `scull_c2`: the fit
  already contains the transient added mass, so running both counts the same
  water twice.
- *One existing test changed on purpose.*
  `test_blade_coefficient_follows_the_rig` pinned the research scull blade at
  58.7. Its intent — the coefficient follows the rig — is kept: it now asserts
  sweep 84.5, scull 140.88, and that the two still differ.
- **The adopted code path reproduces the study that justified it.** The Holt
  probe reached its ×2.4 numbers by multiplying `c2` inside a `ScaledSim`
  subclass. Re-run with no scaling at all, so the value comes from the profile
  instead, it prints c2 = 140.9 and returns the same rows:

  | single | catch / finish slip | speed error | blade eff. |
  |---|---|---|---|
  | M1x default | 14.0 / 20.1° | −1.2% | 0.781 |
  | M1x cr06 shape | 12.6 / 15.7° | −1.0% | 0.785 |
  | W1x default | 19.2 / 24.3° | −4.8% | 0.790 |
  | W1x cr06 shape | 19.0 / 20.3° | −4.5% | 0.793 |

  Identical to the hand-scaled run to the printed precision, by a different
  mechanism, which is what makes it a check rather than a restatement.
- *Suite:* 2025 passed, 17 xfailed, nothing else touched.

**Which blade law does her stroke demand? Tier 2 tested against measured
blade loads, and it fails** (2026-09-18, `cr06/blade_law_vs_her.py`).
- *Why ask.* Adopting the fitted C₂ removed the drive-angle lead and left a
  2.7–4.3° **lag** from 0.30 s: the residual moved to the catch, which is
  exactly where [CR06] say their own fit is worst. Tier 2 predicts a
  catch-specific correction, because the angle of attack there is small
  (the blade is at ~60° and the flow runs along the shaft) and 90° at
  mid-drive, where it reduces to Model 1.
- *The target.* Her oar's moment balance gives the normal blade load
  independently of any blade law:
  F_n = (I_lock θ̈ + ½F_hand·s + m d a_hull cos θ)/ℓ. Each candidate law is
  then evaluated on her own kinematics and scored against it.

  | law | rms | bias | catch → 0.35 s | 0.35 → 0.85 s |
  |---|---|---|---|---|
  | tier 1, computed C₂ 58.7 | 55.2 N | −50.5 | 39.9 | 63.1 |
  | **tier 1, fitted C₂ 140.88** | **25.1 N** | −6.0 | 32.2 | **19.4** |
  | tier 2, lift and drag as shipped | 82.3 N | +45.8 | **124.0** | 36.5 |

- **Tier 2 fails, and in the opposite direction to the hypothesis.** At
  0.10 s it predicts 246 N against her 31 N. The cause is structural: at a
  small angle of attack the *axial* flow (2.6 m/s) dwarfs the slip (0.8 m/s),
  so the dynamic pressure ½ρA(w_n² + w_a²) is large and the lift term turns it
  into a large normal load. The shipped constants make the blade behave like a
  hydrofoil through the catch.
- **It is the lift term, not the structure.** Halving A_l gives rms 38.9 N;
  her drive would choose **A_l = 0.25 and A_d = 4.11** against the shipped
  1.25 and 2.07 — a fifth of the lift and twice the drag.
- **And the second parameter barely earns its place.** That best-fit
  lift/drag pair reaches 21.5 N against the one-parameter slip law's 25.1 N.
- **An independent route to the adopted C₂.** Least squares of a pure slip law
  on her blade load gives **C₂ = 140.3**, against [CR06]'s fitted 140.88
  (0.4%) and their computed 58.7. Adopted today from their fit; recovered here
  from her trace.
  - *Not independent physics.* F_n is dominated by the handle term — it
    correlates 0.996 with her measured handle force — so this is the same
    data reached by a different route, not a second measurement of the water.
- *What it leaves.* The early-drive lag is **not** angle of attack. The
  candidates still standing are the entry transient (immersion and added mass
  developing over the first strokes of the drive) and the body-on-the-oar
  balance at the catch, which 4.3 owns.

**What the fitted C₂ actually is: three tests and a literature check**
(2026-09-18, `cr06/added_mass_vs_her.py`, `wake_growth_vs_her.py`,
`lever_sensitivity.py`).

1. **It is not a constant added mass, and [CR06]'s own explanation fails on
   their own athlete.** They attribute the 2.4× to transient added mass
   (citing Wang 2005). Fitting C₂ and an added mass jointly to her blade load
   gives **m_a = −1.50 ± 0.15 kg** against Patton's 13.1 for her blade — the
   wrong sign and 94 standard errors away — while C₂ stays at 140.3, moving 1%
   of the way toward the computed 58.7. The two basis terms are orthogonal
   (r = 0.005), so the fit can see them separately, and the conclusion holds
   from 8 to 15 harmonics. Forcing m_a to Patton's value leaves C₂ at 140.3
   and raises the rms from 23.7 to 83.4 N.
2. **But the mechanism survives, as a growing wake rather than a constant
   mass.** [G19] measured an accelerating AR-2 plate: a constant added mass is
   *inadequate* because the entrained mass grows with travel, and the transient
   lasts **7–8 plate heights**. Her blade travels **0.54 m through the water in
   the whole drive — 2.5 blade widths — so the drive never leaves the
   transient.** And her effective coefficient does grow: fitting force to slip²
   without ever dividing, **C₂ = 120 over the first half of the travel and 160
   over the second**, a 34% rise, with the second half fitting better (rms 17.3
   against 26.0 N). The trend strengthens under stricter slip filters
   (r = 0.45 → 0.93 as the cut rises), which is the opposite of the
   small-denominator artefact that would otherwise be suspected.
3. **The level is not a clean fluid coefficient, because it is entangled with
   the lever arm.** C₂ = 140.88 implies C_D = 2C₂/(ρA₀) = **3.12**, against
   [G19]'s measured 1.10 at the surface, 1.30 deep and 1.60 at 20 mm. But ℓ
   enters twice — dividing the moment and setting the slip — and mid-drive the
   slip is a small difference between ℓθ̇ ≈ 4.3 and v cos θ ≈ 4.5:

   | ℓ | C₂ | implied C_D | rms |
   |---|---|---|---|
   | 1.700 | 256 | 5.68 | 38.6 N |
   | **1.805** (adopted) | **140** | **3.11** | 25.1 N |
   | 1.900 | 88 | 1.94 | 18.3 N |
   | 2.020 (her outboard) | 53 | 1.17 | 14.2 N |

   *But the lever arm cannot be moved far enough to rescue it.* Her blade spans
   1.590–2.020 m from the pin; its geometric centre is 1.805 and, if the slip
   were purely rotational, the centre of pressure would sit at 1.822. The
   defensible range is 1.805–1.822, where C_D stays 2.9–3.1 — still about twice
   [G19]'s measured maximum.
   - *An unexplained flag, recorded rather than resolved:* the rms keeps
     falling monotonically as ℓ grows past any physical value. The balance is
     absorbing a systematic error that a longer lever partly compensates.
     Candidates: blade immersion (less wetted area than the nominal 0.0903 m²),
     shaft bend putting the blade behind the measured shaft angle, or an
     oarlock water-relative velocity that is not the hull's surge.

**What this means for the value adopted this morning.** It stands on what it
does, not on what it is: it predicts her boat speed, quarters her oar-angle
error, and moves Holt's singles the right way. It should **not** be quoted as a
measured blade drag coefficient — at her geometry it implies a C_D about twice
anything measured, and it is acutely sensitive to a lever arm the data cannot
pin down. The docstring now says so.

**The primary arrives, and the level stops being anomalous** (2026-09-18;
Caplan & Gardner 2007 supplied by the project owner, now [CG07] in SOURCES).
- **The constants this project uses are verified.** Their measured flat-plate
  C_Dmax is ≈ 2 at α = 90° and C_Lmax falls at α = 40–45°. Atkinson's rendering
  evaluates to A_d at 90° and A_l at 45°, so his 2.07 and 1.25 *are* their
  peaks. The amplitudes were never the problem.
- **So the tier-2 failure is re-diagnosed, and more precisely.** What breaks at
  a catch is the *interpolation* between those measured peaks: α is ~12° there,
  far from either, and `sin 2α` at small α is Atkinson's curve rather than
  their measurement. Add that [CG07] is quasi-static with the blade's top edge
  flush with the surface, and the model is being asked for a load in a
  configuration nobody measured.
- **And the C_D = 3.12 that looked wrong is now roughly where it should be.**
  [CG07] ignores non-steady vortex development by construction; [G19] measured
  what that omission is worth — peaks of 3.6 N against 2.6–2.8 N predicted,
  +30–40% — and showed the transient outlasts a whole rowing drive. A
  quasi-static ≈ 2.0 carrying that enhancement is ≈ 2.6–2.8, against the fitted
  3.12. An on-water effective coefficient *should* sit above every flume
  number, and it does, by about the measured amount.
- *What is still unexplained:* the monotone rms improvement toward an
  unphysical lever arm. Reconciling the level does not touch it.

**The unexplained flag, chased down: it is partly the shaft bending**
(2026-09-18, `cr06/shaft_bend_vs_her.py`).
- *Why suspect it.* The residual is phased, not a constant bias: with
  ℓ = 1.805 the law over-predicts early (94 N against her 31 at 0.10 s) and
  under-predicts late (5 N against 40 at 0.80 s). A shaft that bends under load
  does exactly that — the blade trails the pin while the load builds, so the
  true slip is smaller than computed early and larger late, since
  θ̇_blade = θ̇ + δ̇.
- *Modelled* as δ = c·(F_hand/2), with the target left alone: it is measured at
  the handle, so bending changes where the blade is, not what the rower pulled.
- *Constrained by measurement, not fitted freely.* Laschowski, Nolte,
  Adamovsky & Alexander (2015) loaded sculling oars to **201 N at the blade**
  and measured a deflection angle at the blade end of **at most 1.18 ± 0.01°**.
  A free fit here wants 1.43° at a blade load of only 127 N — about **1.9×
  softer than real oars are** — so the free fit is absorbing more than a shaft
  can give.

  | | C₂ | C_D | rms |
  |---|---|---|---|
  | rigid shaft | 140.4 | 3.11 | 25.5 N |
  | **bend at the measured stiffness** | **142.6** | 3.16 | **20.2 N** |
  | bend fitted freely (too soft) | 140.9 | 3.12 | 18.3 N |

- **Verdict: real, correctly phased, worth about a fifth of the residual, and
  not the explanation for the 2.4×** — it moves C₂ by 1.5%. At the measured
  stiffness the peak lag is 0.75°, 2.4 cm at the blade.
- *It also only flattens the lever-arm drift rather than removing it* (rms
  across ℓ = 1.805 → 2.02 falls 25.5 → 14.5 rigid, 18.3 → 16.1 with bend), so
  something smaller is still there. Immersion — less wetted area than the
  nominal 0.0903 m² — is the remaining candidate, and it needs data this
  dataset does not have.

**Model status before advancing: the adopted C₂ is robust.** It is the one
number the next stage rests on, and it does not move under any plausible
elaboration of the model:

| model extension | fitted C₂ |
|---|---|
| none (pure slip law) | 140.3 |
| + constant blade added mass | 140.3 |
| + shaft bend at measured stiffness | 142.6 |
| + shaft bend fitted freely | 140.9 |
| lift and drag instead of slip² | rejected: rms 82 N against 25 N |

Only the assumed centre of pressure moves it, and the blade's geometry pins
that to 1.805–1.822 m. The residual that remains is ~20 N rms against peak
blade loads of 127 N, about 16%, with its phase now understood.

*End-to-end gate passed* (`cr06/her_via_profile.out`). Her stroke re-run with
no manual scaling, so the coefficient comes from the profile rather than a
study subclass, reproduces the study digit for digit: oar-angle rms 2.26°
against her over 0.80–1.10 s, hull 4.0901 m/s (−2.4%), velocity rms 0.0331 m/s,
250.0 W, turn-round 1.072 s at −46.5°. Together with the Holt singles check and
the 2025-test suite, the adopted path is the studied path.

**So 4.1's blade is closed for now, and the live defect is 4.3's.** The model's
oar no longer leads hers; it lags by 2.7–4.3° from 0.30 s, and the catch is
where that sits. Angle of attack is ruled out, added mass is ruled out, shaft
bend is measured and too small to carry it. What is left is the body on the
oar at the catch — which is phase 4.3, and which is partly blocked on the
foot-force data in DATA_REQUESTS §11.

### The drive is 18–28% too long, and the cause is the ergometer
**Impact: high — drive duration sets the time base of the whole stroke.**

`StrokeTiming.drive_fraction` was changed to `0.63067 − 5.20991/rate`,
fitted to Telfer et al. (2023) — who measured **ergometer** rowers —
because [F09]'s quadratic gave a catch fraction of 0.300 against Telfer's
0.394 at 22 spm.

That traded away an out-of-sample validation. [HF09] Table 1 measured
eight elite coxless **pairs on the water**:

| rate | measured drive | model | error | [F09]'s formula |
|---|---|---|---|---|
| 20.6 | 862 ms | 1100 ms | **+27.6%** | 829 ms (−3.8%) |
| 24.2 | 810 ms | 1030 ms | **+27.1%** | 798 ms (−1.4%) |
| 27.7 | 779 ms | 959 ms | **+23.1%** | 772 ms (−0.9%) |
| 31.5 | 752 ms | 886 ms | **+17.9%** | 748 ms (−0.6%) |

On the water the drive is 29.6–39.5% of the cycle; the ergometer fit says
37.8–46.5%. The gap is about **0.08 of the cycle at every rate**, and it
has an obvious cause: on the water a crew has to let the boat run, and on
a stationary ergometer there is no boat to run. Both datasets are right
about their own conditions — and this model is of a boat.

This is defect two of the physics programme, showing up in the single most
directly measurable quantity in the stroke, and it was sitting in a
regression test nobody had run green since 26 August.

*Not fixed here.* Putting it right moves the time base of every calibration
in the project, so it belongs in the `research` profile behind the
scorecard. `tests/regression/test_paper_validation.py` carries the [HF09]
comparison as a strict xfail, so the fix announces itself.


*2026-09-26, a third data point that cuts the other way (SOURCES §158):* [BR24]'s elite
single at 32.4 spm keeps force above 10% of peak for 0.926 s (0.50 of the cycle). The
research model's nominal drive is 0.47, its pull is above 10% for only 0.688 s, and driven by
his force its oar reaches the finish at 0.79 s. For this single the model's effective drive
is too **short**. [HF09]'s pairs are not a general on-water drive time; do not move every
boat to them.

### Handle power moved 53% and nobody noticed
**Impact: high — handle power is what CP and W′ are measured against.**

`a053540` made the blade load perpendicular to the shaft. Its message
says `f_x` is unchanged so "nothing about speed or power moves". That is
true of thrust and of boat speed. **It is false of handle power.**

The handle sweeps about the pin, so its velocity is perpendicular to the
shaft, and `mean_handle_power` dots the *whole* force vector with it.
With the old `+side` sign the force sat at an angle to the handle
velocity and the dot product carried a `cos(2φ)` factor; with the
corrected sign the force is parallel to the handle velocity and the
factor is 1. Measured directly on the eight at rate 28:

| sign | mean handle power |
|---|---|
| old, `+side` | 411.1 W |
| corrected, `−side` | 629.6 W |

**+53%.** Every η in this file moved by the same factor, `mean_handle_
power`'s own docstring ("about 470 W per rower at scale 1.0") is stale,
and so is any pacing or W′ calibration that was fitted against the old
number — including v0.13's per-rower reserve.

The new value is the physically right one: a rower pulling perpendicular
to the oar does full work on it, and the old `cos(2φ)` was an artefact of
the sign bug. So this is a correction, not a regression. But it was
invisible for the same reason everything else here is: **nothing measured
it.** The validation scorecard caught it on its first run, which is what
the scorecard is for.

#### What it does to every published speed

`mean_handle_power` is the unit that converts watts into `power_scales`.
Both the trainer (`fpv.py`: `base_scale = nominal_power / reference_power`)
and the report (`hocr_four`: `power_scales = target / unit`) divide by it,
so a 53% rise in the unit is a **35% cut in the force scale** for the same
stated wattage. On this project's own four, at its measured 131.2 W per
rower:

| | handle power at scale 1.0 | power scale for 131.2 W |
|---|---|---|
| before `a053540` | 486.4 W | 0.2698 |
| after | 744.9 W | 0.1761 |

So the boat is driven **35% softer** for the same crew watts, and every
speed computed before `a053540` is stale — including the report's 2.70 m/s
for the four and the "+7% / +35% optimistic" split, none of which have been
rebuilt since. **The report needs a full re-run before any of its numbers
are quoted again.** The released trainer is *not* affected: v0.13 was cut
at `6836a47`, before this commit.

*Still open, and separate:* `mean_handle_power` dots the **oarlock**
force with the **handle** velocity. Under [F09]'s ideal lever those are
not a conjugate pair — `F_h = −(L − r_h)/L · F_o` — so the product may
carry a gearing factor it should not. Not investigated; flagged because
it sits underneath every power number in the project.

### Still open: should an eight have a combined fin-and-rudder?
**Impact: low, but it is an unexamined modelling change.**

`0b35995` replaced the eight's separate skeg with one `_fin_with_rudder`,
so `catalog.eight()` now has a single controllable surface where the four
still has `[skeg, rudder]`. Physically a shell's rudder *is* a flap on the
fin, so the combined surface is defensible — but the change was made
inside a large commit, it silently disabled 14 tests (see Fixed), and
nothing records why the eight and the four should differ. Not settled.

### The two speed models disagree by a factor of 2.3 in efficiency
**Impact: high — it decides every published target.**
**Cause found:** see the item above — the blade force carries no
velocity term, so the implied efficiency is proportional to speed.

`CoursePacing` (which produces the targets in `scripts/targets.py`)
solves `R(v)v = 0.80 x gate power x rowers`: a flat blade efficiency of
0.80. The 6-DOF simulator, driven through `power_scales`, implies an
efficiency that RISES with power and is far lower:

| gate W/rower | simulator speed | implied efficiency |
|---|---|---|
| 131 | 2.71 m/s | 0.34 |
| 188 | 3.29 m/s | 0.40 |
| 250 | 3.89 m/s | 0.49 |
| 486 | 5.60 m/s | 0.70 |

Some rise is real — a slower boat slips its blades more — but 0.34 is
implausibly low, and the two halves cannot both be right. On the same
crew they predict 23:49 and 29:43 for the same course.

The field says `CoursePacing` is closer: the slowest crew in six years
of results is 25:33, and the simulator would need 188 W a rower just to
row that, which is a strong 60+ erg. But this is inference, not
measurement.

**Localised to the oar-to-hull conversion.** Over one cycle at scale
1.0, the mean forward force the oars put into the hull is 248.7 N, and
the crew's handle power is 1945 W. At the speed the simulator settles
at, that is a propulsive-to-handle ratio of **0.346** -- and 0.346 is
exactly `blade_efficiency x inboard/outboard` (0.78 x 0.445).

The FORCE relation looks right: for the boat-and-crew system the
external propulsive force is the blade's reaction, `F_handle x
inboard/outboard`. What does not reconcile is the POWER. For the
efficiency to reach 0.78 the boat would have to move about 2.25 times
the handle speed; the geometry here gives nearer 1.2-1.6, and the
shortfall is the whole gap.

So the open question is narrow and worth stating exactly: whether the
handle kinematics (and therefore `mean_handle_power`) are consistent
with the force path, not whether the hull or the gearing is wrong.

**Why no existing test caught it.** The Holt validation SCALES oar force
until the delivered power matches a target, so it never exercises the
handle-power-to-force relation at all -- it is calibrated around
whatever that relation happens to be. The 6-DOF path is validated for
force to speed and unvalidated for watts to force, and the trainer uses
the second.

**Partly settled, and it goes against the simulator.** Sammamish rowed
this category in 2024 and finished 12th of 20 in 22:07.9. Asked what
that took:

* `CoursePacing` says **164 W a rower** -- a 5 km erg of 21:27.9, which
  is an ordinary masters women's score and lands mid-field exactly where
  the crew finished;
* the 6-DOF path says **223 W** -- a 5 km of 19:22.6, which for a 60+
  woman is close to national level, for a crew that came 12th of 20.
  That is not believable.

So the flat 0.80 is closer to reality than the simulator's power chain,
and the published targets stand. What is still open is WHY the 6-DOF
path is so lossy -- whether the fault is in `power_scales`, in the
blade model, or in the unsteady losses.

### The balance controller has nothing to do
Found while making the crew's port arms port (the kinematics signature
omitted the side; see Fixed).  With that bias gone, the 6-DOF eight is
**roll-stable at race pace on its own**: from an 8 deg initial roll it
returns to 0.04 deg within 6 s with no balance controller at all, and
every authority — full, oars-only, constant, none — gives the same swing
(0.047 deg matched, 1.331 deg at a 60 ms split).

The terms at 2 deg of heel, 4.6 m/s, eight: buoyancy −36.7 N m, gravity
+75.7 (net +39, destabilising, as `roll_divergence_time`'s surrogate
says) — and **appendage −209.4 N m**, at zero sideslip and zero roll
rate.  A fin restoring roll five times harder than the hull destabilises
it is either a real heel–sideforce coupling or a frame slip in how the
appendage moment is taken about the hull origin; nobody has checked
which.  Until that is settled, "the boat has to be sat" is a surrogate
claim the simulation contradicts, and the two balance tests that assumed
it are expected failures pointing here.

### The crew cannot hold a steady heel through the recovery
The phase-limited balance authority (SOURCES §15, `PhaseAuthority`:
1525 N m on the drive, 93 N m on the recovery) loses an eight to a steady
heeling moment just beyond the recovery figure, where the flat 4000 N m
`max_moment` holds it to hundredths of a degree.  Measured with the
harness in `tests/unit/test_trim.py`: eight at rate 32, 4.6 m/s, 16
strokes, a steady **102 N m** heel (1.1 × the recovery authority, an 80 kg
rower 13 cm off the centreline).  Swing is the mean of the last three
strokes; "over" is a swing past 180°.

| authority | no trim | learned trim | novice | practised |
|---|---|---|---|---|
| flat `max_moment` | 0.069° | 0.020° | 0.058° | 0.025° |
| `PhaseAuthority` | over by stroke 4 | over by stroke 6 | 13.7° | over by stroke 6 |

At 30 N m the phase window never binds and the two authorities give
bit-identical swings (0.033°, 0.013° with trim).  With no balance
controller at all the boat goes over even at 30 N m (290° in the first
stroke), so the self-righting in the item above is for a disturbed but
unloaded boat, not for one carrying a steady heel.

**History.**  §64 diagnosed this on 2026-09-01 from swing growing over
strokes (1.43 → 1.90°).  The load behind that growth was the port rowers'
starboard arms (see Fixed): at `da0a0b9~1` the harness gives 1.39 → 1.99°
without trim and 2.37° with it; at `da0a0b9`, 0.038° and 0.015°.  With the
bias gone both tests XPASSed on a 0.03° swing from 2026-09-08 (the
"2 xpassed" in that commit's full run, and in the full run of 2026-09-13)
while exercising nothing.  They now apply the heel explicitly and are
strict xfails with `raises=AssertionError`.  The harness asserts that the
boat stays under a 45° swing before comparing, because post-capsize
numbers missed passing the trim threshold by only 6% (145° against 137°).  A flat-authority companion,
`test_the_same_heel_is_held_with_a_flat_authority`, must pass, so a strict
xfail cannot be failing for a reason other than the one it names.

**Not settled:** whether a real crew holds a 100 N m heel through the
recovery (a crew visibly sits out a lean), and so whether 93 N m is too
little authority or the missing mechanism is elsewhere, such as more than
2° of trunk lean.  Neither the learning gain nor the test thresholds
should be tuned to hide it.

### The [CR06] digitised traces were lost with the session scratchpad (2026-09-27) — **rebuilt the same day**
*Resolved:* `research/cr06/extract_fig3.py` re-extracts Fig. 3 from the open preprint's vector figure into the committed `data/literature/cr06_fig3_measured.csv` (all five series in one file). It reproduces the recorded checks: mean boat speed 4.190 m/s, velocity 3.061–5.126, leg travel 0.582 m, turn-round −44.36°. The old study scripts in `data/local/cr06_study/` read the two former files and need their paths pointed at it when rerun.

**Impact: medium — the one-athlete studies (SOURCES §§45–60, the finish fix, 4.3a) cannot be rerun until rebuilt.**

`cr06_fig3_measured.csv`, `cr06_fig3_body.csv` and the extraction script
`cr06/traces_fig3_body.py` lived only in the temporary scratchpad and were cleaned away.
The study scripts that read them survive in `data/local/cr06_study/`. The source is open
(Cabrera, Ruina & Kleshnev 2006, Fig. 3, a vector figure), so the traces can be
re-extracted; when they are, they go in `data/literature/` with their provenance, and the
recorded checks (mean boat speed 4.19 m/s against the stated 4.18) are the acceptance test.

## Open — numbers nobody has measured

These are placeholders. Each is labelled in the code as such; this is
the index of them.

| what | value | where | basis |
|---|---|---|---|
| novice power scatter | 0.110 | `crew/variability.py` | extrapolated along the elite→junior slope |
| novice timing scatter | 0.075 | `crew/variability.py` | same |
| square-up fraction | 0.25 of recovery | `crew/blade_contact.py` | "roughly where a crew squares"; put in a constant so it can be moved |
| balance falloff with inexperience | 60→150 N, 0.7→2.0° | `sim/control.py` | judgement; only the *ideal* end is calibrated |
| bias as a fraction of scatter | 0.45 | `crew/variability.py` | judgement |
| blade clearance on the recovery | 0.08 m | `crew/blade_contact.py` | typical, not measured |
| roster rower stature | 1.63 m W, 1.75 m M | `crew/roster.py` | NHANES population mean; the squad sheet logs weight and erg scores but never a height |

The roster stature is the one placeholder that is flagged in the
product as well as here: every rower built from `data/squad_roster.csv`
carries `stature_estimated=True` and the rig editor prints the height
with a leading `~`. Their *power* is measured; their *body geometry* is
a guess, and stature drives every link length in the kinematics, so it
moves the crew's centre-of-mass travel and with it the hull's speed
fluctuation. The two should never be quoted with the same confidence.

Also: `CP = 302.7 W` and `W' = 11.4 kJ` are literature means, not this
crew. They set the *shape* of every pacing answer and should be
measured before any number is quoted to an athlete.

---

## Open — validation gaps

### Michell is still 7–9% slow on singles
Against Holt it is excellent on doubles (+0.6%, +0.5%) and still low on
singles (−6.7%, −9.1%). Better than the constant coefficient on all four,
but the singles gap is real and unexplained.

*The research profile, 2026-09-14* (dynamic oar, tier 1 blade, [CR06]'s catch,
Sretenskii wave drag), at Holt's measured power and rate with the torque
matched: M1x **4.159 against 4.609 m/s (−9.8%)**, W1x **3.647 against 4.182
(−12.8%)**, and the doubles standing in for pairs −1.3% and −2.6%. The singles
gap persists under the research physics and is wider than the shipped model's.
Holt's power is gate power, used as `handle_watts` directly. The first
candidate is sculling-specific: [CR06]'s scull `C₂` = 58.7 is their computed
nominal value, and their own best fit to data was 2.4× that.

*Diagnosed 2026-09-13: this is not a hull or blade defect.* Scratch scripts and
outputs are in the session's `phase4/`. Nothing below changes the model.

| candidate | test | result |
|---|---|---|
| Blade `C₂` too low | `C₂` × 2.4, [CR06]'s own singles fit | M1x −6.0%, W1x −9.7%. Blade efficiency rises 0.71 → 0.79; swing barely moves. Closes 3–4 points, not 10–13. |
| Hull priced too dear | R(v)·v / P at Holt's measured speed, no simulation | Needs 0.97 (M1x) and 1.06 (W1x) of the measured power. The doubles need 0.78. A steady boat can only reach η_blade·η_velocity ≈ 0.63–0.76. |
| Catalogue single too big | Rebuilt at [L9701]'s Empacher (7.925 × 0.274 × 0.101) and King 1X (8.077 × 0.273 × 0.114) | Wetted area only 1% smaller. M1x −8.3% / −8.7%, W1x −11.2% / −11.6%. Closes 1–1.6 points. |
| Our drag differs from real-hull predictions | [L9701]'s own C_t curves, read at Holt's Fnv | Empacher at 4.609 m/s is **69.6 N**; the model gives 67.0 (Empacher dimensions) and 70.5 (catalogue hull). King 2X at Holt's M2- speed gives 124.3 N against the model double's 119.7. The model agrees with Lazauskas within 1–4%. |

So Lazauskas's real-hull prediction says the same as the model: an M1x at
4.609 m/s needs **96% of Holt's 334 W** at the mean speed, before any blade or
surge loss. For the pairs it is 81–85%. The singles gap follows **Holt's
sculling power figure**, and three things in [H20] and around it bear on it:

- **Definition.** Peach power uses the gate force *along the boat's long axis*.
  That is the normal force × cos θ. The model's handle power is the full moment
  × ω. Over Holt's own measured arcs and peak-force angles, the power-weighted
  cos θ is **0.89–0.92 for singles (arc 105°) against 0.94–0.95 for pairs
  (82°)**.
- **Cross-check.** At Holt's own rates, [K00]'s regressions give handle power
  that Holt's scullers reach only **0.73 (M1x) and 0.77 (W1x)** of. The
  sweepers reach **0.89 and 0.98**. The cohorts differ, but the sculling-only
  shortfall matches the definition.
- **Conditions.** The water was **26 °C** ([ITTC11]: ν 0.873 × 10⁻⁶ against
  the catalogue's 1.139 × 10⁻⁶ at 15 °C). The wind was 1.4 m/s
  cross-tail. Both favour every class.

[HLBS18] adds that any moment × ω power is ≥ 10% below true mechanical power.
The model's prescribed body shares that omission, so no factor is applied for
it.

*Measured under Holt's conditions* (`holt_conditions.py`): research profile,
catalogue hulls, torque matched, water at 26 °C from [ITTC11]. The long-axis
share is estimated from Holt's own angles.

| class | 15 °C, Holt P | 26 °C, Holt P | 26 °C, Holt P / share | share | swing (Holt) |
|---|---|---|---|---|---|
| M1x | −9.8% | −8.5% | **−4.9%** (367 W) | 0.910 | 67% (49%) |
| W1x | −12.8% | −11.5% | **−7.7%** (250 W) | 0.894 | 70% (51%) |
| M2- (double hull) | −1.3% | +0.0% | **+2.3%** (404 W) | 0.941 | 62% (55%) |
| W2- (double hull) | −2.6% | −0.4% | **+1.9%** (256 W) | 0.939 | 65% (53%) |

Warm water is worth 1.3–2.2 points on every class. The power definition is
worth 3.6–3.8 points on the singles and 2.2–2.3 on the pairs. Together they
take the singles from −10/−13% to −5/−8%, and the pairs from −1/−3% to +2%.
**A singles-versus-pairs spread of about 7–10 points remains.** Its
candidates:

- The pairs row a double's sculling rig, not a sweep pair. *Measured
  (`holt_pairs_sweep.py`):* the same double hull rebuilt with a two-seat sweep
  rig, using the catalogue's `SWEEP_OAR`, `SWEEP_ARC` and the four's 0.83 m
  span, with no rudder:

  | class | 26 °C, Holt P | 26 °C, Holt P / share | double stand-in | swing (Holt) | blade eff. |
  |---|---|---|---|---|---|
  | M2- | **−3.4%** | **−1.3%** | +0.0% / +2.3% | 66% (55%) | 0.699 |
  | W2- | **−4.3%** | **−2.1%** | −0.4% / +1.9% | 69% (53%) | 0.710 |

  **Withdrawn the same day: these pair rows are not a propulsion result.**
  `pair_yaw_check.py` found the rudderless M2- pair, driven with no helm,
  **turning 5.4° per stroke**. Its mean sideslip was 6.6°, and it was heading
  78.6° off after 16 strokes. So part of its deficit is cross-flow drag from
  going round in a circle. A real pair holds its line by the two rowers pulling
  unevenly. The comparison needs the pair balanced straight, for example by
  trimming stroke against bow power until the mean yaw rate is zero, before it
  can say whether the stand-in flattered the pairs. Until then the spread is
  **not** known to be smaller than 7–10 points.

  *Steered by pressure (`holt_pairs_steered.py`).* The Coxswain's own pressure
  split was driven by a PD loop on heading (gains 2.0 per rad and 1.0 per
  rad/s, a control choice), with torque matched to power with that coxswain in
  the loop. The pair stopped circling but **did not run clean**: over the last
  stroke it still turned 2.7–3.5°, with a mean sideslip of 5.1–5.9°, and it
  needed a mean |split| of 0.35–0.38, about 118% against 82% pressure.

  | class | 26 °C, Holt P | 26 °C, Holt P / share | yaw per stroke | mean sideslip |
  |---|---|---|---|---|
  | M2- | −1.8% | +0.5% | −3.0° | 5.3° |
  | W2- | −2.1% | +0.1% | −3.5° | 5.9° |

  With sideslip left in, these rows **bound** the pairs rather than measure
  them. The pairs lie between the steered sweep pair (−2.1 to −1.8% as
  printed) and the double stand-in, which ran exactly straight (−0.4 to
  +0.0%). **The rig is worth at most about 2 points.** The spread from the
  singles stays **6.4–9.4 points** as printed. Running a pair clean needs its
  real means of balance, which the rig does not have: rigging asymmetry such
  as oarlock span, oar length or inboard per side, and hull shape. These rows
  are not pursued further; the spread's other two candidates are next.

- The swing, too large everywhere and most on the singles. *Priced
  (`holt_swing_cost.py`):* the model's last-stroke speed trace was rescaled
  about its mean to Holt's measured peak-to-peak and priced with the hull's own
  resistance curve. The speed cost follows from the local exponent of
  R(v)·v, measured at 2.84–3.02.

  | class | swing, model (Holt) | η_velocity, model / at Holt's swing | power wasted | speed cost |
  |---|---|---|---|---|
  | M1x | 69% (49%) | 1.128 / 1.065 | +5.9% | **−2.0%** |
  | W1x | 72% (51%) | 1.171 / 1.094 | +7.1% | **−2.2%** |
  | M2- | 63% (55%) | 1.104 / 1.076 | +2.6% | **−0.9%** |
  | W2- | 66% (53%) | 1.132 / 1.089 | +3.9% | **−1.3%** |

  The excess swing costs the singles about **1 point more** than the pairs, a
  small part of the spread. It is also partly circular: the singles are
  slow, and the relative swing grows as speed falls. The shape is the
  model's, so a Holt waveform could differ.

**The spread, accounted for so far** (Holt's power as printed, 26 °C): the
rig ≤ 2 points and the swing about 1 point, leaving **at least 4–7 points
unexplained**. What remains lies in the power figure, not the boat. Holt's
long-axis definition would account for 3.6–3.8 points on the singles against
2.2–2.3 on the pairs, and the scullers read 0.73–0.77 of [K00] against
0.89–0.98 for the sweepers. Neither can be settled without Peach's own
definition or a sculling Peach-versus-oar-shaft comparison. **The singles
item is parked here as a measurement question, not a model defect.** No
research-profile number changes.
- Holt's scullers and sweepers are different cohorts; see the [K00] ratios
  above.
- The swing is too large everywhere, and most on the singles, where the crew
  is the largest share of moving mass.

Holt's stated Peach definition is taken at its word. Peach's own
documentation has not been obtained, and it would settle whether the force
really is resolved along the boat axis. **Two measurements disagree on the
sign** (SOURCES, "What Peach power actually is"). [H21], the same group on a
mechanical rig, found Peach 8–17% *low*. [BR25], on the water in an eight,
found Peach 6–11% *high* against an oar-shaft sensor, and blames the oar's
axial force at non-perpendicular angles. Both tested sweep oars only. So the
cos θ column is what Holt's words imply, not an established correction.

### Surge swing is 10–31% too large
Model against Holt: ratios 1.31, 1.31, 1.10, 1.18. `scripts/unsteady.py`
squares this quantity, so the error is four times worse there.

*The research profile, 2026-09-14:* ratios 1.43, 1.43, 1.16, 1.28 (swing 70, 73,
64, 68% against Holt's 49, 51, 55, 53%) -- larger than the shipped model's on
all four. Largest on the singles, where the crew is the largest share of the
moving mass.

*2026-09-26:* the research leg-placement correction (SOURCES §157, Fixed) adds
0.9-1.05 points to the research swing and ~1 cm to crew CoM travel, which widens
this gap. The model's travel for [BR24]'s athlete is now 0.799 m against his
0.71-0.74 m. The excess is not in where the leg masses sit; the joint-angle
excursions are the next suspect.

*2026-09-26, [BR24] like for like (SOURCES §158):* at his 432 W and on his arc the research
model rows 4.652 m/s against his 4.641 and swings 61.0% against 49.1%. The twelve points
decompose: the crew's timing ~4 (his seat plateaus near 1 m/s, legs before trunk; the
model's CoM peaks at +1.64 m/s relative to the hull against his +1.04), his force shape 1,
averaging under 1, and ~6–7 at the catch, where the model's blade delivers almost nothing
for 0.2 s (8 N·s by 0.3 s against his 26) even with his force and his body. Neither catch
rule reproduces his early loading. CoM travel is not the lever: his trunk sweep cut it 4 cm
and IVV by one point. Next: the catch (blade squared and immersed; [G19]).

*2026-09-27 (SOURCES §159):* the literature's leads, tested. Hull drag is right (his
recovery alone: 73–79 N at 4.64 m/s; model 75.4). The tier 2 lift+drag law fixes the dip
at [CG07] amplitudes but is 4.5% too efficient; scaled to his speed, IVV is 56.4% again.
Patton added mass is worth under a point; the dynamic oar applies no immersion factor.
The model's blade still loads late under every law (1/45/122 N at 0.1/0.2/0.3 s against
his 55/111/174). Remaining suspects: the oar balance's share of his handle torque at the
catch (reflected crew inertia) and the sweep entry. Not a blade coefficient.

*2026-09-27, the crew share split (SOURCES §160):* driving his model with the [CR06] athlete's
on-water timing recovers about 2 of the 4 crew-timing points; the rest is his own legs-first
style. What the two on-water scullers share against the ergometer body is a longer drive
(0.53–0.54 vs 0.47) and a later body swing on the recovery. That is what an on-water driver
should encode.

### The eight is validated only by inference
Holt measured singles and pairs. The boat this project cares about most
has no measured counterpart in the comparison.

---

## Open — presentation

### Headless `--frames` is slow
About 50 s for 122 frames, because it now draws every step so that
catch-driven effects (puddles, splash) can appear at all. Fine for
screenshots; painful for long sequences. CI is unaffected — it passes no
`--frames`.

### Windows CI cannot render
The runner is a headless service session with no usable OpenGL, so the
render check warns instead of failing there and the payload is verified
by file presence. The Windows artefact is built and run locally before
release.

---

## Performance — measured, on this desktop

A rower reported the trainer "unusably sluggish" on an old gaming
laptop.  Profiled headless, 180 frames of an eight on Head of the Lake
at Standard, with nothing else running.  The numbers are this machine's;
the ratios are what matter.

| what | before | after | how |
|---|---|---|---|
| physics per frame | 31–34 ms, every tier | **2.9–3.6 ms** at 50–60 Hz (p50); 4.7 ms at High's 100 Hz | crew kinematics and oar force tabulated once per stroke (`Boat.tabulate_crew`, trainer only); 60 Hz is the same boat to 0.2 mm |
| `seattle._inside` at start | 14.2 s | 0.96 s | matplotlib's C crossing test with a bounding-box pre-filter; identical answers |
| world build at start | 33–37 s | 10–16 s | walls emitted per building in numpy; GC off for the build |
| crew tables at start | — | 0.3–1.1 s | one chain solve per sample; shared by kinematics signature; warmed after the timing scatter |

Per tier, 150 headless frames of an eight on Head of the Lake, medians
after ten warm-up frames, **on this machine's Intel UHD Graphics** — an
integrated part, which is the case the rower reported. `--bench` prints
these; the per-pass GPU timers are what found the water.

| tier | before | after | water pass | world pass |
|---|---|---|---|---|
| Ultra minimal | — (new) | **17.1 ms, 59 fps** | 5.3 ms | 3.8 ms |
| Minimal | 41.1 ms, 24 fps | **16.5 ms, 61 fps** | 3.4 ms (was 7.9) | 3.8 ms |
| Standard | 62.3 ms, 16 fps | **24.2 ms, 41 fps** | 8.5 ms | 5.8 ms |
| High | 56.0 ms, 18 fps | ~27 ms | 8.8 ms | 6.5 ms |

Two things the numbers overturned:

* **Render scale hurt.** Drawing at 0.75 into an off-screen buffer and
  stretching cost 44.7 ms against 27.8 at full size on this GPU: the
  extra pass and copy outweigh the pixels saved on a part whose
  bottleneck is not fill. Every tier is at 1.0; `--render-scale` stays
  as a lever for a GPU where the balance differs.
* **"Flat" water was flat geometry with full per-pixel work.** Inside
  `exact_within` every fragment summed eight waves, sixteen puddles, two
  textures and the wake three times over for a gradient, nearest the
  eye where pixels are densest. The low tiers now take their normal
  from the vertex-baked slope (`water_flat`) and no exact band at all.

Peak working set, 120 headless frames, the interpreter itself:
**648 MB Ultra, 671 Minimal, 868 Standard, 887 High**, against
700–1,040 MB before the CPU mesh copies were dropped after upload.

Found on the way, and fixed: the kinematics signature omitted the side
of the boat, so a matched crew shared the stroke seat's chain and every
port rower wore starboard arms (0.15 m out on eight arm masses).  A
yaw bias, steering the boat.  The golden trajectory was re-recorded for
it — the second re-record, with the reason and the numbers in
`tests/test_stepwise.py`.

What the physics is NOT moved to: the GPU.  It is a small stiff 6-DOF
system on tiny arrays; a GPU's launch latency loses to the CPU JIT that
already runs the hull sweep.  There is no tensor workload here for a TPU.

---

## Done — running on the machine you have

### The offline physics programme — phase 0

Scaffolding for fixing the blade and the rower offline while the shipped
trainer stays frozen. **No physics changed.** See the plan of 2026-09-12.

| what | where | pinned by |
|---|---|---|
| physics resolved by **name**, not constructed implicitly; `shipped` is frozen, `research` and `learned` are the offline programme | `coxswain/physics.py` | `tests/test_physics_profiles.py` |
| `resolve()` and `resolve(None)` both give `shipped`, so forgetting a profile cannot promote research physics into the game | `physics.resolve` | same |
| the blade coefficient follows the rig — [CR06] fit `C2` at 84.5 sweep / 58.7 sculling — and the outboard comes from the boat, not the paper's 2.28 m | `PhysicsProfile.blade_model` | same |
| the trainer's physics pinned in **one place**: every menu route to a boat goes through `build_boat`, which applies `shipped` | `viz/menu.py` | same |
| a **source scan** failing if anything the trainer can execute names a non-frozen profile — a behaviour test cannot cover a path no test exercises, which is how v0.12 shipped its crash | `tests/test_physics_profiles.py` | itself, plus a planted-offence test so the guard can fail |
| one validation battery, every profile scored on it; targets carry provenance, and one that cannot be run is reported `pending` **with its reason** rather than dropped | `coxswain/validation/` | `tests/test_validation_scorecard.py` |
| targets above a profile's blade tier are `n/a`, never `pass` — Kleshnev's 78.5% against a tier-0 model checks the lumped constant against itself | `validation/scorecard.py` | same |
| the η-against-v defect **scored**: the fitted line reaches zero efficiency at 2.0% (eight) and 0.07% (four) of mean speed — through the origin on both hulls — with η/v flat to 2.7% and 1.3% | `validation/targets.py` | same, and `tests/test_blade_velocity.py`'s strict xfail |
| surge swing quoted at the fastest operating point, because it runs 77.4% at 2.81 m/s down to 37.6% at 5.93 and a figure at an unstated speed is not a target | `validation/scorecard.py` | same |
| the efficiency measurement has **one** definition, shared by the scorecard and the test that found the defect | `validation.scorecard.efficiency_at` | `tests/test_blade_velocity.py` imports it |

### The offline physics programme — phase 1, gate failed

Tier 1 was scheduled as "switch the blade model on". It has no viable
operating point, so phase 1 is closed and merged into phase 2.

| what | where | pinned by |
|---|---|---|
| the efficiency-only wiring measured to convergence, not to 70 s: the boat collapses 3.92 → 0.63 m/s from either direction | `validation/scorecard.py` | `tests/test_blade_tier1.py` |
| the sweep shape ruled out as the cause — flatness 0.00 / 0.30 / 0.60 all collapse | — | same |
| the mechanism identified: efficiency at the instantaneous surge, whose minimum coincides with peak oar force; 0.628 at the mean speed against 0.494 at the instantaneous | — | same |
| `SOURCES.md` §7's "5.10 to 4.28 m/s" corrected — it was an unconverged transient | `docs/SOURCES.md` | same |

### The offline physics programme — phase 2, the unit

The oar angle as a state, built and tested standalone before anything is
wired into the simulator.

| what | where | pinned by |
|---|---|---|
| `I·φ̈ = τ_handle(t) + l·F_n(φ, φ̇, v)`, integrated from the catch to the finish angle | `coxswain/crew/oardynamics.py` | `tests/test_oar_dynamics.py` |
| the inertia is a **required argument with no default** — the oar's own is twenty times too small and the rest is the rower's body, so a default would become a fitted parameter nobody remembered fitting | same | same |
| a prescribed angle delivers **+0.2 N·s at 4.85 m/s and −171 at 6.0** under the same force model; the dynamic angle gives +163 and +127 | same | same |
| thrust now **falls** with boat speed — the restoring term the efficiency-only wiring lacked | same | same |
| drive duration becomes an output: 0.970 / 0.720 / 0.634 s at 2.8 / 4.85 / 6.0 m/s for one pull | same | same |
| the integrator checked against a closed form (no water, constant torque → `φ = φ₀ − ½(τ/I)t²`) and for step convergence | same | same |
| the inertia **derived, not fitted**: `Σᵢ mᵢ\|∂xᵢ/∂φ\|²` from de Leva masses and the joint chain, both already in the model | `reflected_inertia` | same |
| it is not a constant — 93 kg·m² early in the drive to 13 at the finish — so the balance carries `½(dI/dφ)φ̇²`, which is a first-order term here and not a refinement | `InertiaProfile`, `OarDynamics.acceleration` | same |
| an unfitted drive-fraction prediction **on the oar alone** (fixed 4.85 m/s, constant pull): 0.319–0.406 across 183–808 W against 0.296–0.395 measured by [HF09]. **Corrected 2026-09-13: it does not hold on the full model** — see the open item on drive time and pace | same | same |

### The offline physics programme — phase 2, the gate

| what | where | pinned by |
|---|---|---|
| the oar balance coupled to hull surge — the smallest model that can answer the gate | `coxswain/sim/oarloop.py` | `tests/test_oarloop.py` |
| **gate passed**: the η-against-v line reaches zero at **8.9x mean speed** against the baseline's 0.020, and η is flat at 0.57–0.60 instead of rising 0.24 → 0.52 | same | same |
| published race pace reached at **380 W per rower**: eight 5.33 m/s (band 5.0–5.6), four 4.76 (band 4.5–5.1); re-measured after the blade-centre correction, 5.21 and 4.66 — where the prescribed model needed 720 W and 795 W and overshot anyway | same | same |
| boat-class ordering falls out at equal power: eight > four > double > single, imposed nowhere | same | same |
| the rower's pull takes the measured front-loaded curve, parameterised by oar **angle** because the drive duration is now an output | `torque_shape` | same |
| a seam in the shipped simulator: the oar block moved verbatim into `RowingSimulator._oar_loads`, and **bit-identical** across the move | `simulator.py` | `tests/test_stepwise.py`, `tests/test_kernels.py`, `tests/unit/test_state_cache.py` — 17 passed |
| the dynamic oar on the **full 6-DOF hull**: one angle per seat, blade load applied at the blade with no gearing factor, slip against the water-relative oarlock velocity so a turning boat's two sides differ | `coxswain/sim/dynamic_oar.py` | `tests/test_dynamic_oar.py` |
| handle power is **closed-form**: work per drive is `peak × ∫shape dφ` whatever the speed, so the torque for a stated wattage needs no search; the integrated measurement agrees to 1% | `DynamicOarSimulator.peak_torque_for_power` | same |
| the test that pins the efficiency-only collapse now builds that wiring **directly**, not through `research` — it is evidence of a failed approach and must not change meaning when the profile is repointed; the direct construction is proven identical to the profile's | `tests/test_blade_tier1.py` | same |
| `research` **repointed at the dynamic oar**: profiles carry an `oar` driver (prescribed / efficiency / dynamic), a contradiction between tier and oar is rejected, and `apply` leaves `blade_model` off for a dynamic oar so the efficiency wiring cannot be laid on top | `coxswain/physics.py` | `tests/test_physics_profiles.py` |
| UNSTABLE retired for **PARTIAL**, which names what is still unfinished: prescribed crew, synchronised crews only, report not ported. The tripwire still requires a warning on every unfrozen profile | same | same |
| **the prescribed oar block refuses a dynamic-oar boat** -- one check that covers the scorecard, the report and anything else, firing only when the prescribed force is about to be applied, so reading a trim state is still allowed | `RowingSimulator._oar_loads` | same |
| the scorecard **settles dynamic-oar boats at stated watts**, routed by the profile stamp, with drag priced by one shared helper; the questionable `mean_handle_power` conversion is kept out of the new physics | `coxswain/validation/scorecard.py` | `tests/test_validation_scorecard.py` |
| the full hull's residual rise in η is **decomposed, not smoothed over**. The guessed cause — the swing starving the blade — was measured and is backwards. Half is power lost to the swing through nonlinear drag (16% at 3 m/s, 6% at 5.5); half is the blade slipping more on a slower boat, which is real physics. The drag half rides on the prescribed crew motion, i.e. defect two | `DynamicRun.drag_power_ratio` | `tests/test_dynamic_oar.py` |
| the report **refuses a dynamic-oar profile at argument parsing**, before any expensive stage runs, rather than at the prescribed oar block's first derivative | `scripts/make_report.py` | `tests/test_dynamic_oar_consumers.py` |
| **`research` scored end to end on the baseline's own harness**: η zero crossing 1.462 (eight) and 1.592 (four) against a floor of 0.15, η/v spread 0.380 and 0.389 against 0.08 — both defect targets that `shipped` fails (0.020, 0.027), passed on both boats | `scorecard.run("research")` | `tests/test_dynamic_oar_consumers.py` |

**Open, found by that run:** the blade-efficiency *level* target
(Kleshnev, 0.754–0.816) scores **n/a** on `research`. It is read from
`boat.blade_model` and the prescribed sweep, and a dynamic-oar boat carries
neither by design. It needs measuring from the dynamic run itself —
force-weighted slip efficiency over the drive.

**Found by checking the level and not just the shape**, which is the point
of carrying both: the rig's **gearing was being applied to the blade force**
as well as the handle force, charging one lever twice — a factor of 3.2,
putting η at 0.18, which no blade has. And a **sculler was charged for one
oar and credited with two**, flattering the single by a factor of two.

**What phase 0 found on the way:** four defects nothing was catching —
handle power moved 53% at `a053540`; the suite had 26 failures reported as
none; the optimiser and simulator disagreed about the rudder; and the drive
is 18–28% too long because it was refitted to ergometer data. All four are
open items above, and the last two were found only because a phase was
spent on measurement before any physics was touched.

### The offline physics programme — phase 4.1, built; closed on a finding

| what | where | pinned by |
|---|---|---|
| the baseline, measured before building: the clock crew's hands up to 0.19 m off the dynamic handle on the eight, 0.74 m on the single — too far for any arm solve | probe, recorded in PHYSICS_PROGRAMME | — |
| the body follows the oar through the drive: stroke time from the angle, pose from the stroke table, velocity and acceleration by the chain rule; hands on the handle to under 2 mm on both boats | `coxswain/crew/follow.py` | `tests/test_crew_follows.py` |
| the oar's inertia built from the velocities the hull is given — one kinetic energy, to 1e-9; power books close to 2% outside the rate floor | same | same |
| recovery retimed from the dynamic finish to the next catch, starting from the finish pose and arriving at the catch pose | same | same |
| `crew="follows"` on the dynamic oar, a study: one elapsed-drive state per seat, the oar computed before the hull, the clock crew's arithmetic and order untouched | `coxswain/sim/dynamic_oar.py` | same, plus the dynamic-oar suites (62 fast, 8 slow) |
| the momentum books: the clock crew closes to 0.005 N·s a stroke; the following crew does not — open item above | same | same (the latter strict xfail) |
| the following crew's velocity jumps at the finish and catch handed to the hull as impulses that conserve the momentum of hull plus crew; the finish detected on the step it happens | `DynamicOarSimulator._hand_jump_to_hull`, `_integrate_stroke` | `tests/test_crew_follows.py` |
| the finish searched for a source: a measured oar turns round there, no deceleration law published in what was found | [FE17], [N-FISA] in SOURCES.md | — |

### The offline physics programme — phase 4.3, groundwork

| what | where | pinned by |
|---|---|---|
| segment principal moments of inertia from de Leva (1996) Table 4 radii of gyration, `I = m (r l)^2` per axis; lumped forearm+hand and shank+foot by the parallel-axis theorem (offset added to the sagittal and transverse moments, not the longitudinal) | `coxswain/crew/anthropometry.py` | `tests/unit/test_segment_inertia.py` |
| the radii transcription verified against the columns already in the code: 20 rows, 0 mismatches | [dL96] in SOURCES.md | same |
| Rongère, Khalil & Kobus (2011) read in full: inverse dynamics with prescribed joints and loops closed by projection, no arms, oars driven independently — and its authors conclude a hybrid inverse/direct approach is needed. PHYSICS_PROGRAMME had credited the torque drive to them; corrected | [RK11] in SOURCES.md | — |

### The research wave model — Michell checked, Sretenskii in, the optimisers on it

| what | where | pinned by |
|---|---|---|
| Michell's integral checked against Lazauskas (2009) Fig. 7.1 on the Wigley hull: the formula agrees analytically; the uniform-weight sum reads 2.8–4.5% high on the production grid; trapezoid weights match within the figure's reading error. `quadrature="uniform"` stays the default for the frozen game (724bb35) | `coxswain/hydro/michell.py` | `tests/test_michell_quadrature.py` |
| Sretenskii's finite-depth thin-ship integral, Wehausen & Laitone (1960) eq. 20.69 with its erratum; deep limit 1.00000, `Fr_h ≤ 0.5` within 0.14% of deep water, peak at `Fr_h` 0.98–0.99 (137bf36) | `coxswain/hydro/finite_depth_michell.py` | `tests/test_finite_depth_michell.py` |
| a depth-aware wave table, within 0.64% of the direct integral between its rows; `PhysicsProfile.wave = "sretenskii"` for `research` and `learned`, applied in `hull_resistance` in place of the chosen shallow factor; shipped keeps the exact old product (137bf36) | `FiniteDepthWaveTable`, `coxswain/physics.py`, `coxswain/hydro/resistance.py` | `tests/test_research_wave_profile.py` |
| on the Charles at constant power the research eight loses 7–33 s less to depth than under the chosen factor at deep-water speeds 4.5–6.0 m/s; the scorecard moves under 1% | measurement, recorded in SOURCES §6 | — |
| the route and pacing optimisers and the CasADi trajectory models take a research boat's depth-aware wave drag; the research deep table samples 301 speeds (8c5abc1, 712a660, b489a8c) | `coxswain/river/route.py`, `coxswain/crew/pacing.py`, `coxswain/river/hydro_casadi.py` | `tests/test_research_optimisers.py`, `tests/test_research_casadi_wave.py` |
| race power sourced in part: Kleshnev's measured power–rate regressions and the 16.8% handle-power gap; 380 W is about elite men's sweep handle power at rate 32 (c769fb7) | [K00], [BR26] in SOURCES.md | — |

### The offline physics programme — phase 4.3a, the catch

| what | where | pinned by |
|---|---|---|
| [CR06]'s entry rule on the dynamic oar: the oar follows the prescribed sweep, blade out, until the blade's normal velocity is zero (their eq. 16), then torque-driven with the sweep's angle and rate; the kinetic energy carried in is counted as rower work; the release rule never bites under it (edb3fd4) | `DynamicOarSimulator(catch="sweep")` | `tests/test_sweep_catch.py` |
| at equal power the parked catch was costing 5–7% of boat speed: eight at rate 28 5.53 → 5.87 m/s, blade efficiency 0.56 → 0.71 | measurement, recorded in the catch item | same |
| the scorecard on it keeps every target's status; the efficiency-proportional-to-speed signature is gone (1a1e9a6) | `coxswain/validation/scorecard.py` | — |
| power matching under either catch: profiles name their catch, and `torque_for_power` matches a sweep-catch crew on a cached settle (c78d38f) | `DynamicOarSimulator.torque_for_power`, `simulator_for`, `settle_dynamic` | `tests/test_torque_for_power.py` |
| `research` and `learned` catch by the sweep: tier 1 blade efficiency 0.714 eight, 0.681 four; tier 2 0.826 eight, **0.795 four, the first pass of the level target** (b923d62) | `coxswain/physics.py` | `tests/test_dynamic_oar_consumers.py`, `tests/test_dynamic_oar_run.py` |
| blade added mass on the sweep catch: an oar in the air follows the sweep with no added mass; entry momentum left out 2.5 N·s per blade at most, against 63 parked; exit 12–24 N·s remains | `DynamicOarSimulator._coupled_system` | `tests/test_blade_added_mass.py`, `tests/test_sweep_catch.py` |

### [LO00] against her stroke — the first same-population check on the stretcher (2026-09-18)

A library sweep for phase 4.3's foot-force blocker turned up
Loschner & Smith (2000) in [WA18]'s reference list; the project owner supplied
it the same day. **Three international female single scullers** — [CR06]'s own
population — with propulsive pin force, **each foot measured separately**, the
oar angle of every peak, and boat velocity. Six checks, none of them fitted:
the stretcher and pin curves come from `stretcher_from_balance_cr06.py`, built
from her measured handle force and hull acceleration plus the segment-mass books.

| check | model, her stroke | [LO00] measured | verdict |
|---|---|---|---|
| peak stretcher force | 715 N | 653–763 N (mean 704) | **passes, 1.02×** |
| boat velocity min/max/mean | 3.06 / 5.13 / 4.19 m/s | subject A 3.06 / 4.80 / 4.01 | **passes** |
| velocity swing / mean | 49% | 43, 46, 51% | **passes** |
| foot and pin peak together | same instant | means −20.0° and −20.2°, 0.2° apart | **passes** |
| oar angle of peak force | −13.1° | −14 to −29°, mean −20.1° | at the late edge of the band, 7° from their mean |
| peak pin force, both gates | 790–800 N | 984–1103 N (mean 1028) | **0.77–0.78** — resolved the next day as [LO00]'s column, not the model's; see below |

*The low pin force is not the oar's inertia.* `gate_inertia_check.py` recomputed
it keeping `I_lock·θ̈` — the term the validated blade balance uses and the
diagnostic's massless lever drops — and got **790 N either way**. The term
cancels at the peak and only bites near the catch (blade load 17 → 5 N at
t = 0.05 s). The gap is in her handle force itself: 553 N peak across both
hands, where [LO00]'s subject A implies about 719 N.

*And it exposes a conflict between the two measurement sets.* Stretcher-to-pin
at peak is **0.68** in [LO00] (0.66–0.70 across three athletes) and **0.94** in
[LE26]. Those cannot both describe the same quantity. Her stroke gives **0.91**
— matching [LO00]'s stretcher *magnitude* and [LE26]'s *ratio*. Pushing her
handle force up until the pin reached [LO00]'s 1028 N would put her stretcher at
about 876 N, matching [LE26]'s 864 and breaking the [LO00] agreement that
currently holds. **Left unresolved; no number from either has changed the model.**

*What it does not do:* answer DATA_REQUESTS §11. [LO00]'s forces are propulsive
only and it defers the vertical and transverse components to a 1999 abstract on
three-dimensional **pin** forces. §11 stays open, now aimed at that abstract
(§11d), and phase 4.3's knee and ankle stay blocked on it.

*Also settled by the same sweep:* the Peach PowerLine — the standard elite
instrument, and [LE26]'s — records gate and stretcher force **along the boat's
longitudinal axis only**, so no PowerLine-equipped squad can answer §11 however
good its data. Recorded in SOURCES and as "What not to ask for" in DATA_REQUESTS.

### Resolved (2026-09-19): the pin gap is [LO00]'s column, and the stretcher is validated three ways

The open question above was whether the 22% pin shortfall was the model or the
source. It is the source, and the test that shows it is cheap: scale her handle
force uniformly until the model *settles at each dataset's own measured boat
speed*, then read the stretcher and gate force there. **This is fair rather than
circular because the hull drag was never fitted to [CR06]** — it is Michell's
integral checked against Lazauskas's printed Wigley curve, Sretenskii's
finite-depth integral and ITTC'57 friction — so it can mediate between one column
of a paper and another. `force_vs_speed_check.py`, from five settled runs:

| source | v measured | v model | stretcher, model vs measured | gate, model vs measured |
|---|---|---|---|---|
| [LO00] A | 4.01 | 4.0139 | 668 vs 696 — **−4.0%** | 749 vs 998 — −24.9% |
| [LO00] C | 3.84 | 3.8583 | 619 vs 653 — **−5.2%** | 693 vs 984 — −29.6% |
| [LO00] B | 3.71 | 3.7458 | 583 vs 763 — −23.6% | 654 vs 1103 — −40.7% |
| [LE26] women | 4.40 | 4.3948 | 796 vs 864 — **−7.9%** | 893 vs 916 — **−2.5%** |

**The stretcher force is validated at three independent points** — [LO00] A and C
and [LE26] — where the model runs a consistent **4–8% low**. That is the real
residual, and it is small. It replaces yesterday's "structure passes, ~120 N
short", which was a catch-time reading; at the peak the model is within 8%
everywhere it can be checked.

**The gate force agrees with [LE26] to 2.5% and cannot be reconciled with [LO00]
at any subject.** Two things point at [LO00]'s pin column rather than at us:

1. *It does not track its own boat speed.* The model's forces fall as the boat
   slows, as they must. [LO00]'s pin column is flat-to-rising — its **slowest**
   boat carries its **highest** pin force. The paper says so itself: "Subject B
   applied the highest propulsive forces on the pin and the stretcher but the boat
   velocity was the slowest." B is also the one subject whose *stretcher* the model
   misses (−23.6% against −4 and −5 for A and C), so B is anomalous in both
   columns by the model's reckoning and by the authors'.
2. *Rate does not rescue it.* Fig. 1's panel captions give each subject a rate the
   table omits — B 28, A 30, C 31 spm. B's lower rate raises per-stroke force at
   equal power, but by about 7%, not 40%.

**Caveat, to be repeated wherever this is quoted:** every run keeps her stroke
shape and her rate (1.94 s, 30.9 spm). [LO00]'s 28–31 spm makes that comparison
effectively rate-matched, which is why it is the clean one. [LE26]'s women rowed
at 34.1 spm, where the same power is spread over a shorter stroke; matching the
rate too would move the model's gate from 2.5% low to roughly 10% low. **So the
2.5% is a rate-matched figure, not a clean pass**, and the honest statement is
that the gate sits within about 2–10% of [LE26] and 25–41% from [LO00].

*What this does not change:* [LO00]'s structural results stand — foot and pin peak
together, and the boat-velocity swing agrees. §11 stays open. And nothing in the
model was altered to reach any of this; the only new code is a default-1.0
`F_SCALE` env knob on two scratchpad scripts, so the recorded runs are unchanged.

### The catch lag taken apart (2026-09-19): not the blade, mostly the entry

The live defect on her stroke is the model's oar running behind hers through the
drive — +2.66 / +3.89 / +4.26° at 0.30 / 0.50 / 0.70 s. Her measured handle force
drives it, so the candidates were the blade load, the oar's inertia, or the hull.
`catch_lag_blade_cr06.py` settles it. Oar inertia was already at its floor (the
oar-only profile on a 1.2 kg oar; the rower's arms would only raise it).

**It is not the blade law.** Compared at the same instant through the drive, the
model's blade load and the load her oar balance implies agree to **0.1%** on the
mean (88 N against 88 N) and within a few newtons pointwise — 92/87, 115/113,
127/127, 68/70. What differs underneath is the hull: 3.400 m/s against her
3.513 through the drive, 3.2% slow, which makes the slip 6.6% more negative,
because `w = ℓθ̇ + v cos θ`.

**And the hull is only about a third of it.** Integrating the oar *alone* on her
clock from her catch, under her measured handle force and the same tier-1 law,
changing only the hull velocity it sees:

| | 0.10 s | 0.50 s | 0.70 s | rms, 0.05–0.85 s |
|---|---|---|---|---|
| full simulation, lag vs her | +1.72° | +3.89° | +4.26° | — |
| isolated oar, **model** hull | −0.25° | +1.05° | +1.35° | 0.88° |
| isolated oar, **her** hull | −0.39° | −0.42° | −0.93° | **0.74°** |

The bottom row is a result in its own right: **the oar equation with the tier-1
law at C₂ = 140.88 and the oar-only inertia, given her handle force and her hull,
reproduces her oar angle to 0.74° rms across the drive.** Blade law, C₂ and oar
inertia are jointly exonerated, by the same machinery that reproduced her
turn-round to 0.67° rms.

So at 0.70 s the lag splits roughly: **1.35° the hull deficit**, and **2.89°
generated inside the full simulation** — of which **1.72° is already there at
0.10 s**, before the hull deficit has had time to accumulate. That early part is
the catch entry, which is 4.3a's own subject, and it sits with the known 13–16°
of catch slip against [H20]'s 7.7–9.7°.

*Correction to an earlier reading in this file:* the catch dip and the oar lag are
**not** the same defect. The dip contributes, but it is the smaller share, and
removing it entirely would leave roughly two thirds of the lag standing. The
entry is where the work is.

*Caveat:* the isolated oar is given her catch state at t = 0.02 s, so by
construction it carries no entry error — that is what makes the comparison
localise the entry, and also why its two rows cannot be read as a full accounting
of the simulation. Nothing was changed in the model; the script only reads it.

### Phase 4.3a: the entry is a RATE defect, and it is worth 3.2° and 1.1 points of speed (2026-09-19)

Following the decomposition above, `entry_arc_vs_her.py` located the entry defect
and `entry_from_her_cr06.py` measured what it is worth. The mechanism is in
`DynamicOarSimulator._integrate_sweep_catch`: while the blade is in the air the oar
is forced onto the prescribed sweep `boat.oar_sweep` after every step, and handed to
the torque drive only when the blade's normal velocity reaches zero ([CR06] eq. 16).

**It is not the catch angle.** Hers is +60.42° and the arc starts at +60.49°.
**It is the rate.** The arc accelerates too gently and runs about 24 °/s slow
through the whole air phase — at 0.04 s hers is −44.6 °/s against the arc's −25.7,
at 0.08 s −75.2 against −50.9. That one defect does three things:

* the blade **enters 29 ms late** — eq. 16 fires at 0.0521 s on her measured trace
  and 0.0815 s on the arc, an air phase 56% longer than hers, because the slower
  rate delays the slip reaching zero;
* the torque drive **inherits an angle 1.44° behind** hers;
* and **inherits a rate 24 °/s slower**, which it must then make up under load.

**What it is worth.** Replacing *only* `_sweep_pose` with her measured angle and
rate — her handle force, the tier-1 blade at C₂ = 140.88, the body field and the
hull all untouched, eq. 16 still the entry rule:

| | entry | 0.30° | 0.50° | 0.70° | speed |
|---|---|---|---|---|---|
| prescribed arc, as built | 0.0828 s | +2.63 | +3.84 | +4.24 | 4.0918 (−2.37%) |
| her measured hand path | **0.0516 s** | +0.50 | +1.02 | **+1.03** | **4.1375 (−1.28%)** |

The drive lag collapses by **76%** and the speed deficit nearly halves. The entry
fires within **0.5 ms** of her own eq.-16 entry, which confirms [CR06]'s rule was
right all along and only the rate fed into it was wrong. And the residual +1.03°
sits where the isolated-oar test predicted the hull deficit would leave it
(+1.35°), so the three pieces close: **entry ≈ 3.2°, hull ≈ 1.0°, blade law 0.**

*This is a study, not a fix, and it cannot ship.* Feeding the model her measured
hand path is not predictive; it sets the budget a genuine body-driven catch has to
buy back. The real fix is the rower's body driving the oar through the entry —
phase 4.3's subject — and would go through `PhysicsProfile` for research and
learned only, never touching the frozen shipped game.

*It also joins up with what was already on record.* "The ergometer-fitted body
reverses too gently" and the legs reversing at 6.9 m/s² against a measured 12–15
are this same defect seen at the body; this is it seen at the oar. The standing
conclusion that the four-keyframe ergometer body cannot make a measured catch
spike is now quantified at the oar: 24 °/s, 29 ms, 3.2°, 1.1 points of boat speed.

### 4.3a: can the body drive the entry? Not naively — it overshoots (2026-09-19)

Having established that a correct entry is worth 3.2° and 1.1 points of speed, the
principled fix is that the hands are on the handle, so the body should set the oar
angle through the air phase. `entry_from_body_cr06.py` asks whether it can.

**An architectural obstacle first.** `Boat._hand_targets` derives the hands *from*
the oar sweep — "a rower holds the oar, so the hands are not free" — and
`kinematics.py` then constrains the arms to it: "the hand track IS the handle". So
in the coupled model the hands follow the oar and cannot drive it. A first probe
that read `rower.joint_positions` was therefore circular, and said so: it recovered
the arc's own rate to 4% (36.7 against 38.2 °/s), which is a check on the method,
not an answer.

**Rebuilt with `hand_targets=None`**, so the hand comes from the stroke dataset's
own arm and body angles, over the air phase 0.02–0.10 s:

| | mean rate | vs her |
|---|---|---|
| prescribed arc | 38.2 °/s | **0.65×** |
| her measured oar | 59.0 °/s | — |
| free-arm body's own hand path | 101.8 °/s | **1.73×** |

**So coupling the oar to the body is not the fix.** It would trade a 35% deficit
for a 73% excess. The body model and the oar arc disagree about hand speed at the
catch by a factor of 2.7, and because the architecture slaves the hands to the oar,
the body's own preference is silently discarded — which is *why* the arc's gentle
catch survives into the simulation unchallenged. Her measurement sits between them.

*Provisional, and marked so.* The free-arm rower is a configuration the model never
runs, and its dataset arm angles were fitted with the hand constraint present, so
the 1.73× may be part artefact. **The arc's 0.65× is the solid half** — it comes
straight from her measured trace and is what the two studies above rest on.

*Where that leaves the fix.* Not the arc alone: correcting its catch rate to hers
would be fitting to one athlete and would not generalise, which the "no unsourced
number" rule forbids. Not the body as it stands either. The remaining route is the
one the programme already names — a torque-driven body whose own dynamics produce
the catch, with the hands on the handle as a constraint rather than a slaving — and
that is the rest of 4.3, not a patch to the entry.

### 4.3a: a predictive entry fix that works in the drive and exposes the finish (2026-09-19)

The entry defect had been read as the ergometer body reversing too gently. **It is
not.** `entry_body_speeds_cr06.py`, from measurement alone:

| over the air phase 0.02–0.10 s | speed | |
|---|---|---|
| her hip | 0.487 m/s | **1.07× the hand her oar needs** |
| her shoulder | 0.982 m/s | 2.16× — not what moves the hand |
| the hand her oar needs | 0.455 m/s | — |
| **the model's hip** | 0.510 m/s | **1.05× hers — already right** |

**Her hand speed is her hip speed, not her shoulder's** — at the catch the arms are
straight and the hands ride the seat. And the model's seat is already moving at the
right speed. The defect is that the prescribed arc drags the hands at 0.284 m/s,
**0.58× the model's own hip**, and `hand_targets` makes the arms absorb the
mismatch instead of the oar. The body was fine; the coupling discarded it.

**The candidate fix**, in `entry_hip_driven_cr06.py` and behind `HIP_ENTRY=1` on the
gate: during the air phase put the handle where the seat has carried it. With the
hand at radius `r_h` from the pin, `d(hand_x) = −r_h cos θ dθ`, so

> `sin θ(t) = sin θ_catch − Δx_hip(t) / r_h`,  `θ̇ = −v_hip / (r_h cos θ)`

`Δx_hip` comes from the model's own leg chain — which is independent of the oar,
since only the elbow and hand are slaved by `hand_targets`. **No free parameters and
nothing read from her oar**, so unlike the measured-path study this one could ship.

**On the full gate, against the recorded run:**

| | speed | vel rms | power | drive lag 0.30/0.50/0.70 | finish rms 0.80–1.10 |
|---|---|---|---|---|---|
| recorded | 4.0901 (−2.4%) | 0.0331 | 250.0 W | +2.66 / +3.89 / +4.26° | **2.26°** |
| `HIP_ENTRY=1` | **4.1457 (−1.1%)** | **0.0317** | 257.6 W | **−0.03 / +0.34 / +0.25°** | **3.93°** |

**The drive is fixed and the finish is worse**, and the signs say why. The recorded
run crossed from *behind* her in the drive (+1.60° at 0.894 s) to *ahead* at the
turn (−1.95°). With the entry corrected it is already ahead at 0.894 s (−1.97°) and
reaches −4.99° by 1.025 s. Release −42.20° against her −39.1° (was −38.68°, nearly
exact); turn-round −49.38° against her −44.34°, error 2.15° → 5.04°.

**So the finish overshoot was always there and the drive lag was masking it.** The
recorded gate's respectable 2.26° finish rms was in part two errors cancelling —
worth knowing, because it means that number was never as good as it looked.

*Not promoted, and not promotable yet.* It is a clear net gain on speed and on the
whole drive, but it trades a finish that was accidentally right for one that is
honestly wrong. Promotion needs the finish handled, and that is the standing blocked
item — a **sourced recovery-push law** for the pull shape past the release. It would
also need testing on the eight and the four, and would go through `PhysicsProfile`
for research and learned only. `HIP_ENTRY` defaults to 0, so every recorded run is
unchanged.

### The recovery-push law: the premise was wrong, and the finish is a rate defect (2026-09-19)

Promotion of the finish fix has stood blocked on "a sourced recovery-push law for
the research pull shape". Going after it changed the question.

**1. There is no published negative-force law to find.** A regex sweep of the whole
local library for a negative gate/handle/pin force, or force "becoming negative",
returns **zero hits in every paper**. A second sweep for the physical alternative —
blade immersion, blade depth, vertical oar angle, washing out — returns almost
nothing either, and **no paper in the library reports a vertical oar angle at all**.
So neither route is sourceable from what is on hand.

**2. [CR06] does not need one, because it enforces the transition.** "We end the
drive when the force on the blade is exactly zero which … corresponds to
`v_O·ê_h = 0` in both oar models." And the paper flags the consequence itself:
"the deviations between the predicted and measured angle are greatest near the catch
and release [which] indicate that perhaps our enforcement of the instantaneous
transition from drive to recovery … is imperfect." **The primary source names the
very defect this programme has been chasing at both ends of the stroke.**

**3. The push is real and now measured, from her digitised trace:**

| | |
|---|---|
| peak handle force | 553 N at 0.508 s |
| push minimum | **−49.9 N at 0.965 s, −9.0% of peak** |
| force crosses zero | **0.8895 s** (10-harmonic fit; 0.8905 raw) |
| her release | **0.894 s** |
| push impulse | −8.9 N·s against a drive impulse of +291 N·s, **3.1%** |

The zero-crossing and the release agree to about 4 ms — a tenth of one sample
spacing of the digitised figure, so **indistinguishable at its resolution**. And the
research Beta shape, `u^a (1−u)^b`, reaches exactly zero at the end of
`drive_duration`. So for her the two candidate release triggers — "the torque
reaches zero" and "the drive ends" — are the same instant.

**4. But the finish defect is not the missing push.** The `HIP_ENTRY` gate run
already used her force **unclipped**, so the −9% tail was present, and the finish
still overshot by 5°. The release rates say why:

| | release rate |
|---|---|
| her measured | **−80 °/s** |
| baseline model | −107.9 °/s |
| with `HIP_ENTRY` | −103.8 °/s |

**The model's oar arrives at the release about 30% too fast**, in both runs, and a
3%-of-impulse push cannot absorb that. A bigger push tail would be fitting a number
to hide a rate error.

*So the finish is a rate defect, exactly as the catch was.* What decelerates a real
handle at the finish is the hands arriving at the body — the arms draw in and the
handle stops — and the model represents none of that, because `hand_targets` slaves
the hands to the oar instead of letting them constrain it. **The oar is
under-constrained by the body at both ends of the stroke, and the entry fix and the
finish fix are the same fix seen twice.**

*Status.* The blocker as worded — "a sourced recovery-push law" — should be retired:
it asks for a number that does not exist in the literature and that would not fix
the defect if it did. What replaces it is the hands-on-handle constraint, which is
the architectural change already named at the end of 4.3a. No number was adopted and
nothing in the model changed.

### WITHDRAWN, same day — see the entry below. Correction (2026-09-19): the entry fix's justification is refuted; the rule is empirical, not derived

Going at the hands-on-handle constraint properly tested the reason I had given for
the entry fix, and the reason does not hold. `arm_reach_cr06.py` measures
shoulder-to-handle distance against arm length across the cycle, on her boat:

| | d / arm length | oar angle |
|---|---|---|
| catch, t = 0 | **0.644** | +60.5° |
| **straightest** | 0.933 | **+22.5°, mid-drive** |
| most drawn | 0.503 | −44.3°, the finish |

**The arms are not straight at the catch.** They sit at 64% of full extension and are
straightest in mid-drive. So:

1. **The stated justification for the entry fix was wrong.** The commit and the 4.3a
   entry above say "at the catch the arms are straight and the hands ride the seat".
   The model's own geometry says otherwise. What survives is the *measurement* — her
   hand speed is 1.07× her hip speed over the air phase — but that is now an observed
   coincidence in one athlete, not a consequence of a principle. **The rule is closer
   to fitted than claimed, and must not be promoted on the old reasoning.** Its
   numbers (lag 4.24° → 0.25°, speed −2.4% → −1.1%) are unaffected; only its status is.
2. **A reach limit cannot be the finish constraint either.** The arms never approach
   full extension, so nothing runs out to stop the oar. The hypothesis that the oar
   is bounded at both ends by arm reach is dead.

**And it exposes why a kinematic hands-drive-oar rule is hard in general.** The hand
sits between shoulder and handle, so its position needs the *arm angles*. The model
solves those by IK **from the oar**, which is circular; the only independent source
is the stroke dataset's own arm angles, and those imply **1.73×** her oar rate. So:

* arms from the oar → circular, cannot drive it;
* arms from the ergometer dataset → 73% too fast;
* arms from her measured hip → works, but is the empirical relation above, on one
  athlete, with no mechanism behind it.

*This is a genuine wall, not a missing afternoon's work.* Driving the oar from the
body kinematically needs an arm model the project does not have. The route that does
not need one is the **torque-driven arm** — give the arms their own dynamics so the
hand position is an outcome rather than an input — which is tier 3 and the rest of
4.3, exactly where the programme already said this leads.

*Nothing was adopted and nothing in the model changed.* `HIP_ENTRY` stays at 0 and
is now recorded as empirical.

### The catch reach is a rig bug: `station_x + 0.30` (2026-09-19)

**This withdraws the correction above, which was wrong in two ways.** The project
owner pushed back that the arms are straight at the catch, which is correct
technique, and checking it properly found a defect in the model rather than in the
claim.

**Error 1 — I reported the model's own cap as a measurement.** The "arms are
straightest mid-drive at d/L = 0.933" figure is `_arm_reach_margin`, which the elbow
solve holds at **0.9326** by design to stay off the straight-arm singularity
(`kinematics.py`, "the elbow solve keeps a usable margin"). It is a hard limit, not
an observation, so it says nothing about where the arms are straightest. Withdrawn.

**Error 2 — I used the model's geometry to refute a claim about real rowing.** The
catch bend is real *in the model* (verified: shoulder `(−0.482, −0.200, 0.525)`,
hand `(−0.772, −0.391, 0.320)`, separation 0.403 m of a 0.626 m arm, port paired
with port, and the elbow chain confirms 0.279 + 0.347 = 0.626 with a genuine bend).
But real rowers *do* take the catch with straight arms, so this is the model failing
to represent the catch — it cannot adjudicate the premise. **The entry fix's
straight-arm justification is not refuted; it is untestable in the current model.**

**The cause, and it is a shipped constant.** `build_sculling_rig` and
`build_sweep_rig` both place the pin at

> `position = np.array([station_x + 0.30, side * span, oarlock_height])`

and `RowerStation(x_ankle=seat.station_x)` puts the feet at the same station, so
**the pin is always 0.30 m bow-ward of the feet**. At her 60.5° catch with a 0.83 m
inboard the handle swings `r_h sin θ = 0.722 m` aft of the pin, so the hands land
**0.422 m behind the feet** — which no sculler does. Straight arms would need 0.591 m
of horizontal reach against the 0.347 m available, a **0.244 m shortfall**, and the
oar cannot close it: the handle would need `sin θ = 1.16`. For the hands to arrive
near the feet the pin wants to be about **0.72 m** bow-ward of them, not 0.30.

The constant carries **no comment, no test and no mention anywhere in the docs** —
an undocumented magic number in the geometry, which is exactly what the programme's
sourcing rule exists to prevent, and it is in the sweep builder too.

**Reframed 2026-09-19, on the project owner's correction: the stretcher moves.** It
is adjustable fore-aft in any real shell, and setting it is exactly how a rower
places their catch. So `station_x + 0.30` is not "the pin in the wrong place" — it is
**a rigging variable hard-coded to a single value**, and that value produces a catch
posture that cannot occur. The fix is therefore *derivable rather than fitted*: put
the stretcher where the catch closes, which is what a rower does when rigging the
boat. No free parameter — the catch angle, the inboard and the arm length determine it.

**Which element is wrong is not yet settled, and should not be asserted.** What is
solid is that the set is mutually inconsistent at the catch:
* the arms sit 36% bent where real technique has them straight;
* the shoulder (−0.482) is 0.13 m stern-ward of the ankle (−0.35), a very large lean;
* **the finish, by contrast, looks right** — the handle arrives at +0.53, between the
  hip (+0.451) and the shoulder (+0.612), which is where a finish should land.
So the defect is at the catch end, and the candidates are the stretcher position, the
body's lean at the catch, and the catch angle. Moving the stretcher **bow-ward**
(feet nearer the pin) is what increases the shoulder-to-handle separation; moving the
pin bow-ward makes it worse, which is the opposite of what I first assumed.

*Consequences, not yet acted on.*
* **`HIP_ENTRY` is reinstated to its earlier status** — a rule whose justification is
  plausible and whose numbers stand (lag 4.24° → 0.25°, speed −2.4% → −1.1%). It is
  still only validated on one athlete.
* **The hands-on-handle programme is blocked on this, not on arm dynamics.** Any
  constraint tying the oar to the hands is being fed a catch posture that cannot
  occur, so the earlier conclusion that it "needs a torque-driven arm" was premature.
* **Fixing it changes shipped geometry.** The pin position moves the crew kinematics
  and the oar's moment arm, so it would change the frozen game. That is the project
  owner's call, and nothing has been altered.

### PARTLY SUPERSEDED, same day — see the next entry. The rig question is [CR06] eq. 8, and our 3-D arm breaks their degeneracy (2026-09-19)

Chasing the catch geometry landed on [CR06]'s own hand-on-handle constraint, §2.6
eq. 8:

> `d_L/F − s sin θ = x_B/F + x_S/B + x_H/S`

left side the **handle** relative to the foot stretcher, right side the **hand**.
Three things follow.

**1. `d_L/F` — oarlock fore-aft position relative to the stretcher — is an explicit
rigging input.** "Oar rotation is known once the rower's coordination and the oarlock
fore-aft positioning are specified." That is exactly the quantity
`build_sculling_rig` hard-codes as `station_x + 0.30`, and the project owner is right
that it is an adjustable setting rather than fixed geometry.

**2. [CR06] cannot supply its value.** Appendix A.7: `d_L/F` and `x_H/S(0)` "cannot
be chosen independently … therefore we fix `x_H/S(0)` and allow `d_L/F` to be free".
It is **fitted per trial and degenerate with the initial hand-to-shoulder distance**,
so only their sum is identifiable. There is no measured number to take.

**3. Our model breaks that degeneracy, and [CR06]'s cannot.** They state they are
"only considering the movement of the arm in the fore-aft direction … neglecting the
projected arm shortening due to the arms not being parallel to the boat", so their
`x_H/S` is unconstrained — nothing checks the arm can reach. **Ours is a 3-D arm of
finite length**, so `x_H/S` is determined by the anthropometry, and `d_L/F` follows
from eq. 8. Evaluated at her catch:

> `d_L/F = s sin θ + x_B/F + x_S/B + x_H/S = 0.722 + 0.220 − 0.352 − 0.512 = **0.078 m**`

against the hard-coded **0.300 m**. The 0.222 m gap is the same number the stretcher
sweep produced independently, which is a consistency check on both.

**But the catch and finish still do not both close.** Setting `d_L/F = 0.078` leaves
the finish **0.142 m short of the hip**, and the underlying sum is rig-independent:

| | |
|---|---|
| the oar offers | `r_h (sin θ_catch − sin θ_finish)` = **1.303 m** of handle travel in x |
| the body wants | shoulder 0.978 + arms (0.512 straight at the catch − 0.045 drawn) = **1.445 m** |
| difference | **0.142 m** (0.222 m using the rower's native trunk rather than her 0.398) |

So **the body's stroke is longer than the oar's sweep**, whatever the stretcher does,
and the model has been absorbing it by bending the arms 36% at the catch. The
remaining suspects are the ones not pinned by her measurements: the **arm length**
from the anthropometry at the fitted stature, and the **span of 0.80 m**, which was
chosen in `her_boat` and is not from [CR06] — the lateral offset costs 0.079 m of
fore-aft reach on its own.

*Also corrected:* `s = 0.83 m` is right but is **not the geometric inboard**. Table 1
defines it as `s* − 0.06 m` for sculls, "assuming that the rower applies a force at
the oar handle 6 cm from the end", so the actual inboard is ≈ 0.89 m — consistent
with the usual 0.88 — while 0.83 is the force/hand radius the model correctly uses.

*Status: derivable, not fitted.* `d_L/F` now has a route that needs no new data, and
the residual is a well-posed 0.14 m to chase rather than a mystery. Nothing changed
in the model; fixing `d_L/F` moves shipped geometry and is the owner's call.

### The bent catch is the trunk over-travel, not a rig bug (2026-09-19)

Solving [CR06] eq. 8 for `d_L/F` properly — letting the travel balance fix the arms
instead of assuming a straight catch — closes the question and retracts most of what
the last two entries claimed.

**Eq. 8 must give one `d_L/F`, and it gives two.**

| evaluated at | `d_L/F` |
|---|---|
| the catch, with **her** measured leg and back travel | **0.184 m** |
| the finish, with the **model's** body | **0.300 m** — the coded value |

The 0.116 m between them has a single cause:

| | leg | trunk | total |
|---|---|---|---|
| the model's body | 0.581 | **0.515** | 1.096 m |
| her measurement | 0.581 | **0.398** | 0.979 m |
| | | | **0.117 m apart** |

**The model's trunk travels 29% further than hers** — which is already on the record
from the [LE26] work as "the model's trunk … travels 30% further". The scratchpad
applies her `K = 0.771` to the body *field* that feeds the momentum books, but
`joint_positions` — and so the hand tracks, and so the entire oar coupling — still
runs the unscaled trunk.

**So the bent catch is a symptom, not a cause.** The over-travelling trunk consumes
0.117 m of hand travel the arms would otherwise supply, so the arms start the catch
less extended: `x_H/S(catch)` comes out −0.289 m, 56% of the available fore-aft
reach, instead of the −0.406 m and 79% that her body implies.

**Retractions.**
* *"`station_x + 0.30` is an undocumented magic number producing an impossible
  catch."* Overstated. It is **consistent with the model's own body** — eq. 8 at the
  finish returns exactly 0.300. It is still undocumented and still a rigging variable
  frozen at one value, but it is not itself the defect.
* *"No single stretcher position closes both ends."* **Wrong**, and wrong because I
  held the catch arm at full extension and then found the finish would not fit. Eq. 8
  does not work that way: the travel balance determines how far the arms draw in, and
  the stretcher follows. With her trunk the geometry **does** close, arms at 79% at
  the catch.
* *"The body's stroke is longer than the oar's sweep."* Only under the straight-arm
  assumption. The arms must draw in by `1.303 − 0.979 = 0.324 m`, and the 0.512 m of
  fore-aft reach available covers it comfortably.

**What stands.** `d_L/F` is a genuine rigging input, [CR06] cannot supply it (fitted,
and degenerate with `x_H/S(0)` — Appendix A.7), and our 3-D arm breaks that degeneracy
where their fore-aft-only formulation cannot. **Once the trunk travel is corrected,
`d_L/F` must move from 0.300 to 0.184 m**, derived rather than fitted.

*So the catch geometry is downstream of a defect already on the list with independent
measurement behind it.* Fixing the trunk travel should fix the catch arm posture as a
consequence — a real test of that connection, and it needs no new data.

### The trunk prediction fails, and the failure localises it better (2026-09-19)

`dec6d9d` predicted that scaling the trunk to her measured 0.398 m travel would fix
the catch posture, with `d_L/F` following to 0.184 m. `trunk_fix_geometry_cr06.py`
tests it by working [CR06] eq. 8 forward — `x_H/S(t) = d_L/F − s sin θ − x_B/F −
x_S/B` — so the arm posture is determined at every instant for each candidate rig.

**It fails, and so does every other `d_L/F`.** Reach demanded, as a fraction of what
the arm can give:

| t / T_drive | native trunk, `d_L/F` 0.300 | her trunk, 0.300 | her trunk, 0.184 |
|---|---|---|---|
| 0.00 (catch) | 56.6% | 72.2% | 94.9% |
| 0.20 | 76.8% | 89.2% | 111.4% |
| **0.41** | **100.0%** | **107.6%** | **132.2%** |
| 0.70 | 41.2% | 39.5% | 65.0% |
| 1.00 (finish) | 16.5% | 9.1% | 32.4% |

**The model's arm runs backwards through the early drive** — 57% extended at the
catch, rising to **full stretch at 41% of the drive**. Real technique is the
opposite: straight at the catch, held straight through the legs and back, bending
only for the arm draw.

**And that 100.0% is the solver's own cap, not a coincidence.** It equals
`_arm_reach_margin` × arm length (0.9326 × 0.626 = 0.584 m), and it falls at
t = 0.369 s, which `arm_reach_cr06.py` independently found as the point of maximum
extension. **The elbow solve is already clamped in mid-drive**, so the geometry has
no slack anywhere, and any change that lengthens the reach demand overruns it.

**So `d_L/F` cannot be the fix.** Moving it to 0.158 m puts the catch at full stretch
— correct — but demands **138%** of reach at mid-drive, which no arm can do. The
catch and the mid-drive pull in opposite directions and the arm is already at its
limit between them.

*What this rules out, and what it leaves.* Ruled out: the stretcher position alone,
and trunk travel alone. What remains are the inputs her measurements do not pin and
that all bear on mid-drive reach —

* the **catch lean** (the model's shoulder sits 0.35 m stern-ward of the hip at the
  catch, 0.27 m with her trunk scaled);
* the **arm length** from the anthropometry at the stature fitted to her leg travel;
* the **oarlock height** (0.32 m, chosen in `her_boat`): the shoulder sits ~0.30 m
  above the handle at mid-drive and that vertical drop costs 0.08 m of fore-aft
  reach right where the budget is tightest;
* the **span** (0.80 m, also chosen rather than taken from [CR06]).

*The honest summary is that the one-athlete rig has four unpinned geometric inputs
and no slack, so the catch posture cannot be attributed to any one of them yet.*
Nothing changed in the model.

### v0.13 — asked for, and done

| what | where | pinned by |
|---|---|---|
| the rig you pick is the rig that rows: `main` reads the lineup off the setup menu, and the plan's shell wins | `main`, `fpv.py` | `tests/test_rig_reaches_the_water.py` — every four and eight rig built and compared, and the real menu walked key by key into the editor and back to Start |
| age, skill and experience per rower in the editor; blank defers to the crew slider (skill and experience store -1, not 0, which would mean novice) | `Rower`, `FIELDS`, `rigview.py` | `tests/test_rower_fields.py` |
| CP from each rower's own erg, `CP = P − W′/t`; W′ scaled by age, not power; the trainer's reserve is the crew's (HOCR four 125.6 W, W′ 7.75 kJ) instead of the literature 302.7 W | `coxswain/crew/ageing.py`, `boat_from_lineup` | `tests/test_ageing.py` |
| per-seat skill scales that seat's deviation, not its value; per-seat experience averages into the crew's balance | `seat_scale_for_skill`, `variability.py`, `fpv.py` | `tests/test_rower_fields.py` |
| full screen: flag, remembered setting, menu row, F11, falls back to a window if refused | `display_flags`, `set_display_mode`, `fpv.py` | `tests/test_rower_fields.py` |

### v0.12 — asked for, and done

| what | where | pinned by |
|---|---|---|
| Q and Quit ask first; only Yes ends the session | `confirm_quit_menu`, `fpv.py` | `tests/test_quit_minimap.py` |
| minimap in the corner, north up, under the HUD's change key (position to 2 m, heading to 5°); off in settings or `--no-minimap` | `draw_minimap`, `fpv.py` | same; Minimal benched at 9.1 ms with it on |
| ~~the rig editor's lineup **races**~~ — **not true as shipped**: `boat_from_lineup` was right and never called with a lineup, because `main` never read it off the setup menu. Fixed in v0.13 (see Fixed) | `boat_from_lineup`, `menu.py` | `tests/test_presets.py`, and now `tests/test_rig_reaches_the_water.py` |
| saved presets in `presets.json`; Save as preset row; built-ins cycle first | `presets.py`, `rigview.py`, `fpv.py` | same |
| Genevieve's Pink Ribbon built in; editor opens on a named default | `rigview.py` | `tests/test_rigview.py` |
| bonus run behind the word "boost": 77 coins / 8 boosts on HOTL, +0.30 on the call for 8 s, score and best kept | `bonus.py`, `fpv.py` | `tests/test_bonus.py` |


### Measured, v0.11 round (Intel UHD, 1180x680, hazy, eight; `--bench 150 --bench-passes`)

| change | evidence | effect |
|---|---|---|
| packed vertices, 20 B | pixel-identical to reference (max 2/255 at Standard) | world pass 4.7→4.2 ms Minimal, 6.3→4.5 Standard |
| sky drawn last (plain-water tiers) | same | sky pass 0.2→0.1 |
| water grid 80/120/240 below High | linear in vertex count: 5.7/3.5/2.2/1.5 ms at 240/160/120/80; 120 vs 240 indistinguishable by eye | water 7.3→2.2 Minimal; Standard 8.2→6.3 same session |
| tile culling, 350 m | pixel-identical at equal water density (max 5/255); 365/560 tiles | world 4.2→3.0 Minimal, 4.5→2.8 Standard |
| oar loop vectorised | oar-table, stepwise, six-DOF tests on both paths | derivative → 0.77 ms |
| world build across cores — **reverted** | Standard: 14.7 s serial, 14.8 s on four workers, and a different world (1,277,564 tri in 13 parts vs 1,291,399 in 10) | none; Windows spawn + reload per worker ≈ the slice of work |

Two findings that changed what was done: at a **quarter of the pixels
neither the water nor the world pass moved**, so both are vertex-bound
on this GPU and fill-rate levers (render scale) cannot help; and this
iGPU's clock **drifts ~25% between sessions** (Minimal water read 5.7
and 7.3 ms on identical code an hour apart), so every A/B here was
taken within one session.


### Measured, this round (Intel UHD, 1180x680, hazy, eight; `--bench 150`, no timers)

| tier | physics | draw+GPU | frame p50 | fps | v0.9 frame |
|---|---|---|---|---|---|
| Ultra minimal | 1.7 ms (Heun, 50 Hz) | 7.9 | **9.7** | 104 | 16.6 |
| Minimal | 2.9 (Heun, 60) | 14.3 | **17.2** | 58 | 18.7 |
| Standard | 3.8 (Heun, 60) | 18.0 | **21.9** | 46 | 25.9 |
| High | 6.8 (RK4, 100) | 22.4 | **29.3** | 34 | 42.8 |

Per pass (`--bench-passes`): water 3.9 / 7.3 / 7.6 / 11.3 ms, world 3.0 /
4.3 / 5.8 / 5.6, sky 0.2–0.3, boat 0.1. The water is still the largest
pass at every tier; the next lever is its fragment cost at Minimal and
above, not triangles. The `.exe` is within 0.3 ms of the script.

| what | where | pinned by |
|---|---|---|
| **Four graphics tiers, each cheaper than the last.** `ultra` is the floor: flat water, no SSR/refraction, one shadow tap from a 2048² map, plain distance fog, no skyline, every tree an impostor, 400 m reach at 12 m step, no MSAA, 50 Hz physics. Render scale is a flag, not a tier default — measured, it hurt on an iGPU. Every knob is a `Tier` field; a typed flag always beats the tier. | `coxswain/viz/menu.py` `QUALITY_TIERS`; `scripts/fpv.py` tier block; shader uniforms `shadow_taps`, `fog_simple`, `reflect_steps` | `test_the_tiers_do_not_go_below_fifty_hertz`; per-tier `--bench` |
| **The tier is chosen for you.** `--quality auto` (the default) reads the adapter list before the build and the live GL renderer after the context exists; a dedicated GPU that is not the one drawing is reported with the fix, and `--prefer-dedicated-gpu` writes the per-user Windows preference — only when asked. | `coxswain/viz/hardware.py` | `tests/test_hardware_telemetry.py` |
| **The `.exe` is not slower than the script.** v0.9 frozen build against v0.9 source on the same iGPU, same tier: Minimal 19.3 vs 19.1 ms, Standard 23.4 vs 23.6, physics 2.9 ms in both — numba is live in the frozen build. Re-check after any build-tooling change with `Coxswain.exe --bench 150`. | `tools/build_exe.py`; `--bench` in the exe | measured 2026-09-09 |
| **Timer queries cost 2.3 ms a frame themselves** on an integrated part (11.8 → 9.5 ms at ultra). `--bench` takes its headline WITHOUT them; `--bench-passes` adds the per-pass breakdown, which is the tool that found the water pass and the sky noise. | `PassTimer` in `scripts/fpv.py` | `--bench` vs `--bench --bench-passes` |
| **The loop cannot spiral.** `max_frame` alone allowed fifteen steps in one late frame at 60 Hz; the loop now takes `max_steps` (4) and drops the rest, counting it (`dropped`) into the diagnostics and the report. The boat runs briefly slow; the program does not stop. | `coxswain/sim/realtime.py` | `tests/test_any_machine.py`, `tests/test_stepwise.py` |
| **The HUD is composed and uploaded only when it changes** — its text and the rudder knob's pixel. A full-window RGBA upload plus font rendering every frame was ~3 ms on an iGPU. | `scripts/fpv.py` HUD block | `test_the_hud_is_uploaded_only_when_it_changes` |
| **Low tiers stop paying for sky noise in the water** (`sky_detail`) and use the vertex-baked slope as the water normal (`water_detail`). Sky pass 1.1 → 0.3 ms, water 4.9 → 4.1 at ultra. | shader uniforms `sky_detail`, `water_detail`; `Tier.sky_detail` | `tests/test_any_machine.py` |
| **Heun below High.** Two derivative evaluations a step instead of four; at 60 Hz it is RK4 at 100 Hz to 3 cm over 367 m (0.0002 m/s, same roll). High keeps RK4; the studies and the golden never see Heun. | `coxswain/core/integrators.py`, `Tier.physics_scheme`, `--physics-scheme` | `tests/unit/test_heun.py` |
| **Performance reports come home.** Off until switched on in the menu (remembered); at close, a JSON of numbers and product names — never a name or path, the sender refuses — goes to a URL the program is told (`--report-url`, `COXSWAIN_REPORT_URL`, `report_url.txt` beside the exe), intended for a Google Apps Script that appends to a sheet; with no URL it is written beside the logs to paste. | `coxswain/viz/phonehome.py`, `settings.py`, `packaging/phonehome/` | `tests/test_phonehome.py` |
| **A diagnostics file.** Build, hardware, settings, world timings, ten-second frame summaries with the physics/draw/present split, stalls over 100 ms with context, tracebacks. Local only, in the OS log folder, printed at start-up; `--no-diagnostics` turns it off. | `coxswain/viz/telemetry.py` | `tests/test_hardware_telemetry.py` |
| **`--bench N`** runs N headless frames with `ctx.finish()` and prints the split, so a regression is a number. | `scripts/fpv.py` | — |
| **Startup**: the wall builder emits each building in one numpy block instead of two Python lists per edge; CPU mesh copies are dropped after the GPU upload (140 MB); the cyclic GC is off during the build and the world is frozen after. | `coxswain/viz/worldmesh.py` `building_walls`; `scripts/fpv.py` upload loop | — |
| **The release tag is in the build**, so the diagnostics file names it. | `tools/build_exe.py` → `packaging/VERSION`; `release.yml` | — |

---

## Done

**Read this before touching anything a summary, a hook, or a memory
says is missing.** Everything below is in the code. Each row says
where it lives and what stops it silently going away. If a request
matches a row here, the answer is a pointer to the row, not a second
implementation — a second splash system, a second wind bed, a second
sky-noise term would *break* the constraints the first one was built
to.

### Scene and audio

| what | where | pinned by |
|---|---|---|
| **Shadows cast by everything, onto everything.** `static` and `shadow_casters` are built in the same loop over every mesh part, so buildings and trees shadow each other and themselves — not just the ground. 4096² map over 3965 m, 1.0 m a texel, baked once because the sun does not cross the sky during a 5 km race. | `scripts/fpv.py` `shadow_casters` | `test_everything_in_the_world_casts_a_shadow` |
| **Shadows received on the surface's own normal.** The world shader passes the fragment normal, because the lookup is pushed a texel along it to clear its own cell. A fixed up-vector would offset every wall as though it were flat ground. | `scripts/fpv.py` `sun_visibility(v_world, n, …)` | `test_the_world_receives_shadows_on_its_own_normals` |
| **Bow disturbance and the hull breaking the water.** `waterline_foam` paints white where the hull meets the surface, tied to the hull frame and to speed, so a stopped boat has none. | `scripts/fpv.py` `waterline_foam` | `test_the_bow_foam_is_tied_to_the_hull_and_the_speed` |
| **Catch splash at blade entry.** A small pooled particle burst, deliberately not dramatic. Off at standard graphics, on at high; `--no-particles` forces it off for A/B. | `scripts/fpv.py` `SplashSystem` | `test_the_splash_pool_exists_and_empties` |
| **Splash off a blade dragging on the recovery.** Every frame, each oar on its own seat clock asks `BladeContact.immersion` at the hull's roll — the same immersion the skim drag uses — and a blade riding below the surface trickles three drops every 80 ms. A clear blade throws nothing; a blade on the drive is left alone. | `scripts/fpv.py` `DRAG_SPLASH_*` | `test_a_dragging_blade_trickles_and_a_catch_bursts`, `test_the_drag_splash_is_gated_on_the_physics_not_on_the_picture` |
| **Ambient world audio** — wind gusting on a slow envelope and brightening with wind speed, water working at the bank, the odd gull. About a fifth of the stroke's level so it never competes with the boat; follows the wind setting live. | `coxswain/viz/ambient.py` | — |
| **Low-frequency variation in the grey sky.** Two octaves of simplex over the exponential gradient, amplitude 0.045, gated on overcast, so a clear day takes almost none and the gradient still dominates. | `scripts/fpv.py` sky shader | `test_the_sky_noise_stays_under_the_gradient` (≤ 0.08) |
| **Diagonal line on the water** — the detailed patch met the far plane at a hard edge. Fixed. | `763e57b` | — |
| **Charles arch bridges: piers under the arches.** Three faults: solid piers drawn at even fractions while the openings followed the measured stations (Western Avenue 17.3 m out); Weeks skipped by a gate on an NBI length a footbridge does not have (13.4 m out, 24 m of arch on dry land); the structure centred on the deck line while the comment said channel (Western 26 m up the Boston bank). Piers now stand at the edges the arches are cut between; every arch bridge is trimmed to the water. | `coxswain/viz/worldmesh.py` `arch_bridge`, `bridge_solids` | `test_every_arch_bridge_has_its_piers_where_navigation_has_them`, `test_no_arch_bridge_stands_on_dry_land_past_its_abutments`, `test_arch_piers_stand_under_the_arch_springings` |
| **Grand Junction trestle** drawn from the surveyed piers, not at thirds. | `truss_bridge` | `test_the_surveyed_piers_are_where_the_survey_puts_them` |

### The crew, as drawn

| what | where | pinned by |
|---|---|---|
| **Per-seat timing in the picture.** Every rower and oar is posed at its own `t − phase·period`, the same offset the physics applies. | `coxswain/viz/worldmesh.py` `oar_pose`, `crew_solids` | `tests/unit/test_viz.py` timing tests |
| **Blade enters and leaves over a window**, not in one frame (was a 190 mm step). | `BLADE_ENTRY`, `BLADE_EXIT`, `_smoothstep` | `test_viz` |
| **Blades square on the drive, feathered on the recovery.** | `oar_pose` | `test_viz` |
| **Bucket rig drawn as a bucket.** S-P-P-S from a starboard stroke: riggers, handles, blades *and* the direction each trunk winds all flip with the side. Measured on the HOCR four. | `coxswain/boats/rig.py`, `crew_solids` | `test_the_plan_puts_riggers_on_the_side_the_rower_rows` |
| **Each rower their own size.** Link lengths from their own de Leva segments (already); girth now ∝ √(mass/stature), so a 70 kg bow is no longer drawn as slight as a 54 kg stroke. Default rower returns exactly 1.0. | `_build_factor` | `test_a_heavier_rower_is_drawn_heavier`, `test_the_drawn_crew_are_not_all_the_same_size` |
| **Riggers on every shell.** Three struts from the gunwale to the pin per oarlock, on the oarlock's own side, read off the hull outline at that station; merged into the hull mesh so they ride with it. There were none before -- the loom pivoted in mid-air. | `coxswain/viz/worldmesh.py` `rigger_solids` | `tests/test_riggers.py` |
| **Body parts where the physics puts them.** Hands on the handle, elbows split the lift, trunk rotation from the kinematics — verified joint by joint. | `crew_solids` | `test_viz` |

### Physics wired into the trainer

| what | where | pinned by |
|---|---|---|
| **Michell wave resistance** in the physics, tabulated once and interpolated. | `coxswain/boats/boat.py` `USE_MICHELL` | Holt calibration tests |
| **Wind as the excess over still air** — no double count. 12.0 % vs the 12.2 % measured. | `coxswain/hydro/wind.py` `excess_loads` | `0111af8` tests |
| **Blade contact**: skim drag, righting moment, lost sweep, early/late catch bounded by `square_up_fraction`. | `coxswain/crew/blade_contact.py` | `0240616` tests |
| **Crew variability** (skill slider), **fatigue reserve** (CP / W′) with coxswain power calls. | `coxswain/crew/variability.py`, `exertion.py` | `dc93060` tests |
| **Balance by experience** (ILC trim, phase authority), handle heights and lean matching. | `coxswain/sim/control.py` `balance_for_experience` | `5dffacb` tests |
| **Coupled crew timing** — the hull and the crew's clocks close the loop. | `coxswain/crew/coupled.py` | `2be526d` tests |
| **Recovery from a perturbation** measured in strokes. | `scripts/recovery.py` | — |

### The boat, the crew, the menu

| what | where | pinned by |
|---|---|---|
| **The HOCR four**, rower by rower, at its own weight and power; report and time predictions. | `scripts/my_crew.py`, `scripts/hocr_crew.py` | — |
| **Rig editor**: top-down plan, riggers on the rowing side, stat boxes linked to each seat, presets, shell and rig change, switch side. | `coxswain/viz/rigview.py`, `fpv.run_setup_menu` | `tests/test_rigview.py` |
| **Text entry** — type a rower in, field by field; bad input refused, not zeroed; a typed height is a measurement. | `rigview.FIELDS`, `parse_field`, `Lineup.commit_edit` | `test_a_rower_can_be_typed_in_from_the_menu` |
| **Anonymous squad roster** — 82 rowers, 305 pieces, names removed, weight to 5 lb, time to the second, order shuffled; the sheet itself never committed. Squad presets built from it. | `scripts/build_squad_roster.py`, `data/squad_roster.csv`, `coxswain/crew/roster.py` | `tests/test_roster.py` (no-identities test) |
| **Calibration against a real result** — 2024 Sammamish W Vet 4+ (12th, 22:07.9). | `97caa8f`, `4fe7e7e` | — |

### Distribution

| what | where |
|---|---|
| **Windows, macOS (arm64) and Linux builds** on a three-platform matrix, ad-hoc codesigned `.app`, glibc-compatible Linux, `gh` upload with retry. v0.7 live on all three. | `.github/workflows/release.yml` |
| Release notes taken from the repository. | `packaging/RELEASE_NOTES.md` |

---

### Coxing research — calls and boat

| what | where | pinned by |
|---|---|---|
| **Calls and boat modelled as one coupled process.** A causal transformer carries both factors of p(C,B); stream dropout gives own-past and both-pasts predictions from one set of weights, so C→B and B→C are measured inside one model and call-only transcripts train the call self-term. A Hawkes-style linear autoregression is fitted on the same folds as the reference. | `research/callmodel/` (`model.py`, `train.py`, `evaluate.py`, `run.py`) | `selftest.py`: both models recover planted coupling (p = 0.032 at the 30-draw floor) and give chance p-values with none |
| **Plug-and-play sessions.** A recording is a folder: `meta.json`, plus any of `words.jsonl`, `boat.csv`, `labels.json`. `ingest.py captions` builds call-only sessions from a folder of .vtt; `ingest.py session` builds a synchronised one from a transcript and a boat log with a clock offset. Nothing else changes to add data. | `research/callmodel/sessions.py`, `ingest.py`, `README.md` | — |
| **First run, 2026-09-25:** 38 sessions (3 synchronised races, 35 transcripts), 11,313 grid steps, 647 with boat data. Self-prediction boat +0.32 / +0.28 nats, calls +0.35 / +0.32 (linear / transformer); coupling negative in both directions for both models, p 0.20–0.67; a second seed agrees. The race-1 catch effect is explained by the boat's own past. | results in `data/local/coxing/` (gitignored) | the working paper's coupled-process section |
| **Email replies recorded.** Kleshnev sent an elite M1x averaged stroke ([BR24], SOURCES §156, data gitignored); Buckeridge referred us to McGregor. Drafts of both follow-ups are in DATA_REQUESTS. | `docs/DATA_REQUESTS.md`, `docs/SOURCES.md` | — |

### [BR24] like-for-like tooling (one athlete, measured stroke)

| what | where | pinned by |
|---|---|---|
| **Channel calibration from the data**: seat zeroed at the catch, trunk relative to the seat at shoulder height (momentum fit 0.583 m), rower CoM travel 0.71–0.74 m. | `research/biorow/calibrate_channels.py` | SOURCES §156 |
| **The research model as his boat** at his 432 W: variants swap in his arc, trunk sweep, slide and drive fraction; the nearest reachable oarlock position is searched and reported. | `research/biorow/like_for_like.py` | SOURCES §158 |
| **Shape and momentum diagnostics**: speed and force timed from the catch; his force shape as the pull; the model's body re-timed onto his seat and trunk curves; the [CR06] finish study on his force. | `research/biorow/shape_br24.py`, `measured_body.py`, `finish_br24.py` | SOURCES §158 |

The measured stroke is commercial and stays in `data/local/biorow/`; every script prints derived numbers only.

## Fixed

| what it was | how it was found |
|---|---|
| **The leg masses were measured from the wrong end (research and learned only; shipped frozen).** `segment_state` placed the thigh CM 40.95% from the *knee* (de Leva: from the hip), and applied the lumped shank+foot fraction to the shank link from the *ankle*. Now `PhysicsProfile.segment_com = "de_leva"` for `research` and `learned`: thigh from the hip, shank 44.59% from the knee, foot mass held at the ankle because it is strapped to the stretcher. `shipped` stays `"legacy"`. The placement is in `kinematics_signature`. **What it moved:** crew CoM travel +0.009–0.010 m (1x 0.771 → 0.780, 8+ 0.728 → 0.737, [BR24]'s 97 kg / 1.91 m athlete 0.789 → 0.799); IVV +0.5–0.75 points on the prescribed oar and +0.9–1.05 points on the research oar (1x 67.7 → 68.7%, 8+ 56.5 → 57.4% at 260 W); speed +0.1–0.2%. It moves the model *away* from [BR24]'s 0.71–0.74 m, and is kept because it is anatomy, not a fit (SOURCES §157). Tests: `tests/unit/test_kinematics.py::test_de_leva_thigh_cm_is_measured_from_the_hip` and `…shank_is_measured_from_the_knee_with_the_foot_fixed` fail without it; `tests/test_physics_profiles.py::test_the_leg_correction_moves_the_research_crew_not_the_shipped_one` checks both sides of the freeze. | Chasing the crew CoM travel gap against the new [BR24] target: `com_fraction` is documented from the proximal joint, and two of the four limb placements read it from the distal one. Upper arm, forearm+hand and trunk checked and correct. |
| **Crews that moved differently shared one matched torque.** `DynamicOarSimulator._match_key` covered mass, rig, hull, pull and stamp, but not how the crew moves. So a research boat with a different leg placement, or a different stature at the same mass, was handed the first crew's cached sweep-catch torque. The key now carries every rower's `kinematics_signature`. `tests/test_torque_for_power.py::test_a_crew_that_moves_differently_is_a_different_cache_entry` fails without it. | Measuring the leg fix above: the second research run finished in 21 s against 49–96 s, a cache hit. Those numbers were discarded and re-run in separate processes at matched power. |
| **Two pull shapes shared one matched torque.** `_match_key` still omitted the force profile, so a boat built with `OarForceProfile(shape="cr06")` was handed the default shape's cached sweep-catch torque. That shape's mean is 1.108× the default's, so the boat rowed **11–12% above its stated power** on every Holt class. The key now includes the profile's shape, peak shift, shift per spm and reference rate. `tests/test_torque_for_power.py::test_a_different_pull_shape_is_a_different_cache_entry` fails without it. The first `holt_cr06_shape.py` rows were measured at that wrong power and are discarded. | The study's settled power read 371.5 W against a 334 W target, and its matches finished in 15–21 s against 55–69 s: a cache hit, not a settle. The hull-shape omission fixed hours earlier was the same class of bug. |
| **Two hulls with the same name shared one matched torque.** `DynamicOarSimulator._match_key` covered mass, timing, rig, profile, wave model and depth, but not the hull's offsets. So a boat rebuilt with a different hull under the same name would be handed the first hull's cached sweep-catch torque, at the wrong power, without complaint. The key now includes the offsets' station, beam and depth arrays. `tests/test_torque_for_power.py::test_a_different_hull_shape_is_a_different_cache_entry` fails without it. | Planning the [L9701] hull study, which rebuilds the single at other dimensions. The study's boats carry distinct names, so its numbers were never affected. |
| **`BladeModel`'s docstring said [CR06] fitted `C₂`.** They computed it, as ½ρC₀A₀ with C₀ ≈ 1.3 from Hoerner and measured blade areas. The value that best fitted their singles was about 2.4× that. Docstring corrected; no number changed. | The ledger's audit question on whether `C₂` is a fit. |
| **The research blade acted at the oar's tip, not its centre.** Concept2 measure an oar's overall length "from the end of the grip … to the edge of the blade", and the rig's 3.70 m and 2.88 m are those overall lengths — but `Oar` documented `length` as blade centre to handle end, and the dynamic oar took `length − inboard` as the blade lever arm: 2.56 m sweep and 2.00 m scull, half a blade too far out. [CR06], whose `C₂` the research blade uses, applies its force at `outboard − blade_length/2`; its Table 1 implies 0.52 m and 0.43 m blades, matching Concept2's Big Blade. Now `Oar.blade_centre_outboard` (2.30 m sweep, 1.785 m scull), read by `OarDynamics.from_boat` and so by the reduced model, the dynamic oar and the figure; the shipped trainer never reads it. **What it moved, at 380 W:** eight rate 28 5.619 → 5.530 m/s, drive fraction 0.400 → 0.376, blade efficiency 0.586 → 0.559; eight rate 32 5.56 → 5.461, 0.467 → 0.440; coxed four rate 32 4.900 → 4.829, 0.510 → 0.481, 0.592 → 0.566; single rate 30 4.217 → 4.171, 0.559 → 0.525, 0.636 → 0.614. Every boat stays in its published pace band; the drive shortens toward on-water timing; blade efficiency falls further below Kleshnev's band. Reduced-model gate 8.9 → 6.4, full-hull gate 1.81 → 1.57, both passing. A prescribed sweep's impulse now crosses zero at 4.36 m/s on the eight at rate 28 (it was ~4.85). | Answering the ledger's audit question on where ℓ is measured, against Concept2's published definitions and the USRowing rules. |
| **A sculler's body was counted once per oar.** The dynamic oar averaged each oar's own balance, and each carried the rower's whole reflected inertia, so a sculler's two oars did not balance as one body: 2.2% of the handle work over a drive went unaccounted for. Now one seat balance, `(I_crew + n I_oar) φ̈ = −n τ + Σ blade`, in both the dynamic simulator and the reduced model; sweep seats are bit-identical. The single at 380 W went from 3.99 m/s — the recorded 2.6% miss below its published band — to **4.22, inside 4.1–4.7**, and its drive from 0.628 of the stroke to 0.559. | Drawing the blade-path figure: the single's drive filled most of the stroke, and closing the energy books over one drive showed the residual. |
| **The optimiser and the simulator disagreed about the rudder.** `hydro_casadi.surface_load` used `lift_curve_slope * angle` where the numpy path uses Whicker–Fehlner — dropping both the `cos` that stalls the surface and the cross-flow term. Up to **6.5% low at 15°**, and unbounded growth past stall, which is exactly the sort of thing an optimiser exploits. The CasADi path is what `PathMPC`, `ReducedModel` and the route optimiser run on, and `SOURCES.md` §10 says steering is already marginal against the tightest bends. | Investigating why 14 CasADi tests had been failing. The comparison against numpy was there and had been **unable to run since 26 August**, so nothing was checking. Now agrees to 0.0 across every appendage, deflection and sideslip tried, and `tests/unit/test_hydro_casadi.py` sweeps the lift curve degree by degree from −45° to +45° instead of sampling it, carries a planted-failure test so the guard can fail, and checks the surface stalls instead of growing without limit. |
| **14 CasADi appendage tests silently disabled for 17 days.** `0b35995` gave the eight one combined `_fin_with_rudder`, and the module fixture was `catalog.eight()`; every test picking `[s for s in appendages if not s.controllable][0]` raised IndexError or compared the wrong surface. The suite was reported green when it was not. | Running the full suite before starting the physics programme. The fixture is now a four (which still has one of each) and **asserts** it has both, so the next catalogue change fails with a sentence rather than an IndexError. |
| **v0.12 crashed before the first frame.** The minimap read the boat as `state` in the windowed loop, where it is `pose`. | A rower's crash log. Every bench and screenshot used `--shot`, which returns before that loop, so nothing here ever ran it. Guarded by `tests/test_no_undefined_names.py`, and by starting the downloaded build before each release. |
| **The pause menu called a function that did not exist** (`_blit`, since `4103ad3`), and **the setup menu used a settings module imported in a different function.** Both crashed on use. | pyflakes, run over the package after the crash above. |
| **A bucket rig rowed as a standard rig.** The setup menu returned the lineup and `main` read `boat`, `race` and `rate` from it and nothing else; a sub-menu also reset it to `None`. So no editor lineup ever raced. | Reported from the boat. |
| **Two tests pinned bugs in place.** One asserted the crashing line was present; one sliced from a later anchor to an earlier one, got an empty string, and passed every assertion about it. | Fixing the crash made the first fail. Both now assert the text that works, and that the slice is not empty. |
| **The released build was far behind the code** — v0.6 predated every physics change. | Releases are now cut per change set, checked on each platform's build, and started from the download. |
| **Diagonal line on the water.** The 110 m detailed water patch met the flat far plane at a hard edge; world-axis-aligned and following the boat, so it read as a moving diagonal. | Appeared in an overhead shot taken for something else, after three failed attempts to reproduce it from the seat — where the edge is past the horizon. |
| **The drawn crew ignored the timing.** Physics offset every seat by `phases[i] * period`; the renderer posed every rower and oar at the same `t`, so the crew was drawn in perfect time whatever the model did. | Reported as "it didn't really look like it". |
| **Seven models written, tested, and never called.** `CrewVariability`, both halves of `BladeContact`, `PhaseAuthority`, `StrokeTrim`, `CoupledCrew`, the Michell table, the wind model — each reachable behind a default of `None` or an unset attribute. | Asked what was actually wired, rather than what existed. |
| **The catalog rowed at ~470 W/rower**, above world class, sustainable for 68 s. Nothing noticed because nothing related the force *scale* to watts. | Tracking the reserve made it immediate. |
| **Wind would have double-counted.** The flat wave coefficient was absorbing the unmodelled still-air drag. Fixed by applying only the *excess* over still air. | Adding the full aero load put an eight 9% above its own calibration reference. |
| **The blade teleported in and out of the water.** Depth was taken straight from the drive flag, so it stepped 190 mm at the catch and again at the finish, inside one frame. | Asked to confirm oar heights through both phases. |
| **Blades were a quarter-turn out of phase**, flat through the drive and on edge through the recovery. | Looking at it. |
| **Windows CI job hung for 52 minutes**, silently, blocking the whole release. | The macOS job did the same work in 5. |
| **`macos-13` runner retired**, so the Mac job queued forever. | Asking why it was slower than the Linux build had been. |
| **Every port rower wore starboard arms.** The kinematics signature that shares one chain across identical rowers omitted the side, so a matched crew was one group led by the stroke seat. 0.15 m on eight arm masses; a yaw bias. | Tabulating the crew made the exact path's error measurable against a per-seat table. |
| **Startup spent 14 s in a pure-Python point-in-polygon.** | The first profile, sorted by internal time. |
| **The physics ran at 100 Hz for no reason.** 60 Hz is the same boat to 0.2 mm over 24 s. | Counting derivative evaluations per frame. |
