# Bugs, gaps and things being tracked

What is known to be wrong, known to be missing, or known to be a
placeholder. Kept separate from `SOURCES.md`, which records what was
built and why; this file records what has *not* been settled.

Anything fixed moves to the bottom with the evidence that fixed it, so
the file is also a record of what kind of thing goes wrong here.

---

## Open — correctness

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

### `mean_handle_power` ignores `power_scales`
Its docstring says it reports power "at the boat's current scale". It
does not — it integrates `oar_force` without the scale, so it always
returns the scale-1.0 figure (486 W for the catalogue four). Callers
that use it to *calibrate* a scale are fine, since power is linear in
scale; callers that use it to *check* one silently get the wrong answer.

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

### Surge swing is 10–31% too large
Model against Holt: ratios 1.31, 1.31, 1.10, 1.18. `scripts/unsteady.py`
squares this quantity, so the error is four times worse there.

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
| published race pace reached at **380 W per rower**: eight 5.33 m/s (band 5.0–5.6), four 4.76 (band 4.5–5.1) — where the prescribed model needed 720 W and 795 W and overshot anyway | same | same |
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

### The offline physics programme — phase 4.1, built; gate failed

| what | where | pinned by |
|---|---|---|
| the baseline, measured before building: the clock crew's hands up to 0.19 m off the dynamic handle on the eight, 0.74 m on the single — too far for any arm solve | probe, recorded in PHYSICS_PROGRAMME | — |
| the body follows the oar through the drive: stroke time from the angle, pose from the stroke table, velocity and acceleration by the chain rule; hands on the handle to under 2 mm on both boats | `coxswain/crew/follow.py` | `tests/test_crew_follows.py` |
| the oar's inertia built from the velocities the hull is given — one kinetic energy, to 1e-9; power books close to 2% outside the rate floor | same | same |
| recovery retimed from the dynamic finish to the next catch, starting from the finish pose and arriving at the catch pose | same | same |
| `crew="follows"` on the dynamic oar, a study: one elapsed-drive state per seat, the oar computed before the hull, the clock crew's arithmetic and order untouched | `coxswain/sim/dynamic_oar.py` | same, plus the dynamic-oar suites (62 fast, 8 slow) |
| the momentum books: the clock crew closes to 0.005 N·s a stroke; the following crew does not — open item above | same | same (the latter strict xfail) |

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

## Fixed

| what it was | how it was found |
|---|---|
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
