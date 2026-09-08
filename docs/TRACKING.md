# Bugs, gaps and things being tracked

What is known to be wrong, known to be missing, or known to be a
placeholder. Kept separate from `SOURCES.md`, which records what was
built and why; this file records what has *not* been settled.

Anything fixed moves to the bottom with the evidence that fixed it, so
the file is also a record of what kind of thing goes wrong here.

---

## Open — correctness

### The released build is far behind the code
**Impact: high.** `v0.6` predates every physics change since: Michell
wave drag, wind, crew skill, fatigue, balance authority, blade contact,
crew timing, and the water-seam fix. Anyone who downloads the link is
running none of it.

*Next:* cut a release once the physics settles.

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

### The two speed models disagree by a factor of 2.3 in efficiency
**Impact: high — it decides every published target.**

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

## Done — scene and audio

Not bugs; asked-for work, recorded here because the fix log had no
trace of it and so it kept being asked for again. Each line says where
it lives and what stops it silently going away.

| what | where | pinned by |
|---|---|---|
| **Shadows cast by everything, onto everything.** `static` and `shadow_casters` are built in the same loop over every mesh part, so buildings and trees shadow each other and themselves — not just the ground. 4096² map over 3965 m, 1.0 m a texel, baked once because the sun does not cross the sky during a 5 km race. | `scripts/fpv.py:2599` | `test_everything_in_the_world_casts_a_shadow` |
| **Shadows received on the surface's own normal.** The world shader passes the fragment normal, because the lookup is pushed a texel along it to clear its own cell. A fixed up-vector instead would offset every wall as though it were flat ground, and the shadow would creep up its own face. | `scripts/fpv.py:376` | `test_the_world_receives_shadows_on_its_own_normals` |
| **Bow disturbance and the hull breaking the water.** `waterline_foam` paints white where the hull meets the surface, tied to the hull frame and to speed, so a stopped boat has none. | `scripts/fpv.py:954`, applied `1045` | `test_the_bow_foam_is_tied_to_the_hull_and_the_speed` |
| **Catch splash at blade entry.** A small pooled particle system, deliberately not dramatic. Off at standard graphics, on at high; `--no-particles` forces it off for A/B. | `scripts/fpv.py:1384` | `test_the_splash_pool_exists_and_empties` |
| **Ambient world audio** — wind gusting on a slow envelope and brightening with wind speed, water working at the bank, the odd gull. Held at about a fifth of the stroke's level so it never competes with the boat, and following the wind setting live. | `coxswain/viz/ambient.py` | — |
| **Low-frequency variation in the grey sky.** Two octaves of simplex over the exponential gradient, amplitude 0.045 and gated on overcast, so a clear day takes almost none and the gradient still dominates. | `scripts/fpv.py:287` | `test_the_sky_noise_stays_under_the_gradient` (amplitude ≤ 0.08) |

The sky-noise bound is the load-bearing one: the whole request was that
the noise not dwarf the exponential map, and an amplitude nudged from
0.045 to 0.45 is a one-character change that no other test would catch.

---

## Fixed

| what it was | how it was found |
|---|---|
| **Diagonal line on the water.** The 110 m detailed water patch met the flat far plane at a hard edge; world-axis-aligned and following the boat, so it read as a moving diagonal. | Appeared in an overhead shot taken for something else, after three failed attempts to reproduce it from the seat — where the edge is past the horizon. |
| **The drawn crew ignored the timing.** Physics offset every seat by `phases[i] * period`; the renderer posed every rower and oar at the same `t`, so the crew was drawn in perfect time whatever the model did. | Reported as "it didn't really look like it". |
| **Seven models written, tested, and never called.** `CrewVariability`, both halves of `BladeContact`, `PhaseAuthority`, `StrokeTrim`, `CoupledCrew`, the Michell table, the wind model — each reachable behind a default of `None` or an unset attribute. | Asked what was actually wired, rather than what existed. |
| **The catalog rowed at ~470 W/rower**, above world class, sustainable for 68 s. Nothing noticed because nothing related the force *scale* to watts. | Tracking the reserve made it immediate. |
| **Wind would have double-counted.** The flat wave coefficient was absorbing the unmodelled still-air drag. Fixed by applying only the *excess* over still air. | Adding the full aero load put an eight 9% above its own calibration reference. |
| **The blade teleported in and out of the water.** Depth was taken straight from the drive flag, so it stepped 190 mm at the catch and again at the finish, inside one frame. | Asked to confirm oar heights through both phases. |
| **Blades were a quarter-turn out of phase**, flat through the drive and on edge through the recovery. | Looking at it. |
| **Windows CI job hung for 52 minutes**, silently, blocking the whole release. | The macOS job did the same work in 5. |
| **`macos-13` runner retired**, so the Mac job queued forever. | Asking why it was slower than the Linux build had been. |
