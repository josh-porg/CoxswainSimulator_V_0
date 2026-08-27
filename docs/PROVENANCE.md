# Provenance register

Every number in the model belongs to one of four classes. The class
determines what a test against it actually proves, and **what has to be
re-derived when something upstream changes**.

| class | meaning | what a test against it proves |
|---|---|---|
| **D** derived | follows from physics or geometry | that the derivation is implemented correctly |
| **L** literature | measured by someone else, published | that we match a published measurement |
| **M** measured | measured by us, from raw data | that we match our own data |
| **F** fitted | tuned so the model matches something | **nothing about the model** — it is a free parameter absorbing error |

**The trap this register exists to prevent.** A fitted parameter is
fitted *against a particular model*. Change the model and the fit is
stale, so a test pinned to it fails — and the failure says nothing about
whether the change was right. Twice now a correct change has been
rejected on exactly that basis (see §49–50 of `SOURCES.md`).

**Rule: when anything in a fitted parameter's "depends on" column
changes, the parameter must be re-fitted before any test using it is
believed.**

---

## F — Fitted (re-fit when dependencies change)

### `PEAK_OARLOCK_FORCE`, per boat class
*Fitted so each class settles at its known cruising speed.*

- **Depends on:** drive fraction, oar sweep arc, drive force-curve shape,
  hull drag model, blade efficiency, crew mass, added mass.
- **Re-fitted:** at the Kleshnev force curve (§38), the 2x speed
  calibration (§32), the sculling arc (§48), and the drive fraction
  (§50). Four times.
- **Consequence:** boat speed passing validation proves *nothing* on its
  own — it is fitted to pass. Only speed predicted from an unfitted
  propulsion model is evidence, which is what §47's 1-DOF test provided
  (4.10 m/s predicted against 3.82 measured, nothing fitted).

### `OarAngleSweep.flatness`
*Was fitted to put blade efficiency inside Kleshnev's 0.80–0.85.*

- **Depends on:** drive duration, sweep arc, blade model, boat speed.
- **Status: obsolete.** It existed to patch an efficiency of 0.747 that
  was too low *because the drive was too short*. With the corrected drive
  fraction the unfitted sweep gives 0.828, inside the band. The default
  is 0.0 and no fit is needed.
- **This parameter caused a wrong rejection.** §49 rejected the corrected
  drive fraction partly because blade efficiency left the band — using a
  flatness fitted against the *old* drive. Circular. Corrected in §50.

### `FIN_SPAN = 0.1471 m` — **assumed, one pixel above the measurement**

*The most load-bearing unmeasured number in the model. Read this before
quoting any absolute turn rate.*

The fin was measured off a scale drawing at **7 px** where the 17.3 m hull
spans 941 — 128.7 mm. The model uses **8 px**, 147.1 mm. That is not a
measurement and it is not a fit to data; it is a choice forced by a
physical criterion, and the reason is the sensitivity:

| px | depth | Nv | C (1e6) | behaviour |
|---|---|---|---|---|
| 6 | 110.3 mm | −161 | −0.90 | unstable |
| **7 (measured)** | **128.7 mm** | **−58** | **−0.06** | **neutral** |
| 8 (used) | 147.1 mm | +60 | +0.90 | stable |
| 9 | 165.5 mm | +193 | +1.99 | stable |

**The choice is narrower than it first looked, and this entry has
overstated it twice.** What the runs actually show, with `munk_factor` at
0.15:

| px | depth | C (1e6) | turn deg/s | bistable? |
|---|---|---|---|---|
| **7 (measured)** | 128.7 mm | −0.06 | **1.598** | no |
| 8 (used) | 147.1 mm | +0.90 | 1.531 | no |
| 9 | 165.5 mm | +1.99 | 1.456 | no |

- At 7 px the boat is **marginally unstable by the linear criterion** and
  wants continuous correction.
- It is **not bistable** there. The hull's quadratic cross-flow damping
  bounds the divergence. The bistability documented in SOURCES §60
  belonged to `munk_factor = 0.35` and was cured by lowering it — **not by
  the fin**. An earlier version of this entry attributed it to fin depth.
- Deeper is monotonically more stable and monotonically **worse at
  turning**. The measured 7 px gives the best steering on the table.

So 8 px rests on one argument: a shell with its skeg tracks, and losing
the skeg is a catastrophe *because* the skeg normally provides a margin.
Neutral stability sits badly with both. That is worth something; it is not
worth much, and it does not survive a measurement disagreeing with it.

**Depends on:** nothing upstream — it is geometry. **Everything downstream
depends on it**: directional stability, rudder authority, whether a
trajectory through the Eliot bend is feasible at all, and hence any
optimised racing line.

**Sensitivity is one-sided and steep.** Deeper is stable and progressively
less manoeuvrable (turn rate falls from 3.80 to 1.44 deg/s between 8 and
12 px); shallower is unstable. Note *span* is the sensitive dimension, not
chord — lengthening chord drops the aspect ratio faster than it adds area.

**To retire this entry:** measure the real boat — fin depth below the
hull, fore-and-aft length, and how far the rudder hangs below the fin.
One measurement settles three separate failures.

### `munk_factor = 0.15` — **no longer fitted** (was 0.35)

*Bounded by a physical stability criterion. Moved out of the F class.*

- **Depends on:** added-mass matrix, cross-flow drag, skeg and rudder
  geometry and lift model.
- **Lives in** `hydro/addedmass.DEFAULT_MUNK_FACTOR`, referenced by both
  `sim/simulator.py` and `river/sixdof.py`, which have silently diverged
  before.

**Why it changed.** The old 0.35 was calibrated against a coxswain's
reported turn rates — but the simulated turn rate it was matched to
*silently included the sweep rig's own ~1 deg/s yaw bias* (SOURCES §60),
so the fit was made against a contaminated quantity and came out roughly
twice too high.

**What that cost.** At 0.35 the Munk moment ran about **1.8× the combined
weathervane moment of skeg and rudder**. The weathervane derivative `Nv`
measured **−464 N·m/(m/s)** where `strokemodel.HydroCoefficients` has
always documented that it must be positive, the straight-line criterion
`C = Yv·Nr − Nv·(Yr − mU)` went negative, and the boat had **no
straight-line equilibrium**. Yaw settled into one of two attractors near
±0.55 deg/s depending only on where the transient started, with a jump
discontinuity between them sitting exactly where a coxswain holds the boat
straight. Seeding the yaw rate from −3, 0 and +3 deg/s gave a spread of
1.16 deg/s at 0.35 and 0.011 at 0.15.

**The bounds are now physical, not fitted:**

| quantity | turns positive below |
|---|---|
| `Nv` (weathervane) | ≈ 0.196 |
| `C` (straight-line criterion) | ≈ 0.227 |

0.15 sits clear of both. The change costs **~3% of real rudder
authority** (1.238 → 1.201 deg/s at 25°) because the old value was never
buying steering — it was buying wander, which the old validation checks
miscounted as steering.

**The skeg-loss anchor does not pin this value.** Drift over 25 s with the
appendages removed is non-monotonic in the factor — 22.2° at 0.15, 15.8°
at 0.20, 22.8° at 0.25 — because a bare hull is *itself* directionally
unstable, so a single run is path-dependent. It is satisfied across
0.10–0.50 and cannot discriminate. Treat it as a sanity check, not a fit
target.

- **Remaining weakness, now isolated:** with the Munk moment set honestly
  the model gives **1.66 deg/s total** at full rudder against a reported
  ~3. That deficit belongs to the **rudder**, not the hull. Closing it
  needs roughly 4× the blade area (9×12 cm → about 18×24 cm), so either
  the modelled rudder is too small or `flap_effectiveness = 1.0` is wrong
  for a rudder hinged on the skeg rather than all-moving. Needs the real
  boat's dimensions.

### `2x PEAK_OARLOCK_FORCE = 485 N`
*Fitted to the club session's own measured 3.82 m/s.*

- Makes the fluctuation comparison like-for-like in mean speed, which
  matters because IVV is a ratio. Not independent evidence of propulsion.

---

## L — Literature (with the assumptions each carries)

| quantity | value | source | assumptions |
|---|---|---|---|
| segment masses | head 6.9%, trunk 43.5%, thigh 14.2%, shank+foot 5.7% | de Leva (1996) | standard anthropometry; scaling by stature and mass |
| joint angles at 4 keyframes | shank/knee/hip/trunk | Caplan & Gardner (2010) Table II | **4 instants per stroke only**; SD 5.5–11°; timing between joints absent |
| trunk swing | 50.8° | Kleshnev | elite crews |
| segment power shares | legs 43 / trunk 33 / arms 24 | Kleshnev | shares of *handle* work |
| drive force peak | 40% of drive; 74% of peak by 60% | Kleshnev | on-water |
| blade efficiency | 0.80–0.85 | Kleshnev | on-water, good crews |
| **drive fraction** | 0.394 / 0.430 / 0.468 at 22 / 26 / 32 spm | **Telfer et al. (2023), n=11** | **ergometer**, rates 22–32; below 22 is extrapolation |
| sweep arc | sweep 87–90°, sculling 104–110° | rigging literature | catch 53–58 / 60–66 |
| added mass coefficients | Lamb `k₁` table | Lamb (1932) art. 71 | potential flow, prolate spheroid |
| blade force | slip-based | Cabrera & Ruina | quasi-steady |
| 6-DOF equations | mass matrix eq. 14 | Formaggia et al. (2009) | rigid hull, moving-mass crew |
| hull minimum masses | 96 / 27 / 15 kg | World Rowing rules | class minima, not actual |
| velocity fluctuation | ~20% about the mean | Day et al. | racing boats |
| within-stroke attitude | pitch <1°, roll and yaw <5° | rowing literature | racing shells |

---

## M — Measured (our own, from the figshare CC0 dataset)

All from the **club 2x session, 20 April 2018** — one crew, one outing,
so speed and fluctuation are like-for-like.

| quantity | value | how |
|---|---|---|
| mean speed | **3.82 m/s** (7 logs, 3.39–4.19) | DGPS baseline logs |
| IVV | **37.3%** | DGPS velocity |
| IVV, cross-check | 35.3 / 37.8 / 35.8 / 35.7% | accelerometer integrated, 4 windows |
| hull surge accel ptp | **8.88 m/s²** | boat IMU, axis chosen by stroke-band power |
| catch-aligned accel profile | 162 cycles | per-stroke catch detection |
| stroke rate detection | MAE 0.21 spm | against logged rates |

**Not usable:** the pelvis IMU. The device frame rotates with the rower,
and the attitude correction does not reproduce the validated hull figure
(1.28 m/s² against 8.88). Anything derived from it is withdrawn (§38).

---

## D — Derived (physics; a test proves the implementation)

- added mass by strip theory from hull offsets — Lamb / Newman / Korotkin
- distributed cross-flow drag — Hoerner; reduces exactly to the lumped
  form at zero yaw rate
- added-mass Coriolis and the Munk moment — Fossen §6.3
- crew–hull momentum coupling — verified to 0.24 m/s², r = 0.9993
- oar lever transmission `F_blade = (r_h / L) F_pin` — massless oar;
  the handle force cancels identically (§42)
- hull hydrostatics from the mesh

---

## Validation status by component

| component | validated against | class of evidence | status |
|---|---|---|---|
| hull 6-DOF equations | Formaggia et al. | reproduces published model | **sound** |
| added mass | Lamb's table (0.500 / 0.209 / 0.059) | L | **sound** |
| propulsion | 4.10 m/s predicted vs 3.82 measured, unfitted | M | **7%, sound** |
| steering | coxswain turn rates, skeg-loss event | L + anecdote | **plausible, munk_factor fitted** |
| attitude | roll 1.57° / pitch 0.58° vs <5° / <1° | L | **sound** |
| boat speed | fitted per class | **F** | **proves nothing** |
| segment masses | de Leva | L | **sound** |
| trunk swing | 54.5° vs Kleshnev 50.8° | L | **sound** |
| seat travel | 0.597 m vs 0.60–0.70 | L | **sound** |
| drive fraction | Telfer (erg) | L | **adopted, erg caveat** |
| **crew CoM travel** | 0.744 m vs ~0.65 implied | M | **FAILS** |
| **intracycle velocity variation** | 51.9% vs 37.3% | M | **FAILS** |
| arm posture | nothing | — | **never validated**; inert when rigged |
| crew acceleration *phase* | implied from hull inversion | M | **FAILS** — opposite sign mid-drive |
| stroke-to-stroke timing scatter | Cuijpers | L | sound |
| blade efficiency | 0.828 vs 0.80–0.85, unfitted | L | **sound as of §50** |

---

## Open: no ground truth for rower motion

The unvalidated components are all **rower** components, and the reason
is that there is no measured rower motion to check against — only four
keyframes with 5–11° standard deviations and no inter-joint timing.

Consequences worth stating:

1. We cannot distinguish a bug from a bad modelling assumption from a
   correct model, because there is no target.
2. Single fixes keep being found that move the fluctuation 2–9 points and
   then get rejected as insufficient. **They may be additive.** Nothing
   has tested them stacked, and the register above shows they touch
   different quantities (timing, geometry, force shape), so there is no
   reason to expect them to be alternatives.
3. Musculoskeletal models of the rowing stroke are the obvious source of
   a target and have not been investigated.

The first two are actionable now. The third needs a literature pass.

---

## Unvalidated inputs, ranked by lever

Added after the Telfer mocap arrived (SOURCES §50–51). The outstanding
failure is crew centre-of-mass travel: **0.760 m where the measured hull
motion implies about 0.65**. This is where that travel comes from:

| group | contribution | share |
|---|---|---|
| **trunk + head** | **0.469 m** | **62%** |
| legs | 0.162 m | 21% |
| arms | 0.130 m | 17% |

Per segment, the movers are upper trunk (1.054 m of travel), head
(1.286 m) and mid trunk (0.848 m) — all of them consequences of treating
the trunk as one rigid link.

### Ranked

1. **Trunk segmentation — 62% of the lever, unvalidated.**
   One rigid hip-to-shoulder link swinging 54.8°. Telfer measure the
   pelvis and spine separately: 32.2° lower back plus 32.4° upper back at
   26 spm. Closing the gap needs the trunk+head contribution down about
   23%, which is the right size. **Blocked on** the reference pose and
   sign convention for `LowerBack`/`UpperBack` — a curling spine moves
   the trunk CoM less *or* more depending on that phasing, and the test
   in §51a came out the wrong way with a guessed distribution.

2. **Crew anthropometry — worth about 5 points, unknown.**
   The model uses 82 kg / 1.84 m. The club 2x crew's actual mass and
   stature are not in the dataset. Sensitivity: 82/1.84 → 51.9% IVV,
   70/1.72 → 48.1%, 65/1.68 → 46.6%. Note **crew mass fraction is
   insensitive** (0.859 → 0.828) because the hull scales with the crew;
   the effect is entirely through stature setting limb lengths.

3. **Arm posture — 17% of the lever, never validated and currently
   inert.** `DEFAULT_ARM_POSTURE` is bypassed whenever `hand_targets` is
   set, which is every rigged boat. Releasing it (§45) gave a hand path
   17% too long. It becomes live the moment the causal chain is inverted.

4. **Link lengths.** Shank 0.459, thigh 0.446, trunk stack 0.638 m,
   scaled from stature by de Leva ratios. Never checked against the
   subjects whose joint angles drive the model.

5. **Trunk pivot location.** The trunk is taken to rotate about the hip
   joint. Real trunk motion is pelvic rotation plus spinal flexion about
   a distributed axis.

### Validated since this register was written

- knee range: Caplan & Gardner 130.2° vs Telfer 129.9° — **two
  independent datasets, 0.2% apart**
- joint waveform *shapes*: knee +0.974, trunk +0.979, ankle +0.972
  against Vicon
- drive fraction: Telfer, confirmed independently by their catch-at-57%
- blade efficiency: 0.828 in Kleshnev's band with **no fitted parameter**

---

## Resolution (SOURCES §57)

**The catch-aligned hull *acceleration* profile (`hull_profile.npz`) is
NOT USABLE.** The pipeline is clean, but model-free inversion shows the
profile implies +350 N of external force with the blades out of the
water, and the DGPS velocity waveform contradicts its phase structure.
Conclusions formerly drawn from it (§43 phase inversion, §55 wrong
distribution, §56 missing unsteady term) are withdrawn.

**The DGPS velocity waveform (`hull_vel_profile.npz`, 37 cycles) is the
shape target.** Textbook curve — minimum after the catch, rising through
the drive, maximum at the finish. The model matches it at **r = 0.92**
(0.95 with club-plausible inputs).

**Amplitude is conditional on one input.** Per-stroke DGPS IVV of 37.3%
is noise-inflated; the true value is ≈33.5%. The model with
static-erg-validated crew kinematics predicts ≈51% — the erg-scale
answer, because the crew input is erg-scale. The measured hull implies
this club crew's relative CoM velocity swing was 0.65–0.75 of the erg
value (gentler travel *and* smoother waveform), consistent with the
documented static-erg-versus-water difference and light paddling. That
swing is now the **only unmeasured quantity in the fluctuation chain**;
the synchronised crew+hull measurement in `DATA_REQUESTS.md` pins it.

Validation table updates: crew CoM travel **PASSES** (0.760 vs 0.727
measured by Vicon markers; the old "~0.65 implied" target was an
inference through the corrupted profile and is withdrawn). Velocity
waveform shape **PASSES** (r = 0.92). IVV becomes a conditional check —
the model must land in the erg band (45–56%) when given erg inputs.
`scripts/validate.py`: **18 of 18 checks pass.**

---

## River furniture: bridges, arches and lanes (SOURCES §59)

Added after the arch model. The classification matters here for the same
reason it does in the physics: two of these look like the same kind of
number and are not.

### M — Measured

| item | value | source |
|---|---|---|
| `PIER_THICKNESS` | 3.32 m | five OSM `bridge:support=pier` polygons on the Grand Junction trestle, 3.14–3.45 m |
| `MEASURED_PIERS["Grand Junction RR"]` | 5 pier centres | same survey |
| `OSM_BRIDGE_DECKS` | deck endpoints | OpenStreetMap, Overpass API |

### L — Literature / official record

| item | source | assumption it carries |
|---|---|---|
| span counts, span lengths, structure lengths | FHWA NBI 2024 MA | item 48 is centre-to-centre, not clear opening |
| `permitted_width` (NBI item 40) | FHWA NBI 2024 MA | **a permitted channel width, not a measured opening** — it exceeds Anderson's own longest span, which is impossible physically. Kept for reference; never used as geometry |
| `HOCR_ARCH_RULE`, `WRONG_ARCH_PENALTY` | HOCR competitor rules | rules as published for the current running |
| Weeks: three arches | Simpson Gumpertz & Heger | — |
| `EIGHT_ROWED_WIDTH` = 6.82 m | derived from the catalog rig | oarlock 0.85 m, oar 3.70 m, inboard 1.14 m |

### D — Derived

| item | how | assumption |
|---|---|---|
| pier positions, all bridges but the trestle | centre span laid symmetrically about the middle of the wet opening | **the bridge is symmetric about the channel.** Supported by NBI: Anderson's side spans are 23.6 m against a 23.5 m centre |
| clear arch widths | span minus one pier thickness, then intersected with water deep enough to row | pier thickness measured on a different bridge |
| waterway extent | wet crossing, clipped to the structure length | the raster reports water at abutments; River Street needed the clip |

**Cross-check that constrains the whole construction:** Eliot's derived
centre opening is 30.2 m against NBI's independently recorded 30.5 m
navigation clearance. Different columns, 1% apart. `test_bridges.py`
pins it.

### The one number known only by report

| item | value | source |
|---|---|---|
| travel lane through the Cambridge Boat Club bend | one boat wide | **local — a competitor who has raced it.** Not published anywhere |

This is not a weaker version of a measurement, it is a different kind of
fact, and it is the only thing that reveals the constraint. The channel
there is 50 m wide, which the bathymetry reports accurately and which is
still misleading, because the bend is double-buoyed and most of that width
is out of bounds. `traffic.TrafficLane.source` records `rules` or `local`
on every entry so this never gets mistaken for a survey.

### Known gaps

1. **Buoy lines are not modelled.** The course boundary in the model is the
   bathymetric navigable edge, which is *wider* than the legal course
   wherever buoys narrow it. An optimiser will therefore find lines that
   are out of bounds. Positions are not published and are re-laid annually.
2. **The travel lane has a width at one place only.** The Boston-side lane
   from the finish down to Weeks takes width everywhere in that reach, by
   an unknown amount. `usable_width` subtracts nothing where nothing is
   known rather than guessing.
3. **BU Bridge arch permission is uncertain.** The rules bar the right arch
   at the trestle, Anderson and Eliot, and allow it at "the three remaining
   bridges" — which leaves four once BU is counted. BU is modelled from the
   Charles River Rowing Committee pattern instead, which puts upstream
   traffic through the second arch from the Cambridge shore.

---

## OPEN CONFLICT: the stroke timing is ergometer timing (found 2026-08-23)

`StrokeTiming.drive_fraction = 0.63067 - 5.20991 / rate` is fitted to
**Telfer et al. (2023), "The effect of foot-stretcher position and stroke
rate on *ergometer* rowing kinematics"**, n = 11 collegiate rowers.

The model simulates a boat. There is on-water data, and it disagrees
systematically. Hill & Fahrig (2009), *Scand J Med Sci Sports* -- elite
coxless **pairs on water**, stepped rates -- implies:

| rate | on-water | model (erg) | error | drive:recovery on water | model |
|---|---|---|---|---|---|
| 20.6 | 0.296 | 0.378 | **+27.6%** | 1:2.38 | 1:1.65 |
| 24.2 | 0.327 | 0.415 | **+27.1%** | 1:2.06 | 1:1.41 |
| 27.7 | 0.360 | 0.443 | **+23.1%** | 1:1.78 | 1:1.26 |
| 31.5 | 0.395 | 0.465 | **+17.9%** | 1:1.53 | 1:1.15 |

A near-constant +0.08 offset in drive fraction. On water at 31.5 spm the
fraction equals the erg value at 22 spm -- a shift of roughly ten rates.
This is the expected direction: on water the boat runs during the
recovery, so the recovery takes more of the cycle than it does on a
machine whose flywheel is decaying.

**This is squarely in the operating range.** A Masters eight races the
Head of the Charles at 28-30 spm, where the drive is about 25% too long.

The same reciprocal form fits the on-water data well:
`0.57391 - 5.82413 / rate`, RMS residual 0.0054.

### Why it has not simply been swapped

The change would not be free, and the reasons the erg fit was adopted are
real and are recorded above:

* Force-weighted blade efficiency comes out **0.828** under the erg
  fraction, inside Kleshnev's on-water band of 0.80-0.85. Under the
  shorter on-water drive it falls to **0.747**, below the band -- which is
  exactly what `OarAngleSweep.flatness` used to be fitted to patch.
* The erg fit extrapolates to a 1:1.00 drive:recovery ratio at 40 spm,
  the race-pace figure coaches quote. The on-water fit extrapolates to
  1:1.33, though that is well outside the data it was fitted to.

So two on-water sources disagree: Hill & Fahrig's measured drive times
say the drive is short, and Kleshnev's measured blade efficiency says a
short drive cannot be right. **This is a stacked-solution case.** Fixing
the drive fraction alone fails the blade-efficiency check; the honest
reading is that the drive fraction and something in the blade model are
both wrong, and neither can be validated while the other is held fixed.

### What would settle it

1. A definitional check on Hill & Fahrig: is their "drive" catch-to-finish
   of handle motion, or force onset to offset? A 20% gap can hide in that.
2. On-water drive fractions for an **eight** rather than a pair.
3. Blade efficiency recomputed from a source that does not depend on the
   drive duration.

### What it already explains

The intracycle velocity work (SOURCES 57) concluded the model gives "the
erg-scale answer for an erg-scale input", with the crew's relative
velocity swing as the one unmeasured quantity. The drive fraction is a
**second erg-scale input pushing the same way**, and it was not counted at
the time.
