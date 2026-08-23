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

### `munk_factor = 0.35`
*Calibrated against a coxswain's reported turn rates and a witnessed
skeg failure.*

- **Depends on:** added-mass matrix, cross-flow drag, skeg and rudder
  geometry and lift model.
- **Known weakness:** turn rate scales 1.7× for 5× rudder where the
  report implies ~3×, so this is absorbing an error that belongs to the
  rudder. Fixing the rudder changes this value.

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
