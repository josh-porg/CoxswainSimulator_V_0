# Calibration register: every value fitted or calibrated to data, and what it was fitted under

*Created 2026-10-27 (user: "check what parts of the model are calibrated/fitted which might be why the boat speed
doesn't match. Did we fit some value to our previous model that no longer holds?"). The ledger's status column is the
term-by-term record of the physics; it filed the one physics fit in the research model as "sourced". This register
exists so that a fit is never mistaken for a source, and so that when the model changes, every fit made under the old
model is re-checked.*

**The rule.** A value fitted or calibrated to data is entered here when it is made, with: what data it was fitted to,
the **model state at the fit** (date, profile, the pieces it was fitted alongside), what it can absorb (the physics
the fit may be standing in for), what invalidates it (**re-check triggers**), and every re-check since. When a change
lands that matches a trigger, the re-check is part of that change. A validation against the same data the value was
fitted to is a consistency check, not a validation, and is never reported as one. (Memory:
`calibration-register`; the project rule "no single-athlete fits" applies.)

## Physics fits in the research model

| value | where | fitted to | model state at the fit | can absorb | re-check triggers | re-checks |
|---|---|---|---|---|---|---|
| **Scull C2 = 140.88** (2.4× [CR06]'s computed 58.7) — **retired from research and learned 2026-10-31** (§202); a study profile may still set it | `PhysicsProfile.scull_c2` (no registered profile sets it since 2026-10-31) | [CR06]'s own singles data (their point-mass model, their fit); confirmed here on her stroke 2026-09-14/18 (her traces imply 2.3–3.4×; her oar angle needs 2.4×) | 2026-09-18: slip law read at the blade centre, sweep catch, clock crew on ergometer kinematics, no added mass, no immersion; before de Leva legs (09-26), Kato roll (09-29), the fin law (09-29), the hands-on-handle crew (09-29) and [K05]'s on-water drive law (09-30) | everything the slip law lacks: the angle dependence of the load (§184: wrong shape on a measured load), the catch build-up, the centre of pressure (C2 88 at l = 1.9 m, 53 at 2.02 m), and the crew model of the day | any change to the blade law, its read point, the catch rule, the crew driver or kinematics, or the hull | 2026-10-27 rung 1: [CR06] −0.4% at her power (**the fitted athlete: a consistency check, not a validation**), [BR24] +2.8%, IVV +6.3 / +11.5 points. Guarded exclusive of strips, added mass and deep depth because it was fitted without them. Filed as "sourced" in the ledger until 2026-10-27; corrected to "fitted" |
| **Cross-flow blade face-on C_Dc = 2.2** (the research profile's blade since 2026-10-31) | `physics.PROFILES["research"].blade_coefficients = "crossflow-normal@2.2"` | **tank datasets, not the test athletes**: the least-error face-on value per measured load ([GTW21] 2.2 at Fr 0.87, [LJH12] 2.3 at 0.49, [RLBQV19] 1.3 at 1.00; §191), so a profile over the tank data; 2.2 is the value for Froude below ~0.9, chosen before either athlete test | 2026-10-27: the cross-flow law in strips, open loop on the measured motions | the free-surface regime (ventilation at full scale), blade shape, unsteady effects | a sourced Froude (or advance-number) function for C_Dc; a blade-shape or immersion change; Barré & Kobus's parameters | 2026-10-31: closed loop, rung 2 on both athletes (§202): IVV +2.1 / +3.3 points, speed −1.2 / +2.6 % at their power (validations: neither athlete entered the value) |
| Pull shape τ(θ) exponents 1.4852 / 2.2278 | `sim/oarloop.py` torque_shape (clock crew) | two Kleshnev points (peak at 40%, 74% of peak by 60%) | 2026-09 (phase 2) | the crew's force time course | a crew driver change (the hands-on-handle rungs do not use it) | failing at the ends (Holt slips, [LE26]); not used by rung 1 or 2 |
| Matched torque (sweep catch) | `dynamic_oar` matched-torque cache | the stated wattage, per boat configuration, at run time | recomputed per configuration | the power level only | **every field that changes the stroke must be in `_match_key`** (memory: new physics switch, cache keys) | automatic; the cross-flow blade's face-on value is carried in its coefficients name (2026-10-27) |
| Telfer drive fraction 0.63067 − 5.20991/r | `crew/stroke.py` StrokeTiming | ergometer data | ergometer | the rhythm | on-water timing in use | failing on the water; research runs use [K05]'s OnWaterTiming |
| Munk factor 0.50 | research profile `munk_factor` | literature value; **refitted** to the coxswain's ~3 deg/s full-helm authority | 2026-09-30, fin law [WF58], reflection 1–2 | the hull's yaw coupling | a fin, rudder or hull-added-mass change | 2026-09-30: 0.49–0.50 at every reflection, the literature value stands |
| Arm draw clock: held flexion 10.4°, onset 0.465 of the drive, return end 0.751 of the recovery (minimum-jerk shape) | `coxswain/crew/arm_draw.py` (`HOLD_FLEXION`, `DRAW_ONSET`, `RETURN_END`); used by the research body's "straight-arms" posture only | **population data, not the test athletes**: Pitto et al. 2024 Fig. 2 mean elbow of 12 male scullers (1.3° rms drive, 3.1° recovery); the finish flexion is solved per boat, never taken from the fit | 2026-11-01, kinematic arm on the keyframed or own-timing body (§209) | the elbow's timing; the finish flexion absorbs any shoulder-path error | a muscle-driven draw replacing the kinematic clock; a sweep or women's elbow source; a shoulder-girdle DOF (it changes the reach the draw must close) | 2026-11-01: finish elbow on own timing 104.6° / 116.3°, inside the source's band (a check, not a fit). 2026-11-02 (§210): on the direct own-channel reconstruction, 98° / 101° |
| **Chain stretch at peak handle force s₀ = 0.124 m** (option, off by default) | `coxswain/crew/arm_draw.PEAK_CHAIN_STRETCH`; `handle_rung1.CHAIN_STRETCH` (None) | **the test athletes**: mean of [BR24] 0.110 m and [CR06] 0.138 m, from a spring fit of the grip-to-shoulder excess against handle force while the arms are held (§210) | 2026-11-02, own-channel shoulders, the arm clock, the boat's force shape | the shoulder complex and upper-back compliance; any trunk-sensor offset from the shoulder; **possibly the oar's inboard bending** (§217: a uniform shaft at [L14]'s EI bends ~5 cm at 500 N) | a published shoulder/upper-back compliance under traction; a third athlete; trunk sensor at the acromion; **a measured inboard stiffness (re-check due, 2026-11-03)** | 2026-11-02: cross-athlete (each predicts the other's whole drive: 2.5° / 1.5° rms) — out of sample across the two, **not** a validation on new data; it must not be switched on and then "validated" on [BR24] / [CR06] |

## Values chosen, not fitted (they set levels; listed so they are not mistaken for either)

From the ledger's "chosen" and "provisional" rows: hull cross-flow C 0.60 (sway, yaw) and 0.80 (heave, pitch); form
factor 1 + k = 1.0; Oswald e 0.8; fin reflection k_r 2.0 (swept 1–2); bluff-body constant m 2.5; rower power 380 W
for race pace; fin depth 0.300 m (from a photograph); blade areas 0.110 / 0.083 m² (provisional); the depth-model blend
bands. A chosen value becomes a fit the moment it is adjusted to make an output match data; it then moves to the table
above.

## Study-only calibrations (enter no profile)

| value | where | fitted to | note |
|---|---|---|---|
| [BR24] channel calibrations (trunk at shoulder height, momentum fit 0.583 m) | `research/biorow/` | his own data | diagnostics; enter nothing |
| [CR06] trunk channel scale K = 0.398/back | `research/cr06/` | her data | rung inputs for her stroke only |
| Trunk controller gains | `research/cr06/coupled_control_cr06.py` | designed on the coupled plant (not fitted to data) | design choice, §182 |

## Race-day product calibrations (products, not physics)

| value | where | fitted to | note |
|---|---|---|---|
| Day factor on the wind | `research/raceday/race_review_opt.py`, `backtest_dayfactor.py` | each race day's field of results | a product correction; the race-card wind model overstates whole-race wind 2–3× (PHYSICS_PROGRAMME) |
| Turning cost k = 0.0071 / 0.0099 | race card `RUDDER_K` | Tail of the Lake 2026 GPS (both squad boats) | measured; a product correction, not simulator physics |

## Re-check log

- 2026-10-27: register created; scull C2's ledger status corrected from "sourced" to "fitted"; its rung-1 agreement on
  [CR06] reclassified as a consistency check. Barré & Kobus (2010)'s abstract ties the maximum drag coefficient to an
  unsteadiness (advance) parameter: the four blade datasets in hand all sit at advance ratio 0.72–0.82 at mid-stroke,
  so it does not separate them (SOURCES §193).
- 2026-10-31: the research and learned profiles' blade changed from the slip law to the cross-flow normal law (strips,
  C_Dc 2.2; SOURCES §199–202, user decision). The fitted scull C2 is retired from both (it was the only physics fit in
  them); the face-on C_Dc enters the physics table above, with its tank-data provenance. Re-check triggers met: the
  matched torque (the blade law, coefficients and span are in `_match_key`), and the validation scorecard re-run
  (PHYSICS_PROGRAMME).
- 2026-11-02: the research and learned profiles' body changed: the trunk link on Pitto et al. 2024's on-water timing
  (`PhysicsProfile.body_clocks`, SOURCES §211, §213; user decision). Re-check triggers met: the matched torque (the
  clock is in `kinematics_signature`, so in `_match_key`), the validation scorecard (re-run, PHYSICS_PROGRAMME), and the
  arm-draw clock's finish-flexion check (the shoulders move later; it is solved per boat, not fitted). No fitted value
  in the physics table depends on the trunk's timing.
- 2026-11-03: oar bending was made a research/learned default (user decision) and **withdrawn the same day** (SOURCES
  §217): the scull law was ~9× too stiff ("1.18°" was a difference between shafts, not a deflection; now [L14]'s
  measured shaft, EI 2099 N·m², as an option); with it rung 2 moves away from both athletes (IVV 53.4 / 60.7 %) and the
  sweep law lags 20–38° into the finish (Kleshnev efficiency 0.39). The profiles row the rigid oar as before. The
  scorecard's blade efficiency is now Kleshnev's energy measure on each boat type's band (S12): rigid eight 0.805, four
  0.777, both pass. **Triggered on a fitted value: the chain stretch s₀**
  (0.124 m, athlete-measured) excluded oar bending because the misread stiffness bent the inboard ~6 mm at 500 N; a
  uniform shaft at [L14]'s EI would bend it ~5 cm. The inboard's EI is unmeasured; until it is, s₀ may include the oar.
  The face-on C_Dc (tank data on rigid plates) is unaffected.
