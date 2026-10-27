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
| **Scull C2 = 140.88** (2.4× [CR06]'s computed 58.7) | `physics.PROFILES["research"].scull_c2` | [CR06]'s own singles data (their point-mass model, their fit); confirmed here on her stroke 2026-09-14/18 (her traces imply 2.3–3.4×; her oar angle needs 2.4×) | 2026-09-18: slip law read at the blade centre, sweep catch, clock crew on ergometer kinematics, no added mass, no immersion; before de Leva legs (09-26), Kato roll (09-29), the fin law (09-29), the hands-on-handle crew (09-29) and [K05]'s on-water drive law (09-30) | everything the slip law lacks: the angle dependence of the load (§184: wrong shape on a measured load), the catch build-up, the centre of pressure (C2 88 at l = 1.9 m, 53 at 2.02 m), and the crew model of the day | any change to the blade law, its read point, the catch rule, the crew driver or kinematics, or the hull | 2026-10-27 rung 1: [CR06] −0.4% at her power (**the fitted athlete: a consistency check, not a validation**), [BR24] +2.8%, IVV +6.3 / +11.5 points. Guarded exclusive of strips, added mass and deep depth because it was fitted without them. Filed as "sourced" in the ledger until 2026-10-27; corrected to "fitted" |
| Pull shape τ(θ) exponents 1.4852 / 2.2278 | `sim/oarloop.py` torque_shape (clock crew) | two Kleshnev points (peak at 40%, 74% of peak by 60%) | 2026-09 (phase 2) | the crew's force time course | a crew driver change (the hands-on-handle rungs do not use it) | failing at the ends (Holt slips, [LE26]); not used by rung 1 or 2 |
| Matched torque (sweep catch) | `dynamic_oar` matched-torque cache | the stated wattage, per boat configuration, at run time | recomputed per configuration | the power level only | **every field that changes the stroke must be in `_match_key`** (memory: new physics switch, cache keys) | automatic; the cross-flow blade's face-on value is carried in its coefficients name (2026-10-27) |
| Telfer drive fraction 0.63067 − 5.20991/r | `crew/stroke.py` StrokeTiming | ergometer data | ergometer | the rhythm | on-water timing in use | failing on the water; research runs use [K05]'s OnWaterTiming |
| Munk factor 0.50 | research profile `munk_factor` | literature value; **refitted** to the coxswain's ~3 deg/s full-helm authority | 2026-09-30, fin law [WF58], reflection 1–2 | the hull's yaw coupling | a fin, rudder or hull-added-mass change | 2026-09-30: 0.49–0.50 at every reflection, the literature value stands |

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
| Cross-flow blade face-on C_Dc | `coxswain/crew/liftdrag.CrossflowBlade` (research option) | **profiled, not fitted**: least error per tank dataset (§191) | the [CR06] test used the value predicted beforehand |

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
