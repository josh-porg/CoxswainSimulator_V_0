# The rowing blade's load: a literature review, and what it says about our blade models

*Started 2026-10-27 (user: "find the data you need via a literature review... don't stop at one source... be
thorough and broad and deep... building models from it"). A living review: each source read is added, each model
built from it is tested here before it enters the physics. Evidence sections in SOURCES (§183 onward) carry the
numbers; this page carries the argument.*

## 1. Why this review

The research profile still runs **tier 1**, the one-parameter slip law (normal load `C2 w_n |w_n|`, C2 = 140.88,
2.4× [CR06]'s computed plate, fitted to one athlete). The multi-parameter **tier 2** blade (lift and drag on the
angle of attack, [CG07] amplitudes, Coppel's and Sliasas–Tullis's full-size corrections, strip integration,
Patton/[LB19] added mass, immersion) is built as simulator options but was never promoted: its gate was blocked on a
measured blade load and motion together, and it failed [CR06]'s catch (2026-09-18; PHYSICS_PROGRAMME, phase 3).

Phase 4.3's rung 2 then showed the cost (SOURCES §181, §183): on [BR24]'s own motion every quasi-steady law carries
0.08–0.31 of his torque in the fourth fifth of the drive, so the forward-dynamic rower is scaled up to make up for a
blade that lets go. The questions for the literature:

1. Where in the stroke is a quasi-steady blade law adequate, and where not, on **measured** loads?
2. What unsteady mechanisms are measured or computed for a rowing blade, and which have a usable model form?
3. Is there a measured blade motion with its measured load, so a law can be tested with no uncertainty in the
   boat's speed through the water (the weakness of the [BR24] test)?
4. What about the tangential load, which tier 1 leaves out and tier 2 predicts with a sign never checked?

## 1a. Prior work in this project (checked 2026-10-27, §190)

The low-aspect-ratio lift this review applies to the blade was already in the code for the fins and rudder:
Polhamus/Helmbold's slope (`coxswain/hydro/appendages.py`) and Whicker & Fehlner's cross-flow form ([WF58], §169).
Hoerner's near-surface plate is [CR06]'s computed C0 = 1.3, and §164 raised ventilation and the unmeasured
near-surface regime. New here: applying these to the blade, and testing every blade law against measured loads on
known motions. Before building, search `coxswain/` and SOURCES for the formula's name.

## 2. Sources

**Read and used** (copies in `data/local/literature/<key>/`; numbers digitised to `data/literature/`):

| key | source | what it measured or computed | what it gives us |
|---|---|---|---|
| [GTW21] | Grift, Tummers & Westerweel (2021), *Hydrodynamics of rowing propulsion*, JFM 918 A29 (open) | a 1:2 sweep blade (0.275 × 0.125 m) driven by a robot along an **on-water blade path** (M4−, standard pace), still water, forces (ATI 6-DOF) with time-resolved PIV | the decisive test case: motion and load both measured. Extracted from the vector figures (Figs 4b, 7b, 9a): `data/literature/grift2021_*.csv`. Lift 40% / drag 60% of propulsion; LEVs at the tip early; quasi-2-D only up to t* ≈ 0.5 |
| [G19]/[G20] | Grift, Vijayaragavan, Tummers & Westerweel (2019) JFM 866; Grift (2020) thesis (open) | an AR-2 plate accelerated normal to itself near the surface | the entrainment form `F = m_p a + ½ρV²C_D A + (dm_h/dt) a (t − t_sr) + m_h a` (on file since §159) |
| [LJH12] | Leroyer, Barré, Wackers, Kobus & Queutey, *Vers le calcul des efforts sur les palettes d'aviron en fonctionnement réel*, 12th Journées de l'Hydrodynamique, Nantes (open, French; year not printed on the paper) | tank tests of a flat blade on a rotating arm against CFD: dynamic vs **static** (quasi-static) computations, with and without the free surface | quasi-static agrees up to ~30° incidence, **underestimates strongly between 60° and 120°** (the working range); static C_D at 90° is 1.15 (no surface) or 1.55 (surface) against the dynamic measurement's **2.05**; history matters at the end; viscous effects minor |
| [RLBQV19] | Robert, Leroyer, Barré, Queutey & Visonneau (2019), *Validation of CFD... full-scale rowing blade with realistic kinematics*, J. Mar. Sci. Tech. 24 (open, HAL) | a full-size blade on a real flexible oar, rotating arm on a carriage at 4 m/s, five-phase simplified stroke; six-component forces at 500 Hz; RANS with free surface and oar bending | a **second** measured load on a known motion (Figs 3, 7: vector). Oar bending "essential"; CFD over-predicts the maxima by 5–12%; experimental uncertainty ~11% |
| [L25] | Leroyer (2025), MARINE conference (open) | review of the SPRing CFD-coupled rowing simulator | the position of the field's leading CFD group: a quasi-static blade "doesn't really make sense" near the free surface |
| [LTW19] | Lintmeijer, Onneweer, Hofmijster, Wijgergangs, de Koning, Clairbois, Westerweel, Grift, Tummers & van Soest (2019), PLoS ONE 14(5) e0215674 (open) | a strain-gauge method for the blade force and its **point of application**, validated on a bench | method only: perpendicular force ICC .999, point of application ICC .93 above 5 N; parallel force not obtainable. No on-water centre of pressure yet |
| [vV22] | van Veen, van Leeuwen, van Oudheusden & Muijres (2022), JFM 936 A3 (open) | CFD of accelerating wings; a calibrated unsteady force model | a **model form**: translational + added mass (with a viscous part) + a term for the **delay of circulation build-up** under acceleration (Wagner-like), as an acceleration-dependent correction to the translational force |
| [WBD04] | Wang, Birch & Dickinson (2004), J. Exp. Biol. 207 (open) | 2-D CFD, 3-D robot wing, and quasi-steady estimates on the same kinematics | when quasi-steady works (drag well, lift depends on phase) and the size of the acceleration term |
| [vN23] | van Nieuwburg, van Spreuwel, Tran et al. (2023), Front. Sports Act. Living 5:1109494 (open) | 1:5 towing tank, a modified blade | +0.4% boat speed at equal power: the scale of blade-shape gains |

**Already on file** and re-read for this question: [CG07] (quasi-static flume, a 0.64 m wide channel), [CO10]
Coppel (full-size CFD corrections), [ST09] Sliasas & Tullis, [LB19] Labbé et al. (added mass on a blade),
[LNA15] (shaft stiffness).

**Cited, not obtained** (paywalled or the host refused automated access; to request or read in a library):
Ringuette, Milano & Gharib (2007) JFM 581 (tip vortex and force on low-AR normal plates; the Caltech thesis host
refused access); Pullin & Wang (2004) JFM 509 (accelerating plate, vortex-sheet similarity); Fernando & Rival
(2016) JFM (vortex pinch-off on low-AR propulsors); Kriegseis & Rival (2014) JFM; Sane & Dickinson (2002) J. Exp.
Biol. (revised quasi-steady model with rotation); Eldredge & Jones (2019) Annu. Rev. Fluid Mech. (LEV models);
Barré & Kobus (2010) (towing-tank comparison of blade force models); Leroyer et al. (2008, 2010); Sliasas &
Tullis (2010, shaft bending); Hofmijster et al. (2010, energy loss at the blades).

## 3. What the measured loads say

**3.1 On an exactly known motion, quasi-steady laws get the middle and late drive roughly right**
(`research/blade/grift2021_quasisteady.py`, SOURCES §184). Each project law integrated in strips across
[GTW21]'s blade on their motion, against their measured F_n (law / measured, by fifths of the drive):

| law | 1 | 2 | 3 | 4 | impulse | peak t* (meas. 0.59) |
|---|---|---|---|---|---|---|
| tier 1 slip, C0 1.3 (computed) | 0.26 | 0.19 | 0.35 | 0.64 | 0.45 | 0.63 |
| tier 1 slip, C0 3.12 (research profile's level) | 0.62 | 0.46 | 0.83 | 1.53 | 1.09 | 0.63 |
| tier 2 CG07 | 1.91 | 0.80 | 0.83 | 1.09 | 1.07 | 0.62 |
| tier 2 Coppel | 1.40 | 0.62 | 0.68 | 0.83 | 0.84 | 0.61 |
| tier 2 Sliasas–Tullis | 1.51 | 0.62 | 0.62 | 0.78 | 0.80 | 0.61 |

- The slip law has the **wrong shape**: a single coefficient would have to run 5.0, 6.8, 3.8, 2.0 through the
  fifths. Tier 2's angle dependence carries the shape; its level is within 10–20% in the middle of the drive.
- **The catch** (first fifth) is over-predicted 1.4–1.9× by tier 2: the same failure as on [CR06] (2026-09-18).
  [LJH12] find quasi-static adequate at the small incidences of the catch, so the excess is more likely the
  blade's entry (immersion, ventilation) than its incidence.
- **The release** (last fifth) has a positive F_n spike that no quasi-steady law produces.
- **Why tier 2's level is close in the middle:** [CG07]'s "quasi-static" C_D ≈ 2 at 90° is high for a static plate
  (unbounded, ~1.15–1.2), and sits where [LJH12]'s *dynamic* measurement is (2.05). A narrow flume (0.64 m) adds
  blockage. So tier 2 may carry the unsteady mid-drive enhancement by coincidence, which matters when a law is
  "corrected" toward static values (Coppel's full-size ratios lower it, and score worse here).

**3.2 The tangential load is drag-like, and tier 2 has its sign wrong.** In [GTW21], F_t · w_a > 0 in every fifth
(as does F_n · w_n): the blade does work on the water along its span as well as across it, so the water's
tangential load opposes the blade's sliding. Its size: C_T ≈ 0.2–0.36 against C_N ≈ 1.2–2.05 in the second and
third fifths. Tier 2's resolution of lift and drag gives `C_T = (A_d − 2A_l) sin²α cos α`, negative for the Big
Blade (−0.43 amplitude), a load *along* the sliding: the prediction `coxswain/crew/liftdrag.py` flagged as "the
first thing tier 2 should be tested on" is **rejected**. Added mass in Kirchhoff's form gives a tangential term
`−m_a U_n Ω` of the wrong sign too; potential flow does not describe a separated blade.

**3.3 Added mass at full strength from the catch over-predicts the catch** (×4.8 Patton, ×7.2 [LB19] in the first
fifth) while it fixes the second fifth's shortfall (0.80 → 1.12 with CG07). This repeats §172: the entrained water
is not there at entry. A history (build-up) is needed, which is what [G19]'s entrainment and [vV22]'s
circulation-delay term are.

**3.4 [BR24] against [GTW21]: the kinematics differ, not only the load.** The blade's angle of attack by fifths
(tip; centre in brackets): [GTW21] 16 (6), 35 (22), 56 (45), 79 (73)°; [BR24] 10 (7), 22 (14), 48 (36), 48 (34)°.
In the measured on-water path the blade turns face-on late in the drive; [BR24]'s oar angle and boat speed imply a
blade still sliding along itself, with |w| at the centre 0.87 m/s against a force his data put at 110 N. At the
tip the coefficient his load implies is 1.7 (plate-like); at the centre 3.7. So either his load sits at the tip
(tip vortex, [Ringuette 2007]; LEVs at the tip, [GTW21]) or his speed through the water and oar-angle channels do
not describe the blade's motion late in the drive (a sculling single accelerates far more within the drive than
[GTW21]'s four). §183's open loop survives a ±0.4 m/s offset; a channel *lag* has not been tested.

**3.5 The full-size cross-check, [RLBQV19] (SOURCES §185).** A real oar on a towing carriage, its bending computed
from the measured moment through the paper's own bench law. Two results carry across both sources: tier 2's
early-stroke over-prediction (+0.5 to +0.8 of the peak at 50–80° here; 1.4–1.9× in [GTW21]'s first fifth), and added
mass at full strength making things worse. One does not: the **level**. On the full-size blade, awash, the computed
plate (C0 1.3) gets the peak (198 against 199 N) and tier 2 over-predicts 1.4–1.8×; on [GTW21]'s half-scale blade
the reverse. The free surface is the leading candidate ([GTW21] Fr 0.65 against 0.93 on the water; [RLBQV19]'s air
cavity). **Bending improves every law** (rms −10 to −30%, peaks toward the measured).

**3.6 The candidate research blade (§188–190).** The project's own fin law on the blade: Whicker & Fehlner's
cross-flow form, C_N = C_Lα sin α cos α + C_Dc sin²α, with C_Lα the fins' Polhamus/Helmbold slope at the blade's
along-flow aspect ratio and C_Dc the face-on coefficient (no cap needed); tangential drag-like from Helmbold's lift.
On the full-size blade with Hoerner's 1.3 it is the best law tested (rms 25 N, peak at 90° against 88°). (§189's
Polhamus–Lamar variant, K_v = π with a cap, is the same structure and is superseded.) Every piece is sourced;
the one open number is the face-on cap, set by the free-surface regime (~2 at the half-scale tank's Froude number,
~1.3 at full scale). Against the research profile's slip law it has the right shape on both measured loads; against
tier 2 it removes the catch excess and the tangential sign error.

## 4. What this means for the models

1. **Tier 1 should not stay the research blade.** On the one measured load with a known motion it has the wrong
   shape at any level; its fitted C2 makes the impulse come out right (1.09) by trading the catch against the late
   drive.
2. **Tier 2's normal law is the better base**, but its tangential sign is wrong and its catch too strong.
3. **Next models, in order, each tested on [GTW21] and [RLBQV19] before any athlete:**
   - (a) a drag-like tangential coefficient (the measured C_T(α) from [GTW21]; sourced, not athlete-fitted);
   - (b) the catch: immersion and an entrainment/circulation build-up, [G19]'s form and [vV22]'s delay term;
   - (c) the release spike: the blade's deceleration and extraction (added mass with the history of (b));
   - (d) the point of application: the strip integration's centre of pressure against [GTW21]'s tip-referenced
     loads, and whether a tip load is needed ([BR24]'s 1.7 at the tip).
4. **[RLBQV19] is the cross-check** that keeps this from being a one-paper fit (done, §185).
5. **Oar bending into the simulator** (user, 2026-10-27: "blade bending is good to model"): a quasi-static shaft
   compliance, f = M/K_lin and phi = M/K_ang, with [RLBQV19]'s bench constants for a sweep oar and [LNA15]'s for
   sculls, as a research option; tested first on [RLBQV19], where the moment is measured.
6. **The free surface** (ventilation against Froude number, immersion): sized from [RLBQV19] Fig. 8 and [G20]'s
   depth curve before any law's level is changed.

## 3a. Strips, or blade-element momentum theory? (user, 2026-10-27; §193)

Strips are BEMT's blade-element half: each strip's own velocity and angle of attack, quasi-steady coefficients, no
induced velocity. The momentum half adds the velocity the blade induces in the water it acts on; for a rowing blade
that is not a propeller's steady inflow but the water the blade has set moving through the drive: [G19]'s entrained
wake, F = m_p a + ½ρV²C_D A + (dm_h/dt) a (t − t_sr) + m_h a (model-scale rates in hand). Strips do not include it.
It is the sourced candidate with the sign of two observations (load per unit dynamic pressure falling with speed in
[GTW21]; fixed coefficients over-loading the catch), and the next model item after the tangential level.

## 4a. Towards a paper (noted 2026-10-27, user: "we ought to validate it and note it to later produce a paper")

Not a new dynamic model: quasi-steady low-aspect-ratio theory plus a quasi-static bending law. The possible
contribution is the validation, and it has to be earned:

| step | why | status |
|---|---|---|
| read Barré & Kobus (2010), Proc. IMechE P 224:37–50 (DOI 10.1243/17543371JSET43) | the closest prior work: both model families against towing-tank data | **requested from the authors by the user (2026-10-27)**; the abstract: maximum drag and lift coefficients depending on an unsteadiness (advance) parameter; the four datasets in hand sit at advance ratio 0.72–0.82, so it does not separate them (§193) |
| a third measured load | not a two-paper result | **done**: [LJH12]'s flat blade (§191); the fins' law reaches its floor (F_x rms 6.2 N) |
| the one open number profiled, one Froude definition | what separates the datasets | **done** (§191): C_Dc 2.3 / 2.2 / 1.3 at Fr 0.49 / 0.87 / 1.00 |
| [GTW21] Fig. 9(b): three speeds | a speed effect inside one source | **done** (§191): impulse/κ² −10% over a doubling; Reynolds confounded |
| [CR06]'s blade load (an athlete, a scull) | held out | **done** (§191): the predicted face-on value gives her impulse to 1%; her rms does not discriminate |
| [RLBQV19] Fig. 9 (roll) and the immersion series as held-out cases | more full-size cases | to do; needs the blade's wetted geometry (area not stated) |
| the law in the simulator, closed loop | the blade inside the boat | **built and run (§192): negative.** −7 to −9% speed at the athletes' power, IVV worse; tier 2 best closed loop. Separate the tangential drag from the weaker lift next |
| the oar's bending in the simulator | sweep oars bend ~7° ([RLBQV19]); sculls less ([LNA15]) | **built** (§193): `oar_bending=`; scull derived from [LNA15] (~1° at 400 N·m); +1–3% at equal power |
| why the cross-flow blade is slow closed loop | before any claim | **found** (§193, corrected §196): the drag-like tangential load costs 6–10% of speed in the boat, but on the measured on-water motion its share of blade work is right (19.6% against 15.7% measured, [H10] ~18%). The boat's blade path slides too much late in the drive; instrument it next |
| a sourced face-on function of Froude number, or a fourth dataset above Fr 1 | the one open number | open |
| independence | nothing tuned to a test set | the rule; C_Dc is profiled, not fitted, and the athlete test used the predicted value |

## 5. Log

- 2026-10-30 (§196): the review and sourced blades combined. On [GTW21]'s measured on-water motion the review blade's
  tangential share of blade work is 19.6 % against 15.7 % measured ([H10]: ~18 %): its level is right. The 6–10 % it
  costs in the boat is the simulated blade still sliding along itself late in the drive (the measured path turns it to
  73° at the centre). Tier 2's closed-loop speed comes from its wrong-signed tangential term offsetting that. Next is
  the blade path in the boat, not a law.

- 2026-10-27 (§193): the closed-loop deficit is the tangential load's level (6–10% of speed; [H10] implies ~1%);
  oar bending in the simulator; Barré & Kobus's abstract (the advance number does not separate the datasets in hand);
  the fits audited (one fit, the scull C2, filed as sourced: docs/CALIBRATION_REGISTER.md); structural issues listed
  and fixed (docs/STRUCTURAL_ISSUES.md).
- 2026-10-27: the cross-flow blade in the core and the simulator (§192); closed loop on both athletes it is 7–9%
  slow at their power and worsens IVV, while tier 2 is best: the open-loop fit does not carry over; not promoted.
- 2026-10-27: paper steps (§191): [LJH12]'s flat blade as a third measured load (the fins' law at the data's floor);
  C_Dc profiled per dataset and ordered by one Froude number (2.3 / 2.2 / 1.3 at 0.49 / 0.87 / 1.00); [GTW21]'s three
  speeds; [CR06] held out (impulse to 1% at the predicted value); a core strip-integration bug fixed.
- 2026-10-27: prior-work check (§190): the slope and the cross-flow form were already the fins'; the fins' law on
  the blade is the best fit at full scale (rms 25 N). Not a novel dynamic model; a possible validation paper, plan
  in §4a.
- 2026-10-27: Polhamus–Lamar vortex lift with a face-on cap (§189): the same sourced law fits both measured loads,
  cap ~2 half-scale and ~1.3 full scale (ventilation the reading); [BR24]'s late drive still collapses under it, so
  his gap is his motion, not missing blade physics.
- 2026-10-27: Helmbold's low-AR lift slope in tier 2 (§188): fixes the catch over-prediction and the tangential
  sign on both measured loads; the moderate-incidence vortex lift (Polhamus, Lamar) and the full-scale drag level
  remain.
- 2026-10-27: [RLBQV19] Fig. 8, the first published force-against-immersion curve for a rowing blade (§187): +11%
  for 3 cm of cover, +16% for 6 cm, full load awash; the project's chosen immersion shape (zero awash) is not
  supported, and [G20]'s slow plate gains far more: the surface's effect depends on speed.
- 2026-10-27: oar bending as a quasi-static hinge solved from the blade's own load (§186): stable; the
  over-loading laws over-bend the oar 1.6–2×, so bending is a second check on a law's level.
- 2026-10-27: [RLBQV19] extracted (motion, bending, measured F_x, F_y, M_z) and every law tested on it (§185).
- 2026-10-27: review started. [GTW21] extracted and every project law tested on it (§184). Tangential sign of tier
  2 rejected; slip law's shape rejected; catch over-prediction confirmed on a second source. [RLBQV19], [LJH12],
  [vV22], [WBD04], [LTW19], [L25], [vN23], [G19] obtained.
