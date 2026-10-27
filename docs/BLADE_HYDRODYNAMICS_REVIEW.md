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
4. **[RLBQV19] is the cross-check** that keeps this from being a one-paper fit: a full-size blade, a different
   motion (constant carriage speed), a flexible oar. Its bending must be modelled first (the blade lags the motor).

## 5. Log

- 2026-10-27: review started. [GTW21] extracted and every project law tested on it (§184). Tangential sign of tier
  2 rejected; slip law's shape rejected; catch over-prediction confirmed on a second source. [RLBQV19], [LJH12],
  [vV22], [WBD04], [LTW19], [L25], [vN23], [G19] obtained.
