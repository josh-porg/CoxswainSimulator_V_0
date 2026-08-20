# Sources

Every number in this model that is not derived from first principles comes
from one of the references below. Each entry records what was taken from
it, where that lands in the code, and — where it matters — what the source
does *not* settle.

Local copies of the open-access PDFs are not committed (they are
copyrighted); the URLs below were live as of August 2026. The
[UCLA Bionics rowing archive](http://bionics.seas.ucla.edu/education/Rowing/)
mirrors many of them and is the single most useful starting point.

Contents:
[1. Boat dynamics](#1-boat-dynamics) ·
[2. Rower kinematics](#2-rower-kinematics) ·
[3. Anthropometry](#3-anthropometry) ·
[4. Boat velocity fluctuation](#4-boat-velocity-fluctuation) ·
[5. Hull hydrodynamics](#5-hull-hydrodynamics) ·
[6. Shallow water](#6-shallow-water) ·
[7. Oar and blade](#7-oar-and-blade) ·
[8. Open questions](#8-open-questions)

---

## 1. Boat dynamics

### [F09] Formaggia, Miglio, Mola & Montano (2009) — primary
*A model for the dynamics of rowing boats.*
**Int. J. Numer. Meth. Fluids 61**(1) 119–143. doi:10.1002/fld.1940
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2009_01.pdf)

The backbone of the whole model. Their eq. (14) is a surge/heave/pitch
formulation; `coxswain/core/rigid_body.py` generalises it to six degrees of
freedom.

Taken from it:
- the coupled equations of motion, including the mass-matrix structure
  `[[Mt·I, −A], [A, I − B]]` with `A = Σm·S(r)` and the crew's **positive**
  contribution `+Σm|x|²` to rotational inertia (eq. 14b);
- the reference-frame definitions (§3): `G_h` is the *hull* centre of mass,
  not the combined centre of mass;
- the 12-segment rower decomposition (§4.2);
- the ideal-lever oar model and the `r_h/L` gearing that determines how
  much oarlock force reaches the hull (eq. 12, 14a);
- the drive-duration fit `τ_a = 0.00015625(r−24)² − 0.008125(r−24) + 0.8`
  (§5), which they attribute to Atkinson and Rekers;
- the oarlock force shape `F_max·sin(πt/τ_a)` with `F_max_x = 1200 N`,
  `F_max_z = 200 N` (eq. 15);
- the steady resistance decomposition (§6.1), see [§5](#5-hull-hydrodynamics);
- validation targets (§7): single scull hull 8.2 m / 15 kg / 66 kg·m²
  pitch inertia; pitch within ±0.02 rad; heave within ~0.08 m.

Two cautions found while implementing:
- the paper **switches rotation-matrix convention** between §3 (where `R`
  maps absolute→hull, eq. 2–5) and eq. (8)/(14) (where it maps hull→absolute).
  `coxswain/core/frames.py` therefore names every direction explicitly.
- its area measures are written `∫q dσ` weighted by submersion depth, which
  makes them volumes rather than areas, so the reference area intended for
  `C_dw` is genuinely ambiguous. See `docs/validation.md`.

### [CR06] Cabrera, Ruina & Kleshnev (2006)
*A simple 1+ dimensional model of rowing mimics observed forces and motions.*
**Human Movement Science 25**(2) 192–220.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2006_01.pdf)

An independently validated model (boat-velocity residuals ≈ 0.13–0.15 m/s).
Used here as a cross-check rather than a source of parameters, with one
important exception: their **eq. (8)** imposes

> "Since the rower always has a grip on the oar handle, the fore-aft
> positions of the rower's hand and the oar handle relative to the foot
> stretcher are the same."

That is the constraint `coxswain/boats/boat.py::_hand_targets` now enforces.
Before it was added, this model's rowers' hands sat up to **0.46 m** away
from their own oar handles.

Also useful: their boat drag coefficients `C₁ = 3.16 N/(m/s)²` (single) and
`1.99` (four), and blade drag `C₂ = 58.7` (scull) / `84.5` (sweep).

### [S10] Serveto, Barré, Kobus & Mariot (2010)
*A three-dimensional model of the boat–oars–rower system using ADAMS and
LifeMOD commercial software.* **Proc. IMechE Part P 224**(1) 75–83.
[Abstract](https://journals.sagepub.com/doi/10.1243/17543371JSET42) ·
[Short version](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2009_03.pdf)

The precedent for driving an articulated rower from **joint coordinates**
rather than from motion-capture marker trajectories, which is the approach
`coxswain/crew/kinematics.py` takes.

---

## 2. Rower kinematics

### [CG10] Caplan & Gardner (2010) — primary driver
*The influence of stretcher height on posture in ergometer rowing.*
**J. Sports Sciences 28**(3) 263–269.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Biomechanics_2010_01.pdf)

Nine male university rowers, Vicon at 120 Hz, 30 spm, angles reported at
catch / mid-drive / finish / mid-recovery for three stretcher heights.
Transcribed in full into `coxswain/crew/stroke_data.py`.

Their definitions matter and are easy to misread:
- **shank** ("ankle angle") is measured between the shank and the ground
  *from the stern side*, so a bow-referenced link angle is `180 − value`;
- **knee** and **hip** are interior angles between long axes;
- **trunk** is from vertical, negative towards the catch.

Table II, stretcher position 1 (standard), mean ± SD:

| | catch | mid-drive | finish | mid-recovery |
|---|---|---|---|---|
| shank | 91.6 ± 8.0 | 140.4 ± 6.3 | 168.5 ± 3.7 | 141.2 ± 4.0 |
| knee | 41.0 ± 13.5 | 125.9 ± 15.6 | 171.2 ± 6.5 | 127.2 ± 8.6 |
| hip | 18.7 ± 8.6 | 65.7 ± 6.1 | 109.1 ± 8.4 | 51.1 ± 8.2 |
| trunk | −38.1 ± 8.5 | −9.4 ± 6.5 | +16.6 ± 8.3 | −24.8 ± 7.2 |

**Why this dataset is trustworthy.** The four angle sets are not
independent — shank, knee and trunk determine the hip. Reconstructing the
hip angle reproduces the separately measured value to better than 0.5° at
mid-drive, the finish and mid-recovery, and to 17.4° at the catch (inside
one SD of the ±13.5° scatter on the catch knee angle). They describe one
consistent linkage. Pinned by `tests/unit/test_stroke_data.py`.

The implied seat travel is **0.605 m**, which independently matches the
0.60–0.70 m slide excursion reported for on-water crews, and the catch and
finish keyframes agree on the hip height above the ankle to **1 mm** — the
level-seat constraint, recovered from the data rather than assumed.

**Known departure.** Taken literally, the two mid-stroke keyframes put the
hip ~7 cm higher than the ends do. A seat on a rail cannot do that, so the
default `thigh_mode="level_seat"` derives the thigh angle from the shank at
constant hip height. `thigh_mode="measured"` reproduces the data verbatim.

### [K19] Kleshnev — on-water elite cross-check
*Analysis of Angles of Body Segments in the World's Best Rowers*, row2k
(2019), summarising the *Rowing Biomechanics Newsletter*.

Used only to check [CG10]'s ergometer values against on-water elite
practice. Catch knee 45.4° (medallists; mean 47.3 ± 8.7, range 33.2–71.8);
trunk 24.5 ± 4.5° forward at the catch and 26.3 ± 4.4° back at the finish;
stroke length 1.52 m. Trunk *sweep* agrees with [CG10] to about 4°, which
is why the ergometer data are used as the driver without apology.

### [OT25] *Kinematic Analysis of Olympic and Traditional Rowing Mechanics
at different Stroke Rates* (2025). [PMC12289236](https://pmc.ncbi.nlm.nih.gov/articles/PMC12289236/)

Recorded in `stroke_data.py` but **not** used as a driver: its sliding-seat
finish trunk angle of 48.6° is more than 25° larger than either other
source. Its peak segment velocities (leg 1.15, trunk 1.34, arm 2.29 m/s at
30 spm) are used as loose sanity bounds.

---

## 3. Anthropometry

### [dL96] de Leva (1996)
*Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.*
**J. Biomechanics 29**(9) 1223–1230.
[PDF](https://ebm.ufabc.edu.br/wp-content/uploads/2013/12/Leva-1996.pdf)

Table 4 verbatim in `coxswain/crew/anthropometry.py`. Reference samples
73.0 kg / 1.741 m (male) and 61.9 kg / 1.735 m (female). Chosen over
[F09]'s cited NASA-STD-3000 because de Leva's lengths are referenced to
**joint centres**, which is what a kinematic linkage needs.

One transcription choice: the upper trunk uses de Leva's *alternative*
CERV→XYPH endpoints (242.1 mm, CM 50.66%) rather than SUPRA→XYPH, so that
lower + mid + upper trunk sums exactly to the 603.3 mm whole-trunk length
and the three can be stacked into one rigid link. Checked by
`test_trunk_thirds_stack_to_the_whole_trunk_length`.

---

## 4. Boat velocity fluctuation

This is the model's largest known discrepancy, so the evidence is set out
in full.

### [HF09] Hill & Fahrig (2009)
*The impact of fluctuations in boat velocity during the rowing cycle on
race time.* **Scand. J. Med. Sci. Sports 19**(4) 585–594.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Technique_2009_03.pdf)

Eight elite coxless pairs, stepped rates. Their Table 1 gives a strong
independent check on **stroke timing**:

| planned rate | 20 | 24 | 28 | 32 |
|---|---|---|---|---|
| measured rate (spm) | 20.6 | 24.2 | 27.7 | 31.5 |
| measured drive (ms) | 862 | 810 | 779 | 752 |
| [F09] τ_a formula (ms) | 829 | 798 | 772 | 747 |
| error | −3.8% | −1.5% | −0.9% | −0.7% |

The drive-duration formula this model inherits from [F09] is accurate to
better than 1% at racing rates against data it was not fitted to.

They also quantify the *cost* of fluctuation: 4.6–5.1 s over 2000 m
relative to a hypothetical constant-speed boat, rising with rate.

### [IVV25] *Intracycle Velocity Variation During a Single-Sculling 2000 m
Rowing Competition.* [PMC12349136](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349136/)

The cleanest quantitative target found. Elite single scullers, full race:

| | male | female |
|---|---|---|
| mean velocity | 15.40 ± 0.81 km/h (4.28 m/s) | 13.36 ± 0.88 (3.71 m/s) |
| max | 21.39 km/h (5.94 m/s) | 18.77 (5.21) |
| min | 11.15 km/h (3.10 m/s) | 9.03 (2.51) |
| IVV = max − min | 5.78 km/h (1.606 m/s) | 5.50 (1.528) |
| **IVV / mean** | **37.5 %** | **41.2 %** |
| CVV (σ/mean) | 14.13 ± 2.02 % | 11.64 ± 1.93 % |
| cycle rate | 33.65 ± 1.63 | 32.40 ± 4.09 |

### [D11] Day et al. (2011), introduction — see [§6](#6-shallow-water)

For a men's pair at rate 35, from Kleshnev (2002) acceleration data:
maximum deceleration **over 1 g** at the catch, and "the range of the speed
variation is almost **50%** of the mean value".

### Where this model stands

At 33.65 spm, peak-to-peak surge as a fraction of mean:

| | model | measured |
|---|---|---|
| 1x | 65.1 % | 37.5 % [IVV25] |
| 4+ | 56.6 % | — |
| 8+ | 54.7 % | — |
| pair | — | ~50 % [D11] |

### What has been ruled out

**The dynamics are not at fault.** The hull's surge swing should equal
`(m_crew / m_total)` times the crew centre-of-mass velocity swing, by
momentum exchange alone. Measured in the model:

| | crew/total | crew CoM rel. velocity p2p | predicted hull swing | actual |
|---|---|---|---|---|
| 1x | 0.850 | 3.18 m/s | 2.70 m/s | 2.90 m/s |
| 8+ | 0.823 | 3.07 m/s | 2.52 m/s | 2.85 m/s |

The balance closes to 7–13%, the excess being the oar impulse and drag.
So the fluctuation is inherited wholesale from the prescribed crew motion.

**The crew motion follows necessarily from the measured joint angles.**
Per-segment fore-aft travel for the single sculler, and the mass-weighted
total:

| segment | % of body mass | x travel | contribution |
|---|---|---|---|
| upper trunk | 16.0 | 1.09 m | 0.174 |
| mid trunk | 16.3 | 0.88 m | 0.143 |
| head | 6.9 | 1.34 m | 0.093 |
| lower trunk | 11.2 | 0.67 m | 0.075 |
| thighs | 28.4 | 0.52 m | 0.146 |
| arms | 9.8 | 1.17 m | 0.116 |
| shanks + feet | 11.4 | 0.18 m | 0.020 |
| **total** | 100 | | **0.767 m** |

The hip travels 0.615 m and the shoulder 1.214 m. Both are forced: with
[CG10]'s trunk angles (128.1° at the catch, 73.4° at the finish as link
angles) and a 0.65 m trunk, the shoulder *must* travel
`0.615 + 0.65·(cos 73.4° − cos 128.1°) = 1.22 m`. Each trunk mass sits at
its de Leva height above the hip and sweeps the corresponding arc; all
three reproduce to 1%.

To match [IVV25] the crew centre of mass would have to travel about
**0.46 m**, not 0.77 m — which, with a seat travelling 0.615 m, requires
the upper body to move *less* than the seat. Substituting [K19]'s
on-water elite trunk angles changes the excursion by only 5%, so it is not
an ergometer-versus-water artefact of the trunk angles either.

### What is left

The tension is genuine and unresolved. Either

1. the published IVV understates the true peak-to-peak — plausible if the
   instrument smooths, and note the reported extremes are oddly skewed
   (max − mean = 1.66 m/s against mean − min = 1.18 m/s, the opposite
   asymmetry to a real boat-speed curve, which has a sharp catch dip and a
   broad recovery plateau); or
2. rowers on the water swing their upper body substantially less than
   [CG10] measured on a stationary ergometer, in a way the trunk *angle*
   does not capture.

**The decisive missing datum is a published rower centre-of-mass
excursion relative to the boat, measured on the water.** Until then the
regression bound is a wide band, not a target.

Things checked and found not to be the cause: added mass in surge
([F09] §6.4 gives 0.201 kg for a 4x — negligible, as expected for a
slender hull); the hands-on-handle constraint (worth 2.5 percentage
points); the oar sweep discontinuity (worth −2 points, i.e. it made the
fluctuation slightly *worse* while being more correct); and the
hip-to-shoulder link length, which was genuinely wrong — de Leva's trunk
ends at the cervicale, not the shoulder joint — but which turns out not to
move the centre of mass at all, because the trunk masses sit at their own
anatomical heights regardless of where the link ends.

---

## 5. Hull hydrodynamics

### [F09] §6.1 — the resistance decomposition in use

```
R_shape = ½ρ|Γ_X| C_dX V²      C_dX = 0.01
R_vis   = ½ρ|Γ|   C_f  V²      C_f  = C_f0/(log₁₀Re − 2)², C_f0 = 0.075
R_wave  = ½ρ|Γ_Z| C_dw V²      C_dw = 0.02
```

`C_f` is the **ITTC 1957** model-ship correlation line ([F09] ref. [13],
Hadler 1958), which is defined with `log₁₀`. The paper writes "log"; the
legacy code read that as natural log, which understates `C_f` about
eightfold.

The wave term is **not** used literally — see `PAPER_LITERAL` in
`coxswain/hydro/resistance.py` and `docs/validation.md` for why, and for
the towing-data calibration used instead.

### [D11] Day, Campbell, Clelland, Doctors & Cichowicz (2011)
*Realistic evaluation of hull performance for rowing shells, canoes, and
kayaks in unsteady flow.* **J. Sports Sciences 29**(10) 1059–1069.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2011_02.pdf)

Towing-tank measurements with a rig that reproduces realistic speed
profiles, plus hot-film turbulence measurements on a full-scale single.
Their headline caution, which applies directly to this model: a
computational approach predicted unsteady wave resistance well at low
frequency but **substantially under-predicted total resistance** at
realistic oscillation frequencies, attributed to unsteady viscous effects.
This model uses quasi-steady resistance and inherits that limitation.

### Other resistance references
- **Lazauskas (1998)**, *A performance prediction model for rowing races*,
  Tech. Rep. L9702, Univ. Adelaide — source of [CR06]'s drag coefficients.
- **Tuck & Lazauskas (1996)**; **Scragg & Nelson (1993)** — thin-ship
  (Michell) wave resistance for shells, the latter *including* shallow
  water. The principled route to replacing the parametric wave term.
- **Hoerner (1965)**, *Fluid-Dynamic Drag* — cross-flow drag coefficients.

---

## 6. Shallow water

Directly relevant: the target course is the Charles, a river of a few
metres' depth, not an Olympic lake.

### [D11] §"Effect of water depth" — the basis for `coxswain/hydro/shallow.py`

- the governing parameter is the depth Froude number `Fr_h = U/√(gh)`;
- **`Fr_h ≤ 0.5`** — "results are similar to deep water";
- approaching **`Fr_h = 1`** — "wavelengths, wave heights, and wave
  resistance all increase";
- **`Fr_h > 1`** — "the transverse components of the wave pattern
  disappear and the wave resistance may be reduced compared with the
  critical value";
- "On a rowing lake with depth of 3.0 m, the critical speed is around
  **5.4 m/s**; many elite rowers will be travelling at this speed at some
  point in their stroke cycle" — the model gives 5.42 m/s;
- for a pair at rate 35 in 3.0 m, "the depth Froude Number would vary from
  **0.65 to 1.09**" — i.e. a racing shell sweeps *through* critical twice
  every stroke;
- the viscous changes with depth "are less likely to be sensitive to water
  depth", which is why only the wave term is scaled here.

### Schlichting's method

The subcritical correction implemented is Schlichting's matched-wavelength
construction: a hull at `U` in depth `h` makes the same transverse
wavelength as a hull at `U_∞ > U` in deep water, from the finite-depth
dispersion relation, giving `U = U_∞·√(tanh(gh/U_∞²))`. Background:
[Marine Insight overview](https://www.marineinsight.com/schlichtings-method-shallow-water-effect-on-ship-resistance/);
**Faltinsen (2005)**, *Hydrodynamics of High-Speed Marine Vehicles*, CUP,
for the wave-pattern treatment [D11] cites.

Schlichting's **second** term, the speed loss from return flow, scales with
`√A_m/h`. A racing eight has a midship submerged section of roughly
0.05 m², so `√A_m/h ≈ 0.07` in 3 m — negligible, and not modelled.

**What is not sourced.** Schlichting's construction diverges at `Fr_h = 1`,
so the near-critical amplification is capped
(`DEFAULT_MAX_AMPLIFICATION = 3.0`). That figure is a modelling choice
consistent with finite-depth thin-ship results for slender hulls; it has
**not** been measured for a rowing shell. Results that depend sensitively
on `0.9 < Fr_h < 1.1` should be treated as indicative. Pinning it down
needs the tank programme [D11] describes, or a finite-depth Michell
calculation (Scragg & Nelson).

### Verification of the drag increment

Checks the implementation passes, against numbers not used to build it:

| check | source | model |
|---|---|---|
| critical speed at 3.0 m depth | 5.4 m/s [D11] | 5.42 m/s |
| `Fr_h` range, pair at rate 35, 3.0 m | 0.65–1.09 [D11] | reproduced |
| `Fr_h ≤ 0.5` indistinguishable from deep | [D11] | <1% change |
| eight at 3.0 m, speed loss | — | 12.6% |

The first three are genuine validations. **The fourth is not** — no
measured speed loss for a rowing shell at a stated depth was found in the
literature searched, so the 12.6% is the model's output, corroborated only
circumstantially by World Rowing setting 3 m as the minimum legal depth.

An accurate increment near `Fr_h = 1` remains out of reach without either
the tank programme [D11] describes or a finite-depth Michell calculation;
the cap at `DEFAULT_MAX_AMPLIFICATION = 3.0` is where that uncertainty is
parked. Since a racing eight sweeps through critical twice per stroke in
3 m water, this is the single largest uncertainty in any Charles result.

### Regulatory corroboration
World Rowing (FISA) Bye-Law rule 2.4 requires **3 m minimum** depth across
all lanes, with 3.5 m recommended for international courses. The model
puts an eight's loss at 3.0 m at roughly 13% of speed, which is consistent
with a depth chosen to be the threshold of acceptability rather than a
comfortable margin.

---

## 7. Oar and blade

### [F09] §4.3 — the ideal-lever model in use
Oar massless and infinitely rigid, blade a fixed fulcrum ("perfect
blades"), so hand and oarlock forces are proportional: `F_h = −(L−r_h)/L·F_o`
and the net force reaching the hull is `(r_h/L)·ΣF_o`. The paper notes this
"can be weakened by using a more detailed model of the blade action".

This model keeps the ideal lever but multiplies by a `blade_efficiency`
factor (default **0.78**) standing in for slip — see below.

### [H10] Hofmijster, de Koning & van Soest (2010)
*Estimation of the energy loss at the blades in rowing: common assumptions
revisited.* **J. Sports Sciences 28**(10) 1093–1099.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Rigging_2010_01.pdf)

Instrumented oars and oarlocks, world-class female sculler at racing pace.
Two findings that bear directly on the assumptions here:
- estimated blade power losses are **18% higher** when a blade force
  component *parallel* to the oar is included — i.e. the
  "force normal to the blade" assumption (used by [CR06] and here)
  understates losses;
- oar **deformation** substantially changes reconstructed blade
  kinematics, but has **no effect** on the power-loss estimate — so
  modelling oar flex is not needed for an energy budget.

### [ST09] Sliasas & Tullis (2009)
*Numerical modelling of rowing blade hydrodynamics.*
**Sports Engineering 12**(1) 31–40.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2009_02.pdf)

Three-dimensional CFD of a blade interacting with the free surface,
validated against quarter-scale flume experiments. The reference for
replacing the efficiency factor with a real blade model.

### The [CR06] blade model, as implemented

`coxswain/crew/oarlock.py::BladeModel` implements their **Model 1** (due to
Pope, and to Alexander 1925). Blade velocity resolved on the oar's polar
basis, force normal to the blade and quadratic in the normal slip:

```
v_O   = v_b sin(theta) e_r + (l theta_dot + v_b cos theta) e_theta
F_oar = C2 (l theta_dot + v_b cos theta)^2          [CR06] eq. (11)
C2    = ½ rho C0 A0
```

`theta` is the oar angle from the boat's transverse axis, `l` the outboard
length, `A0` the blade face area, `C0` a shape constant. Fitted values
**C2 = 58.7** (single scull) and **84.5** (sweep).

Two of their findings are load-bearing for this choice: the fit quality is
more sensitive to the blade and hull drag coefficients than to any other
parameter, and allowing slip at all is *necessary* — a non-slipping blade
(their `C_D = 1`) cannot reproduce the data.

What this buys over the prescribed half-sine: the force now depends on boat
speed, so the loop is closed. The model reproduces, without being told to,
the retarding force at the catch and the wash-out at the finish — at both
ends the oar's angular rate passes through zero, the boat's own speed
dominates the slip, and the blade is dragged rather than driving. Through
mid-drive it gives 66–81% blade efficiency, bracketing the fixed `0.78`
factor it is designed to replace.

Their **Model 2** resolves lift and drag against angle of attack, after
Wang, Birch & Dickinson's hovering-insect-wing treatment. Not implemented.
[H10] separately bounds the error the pure normal-force assumption carries
at about 18% of blade losses.

**Status: implemented and unit-tested, not yet wired into the force path.**
The whole regression suite is pinned to the prescribed-profile numbers, and
switching the force model changes every one of them; that swap wants to be
its own change with its own re-validation.

### Depth of water around the blade

Two effects, acting in opposite directions, both implemented on
`BladeModel`.

**Ventilation (blade cover).** A blade whose top edge sits at the surface
draws air down its low-pressure face, collapsing the pressure difference it
works by. Kleshnev puts the optimum at about **half a blade width of water
over the blade** and reports that modelling favours deeper immersion over
holding the blade at the surface. W. C. Atkins,
[*Blade Immersion Depth vs. Puddles*](http://www.atkinsopht.com/row/bladepth.htm),
argues the same from the other side: at constant propulsive force, deeper
immersion needs *less* slip, which is the definition of a more efficient
blade.

`immersion_factor(cover) = 1 − exp(−k·cover/W)`, with `k = 4.605` set so
that half a blade width returns 0.90. **This is a shape, not a fit** — it
respects monotonicity and saturation near the reported optimum, and nothing
more. No published force-versus-immersion curve for a rowing blade was
found. Over-immersion is not penalised here: Kleshnev's 3.5% speed loss for
six degrees of extra blade depth is borne by the shaft and the vertical
handle force, not the blade face.

**Blockage (water depth).** The blade must push water around itself, and
with the free surface above and the bed below that flow is confined to the
water column. Maskell's bluff-body correction,
`C_D(confined) = C_D(1 + m·σ·C_D)` with vertical blockage `σ = W/h` and
`m = 2.5`:

| water depth | blockage factor |
|---|---|
| deep | 1.000 |
| 4 m | 1.172 |
| 3 m | 1.229 |
| 2 m | 1.344 |

**Read as an upper bound.** At `σ = 0.05` this gives 1.14 against the
"under 10%" reported for bluff bodies in ducts. Two reasons it should be
conservative: that figure is for all-round confinement, whereas a blade is
confined only vertically; and `m = 2.5` is the generic bluff-body constant,
not one measured for a blade. Matching the cited datum exactly would need
`m ≈ 1.8`. The standard constant is kept and the discrepancy stated rather
than tuned away.

[ST09] models the blade flush with the surface but does not vary immersion
parametrically, so it does not settle either question.

### [B09] Brearley (2009)
*A method of improving oar efficiency.* **ANZIAM J. 50** 534–540.
[PDF](http://bionics.seas.ucla.edu/education/Rowing/Math_Model_2009_05.pdf)

Blade efficiency is worse at large oar angles, so the traditional stroke —
much larger angle at the catch than at the finish — wastes more at the
catch than it need. Directly relevant to the asymmetric
`OarAngleSweep(catch=+55°, finish=−35°)` used here.

### [K07] Kleshnev, *Propulsive efficiency of rowing*
Reports that improving **blade** efficiency offers 3–5% performance gain
against 0.5–0.8% for boat-velocity efficiency — i.e. the blade model
matters more than the fluctuation this document opens with.

### Other
- **Millward (1987)**, *A study of the forces exerted by an oarsman and the
  effect on boat speed*, J. Sports Sci. 5 93–103 — [F09]'s air-resistance
  reference and an early force-measurement source.
- **Affeld, Schichl & Ziemann (1993)** — resistance ∝ V^1.8 rather than V²,
  cited by [HF09].

---

## 7b. River fields (depth, current, channel)

`coxswain/river/` holds the spatial description a route optimisation needs:
`DepthField` (feeds the shallow-water correction), `CurrentField`
(depth-averaged water velocity), and `Course` (centreline, navigable
half-width, station/offset parameterisation).

`RowingSimulator(boat, course=...)` looks depth up at the boat's position
each step and takes hydrodynamic forces on the **water-relative** velocity
while keeping the trajectory and crew reactions in the ground frame.
Measured effect for an eight at rate 32, from 5.355 m/s deep and still:

| condition | mean ground speed | vs deep, still |
|---|---|---|
| 3 m depth, still | 4.683 m/s | −12.6 % |
| 3 m depth, 0.5 m/s head | 4.257 m/s | −20.5 % |
| 3 m depth, 0.5 m/s tail | 5.118 m/s | −4.4 % |

**No Charles bathymetry is loaded.** `charles_river_sketch()` is a
caricature with invented depths and geometry; `Course.is_survey` is
`False` for it and `Course.require_survey()` raises, so it cannot quietly
become a routing number. Replacing it needs real soundings projected
through `local_tangent_plane`.

---

## 9. Charles River data

### Bathymetry — [CRAB18]
C. Zimba, M. J. Sacarny, M. Yoder, B. Bray and C. Chryssostomidis,
*Changes in the Depth of the Lower Charles River Basin*, Charles River
Alliance of Boaters / MIT Sea Grant (2018).
[Report](http://www.charlesriverallianceofboaters.org/docs/ChangesintheDepthoftheLowerCharlesRiverBasin.pdf)
· [Chart KMZ](http://www.charlesriverallianceofboaters.org/chart/charles.kmz)
· [Method](https://repository.library.noaa.gov/view/noaa/46058/noaa_46058_DS1.pdf)

The **first detailed bathymetric chart of the river since 1902**. Sonar
survey of the Lower Charles, New Charles River Dam to Watertown Dam
(~14.5 km), 2016–17: Lowrance HDS-7 broadband sonar with Point-1 GPS on
track lines 9–18 m apart, processed in ReefMaster, corrected for transducer
depth.

Extracted to `data/charles_isobaths.csv`: 12,164 contour vertices at
1-foot intervals, **0.30 m to 10.36 m**, median 3.66 m. Depths are below
the basin's normal pool, which the New Charles River Dam holds nearly
constant — so they are already depth below the surface the boat floats on,
with no tidal reduction. A tidal estuary would not be this kind.

CRAB also document four shoaling areas of concern — Muddy River / Stony
Brook outlet, Magazine Beach, Faneuil Brook outlet, Sunset Bay — and note
the up-river Community Rowing docks that "were in 18–24 inches of water
when the boathouse opened in 2008 are now resting on mud". The model
reproduces that: median depth along the traced deep channel is 3.32 m
overall but only 1–2 m at the Watertown end.

### Discharge — USGS 01104500
[CHARLES RIVER AT WALTHAM, MA](https://waterdata.usgs.gov/monitoring-location/USGS-01104500/),
daily statistics over the **1931–2026** period of record, condensed to
`data/charles_discharge_waltham.csv`. Waltham is immediately above the
reach; it misses a small ungauged inflow below, a few percent for this
catchment, and is the best available proxy.

| month | median (m³/s) | p10 | p90 | max |
|---|---|---|---|---|
| March | 15.27 | 7.51 | 30.52 | 58.4 |
| **October** (HOCR) | **2.84** | 0.81 | 11.44 | 33.2 |
| August | 1.92 | 0.50 | 8.25 | 39.5 |

### Flow model — continuity
The lower Charles is an **impoundment**, not a free-flowing river: the dam
sets the level and the water is near slack. Flow speed therefore comes from
continuity, `U(s) = Q / A(s)`, with `A` integrated across the channel from
the surveyed bathymetry. Two measured inputs, no fitted parameters. It is
the mechanism CRAB themselves invoke: "As water over a given cross section
becomes shallower, water flow velocity must increase."

**Manning's equation is deliberately not used.** It needs an energy slope,
and the slope across an impounded basin is neither measured here nor
meaningfully constant.

Result, along the traced deep channel:

| condition | Q (m³/s) | flow speed |
|---|---|---|
| October median | 2.84 | 0.4–5.2 cm/s |
| October p90 | 11.44 | 1.6–20.8 cm/s |
| March median | 15.27 | 2.1–27.8 cm/s |
| October max of record | 33.2 | 4.6–60.4 cm/s |

**At race conditions the Charles is effectively slack** — 5 cm/s against a
5 m/s shell is one part in a hundred. The current only earns its place in a
route calculation in a wet year, and then it concentrates where the section
is smallest, which is exactly where a line choice exists to be made.

### Known limits
- The centreline is the **thalweg** — the deepest water, traced through the
  survey — not a surveyed navigation channel and not a race line. Callers
  wanting a specific line should pass their own centreline.
- Channel half-width is a uniform 55 m placeholder; the true navigable
  width varies and pinches hard at the bridges.
- Discharge is monthly climatology, not a live feed. The USGS instantaneous
  service is one call away when a specific day matters.

---

## 8. Open questions

Ordered by how much they would change a result.

1. **Boat velocity fluctuation is ~1.7× measured** (§4). Traced to crew
   CoM velocity amplitude; cause not yet identified. The decisive missing
   datum is a published rower centre-of-mass excursion relative to the
   boat.
2. **Charles bathymetry is loaded** (§9); what is missing is a surveyed
   navigable width and a race line. The thalweg stands in for both.
3. **Near-critical shallow-water amplification is capped by choice, not
   measurement** (§6). Affects any Charles route optimisation that runs
   near `Fr_h = 1` — which 2–4 m water and a racing eight do.
4. **The blade model is implemented but not wired in.** `BladeModel`
   ([CR06] eq. 11) is tested and ready; the force path still uses the
   prescribed half-sine and the fixed 0.78 efficiency. Switching over
   re-baselines every regression number, so it is its own change.
5. **Resistance is quasi-steady.** [D11] measured substantial
   under-prediction at realistic oscillation frequencies from unsteady
   viscous effects.
6. **The oar sweep is piecewise-linear in stroke phase**, so the handle
   path has a velocity discontinuity at catch and finish. Now that the
   hands are constrained to the handle, that discontinuity propagates into
   crew accelerations; it is currently smoothed only by the Fourier
   truncation of the hand track.
7. **Added mass and wave damping are absent.** [F09] §6.4 computes them
   from a radiation problem and reports ~10% of total energy dissipation
   from secondary motions; their published matrices for a 4x are in the
   paper if a first approximation is wanted.
7. **The 1x catalog rig runs the arms to full extension** mid-drive,
   needing ~6 cm of modelled shoulder protraction. Either the rig geometry
   or the reference athlete's proportions want revisiting.
