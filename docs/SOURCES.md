# Sources

Every number in this model that is not derived from first principles comes
from one of the references below. Each entry records what was taken from
it, where that lands in the code, and — where it matters — what the source
does *not* settle.

Local copies of the open-access PDFs are not committed (they are
copyrighted); the URLs below were live as of August 2026. The
[UCLA Bionics rowing archive](http://bionics.seas.ucla.edu/education/Rowing/)
mirrors many of them and is the single most useful starting point.

[1. Boat dynamics](#1-boat-dynamics) ·
[2. Rower kinematics](#2-rower-kinematics) ·
[3. Anthropometry](#3-anthropometry) ·
[4. Boat velocity fluctuation](#4-boat-velocity-fluctuation) ·
[5. Hull hydrodynamics](#5-hull-hydrodynamics) ·
[6. Shallow water](#6-shallow-water) ·
[7. Oar and blade](#7-oar-and-blade) ·
[8. River fields (depth, current, channel)](#8-river-fields-depth-current-channel) ·
[9. Charles River data](#9-charles-river-data) ·
[10. Steering authority](#10-steering-authority) ·
[11. Open questions](#11-open-questions)

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

### A better hull source, not yet obtained

**L. Lazauskas**, *Rowing shell drag comparisons* (Dept. Applied
Mathematics, University of Adelaide) compares the calm-water resistance of
real racing shells using Michell's integral for wave resistance, the ITTC
1957 line for skin friction and an empirical form-drag term — the same
decomposition [F09] uses, applied to actual hull forms rather than a
parametric approximation. Indexed at
[IAT Leipzig](https://iat.uni-leipzig.de/datenbanken/iks/sponet/Record/4000726).

The hull offsets behind it would replace this model's parametric hull with
measured geometry. `cyberiad.net`, which hosted the data, currently returns
403 to automated requests, so it has **not** been obtained.

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

### Wiring the blade model in — and what it revealed

`Boat.blade_model` is optional and **off by default**. When set, the
simulator replaces the oar's fixed `blade_efficiency = 0.78` with the
instantaneous value from slip and water depth. The rower still pulls the
measured handle force; what changes is the fraction of it that becomes
thrust.

Turning it on exposed an inconsistency that nothing else in the model
could see. Force-weighted blade efficiency over the drive, at 5.5 m/s:

| sweep flatness | peak oar rate (rad/s) | blade efficiency |
|---|---|---|
| 0.00 (raised cosine) | 3.31 | 0.73 |
| **0.30** | 2.95 | **0.82** |
| 0.50 | 2.71 | 0.88 |
| 1.00 (linear) | 2.11 | 0.90 |

Kleshnev reports 0.80–0.85 for good crews. **The default raised-cosine
sweep is too peaked**: at 1.57× the mean angular rate it drives the blade
through the water faster than the boat runs, so it slips hard through
mid-drive. Flatness near 0.30 reproduces the measured figure.

That is an independent constraint on the oar kinematics which the blade
model supplies and no other part of this model does. The default is left
at 0.0 only because the regression suite is pinned to it.

#### The double count, found and fixed

The 14% speed loss recorded here originally was a bookkeeping error, not
physics. The lumped `blade_efficiency = 0.78` is a measured *total* blade
efficiency at nominal cover, so it already carries the immersion and
ventilation loss. Multiplying the slip efficiency by `depth_factor` —
which is `immersion_factor` × `blockage_factor` — charged for that loss a
second time, costing a flat 10% (`immersion_factor()` returns 0.90) on top
of whatever the slip model said.

Both paths now apply **`blockage_factor` only**. Immersion is applied only
when a caller sets `Boat.blade_cover` explicitly and thereby takes over the
bookkeeping. `RowingSimulator._blade_efficiency` and
`coxswain.river.sixdof._blade_efficiency` agree term for term.

#### What the blade model is for

Not a correction to the mean — for that, a constant would do. It is the
*variation* the constant cannot carry:

| speed | flatness 0.00 | flatness 0.30 |
|---|---|---|
| 4.6 m/s | 0.651 | 0.728 |
| 5.2 m/s | 0.707 | 0.793 |
| 5.8 m/s | 0.754 | 0.842 |

Force-weighted over the drive; instantaneously it ranges 0.40 to 0.89
within a single drive at fixed speed. Two things follow that a fixed
factor cannot represent:

- **Blade efficiency rises with boat speed.** A slower crew slips more for
  the same handle force. So it is not a boat-independent constant, and a
  masters eight at 4.6 m/s genuinely has a worse blade than an open eight
  at 5.8 — which is a real term in the power budget, not a modelling
  artefact.
- **It is lowest near the catch**, where the handle force is highest and
  where `cos(theta)` already wastes most of the blade load laterally. The
  two losses coincide, which is exactly [B09]'s argument.

Because efficiency depends on the boat's *speed*, it is not a function of
time and cannot be folded into the periodic fits. In the CasADi path it is
therefore evaluated symbolically from the state at every collocation point,
with the oar angle and rate carried as their own Fourier fits.

#### Why the efficiency is low: the sweep over-drives the blade

Decomposing the slip by sign, force-weighted over the drive at 5.06 m/s,
separates the two ways a blade can waste energy — moving through the water
*faster* than it needs to (`slip < 0`) and failing to anchor (`slip > 0`):

| flatness | blade efficiency | over-drive (m/s) | under-drive (m/s) |
|---|---|---|---|
| 0.00 | 0.695 | **1.614** | 0.169 |
| 0.30 | 0.779 | 1.136 | 0.099 |
| 0.45 | 0.837 | 0.827 | 0.058 |
| 0.60 | 0.864 | 0.678 | 0.043 |

**Over-driving is ~90% of the loss.** Over one drive the blade sweeps a
3.58 m arc while the boat travels 3.77 m, so on average the blade nearly
anchors — the loss is not a shortage of arc, it is the *rate* being wrong
moment to moment. A raised cosine peaks at 1.57x the mean angular rate, so
mid-drive the blade is driven through the water 2.6 m/s faster than it has
to be, and near the ends it stalls to a tenth of mean rate and slips at
almost full boat speed.

At flatness 0.30 the slip model returns **0.779** at the model's own
operating speed, against the **0.78** constant it replaces — and the
thrust impulse ratio is **1.005**. The two independent calibrations agree
to within half a percent. That is meaningful: the lumped constant was
right, and the sweep should be flat at about 0.30.

#### What is still unexplained

Switching the blade model on at flatness 0.30 nonetheless costs speed —
5.10 to 4.28 m/s — despite that thrust-neutral steady-state balance. The
gap is the *within-stroke* speed variation. Efficiency is evaluated at the
instantaneous hull surge, which swings 54% peak-to-peak; the boat is
slowest at the catch, which makes the mid-drive slip more negative and so
costs more than the constant-speed estimate predicts. The effect is real
and correctly modelled, but its size rides on the model's speed
fluctuation, which is itself the known open discrepancy of section 4
(model 55-65%, measured 37.5-50%). **Until that is closed, the blade
model's absolute level cannot be calibrated**, so it stays opt-in.

This is the first place the fluctuation gap has been shown to change a
physical prediction rather than just a diagnostic number.

#### The flatness question is still open

The table above is also an independent constraint on the oar kinematics
that nothing else in this model supplies. Kleshnev reports 0.80–0.85 for
good crews, which flatness 0.30 reproduces at racing speed and the default
raised cosine (flatness 0) misses low. **The default is still 0.0.**
Changing it is a substantive change to the crew kinematics that would
re-pin the regression suite, and it should be made on the strength of a
measured oar-angle trace rather than on one indirect constraint.

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

## 8. River fields (depth, current, channel)

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

### Lateral flow distribution

Section-mean `Q/A` is not what a boat feels: water runs fastest in the deep
channel and slowest over the shoals. That spread is the entire reason a
line choice exists — upstream a crew wants the slack near the bank, coming
down they want the thread of the current.

The distribution follows the standard conveyance argument. Locally Manning
gives `u = h^(2/3) S^(1/2) / n`, so across a wide section

```
u(y) = Q · h(y)^(2/3) / ∫ h(y)^(5/3) dy
```

and the slope `S` and roughness `n` **cancel**. That is what makes it
usable here: the *shape* needs only the surveyed bathymetry, the
*magnitude* is still pinned by measured discharge, and `∫ u h dy = Q`
holds exactly at every section. Manning is used for the lateral shape,
where its unknowns drop out, and still not for the magnitude, where they
would not.

Measured spread at a shoaled section, October maximum of record:

| position | flow speed |
|---|---|
| deep channel | 19.1 cm/s |
| centreline | 13.3 cm/s |
| far bank | 6.6 cm/s |
| *section mean (what the uniform model gives everywhere)* | *16.4 cm/s* |

A factor of ~3 across the section. The uniform model overstates the adverse
current at the bank by 2.5×, which is exactly the error that would hide a
line choice from a route optimiser.

### Known limits
- The centreline is the **thalweg** — the deepest water, traced through the
  survey — not a surveyed navigation channel and not a race line. Callers
  wanting a specific line should pass their own centreline.
- Channel half-width is a uniform 55 m placeholder; the true navigable
  width varies and pinches hard at the bridges.
- Discharge is monthly climatology, not a live feed. The USGS instantaneous
  service is one call away when a specific day matters.

---

## 10. Steering authority

The trajectory optimiser could not fly the Charles on rudder alone, which
turned out to be physics rather than a solver problem.

### What the river demands

Curvature of the extracted channel centreline (§9), which is derived from
the survey rather than assumed:

| | required turn radius |
|---|---|
| tightest 1% of the reach | ≤ 133 m |
| tightest 5% | ≤ 177 m |
| tightest 10% | ≤ 222 m |
| median | 646 m |

The six tightest bends need **103–146 m**, i.e. 2.0–2.9 °/s at 5.2 m/s.

### What the boat can do

Steady turn radius, measured from the full 6-DOF model:

| control | yaw rate | radius |
|---|---|---|
| rudder 4° | 0.35 °/s | 853 m |
| rudder 8° | 0.71 °/s | 421 m |
| rudder 12° (full) | 1.15 °/s | **259 m** |
| split 10% | 0.28 °/s | 1063 m |
| split 20% | 0.50 °/s | 592 m |
| split 30% | 0.77 °/s | 385 m |
| **rudder 12° + split 30%** | **2.28 °/s** | **130 m** |

**19% of the reach is tighter than full rudder alone can hold.**

### The conclusion

A rudder-only model of an eight silently cannot fly the Charles. Solving a
1.6 km leg containing one of the tight bends:

| | result |
|---|---|
| rudder only | **fails** — IPOPT error, min clearance 0.0 m (leaves the water) |
| rudder + split | optimal, min clearance 8.5 m, max split 0.15, 15 s quicker |

So the port/starboard pressure split is not a refinement on top of
steering. On a river it *is* the steering, and the rudder trims it.

Implemented as `Coxswain.pressure_split` in the 6-DOF model and as the
second control of the reduced model. Applied symmetrically — half added on
one side, half removed on the other — so it is a pure yaw couple and
cannot be used to accelerate.

### A caveat on the reduced model

The two controls combine **super-additively** in the full model: 2.28 °/s
against the 1.92 a linear sum predicts, because the sideslip each induces
raises the other's effectiveness. The reduced model is linear in both, so
it predicts 164 m where the full model gives 130 m. That is conservative,
which is the right direction for a trajectory that has to be flyable — but
a solution sitting near the steering limit should be checked back in the
full model.

### Not resolved

Whether the *rudder* coefficient itself is right. The modelled rudder is
108 cm² at a 6.6 m arm, giving 230 N·m at 12° and 5.2 m/s. No published
rudder-force or turning-circle measurement for a racing eight was found to
check it against; the qualitative descriptions that do exist ("it takes a
stroke or two before it starts to turn") are consistent with a slow-turning
boat but do not pin a number. The split coefficient is on firmer ground,
being derived from oarlock forces the model already reproduces.

---

## 10b. Pacing and power

Holding the crew at constant power removes the decision a crew actually
makes: how hard to push, and where. On a head course that is not separable
from steering, because a pressure split spends thrust on a couple — so the
optimiser must be able to answer *"push through the bend, or ease and
steer"* rather than having the first half of it fixed.

### The model

The **critical-power model**, the standard two-parameter description of
endurance performance:

- **Monod, J. & Scherrer, J. (1965)**, *The work capacity of a synergic
  muscular group.* Ergonomics 8, 329–338.
- **Morton, R. H. (2006)**, *The critical power and related whole-body
  bioenergetic models.* Eur. J. Appl. Physiol. 96, 339–354.

Power `CP` is sustainable indefinitely; anything above it draws on a finite
anaerobic capacity `W'` which depletes at `P − CP`. `W'` is carried as a
sixth state, constrained non-negative, so the optimiser cannot spend energy
the crew does not have. Without that budget minimum time is trivially "row
flat out everywhere".

Values used: `CP = 3040 W` for the crew (≈380 W per rower, a club eight;
world level is nearer 450) and `W' = 176 kJ` (≈22 kJ per rower). Both are
order-of-magnitude right from the literature but **not fitted to a specific
crew** — they set the shape of the pacing answer and should be measured
before any number is quoted to athletes.

### What it changes

On a 2.6 km Charles leg:

| | |
|---|---|
| free pacing | 618.5 s |
| constant power at CP | 637.3 s (**+18.8 s**) |
| constant power at 1.1 CP | infeasible — exhausts `W'` before the finish |

Nearly 3%, and previously unmodellable.

`W'` drawdown is front-loaded: 176 kJ → 162 at a quarter distance → 98 at
half → 5.6 at three-quarters → 0 at the line. The last quarter is rowed at
CP.

### The counterintuitive bit

The crew pushes **harder** in the bends, not easier:

| | power | rudder | split | yaw rate |
|---|---|---|---|---|
| in the bends | 1.057 | 1.00 | 0.20 | 1.07 °/s |
| on the straights | 1.023 | 0.23 | 0.00 | 0.21 °/s |

Because the split spends thrust on a couple, holding speed through a turn
costs extra power. The effect is weak — 3%, correlation +0.12 — so it is
worth stating as a tendency rather than a rule, but it is the opposite of
what I would have guessed.

---

## 11. Open questions

Ordered by how much they would change a result.

Closed since the last revision: the navigable channel is now **derived from
the survey** rather than assumed (§8, §9) — an alpha-shape water mask and a
distance transform give the centreline and half-width, and the centreline is
100% navigable against 73.7% for the old thalweg. And steering authority is
resolved (§10): the rudder alone cannot fly the river, and the crew pressure
split that closes the gap is now a modelled control.

1. **Boat velocity fluctuation is ~1.7× measured** (§4). Traced to crew
   CoM velocity amplitude; cause not yet identified. The decisive missing
   datum is a published rower centre-of-mass excursion relative to the
   boat.
2. **The rudder coefficient itself is unverified** (§10). The modelled
   rudder gives 230 N·m at 12° and 5.2 m/s; no published rudder-force or
   turning-circle measurement for a racing eight was found to check it
   against. The crew-split coefficient is on firmer ground, deriving from
   oarlock forces the model already reproduces. Now that steering is known
   to be marginal against the tightest bends, an error here matters.
3. **Near-critical shallow-water amplification is capped by choice, not
   measurement** (§6). Affects any Charles route optimisation that runs
   near `Fr_h = 1` — which 2–4 m water and a racing eight do.
4. **The blade model is wired in but not calibrated, and stays opt-in**
   (§7). It exposed that the raised-cosine oar sweep is too peaked — a
   real finding — but its absolute level double-counts ventilation against
   the lumped 0.78 it replaces. Needs a measurement separating slip from
   ventilation.
5. **Resistance is quasi-steady.** [D11] measured substantial
   under-prediction at realistic oscillation frequencies from unsteady
   viscous effects.
6. **The oar sweep is piecewise-linear in stroke phase**, so the handle
   path has a velocity discontinuity at catch and finish. Now that the
   hands are constrained to the handle, that discontinuity propagates into
   crew accelerations; it is currently smoothed only by the Fourier
   truncation of the hand track.
8. **Added mass and wave damping are absent.** [F09] §6.4 computes them
   from a radiation problem and reports ~10% of total energy dissipation
   from secondary motions; their published matrices for a 4x are in the
   paper if a first approximation is wanted.
9. **The 1x catalog rig runs the arms to full extension** mid-drive,
   needing ~6 cm of modelled shoulder protraction. Either the rig geometry
   or the reference athlete's proportions want revisiting.

## 12. Crew balance, and what a sweep rig does to it

### The reduction that was there

Balance was applied as a saturated PD couple about the hull `x` axis,
added straight to the moment vector. `BalanceController`'s docstring
justified it: "the underlying handle-height trim is equal and opposite
across the boat, so it produces no net force."

The force claim is true and is now tested. The couple claim is not.

### A crew cannot apply a pure roll couple

What a crew can do is change handle height. That loads the oar as a lever
about its oarlock and puts a **vertical force at the rigger**. The riggers
are at fixed points on the hull, so the moment the crew produces is
whatever that particular set of point forces makes.

An eight is rigged alternately, so the four port oarlocks and the four
starboard ones sit at different mean longitudinal stations — by exactly
one seat spacing:

| | mean oarlock x |
|---|---|
| port | −0.34 m |
| starboard | +0.88 m |
| offset | **1.22 m** = one seat |

A balance effort pushing one side down and lifting the other therefore
applies four downward forces sitting 1.22 m from the four upward ones.
That is a couple in the vertical–longitudinal plane as well as the
intended one in the transverse plane. Writing the moment of a vertical
force `F_z` at `(x, y)` as `(y F_z, −x F_z, 0)` and summing over the rig:

| boat | roll per unit | pitch per unit | **pitch coupling** |
|---|---|---|---|
| eight | 6.80 | 4.88 | **0.718** |
| coxed four | 3.32 | 2.44 | **0.735** |
| single scull | 1.60 | 0.00 | **0.000** |

**Balancing a sweep boat pitches it, by about 0.72 of the roll moment.**
The sculler's exact zero is the control: two oarlocks at the same station
and opposite `y` do make a pure roll couple, which is what the old model
assumed for every boat. The coupling is a property of sweep rigging, not
a modelling artefact.

`coxswain.crew.balance.BalanceRig` carries the three coefficients. They
are exact geometry — linear in the demand, no fit and no smoothing — so
CasADi differentiates them trivially, and both the numpy simulator and the
symbolic 6-DOF model take their balance loads from the same object. A test
asserts the two agree.

### What it changes

Over 20 s at rate 32, applying balance through the riggers rather than as
a pure couple:

| | pitch swing | roll swing | mean speed |
|---|---|---|---|
| eight, pure couple | 0.5152° | 1.0569° | 5.162 m/s |
| eight, through riggers | **0.4991°** | 1.0566° | 5.162 m/s |
| single scull, either | 0.9533° | 0.0000° | 4.411 m/s |

A 3.1% reduction in pitch swing, no change to roll or speed, and the
sculler bit-identical. Modest, but it moves the right way against the
model's known pitch-swing excess, and it was obtained from geometry rather
than by tuning anything.

The roll response also changes very slightly (about 0.01%), because the
mass matrix couples roll and pitch and the new pitch moment feeds back.
That feedback is part of what a pure `x` couple discarded.

### Still open: the authority is not phase-dependent

`max_moment = 4000 N m` applies across the whole stroke. Physically it
should not. On the drive the blade is buried, so the water reacts the
handle force and the rower can genuinely push against it; the oarlock then
carries `(1 + inboard/outboard)` times the handle force. On the recovery
the oar is a free lever in air, so a sustained vertical handle force is
impossible — only the oar's own inertia and the rowers' lateral body lean
are available, which is perhaps a few hundred newton-metres against
roughly 1500 on the drive.

That is why a boat is hardest to balance on the recovery, and the model
does not currently know it. Fixing it needs a sourced figure for the
vertical handle force a rower can apply while also pulling, which has not
been found; the geometry above is ready for it when it is.


## 13. The rowers were not holding their oars sideways

`Boat.hand_positions` fed the oar moment arms and, through them, every
oar-generated force and moment on the hull. It returned `y = 0` for every
seat, at every instant, in every boat.

Two independent causes, which is why it survived:

1. The rower's joint chain is **sagittal**. `joint_positions` solves a
   planar linkage in the boat's centreplane, so the hand it returns has no
   lateral component at all.
2. `hand_positions` batches seats into kinematics groups, evaluates the
   group leader once, and offsets the others. It offset **`x` only**, so
   even a lateral position would have been copied unmirrored to every seat
   in the group.

The consequence is not subtle. A sweep handle sweeps a wide lateral arc:
on this rig `+0.187 m` at the catch through `-0.280 m` at mid-drive, and
mirrored between port and starboard. The oar's lateral moment arm, taken
from the oarlock at `y = ±0.85 m` to the hand, was therefore a constant
0.85 m against a true value swinging from 0.71 m to 1.13 m:

| stroke fraction | true lateral arm | with hands at y=0 | error |
|---|---|---|---|
| 0.05 (catch) | 0.709 m | 0.850 m | **−19.9%** |
| 0.25 (mid-drive) | 1.130 m | 0.850 m | **+24.7%** |

**The error changes sign through the drive**, so the oar yaw moment was
distorted in *shape*, not merely scaled — and that moment is what the
entire steering model rests on.

The fix takes `x` and `z` from the joint chain, which already agree with
the rig geometry exactly, and the lateral component from the oar. For a
sculler the two handles mirror and the mean is on the centreline, which is
the right answer there and was the only case the old code got right.

### A side effect worth recording

`yaw_per_split` needed 32 Fourier harmonics to hit tolerance and now needs
16. The constant moment arm had been imposing a spurious step at the catch
on top of the real discontinuity; removing it halved the spectrum. A model
error showing up as a harder Fourier fit is a useful smell.


## 14. Bridges

Six bridges cross the racing reach, and they are the tightest constraint on
the course. `charles.BRIDGES` held only their centre coordinates, which is
enough to label a plot and nothing else — an optimised trajectory was free
to pass straight through a pier.

`coxswain.river.bridges` makes them constraints.

### What is measured and what is not

**The deck geometry is real.** Each bridge's carriageway centreline comes
from the OpenStreetMap Overpass API, projected onto the same local tangent
plane as the depth survey:

| bridge | deck span | navigable opening |
|---|---|---|
| Eliot Bridge | 123.2 m | 63.0 m |
| Larz Anderson | 84.6 m | 61.1 m |
| Weeks Footbridge | 108.7 m | 84.2 m |
| Western Avenue | 152.0 m | 84.2 m |
| River Street | 134.6 m | 77.9 m |

**The pier positions are not surveyed.** No published source for the arch
spans of these particular bridges was found. Rather than invent them, the
opening is derived from data that does exist: a gate is open exactly where
the bridge line crosses water the channel raster calls navigable. That
keeps a boat off the abutments and out of the shallows without pretending
to know where a mid-river pier stands. `BridgeGate.piers` subtracts any
piers that are supplied, so the constraint is ready for survey data.

The openings above are therefore **full water widths, not arch spans** —
the real constraint at Weeks and Anderson is tighter than this, and those
are exactly the two bridges where Head of the Charles crews collide.

### Why a gate rather than an obstacle

A bridge is a line the boat crosses once, at a point that must lie inside
an opening with clearance. That gives the optimiser one scalar constraint
per bridge evaluated at a single crossing, rather than a keep-out region it
must be excluded from along the whole trajectory — cheaper and better
conditioned.

Clearance must be compared against the **blade tips**, not the hull. An
eight is 0.57 m in the beam, but its oarlocks sit at ±0.85 m and the blades
reach 2.56 m beyond that.


## 15. Phase-dependent balance authority

### [D96] The primary source

**"Balance of Racing Rowing Boats"**, Furnivall Sculling Club, 1996;
lightly revised 2013 PDF, 15 pp. Hosted at
<https://eodg.atm.ox.ac.uk/user/dudhia/rowing/physics/Balance_of_Racing_Rowing_Boats_v3.pdf>

The article extends classical ship hydrostatic stability theory to racing
shells. Its conclusions are stated flatly and are the foundation of this
section:

> "No racing rowing boat is statically stable with the crew rigid in the
> boat and the oars off the water. Not even close, stationary or moving."

and, on the recovery specifically:

> "Lifting or dropping the hands during the recovery to keep the blades at
> a constant height off the water with a non-flat boat tends to make it
> even less flat. The system has positive feedback."

Its Table 1 gives, for an eight with crew aboard, all referred to the
waterline: metacentre 13 cm, centre of gravity 28 cm. The centre of
gravity is 15 cm **above** the metacentre, hence unstable.

### Independent reproduction

This model was built from de Leva (1996) segment inertias and a
triangulated hull mesh, with no reference to [D96]. It agrees:

| quantity | this model | [D96] Table 1 |
|---|---|---|
| CoG above waterline, eight | **25.4 cm** | 28 cm |

and computing the net static roll moment from buoyancy against gravity
with the crew rigid gives, for every boat in the catalogue, a **positive**
(destabilising) stiffness:

| boat | static roll stiffness | roll inertia | e-folding time |
|---|---|---|---|
| eight | **+2122 N·m/rad** | 101 kg·m² | 0.218 s |
| coxed four | +1166 | 50 | 0.207 s |
| single scull | +329 | 11 | 0.185 s |

`coxswain.crew.balance.static_roll_stiffness` computes it;
`roll_divergence_time` turns it into the timescale.

### What the crew can actually do about it

The two stroke phases work by different mechanisms, and only one of them
is any good.

**Drive — the blade is buried.** A rower who changes handle height loads
the oar as a lever against water that pushes back, so the rigger carries
`(1 + inboard/outboard)` times the handle force. [D96]: "The oars can be
used to force the hull flat during the drive."

**Recovery — the blade is in the air.** There is nothing to push against.
The only reaction available is the oar's own inertia: the rower angularly
accelerates it about the oarlock and the reaction appears at the rigger.
The available angular acceleration is bounded by how far the hands can
travel vertically in the time available, so this is computed rather than
assumed. [D96] confirms the mechanism is weak — hand-height changes "can
only produce transient forces" — and rules out the alternative for crew
boats:

> "crew boats require you to get the spoons right off the water to clear
> the puddles coming down from rowers behind you, hence this strategy is
> not available"

which removes blade-skimming and its ground-effect assist, both of which
[D96] describes as powerful but available only to singles and pairs.

| boat | drive authority | recovery authority | ratio |
|---|---|---|---|
| eight | 1474 N·m | **32.6 N·m** | **2.2%** |
| coxed four | 720 | 15.9 | 2.2% |
| single scull | 346 | 7.9 | 2.3% |

The ratio is nearly constant because it is set by oar geometry and stroke
timing, which barely differ between classes.

The one physiological input is the vertical handle force a rower can apply
while also pulling, taken as 150 N. Instrumented-oar work calibrates
three-dimensional oarlock transducers over a 0–150 N blade-force range,
and the vertical component a rower can spare mid-drive sits at the low end
of what they produce horizontally, so this is deliberately conservative.
Oar mass is 2.7 kg, a composite sweep oar.

### The consequence, and why it matches what crews report

For the eight:

* recovery authority 32.6 N·m against a destabilising 2122 N·m/rad means
  the crew can still arrest **0.88° of heel** with the blades out — and no
  more;
* the recovery lasts 1.130 s against an e-folding time of 0.218 s, i.e.
  **5.2 time constants**, so an uncontrolled roll grows by a factor of
  **179** between the finish and the next catch.

Those two numbers close on each other. A boat set flat to within 0.006° at
the finish arrives at the catch at 1°. **The ~1° roll oscillation this
model produces is therefore not noise — it is the signature of an unstable
plant being caught once per stroke**, flattened on the drive, diverging
through the recovery, and caught again.

Replacing the constant 4000 N·m limit with this phase-dependent one raises
the simulated roll swing from 1.06° to 2.45°. The rowers' objection that
holding the boat during the recovery is the hard part is quantitatively
correct, and it is hard for a specific reason: they are not damping a
stable mode, they are catching a diverging one with 2% of the authority
they had a moment earlier.

### The criterion is bounded growth, not stability

Calling the boat "unstable" is true but sets the wrong control objective,
and it is worth being precise because the distinction changes what the
crew are being asked to do.

Closed loop, the boat does **not** need to be stable in any asymptotic
sense. Nothing needs to converge. The blades return to the water every
1.9 s at rate 32, and the drive re-establishes 1526 N·m of authority --
eighteen times what the recovery has. The requirement is only that roll
does not grow *too much* between the finish and the next catch:

    |phi(t_catch)| <= phi_max,   given |phi(t_finish)| and the growth over
    the recovery

which is a **finite-horizon boundedness** condition, not a stability one.
An unstable open-loop mode is perfectly acceptable provided the horizon is
short enough relative to its e-folding time and the margin is wide enough.

So the quantities that matter are:

| | eight |
|---|---|
| e-folding time of the open-loop roll mode | 0.218 s |
| recovery duration | 1.130 s |
| **growth factor over one recovery** | **×179** |
| heel the crew can arrest on the recovery | 2.27° |
| implied tolerance on heel at the finish | **0.013°** |

That last line is the operative number, and it is the one to design
against. It is not "make the boat stable"; it is "leave the finish flat to
within about a hundredth of a degree, or arrive at the catch outside what
you can hold." That is a demanding but finite requirement, and it explains
why the skill is trained as *set the boat at the finish* rather than as
*correct it during the recovery* -- by the time there is something to
correct, the authority to correct it is gone.

It also identifies the right lever. Growth over the recovery is
exponential in `recovery_duration / tau`, so anything that shortens the
recovery helps disproportionately. Sweeping the rating:

| rate | recovery | growth factor | tolerance on heel at the finish |
|---|---|---|---|
| 24 | 1.700 s | ×2464 | 0.0007° |
| 28 | 1.373 s | ×547 | 0.0036° |
| 32 | 1.130 s | ×180 | 0.0126° |
| 36 | 0.942 s | ×76 | 0.0351° |
| 40 | 0.790 s | ×38 | **0.0846°** |

**A boat is about 120 times more forgiving in balance at rate 40 than at
rate 24.** Nothing about that was put into the model -- it follows from an
exponential whose exponent is the recovery duration over a timescale set
by hydrostatics and inertia. It reproduces one of the most universally
reported facts in the sport: boats fall over at low rate, and feel solid
at racing rate. Crews and coaches usually attribute this to concentration
or to "having something to sit on"; this model says it is mostly that the
unstable mode has less time to run.

It is also a falsifiable prediction with an obvious experiment: measure
heel variance against rating for a fixed crew. The predicted scaling is
exponential in the recovery duration, not linear, and it should hold
regardless of crew skill -- skill sets the disturbance amplitude at the
finish, the rating sets the amplification.

### Bounded smoothing

The limit is physically a square wave. The phase-locked mesh puts nodes
exactly at the catch and the finish, where a square wave has no
derivative, so it is smoothed with a pair of logistic edges.
`PhaseAuthority.window_error` measures the departure from the square wave
away from the transition bands: **1.7% of drive authority** at the default
sharpness. The transition occupies about 0.19 s at rate 32, comparable to
the time a blade actually takes to enter and leave the water, so the
smoothing corresponds to something physical rather than being pure
numerical convenience.

### What this exposed: a double rotation in the buoyancy moment

Making the authority honest immediately broke the 6-DOF model — roll
diverged to 300°. The cause was a latent frame bug that the over-generous
4000 N·m had been masking.

`HullMesh.submerged` returns the centre of buoyancy in the **absolute**
frame: its own `buoyancy_moment` equals `cross(centre, force)` against an
unrotated vertical force, which only balances if the centre already
carries the attitude. `HullSurrogate` tabulated that value directly, and
`SixDofModel` then rotated it again by the full attitude — applying roll
and pitch twice.

At one degree of heel this turned an 18.16 N·m righting moment into
2.35 N·m. **It destroyed 87% of the hull's roll stiffness.** It was
invisible at zero roll, where every previous check had looked, and it
stayed invisible while the crew were credited with roughly 100× the
recovery authority they actually have.

The surrogate now stores the centre of buoyancy in the hull frame. The
reconstructed roll moment is −18.154 N·m against the exact mesh's
−18.161 N·m, and the numpy and CasADi paths agree on roll acceleration at
1° of heel to **0.02%**, against 14% before the fix.

This is the clearest case so far of the project's working assumption: an
over-generous parameter does not merely make one number wrong, it hides
errors elsewhere by absorbing them.


## 16. Trunk lean: the other recovery actuator, and the stronger one

The first version of §15 credited the recovery only to oar inertia. That
was too pessimistic, and the omission mattered: it made the boat unsittable
in simulation.

A rower who leans their upper body laterally moves a large mass on a long
lever. For the eight, 704 kg of crew sits above seat height with its
centroid 0.214 m above it, so:

| lean | lateral shift | roll moment | vs oar inertia |
|---|---|---|---|
| 1° | 3.7 mm | 25.7 N·m | 0.8× |
| **2°** | **7.5 mm** | **51.5 N·m** | **1.6×** |
| 3° | 11.2 mm | 77.2 N·m | 2.4× |
| 5° | 18.6 mm | 128.5 N·m | 3.9× |

**Trunk lean dominates hand height on the recovery.** With it included:

| boat | drive | recovery = oars + lean | ratio | heel it can hold |
|---|---|---|---|---|
| eight | 1526 N·m | 84.1 = 32.6 + 51.5 | 5.5% | 2.27° |
| coxed four | 744 | 39.8 = 15.9 + 23.9 | 5.4% | 1.96° |
| single scull | 351 | 13.1 = 7.9 + 5.2 | 3.7% | 2.27° |

Simulated roll swing: 1.06° with the old constant limit, 1.23° with
phase-dependent authority including lean, **2.45° with the oars alone**.
Trunk lean is what makes an eight sittable through the recovery.

This is [D96]'s "body inertia" mechanism, and it is the one it describes a
sculler actually using:

> "if you set the boat up at the finish and swing straight down the hull,
> your upper torso is not going anywhere very quickly and its inertia can
> be used as a reference point to sit the boat flat"

It also matches coaching practice better than the hand-height story does:
[D96] shows that lifting and dropping the hands to hold blade clearance is
*positive feedback*, whereas using the body is not.

Two caveats, both from [D96], and both reasons the term is bounded rather
than free:

* it is **transient**. Shifting mass laterally cannot fix a persistent
  list — "You cannot correct a consistently off-flat racing boat by
  permanently leaning slightly sideways to the uphill side, it will tend to
  reinforce the tilt, not reduce it."
* it moves the crew centre of gravity, which is the quantity that makes
  the boat unstable in the first place, so leaning hard raises the
  stiffness it is fighting.

The ratio is not class-independent: the oar term scales with the rig, but
the lean term scales with crew mass per oar, so a sculler gets
proportionally less from it. [D96] observes the consequence from the other
end — small boats depend on skimming the blades, which crew boats cannot
do.


## 17. Blades on the water: a modelled boundary, stated

The model assumes the blades are clear of the water throughout the
recovery. That is the intent of good rowing and, per the crew this work is
for, what elite crews achieve on nearly every stroke of the Charles. It is
not what always happens, and the difference is worth recording because it
sits exactly on the edge of what is modelled.

An unset boat produces two effects that are outside the current scope:

1. **Blades skipping the surface.** A feathered blade dragging or skipping
   along the water decelerates the boat. [D96] treats the same contact as
   a *balance aid* — the spoon carries some of the rigger's weight, and
   "by exact hand control you can scull a boat dead flat this way" — so
   the same event is simultaneously a stability gain and a speed loss.
   The model has neither.
2. **Truncated drives.** Once a blade is in the water the rower must go
   with it: an unset boat forces an early catch and less than full
   extension. That is a *timing and length* penalty, not a drag penalty,
   and it couples roll error directly into propulsion.

Both make roll error costly in a way the current model does not represent
— here, poor balance costs nothing but roll. Since §15 establishes that
roll is an unstable mode held by a few percent of the drive's authority,
and §16 that the margin is about 2°, these are the mechanisms by which
exceeding that margin would actually show up on a stopwatch. Adding them
would make balance a performance variable rather than a comfort one.

Recorded as a known boundary rather than a defect: the assumption is
stated, its violation is described, and its consequences are named.


## 18. Crew synchronisation as coupled oscillators

Every rower in this model is at exactly the same stroke phase. Real crews
are not, and the deviation is the natural disturbance source for
everything in §15.

**Naomi Ehrich Leonard** (Princeton, Mechanical and Aerospace Engineering,
and Applied and Computational Mathematics) works on phase models of
coupled oscillators and their connection to collective motion, including a
collaboration using improvisational dance as a model system for
in-the-moment collective decision making.
Publications: <https://naomi.princeton.edu/publications/>

The framework transfers to a rowing crew directly, and with an unusually
clean physical justification:

* each rower is a phase oscillator with phase φᵢ around the stroke cycle;
* the **coupling is physical, not perceptual**. Rower *i*'s motion moves
  the hull, and every other rower is rigidly attached to that hull. The
  coupling is therefore mean-field through a shared body, which is exactly
  the structure Leonard's collective-motion work treats — rather than the
  looser visual or auditory coupling usually assumed for human
  synchronisation;
* the stroke seat sets a phase reference, giving a directed coupling
  topology rather than an all-to-all one.

Why it matters here rather than being merely analogous:

1. **Phase spread is a roll disturbance.** §15 establishes that roll is an
   unstable mode with a 0.218 s e-folding time, held by ~5% of the drive's
   authority during the recovery, with about 2° of margin. Port/starboard
   phase asymmetry injects roll moment directly. Synchronisation quality
   therefore sets the disturbance amplitude for a plant that is already
   marginal — the two halves of the problem meet.
2. **Phase spread is a steering disturbance.** Asymmetric timing between
   sides is a yaw moment, i.e. exactly the standing bias whose correction
   the steering study of `studies/steering_strategy.py` compares.
3. **It is the timing half of the stochastic problem.** The stochastic
   optimal control goal already calls for stroke-to-stroke variation in
   rower *power*; variation in rower *timing* is the same question and has
   a ready-made dynamical formalism.

Status: not yet implemented. The model currently hard-codes a single crew
phase, so the first step is a per-rower phase offset in the crew field —
after which a Kuramoto-type coupling can be driven by the hull motion the
simulator already computes, which closes the loop between Leonard's
formalism and this model rather than bolting one onto the other.


## 19. How hand heights are actually used: learned trim

### The paradox

§15 quotes [D96] showing that adjusting hand height to hold blade
clearance is **positive feedback** — if the boat is down on your side and
you lift to keep your blade off the water, you unweight your rigger and it
goes down further. Yet crews plainly do use hand heights to set a boat,
and it plainly works. Taken together with §15's timing argument — the roll
mode e-folds in 0.218 s against a human reaction of 150–250 ms — hand
heights ought to be useless *and* harmful.

They are neither, because the control law rowers actually use is not the
one [D96] analyses.

### What rowers actually do

From the coxswain this work is for:

> "if you notice the boat dips down to your side at the catch, on the next
> stroke you will carry your hands slightly higher near the catch, and
> vice versa. Similarly if the boat is down to starboard at the finish,
> the next stroke starboards will raise their hands more at the finish and
> ports will lower theirs."

Every clause is a specification:

| clause | property |
|---|---|
| "on the next stroke" | error on cycle *k*, correction on cycle *k+1* |
| "near the catch" / "at the finish" | correction applied at the **same phase** as the error |
| "starboards raise, ports lower" | **antisymmetric by side** — a roll moment through the riggers |
| "you will carry" | **anticipatory**: applied before the error recurs |

That is iterative learning control [BTA06]. The stroke is a repetitive
process; the error at phase θ on cycle *k* updates the input at phase θ on
cycle *k+1*. The one-stroke delay that would be fatal in a feedback loop
is not a delay at all here — for a disturbance that repeats every stroke,
it is memory.

`coxswain.crew.trim.StrokeTrim` implements the standard update
`u ← Q u − L K e`, with `Q` the robustness filter of [BTA06].

### Why this resolves it

The learned trim never tries to arrest a roll excursion in progress. It
reduces the **initial condition** — the heel at the finish, from which the
recovery's ×180 amplification runs. §15 puts the tolerance there at about
0.013°, which nothing reactive could hold. Learning can, because it has as
many strokes as it needs and the disturbance it cancels is the repeating
part.

The two mechanisms are therefore complementary rather than contradictory:

| | [D96]'s reflex | learned trim |
|---|---|---|
| timescale | within stroke | across strokes |
| information | blade clearance now | heel at this phase last stroke |
| sign | positive feedback | negative, converging |
| what it fixes | nothing | the recurring finish-line error |

### It works, and it converges

Eight at rate 32, phase-dependent authority throughout:

| crew | roll swing | learned trim |
|---|---|---|
| no learned trim | 1.205° | — |
| novice (L=0.15, Q=0.75) | 1.054° | 13.9 N·m rms |
| club (L=0.40, Q=0.90) | 0.741° | 51.1 N·m rms |
| elite (L=0.60, Q=0.97) | **0.467°** | 87.9 N·m rms |

Monotone in skill, and the elite crew is 61% better than one with no
learned trim at all — using the *same* physical authority, the same boat
and the same disturbance. The difference is entirely what they have
learned about this boat.

Roll swing falls from 1.22° to 0.63° over fourteen strokes with a
practised crew and is still falling. **Skill is three physical parameters**
— learning gain, memory length, and the resulting trim effort — not a
fudge factor, and "hours of drilling together" becomes convergence of an
ILC memory. It also explains why the skill does not transfer: the memory
is specific to that crew, that rigging and that boat.

### Where the authority comes from

The corrections described are made **at the catch and at the finish** —
both points where the blade is in or entering the water, so the crew have
the drive's authority rather than the recovery's. Nothing special-cases
this: the command is saturated by the same `PhaseAuthority` window as the
reactive loop, and a test asserts that even an absurd learned command
cannot exceed the recovery limit mid-recovery. The learning naturally puts
its corrections where the authority to execute them exists.

### A diagnostic that falls out

`StrokeTrim.effort` — the RMS of the learned command — is a measure of how
much trim the crew is carrying. A well-matched crew in a well-rigged boat
converges to a small number. A large converged effort means they are
holding against something structural: a rigging error, or a rower who is
genuinely heavier on one side. That is a measurable quantity distinguishing
"this crew cannot sit the boat" from "this boat is not set up right",
which coaches currently separate by intuition.

### References

[BTA06] Bristow, D. A., Tharayil, M., & Alleyne, A. G. (2006). A survey of
iterative learning control. *IEEE Control Systems Magazine* **26**(3),
96–114.


## 20. Wind

The largest force the model was missing, and the one a Head race on a
river with a prevailing wind most needs.

### Where the drag is

Not on the hull. [K13] puts aerodynamic drag at about **13% of total
resistance** in still air, and splits it:

| component | share |
|---|---|
| oars | 50% |
| rowers' bodies | 35% |
| boat and riggers | 15% |

Five-sixths is crew and oars. That is not bookkeeping: it puts the force
**above the waterline**, and the oar share **outboard at the riggers**, so
a crosswind is a **roll and yaw moment**, not merely a drag. Given §15 —
roll is a marginal mode held by a few percent of the drive's authority
through the recovery — a lumped hull windage term would have put the
single largest environmental disturbance in the wrong place entirely.

`coxswain.hydro.wind.AeroModel` carries the three components separately,
each with its own drag area, height and lateral offset.

### The boundary layer, which is not optional

Wind speeds are quoted at the WMO standard **10 m** anemometer height. A
rower's shoulders are at about 0.6 m and an oar shaft lower still, deep
inside the surface layer. The neutral logarithmic profile
`u(z)/u(z_ref) = ln(z/z₀)/ln(z_ref/z₀)` with `z₀ = 2×10⁻⁴ m` over calm
water gives:

| height | fraction of the quoted wind |
|---|---|
| 0.15 m (hull) | 0.61 |
| 0.40 m (oars) | 0.70 |
| 0.60 m (bodies) | 0.74 |

Driving the drag model with the quoted speed over-predicts headwind force
by about `1/0.74² = 1.8`. **This was the single largest error in the first
version**: it made a 5 m/s headwind cost an eight 17.9% of its speed
against the 12.2% [K13] measures. It is a correction that has to be there,
not a refinement.

### Validation

The **only** calibrated number is the still-air 13%. Everything else is a
prediction from `v_rel²` drag and the boundary layer:

| | model | [K13] measured |
|---|---|---|
| 5 m/s headwind | **−11.3%** | −12.2% |
| 5 m/s tailwind | **+5.8%** | +5.1% |

Both within 15%, and the **asymmetry** — a headwind hurting roughly twice
as much as the same tailwind helps — is reproduced rather than imposed. It
follows from the square law: a headwind adds to the apparent wind the boat
already makes for itself, a tailwind subtracts from it, and the boat is
already moving at about the wind speed.

[K13] also notes the aerodynamic share can rise up to fourfold in a
headwind and fall to zero in a sufficient tailwind. Both come out of the
same square law without being put in; the model gives a *thrust* once the
tailwind overtakes the boat, which is tested.

### What is deliberately left open

`WindField` is an interface, and only `UniformWind` is implemented. A
river is not uniform: a bend turns a tailwind into a crosswind, and the
Charles has bridges, banks and buildings that shelter parts of the reach.
Spatial and temporal variation is the next step and needs no change to the
force model — which is why the interface exists now rather than later. It
is also a prerequisite for the stochastic formulation, where varying wind
is one of the named uncertainties.

### References

[K13] Kleshnev, V. *Rowing Biomechanics Newsletter* — wind effects and the
composition of aerodynamic drag.
[H21] "Rowing Against the Wind: An Analysis of the Impact of Variable Wind
Conditions", Harvard.
<https://dash.harvard.edu/server/api/core/bitstreams/4f46c026-cb8b-4f50-aa82-1a63b2baa8a5/content>


## 21. Per-rower stroke phase (plan step A1)

Every rower shared one stroke phase until now. Sections 15–16 make clear
why that mattered: roll is an unstable mode held through the recovery by
about 5% of the drive's authority, and port/starboard timing asymmetry is
one of the main things that disturbs it. **Setting all phases equal set
that disturbance to exactly zero.**

`Boat.phase_offsets` now carries one offset per seat, as a fraction of a
stroke, positive meaning late. Zero for every seat reproduces every prior
result exactly, which is asserted rather than assumed.

The offset enters in two places, and it has to be both: the rower's
kinematic chain *and* their oar. Evaluating the oar at the boat's time
while the body ran on its own would take the hands off the handle, which
is the constraint the crew model rests on.

Cost is honest: the kinematics batching keys on the offset, so a crew of
individuals costs one chain per seat rather than one per crew. A
synchronised eight still collapses to a single group.

### What a timing split does

Port side lagging starboard, stroke-averaged oar loads on the hull, eight
at rate 32. Recovery balance authority for comparison: **84.1 N·m**.

| split | roll rms | fraction of recovery authority | yaw rms |
|---|---|---|---|
| 0 ms | 0.4 N·m | 0.01 | **199.2 N·m** |
| 10 ms | 12.9 | 0.15 | 88.1 |
| 20 ms | 26.4 | 0.31 | **40.4** |
| 40 ms | 52.6 | 0.63 | 129.2 |
| 60 ms | 78.4 | **0.93** | 260.5 |
| 80 ms | 103.5 | 1.23 | 392.0 |

**A port/starboard split of about 65 ms exhausts the crew's entire
recovery balance authority.** That is 3.5% of a stroke at rate 32. It is
an uncomfortably small number, and it connects the two halves of this
work: §15 says how little authority there is, §21 says how little timing
error it takes to spend it.

### An unexpected result: timing as a steering trim

The yaw column is **not monotonic**. A perfectly synchronised sweep eight
already yaws — the rig is not port-starboard symmetric, and §15's baseline
is 199 N·m rms. A small timing split pushes the other way, and around
**20 ms it very nearly cancels the rig's own bias** (199 → 40 N·m).
Beyond that it overtakes it and grows.

That is a different statement from "timing error is bad". Crew timing is a
**steering trim** as well as a disturbance, and there exists a small
non-zero split that makes a sweep eight track straighter than a perfectly
synchronised one would. Whether real crews find it unconsciously is a
testable question; that it exists is a property of the rig.

### Where the cost is not

Stroke-averaged surge force is essentially unchanged across this whole
range (409.2 → 409.4 N). At these splits the price of poor timing is paid
in **roll and yaw, not in thrust**. That is worth stating plainly because
the usual explanation offered to crews is that being out of time "wastes
power", and at this scale that is not where it goes — it goes into making
the boat harder to hold and harder to steer, both of which cost time by
other routes.

### Next

This is step A1 of `docs/PLAN_SYNCHRONISATION_AND_BLADES.md`. Still to
come: the vertical phase ψᵢ (blade height, separate from the horizontal
sweep — an early extraction changes one without the other), the
Kuramoto-type coupling with both mechanical and sensory channels, and
skill as (σ, K_sensory, ψ–φ offset).


## 22. Crew synchronisation as coupled oscillators (plan step A2)

§21 gave rowers individual phases as *inputs*. Real crews do not have
prescribed phases — they converge on one, imperfectly, and that
convergence is a dynamical process with its own timescale and failure
modes. `coxswain.crew.synchronisation` models it.

### The framework, and what rowing contributes back to it

This is the setting Naomi Ehrich Leonard's group works in [L-PUB]: phase
models of coupled oscillators and their connection to collective motion,
including a collaboration using improvisational dance as a model system
for in-the-moment collective decision making.

Most human-synchronisation systems have one perceptual coupling. A rowing
crew has **two**, and they are physically different:

| | mechanism | latency | topology |
|---|---|---|---|
| **mechanical** | shared hull — rower *i*'s motion moves the shell every other rower is bolted to | none | mean-field |
| **sensory** | watching the blade or back ahead, hearing the catch | 150–250 ms | directed chain toward stroke |

The mechanical channel is what rowing offers *back* to the coupled-
oscillator literature rather than merely borrowing from it: the mean-field
term is not a modelling convenience but an actual mechanical path through
a shared rigid body, with a transfer function this simulator already
computes.

### Two phases, not one

An early extraction changes vertical timing without changing the
horizontal sweep, and they reach the boat through different terms:

* `phi` — horizontal: sweep angle and handle force → surge and yaw;
* `psi` — vertical: blade immersion and extraction → roll and blade forces.

A rower who washes out early has `psi` leading `phi`. Invisible to a
single-phase model, and a roll disturbance arriving exactly when §15–16
say roll authority is lowest.

### A modelling error worth recording

The first version used naive delayed coupling — comparing your phase *now*
against another rower's phase 200 ms ago. **That makes the whole crew row
slow**, by an amount independent of gain: a persistent 1.47 rad/s against
a nominal 3.35, i.e. 44% below the rate called. Gain-independence is the
signature of a structural error rather than a tuning one.

Real sensorimotor synchronisation is *predictive*: people tap **with** a
metronome, not behind it. Advancing the seen phase by the delay before
comparing fixes it, and a synchronised crew then feels no coupling force
at all, which is the correct behaviour.

This mattered beyond tidiness. The naive version produced a dramatic
result — a sharp synchronisation transition, with visually-coupled crews
unable to lock at all — which **did not survive the correction**. It is
recorded here because it was nearly reported as a finding.

### What the corrected model says

Detuning sd 0.10 rad/s (3% of stroke rate), `sensory_gain = 4`:

| chain / mean-field | coherence | spread |
|---|---|---|
| 0% / 100% | 0.9999 | **12.0 ms** |
| 50% / 50% | 0.9998 | 14.7 ms |
| 100% / 0% | 0.9962 | **75.1 ms** |

§21 puts the balance budget at about 65 ms of port/starboard split. So:

* a crew coupled purely through the **visual chain sits right at the edge
  of the balance budget**;
* one coupled through the **shared hull has roughly six times the margin**;
* both channels work — the hull one is far more effective per unit of
  coupling, because it is immediate and reaches everyone at once.

A chain-coupled crew can still get inside the budget, but only with strong
coupling: 187 ms of spread at gain 1, 75 ms at 4, 48 ms at 16. That is a
plausible reading of what drilling together actually trains — not better
eyesight, but a higher gain on what is seen.

### A testable discriminator

The two topologies leave different signatures: a **chain accumulates lag
monotonically from stroke toward bow**; mean-field does not. Per-seat
telemetry can distinguish them directly, which turns "which channel
dominates?" from a matter of opinion into a measurement.

### References

[L-PUB] Leonard, N. E., Princeton University, Mechanical and Aerospace
Engineering / Applied and Computational Mathematics.
<https://naomi.princeton.edu/publications/>
[KUR] Kuramoto, Y. *Chemical Oscillations, Waves, and Turbulence* (1984).
[K-VAR] Kleshnev, V. "Rowing Science: New Analysis of Variability of
Rower's Technique", parts 1–3, row2k.


## 23. The speed-fluctuation discrepancy, diagnosed

The model's intracycle velocity variation has been too high since the
beginning. This section closes the investigation: the cause is identified,
three plausible hypotheses are eliminated, and the remaining fix is a data
problem rather than a code one.

### First, the target was misquoted

`test_speed_fluctuation_is_larger_than_measured` cited "37.5% for elite
male single scullers (max 5.94, min 3.10, mean 4.28 m/s)". Those numbers
do not give 37.5% — they give 66%. Going back to the source
(PMC12349136) resolves it: the max and min quoted are **whole-race**
extremes, and the paper says so explicitly. The intracycle quantity is
separate:

| | males | females |
|---|---|---|
| mean velocity | 15.40 km/h | 13.36 km/h |
| **IVV** (max−min *within each cycle*) | **5.78 km/h** | 5.50 km/h |
| IVV / mean | **37.5%** | **41.2%** |
| CVV (SD/mean) | 14.13% | 11.64% |

So 37.5% is right, but only for the intracycle definition. The note had
put whole-race extremes next to an intracycle percentage, which invites
exactly the error it caused.

### Like for like

Measuring the model with the paper's own definitions, single scull at the
same 33.65 spm:

| | mean | IVV | IVV % | CVV % |
|---|---|---|---|---|
| **measured, elite male 1x** | 4.278 | **1.606** | **37.5** | **14.13** |
| model, ergometer kinematics | 4.442 | 2.895 | 65.2 | 21.43 |
| model, on-water kinematics | 4.451 | 2.659 | 59.7 | 20.39 |

### Three hypotheses, eliminated

**It is not the slide travel.** The recorded suspicion was that the crew
centre of mass travels 0.77 m against 0.4–0.5 m implied by measurement.
That rested on reading segment 0 of the crew field as the seat; it is the
**head**, which legitimately travels 1.33 m because the trunk swing adds
to the slide. The rower's actual `slide_travel()` is **0.633 m**, against
0.60–0.70 m reported for on-water crews. The kinematics are right.

**It is not the dataset.** The default driver is Caplan & Gardner, which
is ergometer data (Concept 2 stretcher) with a catch trunk angle of
−38.1°; Kleshnev's on-water elite telemetry gives −24.5°. Switching to
the on-water set moves IVV from 65.2% to 59.7% — real, but a small part
of the gap.

**It is not crew synchronisation.** A plausible story was that an eight
fluctuates less than a single because its rowers are not quite together,
smoothing the aggregate. Tested directly with per-seat phase offsets:
49.4% at zero spread, 49.2% at 30 ms, and only 46.0% at an unrealistic
80 ms. It does not explain the gap.

### What it is

The fluctuation is set almost entirely by the crew's centre-of-mass
velocity — momentum conservation alone predicts 50.8% against the 53.9%
simulated, so 94% of it. The travel is right and the timing is right, so
the error must be in the **shape** of the velocity profile, and it is:

| | peakiness (peak rate / mean rate) |
|---|---|
| model, drive | **1.589** |
| model, recovery | **1.861** |
| pure sinusoid | 1.571 |
| constant rate | 1.000 |

The reconstruction is essentially a sinusoid, and slightly worse than one
on the recovery. Quantitatively, for the single scull:

| | COM velocity swing |
|---|---|
| slowest possible (exactly constant rate) | **1.664 m/s** |
| implied by the measured 37.5% | **1.887 m/s** |
| what the model produces | **2.470 m/s** |

Real rowers sit **13% above the constant-rate floor**; the model sits
**48% above it**. The measured value is comfortably reachable given the
same keyframes and timing — this is not a contradiction in the data.

### Why it cannot be fixed by interpolating better

`FourierProfile.from_keyframes` fits a periodic cubic spline through
**four** measured instants and truncates to three harmonics. Four points
per cycle cannot help being near-sinusoidal.

A `flatness` parameter now exists to blend the spline towards
straight-line traverse, and it helps: with 8 harmonics it moves an elite
single from 60% to 54%, about a third of the way. It cannot do better,
and pushing it breaks something that matters — the straight-line traverse
has corners at the keyframes, truncating that rings around them, and the
reconstruction **stops passing through the measured angles**. Two unit
tests catch exactly that. Raising the harmonic count to chase the corners
fits spline artefacts instead of rower motion.

So `flatness` defaults to zero and the keyframe fidelity is kept.

### The conclusion

**This is a data-resolution limit, not a modelling error.** Four instants
per stroke do not determine the shape of the traverse between them, and
the shape is what sets the boat's speed variation. Every other input —
travel, timing, mass ratio, synchronisation — checks out.

Fixing it needs a **densely sampled seat-position trace**, not a cleverer
interpolation of four points. A seat or hull IMU at 50 Hz for a few
strokes would determine the profile directly and close this. That is a
single outing's work and it would replace the largest remaining
validation gap in the model with a measurement.

Until then the model overstates intracycle velocity variation by about
1.6x, and any result that depends on the *amplitude* of the surge
oscillation — rather than its mean — should be read with that in mind.
The blade model's absolute calibration is the main one.

## 24. Section 23 was wrong, measured against real data

§23 concluded that the model's excess intracycle velocity variation came
from the **shape** of the crew's centre-of-mass velocity profile — that
four keyframes plus a truncated Fourier series reconstructed something too
peaky, and the fix needed densely sampled kinematics. Measured against
real boat telemetry, that diagnosis does not survive.

### The data

[Accompanying Raw Data for "Adaptive smartphone-based sensor fusion for
estimating competitive rowing kinematic metrics"](https://api.figshare.com/v2/articles/7963643),
**CC0**, 195 MB. A club session and an elite session, each with a
smartphone on the boat, one on the rower, and a Swift Navigation RTK
receiver. The usable channel is the **DGPS baseline log**: boat position
relative to the base station at 10 Hz with 9 mm accuracy.

Not the `velocity_log` or `position_log` files, which carry the base
station's own solution and read a steady 0.008 m/s.

### The pipeline recovers the logged rates

| stamp | logged | detected | error |
|---|---|---|---|
| 091022 | 22 | 22.5 | +0.5 |
| 091604 | 22 | 21.9 | −0.1 |
| 092004 | 24 | 24.3 | +0.3 |
| 092443 | 24 | 24.0 | 0.0 |
| 092925 | 26 | 26.2 | +0.2 |
| 093324 | 26 | 26.1 | +0.1 |

**Mean absolute error 0.21 spm.** The detector is trustworthy, which is
what makes the rest of this section evidence rather than opinion.

### The correction

Comparing hull velocity like for like — the same quantity, the same
definitions:

| | peakiness | IVV |
|---|---|---|
| model, 1x, on-water kinematics, 24 spm | 2.358 | 57.3% |
| model, 1x, ergometer kinematics | 2.453 | 61.5% |
| **measured, club 2x, 22–26 spm** | **2.407** | **37.3%** |
| published, elite 1x, 33.7 spm | — | 37.5% |

**The shape is right.** Measured peakiness 2.407 against the model's
2.358 — within 2%, and the model sits between the two datasets. §23's
central claim, that the reconstruction is too sinusoidal, is contradicted:
the real hull surge is at least as peaked as the model's.

**The amplitude is wrong.** IVV 57.3% against 37.3% measured — a factor of
1.54, and the measurement independently reproduces the published 37.5%
for elite singles from a completely different boat, crew and instrument.

§23's peakiness figures of 1.589 and 1.861 were measured on the **crew
centre of mass**, not the hull, and were compared against a hull number.
That is the error: two different quantities, and the conclusion drawn from
the comparison does not hold.

### What it means for the diagnosis

The gap is amplitude, not shape, so the remaining candidates are the ones
that scale the surge without changing its form:

* **crew centre-of-mass travel.** 0.72 m in the model; roughly 0.55 m
  would give the measured amplitude. But the seat travel is 0.633 m and
  matches the 0.60–0.70 m literature, and trunk swing adds to seat travel
  rather than cancelling it, so 0.72 m is kinematically consistent. This
  does not obviously close.
* **blade anchoring.** During the drive the hull is not a free body — it
  is coupled to the water through blades that are, to first order,
  anchored. That impedance resists surge oscillation. The model applies
  oar *force* to the hull but represents the blade through an effective
  gearing, which may understate how much the anchored blade resists the
  hull surging against it. This is the leading hypothesis and it is
  testable: it predicts the excess is concentrated in the drive, not the
  recovery.

### Method note

The telemetry pipeline needed two fixes that only real data could have
prompted, both now pinned by tests:

* **autocorrelation fails on hull vibration.** A boat-mounted phone picks
  up slap an order of magnitude above stroke frequency and often larger in
  amplitude; the autocorrelation peak pins to the shortest allowed lag.
  Every elite trial read 75 spm — the search floor — for rowing that was
  16 to 34. Replaced with a band-limited spectrum.
* **band-limiting alone is not enough.** Low-frequency energy from
  integration drift and from the boat accelerating inside the window can
  outrank the fundamental; in the 34 spm trial the correct peak at exactly
  34.0 ranked sixth. Harmonic support is now used to weight the spectrum —
  as a weight rather than a product, since a pure tone has no harmonics
  and the textbook harmonic product spectrum would zero it.

## 25. Literature review: rower kinematics and boat velocity fluctuation

The fluctuation gap of §24 -- shape right, amplitude 1.5x too large -- was
diagnosed against two papers. That is not a literature review. This
section is the corpus, what each source constrains, and the three
quantitative checks it makes possible.

### The corpus

**Rower kinematics, measured**

1. **Caplan & Gardner (2010)**, *J. Sports Sci.* 28(3) 263-269. Joint
   angles at four keyframes, Concept 2 ergometer. The model's driver.
   Ergometer, not on-water -- catch trunk angle -38.1 deg.
2. **Kleshnev**, *Rowing Biomechanics Newsletter* / row2k, "Analysis of
   Angles of Body Segments in the World's Best Rowers" (2019). On-water
   elite telemetry: catch trunk -24.5 deg, finish +26.3, stroke length
   1.52 m. Used as the on-water cross-check.
3. **Kleshnev**, "Amplitude and power of body segments". Segment shares of
   **stroke length: legs 33%, trunk 31%, arms 36%**; of **power: legs 43%,
   trunk 33%, arms 24%**. Trunk velocity peaks at ~70% of the drive.
4. **de Leva (1996)**, *J. Biomech.* 29(9) 1223-1230. Segment masses,
   lengths and centre-of-mass fractions. The model's mass distribution.
5. **Lintmeijer et al. (2018)**, *Eur. J. Sport Sci.*, "An accurate
   estimation of the horizontal acceleration of a rower's centre of mass
   using inertial sensors: a validation". Finds CoM acceleration is
   recovered accurately from a **13-segment** IMU suit plus a mass model,
   and explicitly that it is **not** recoverable from pelvis acceleration
   alone.
6. **Lintmeijer et al. (2017)**, *J. Sports Sci.*, "Improved determination
   of mechanical power output in rowing". True power differs from the
   oar-based measure by ``m * a_CoM * v_boat`` -- the same product that
   drives the fluctuation problem here.
7. **Geneau et al. (2024)**, *Sensors* 24(18) 6085. On-water kinematics by
   class: W8+ 37.9 spm / 5.47 m/s, W4- 36.7 / 4.92, W1x 31.9 / 4.09.

**Boat velocity fluctuation**

8. **Caplan & Gardner**, "Modelling the influence of crew movement on boat
   velocity fluctuations during the rowing stroke". Their own single-mass
   simulation matched *mean* velocity but **modelled instantaneous
   velocity poorly**; a five-segment crew model was needed to reproduce
   the features of on-water data.
9. **Kleshnev (2010)**, *Proc. IMechE P: J. Sports Eng. Tech.*, "Boat
   acceleration, temporal structure of the stroke cycle, and effectiveness
   in rowing". Six drive and three recovery microphases; the relative
   magnitudes of boat and rower CoM acceleration switch twice during the
   drive.
10. **PMC12349136 (2025)**, *Sensors* 25(15) 4696. Elite single scull over
    2000 m: IVV 5.78 km/h on 15.40 mean = **37.5%** (males), 41.2%
    (females); CVV 14.13% / 11.64%.
11. **Day et al. (2011)** and the scoping review -- longer boats show lower
    intracycle velocity variation.
12. **"The impact of fluctuations in boat velocity during the rowing cycle
    on race time"** -- the fluctuation implies a **5-6% power loss**.

**Crew coordination**

13. **Cuijpers, Zaal & de Poel (2015)**, *PLoS ONE* 10(7) e0133527,
    "Rowing Crew Coordination Dynamics at Increasing Stroke Rates".
    Coupled-oscillator treatment of a rowing dyad. **SD of relative phase
    2.2 deg in-phase, 4.2 deg antiphase** (trunk-based); by rate 4.13 /
    3.21 / 4.24 / 4.81 deg at 30 / 32 / 34 / 36 spm. Ergometer velocity
    fluctuation SD **0.667 in-phase against 0.221 antiphase**.
14. **Kleshnev**, variability series. Force variation **2.3% elite, 5.1%
    junior**; work per stroke 1.3% / 4.7%.

**Hydrodynamics, blade, environment**

15. **Formaggia, Miglio, Mola & Montano (2009)**, *Int. J. Numer. Meth.
    Fluids* 61:119-143. The 6-DOF formulation; eq. (14) is the mass
    matrix.
16. **Cabrera, Ruina & Kleshnev (2006)**. Slip-based blade model.
17. **Brearley (2009)**, *ANZIAM J.* 50:534-540. Blade efficiency versus
    oar angle.
18. **Lazauskas**, Technical Report L9701, hull drag comparisons.
19. **Dudhia (1996/2013)**, "Balance of Racing Rowing Boats". Static roll
    instability; sections 15-16.
20. **Bristow, Tharayil & Alleyne (2006)**, *IEEE Control Syst. Mag.*
    26(3) 96-114. Iterative learning control; section 19.

### What the corpus changes

**(a) Crew timing spread is now measured, not guessed.** §22 used
plausible values. Cuijpers et al. give SD of relative phase 2.2-4.8 deg,
which at 32 spm is **11-25 ms** -- squarely inside the range §21 tested
and consistent with the finding there that realistic spread does *not*
explain the fluctuation gap (49.4% at zero spread, 49.2% at 30 ms).

**(b) The segment amplitude split is wrong, and it is the right size to
matter.** Against Kleshnev's published shares of stroke length:

| segment | model | Kleshnev |
|---|---|---|
| legs | **38.9%** | 33% |
| trunk | **37.4%** | 31% |
| arms | **34.0%** | 36% |

The model over-weights legs and trunk and under-weights arms. Legs and
trunk carry ~85% of body mass and the arms ~9%, so this inflates the
centre-of-mass travel specifically. Matching the published legs share
would put seat travel at 0.524 m rather than the model's 0.618 m.

The cause is structural and was already documented: **Caplan & Gardner
measured only the legs and trunk**, so the model's arm postures are
hand-specified (``DEFAULT_ARM_POSTURE``) rather than measured. The one
part of the chain not driven by data is the one that is short.

**(c) The failure mode is a known one.** Caplan & Gardner's own single-mass
simulation reproduced mean velocity but not instantaneous velocity, and
needed five crew segments to fix it. This model has twelve, so
segmentation is not the issue here -- which narrows the remaining
candidates to the amplitude of the segment motions themselves, exactly
where (b) points.

### What it does not resolve

Correcting the segment split to the published shares moves centre-of-mass
travel by roughly 15%, which is worth having but is not the factor of 1.5
that §24 measured. The remaining discrepancy is still open. Lintmeijer's
``m * a_CoM * v_boat`` correction is the most promising lead, because it
is the same product and it is measured directly on a force plate rather
than inferred -- but both Lintmeijer papers are paywalled and the
numerical values were not obtained.

## 26. Two ways out of the kinematics gap, and a third

§24 and §25 leave the crew's centre-of-mass motion as the open quantity:
the model has ~1.5x too much of it, the shape is right, and four measured
keyframes cannot pin the amplitude. There are two obvious routes and one
less obvious one.

### Route A -- measure the kinematics properly

**[UHL23] Uhlrich, Falisse, Kidzinski, Muccini, Ko, Chaudhari, Hicks &
Delp (2023).** "OpenCap: Human movement dynamics from smartphone videos."
*PLOS Computational Biology*. Open-source pipeline: two or more
smartphones, pose estimation, then deep learning and biomechanical models
for three-dimensional kinematics, then physics-based simulation for muscle
activations. Kinematic RMSE across lower-extremity degrees of freedom is
comparable to IMU approaches (2.0-12 deg for walking, running and daily
activities) and to eight-camera video systems.

This is the practical route. It needs two phones and no laboratory, it
produces full-cycle joint angles rather than four keyframes, and it is
subject-specific -- so the driver becomes *this crew*, not a 2010
ergometer study.

**[LIN18] Lintmeijer et al. (2018).** *Eur. J. Sport Sci.* Validates
recovering a rower's anterior-posterior centre-of-mass acceleration from a
**13-segment** IMU suit plus a mass distribution model, and shows it is
**not** recoverable from pelvis acceleration alone. Relevant twice over:
it is the direct measurement of the quantity in question, and it is why
the waist-phone channel in the CC0 dataset was not used to settle it.

Also noted: **"Functional Data Analysis of Rowing Technique Using Motion
Capture Data"**, Proc. 6th Int. Conf. on Movement and Computing (2019) --
rowing-specific marker-based capture of six elite rowers on an ergometer.
No large-scale open rowing motion-capture dataset appears to exist.

### Route B -- change the formulation

**[VDB11] van den Bogert, Blana & Heinrich (2011).** "Implicit methods for
efficient musculoskeletal simulation and optimal control." *Procedia
IUTAM* 2:297-316. Poses the dynamics implicitly as ``f(x, xdot, u) = 0``
rather than solving explicitly for accelerations, which suits direct
collocation: exact sparse Jacobians, each constraint touching only two
neighbouring nodes, and -- the relevant part here -- it is specifically
designed for systems where **small masses couple to stiff elements**.

That is this model. §15 established that the crew's balance loop is stiff
(roll e-folds in 0.218 s) and that explicit integration of it diverges at
mesh-interval steps. There is also a concrete efficiency win available:
``SixDofModel.derivative`` currently ends in ``ca.solve(mass_matrix, ...)``
at every collocation point. The implicit form replaces that linear solve
with the residual ``M a - F = 0`` and promotes the accelerations to
decision variables, which removes the solve and improves sparsity.

**[FDR21] "A Review of Forward-Dynamics Simulation Models for Predicting
Optimal Technique in Maximal Effort Sporting Movements."** *Applied
Sciences* 11(4):1450. Surveys the four-stage process -- model
construction, parameter determination, model evaluation, model
optimisation -- and the distinction between **data-tracking** and
**predictive** simulations.

### Route C -- stop prescribing the kinematics

That distinction is the third route, and it dissolves the problem rather
than solving it.

This model is a *data-tracking* simulation: the crew's motion is
prescribed from measured keyframes, so any error in those keyframes is an
error in the answer, and the centre-of-mass travel is an **input** that
has to be right.

A *predictive* formulation makes the crew's motion a **decision variable**
constrained by what a rower can actually do -- joint limits, the seat on
its rail, the hands on the handle, a power budget. The centre-of-mass
travel then comes out of the optimisation as whatever is consistent with
the physics and the physiology, rather than being measured and imposed.
The 1.5x amplitude error would not be correctable in that formulation
because it would not be expressible.

It is also the honest version of the question this project asks. The
current model optimises the *line* for a prescribed stroke; a predictive
one would optimise the line and the stroke together, which is what a crew
actually chooses.

The cost is a much larger optimisation problem and a physiological model
that does not yet exist here. Route A is cheaper and would settle §24
directly; Route C is the better model.

## 27. Route C started: solving for the stroke instead of prescribing it

`coxswain.crew.predictive` implements the predictive formulation of §26:
the rower's motion is a decision variable, not an input.  Configuration is
``(slide, trunk lean, arm extension)`` and the **oar angle follows** from
where those put the hands.  This is the structure of [MSD13], which
represents rower, boat and oars as rigid links and lets the optimisation
determine movement and forces together.

### Three modelling errors it exposed

Each was found by the optimiser exploiting it, which is the point of a
predictive formulation -- a prescribed stroke cannot reveal that its own
assumptions are wrong.

1. **Oar angle free, reach as an inequality.**  The optimiser left the
   trunk still -- 0.2 degrees of swing -- and swept the oar on its own.
   A rower cannot do that.  Making the oar angle *follow* the hands fixed
   it, and the trunk immediately began to matter.
2. **Power omitted work on the water.**  Thrust was free; the answer was a
   6.9 m/s single scull drawing 112 W of a 300 W budget.
3. **Central difference on boat speed.**  Odd and even nodes decoupled and
   the optimiser put a sawtooth in the speed at no cost: 203% of
   intracycle variation.  Trapezoidal integration fixed it.

### What it chooses

Single scull, 24 spm, nothing prescribed:

| quantity | predictive | reference |
|---|---|---|
| mean speed | 3.8-4.7 m/s | right for a 1x |
| trunk swing | **40-52 deg** | Kleshnev **50.8 deg** |
| oar sweep | 67-74 deg | model prescribes 90 |
| slide travel | 0.22-0.37 m | measured **0.60-0.70** |
| crew CoM travel | 0.33-0.43 m | prescribed model 0.79 |
| **boat IVV** | **36.8-40.6%** | **measured 37.3%** |

**The IVV result is the one to watch.**  §24 measured 37.3% on
differential GPS against 54-58% for the prescribed-kinematics model, and
§25 traced the excess to the crew's centre-of-mass amplitude.  The
predictive formulation, told nothing about kinematics, lands on 37-41%.

That is exactly what §26 predicted would happen, and it is the strongest
evidence so far that prescribing the stroke from four keyframes was the
source of the error rather than any of the hydrodynamics.

### Why it is not yet a result

**None of these solves converged.**  They terminate on
``Maximum_Iterations_Exceeded``, not infeasibility -- the power constraint
is satisfied at the returned point (298 W against a 300 W budget) and the
answers are stable across budgets, but IPOPT does not reach its tolerance.
A number from an unconverged solve is suggestive, not established, and the
IVV agreement above is quoted on that basis.

The slide travel is also plainly wrong: 0.22-0.37 m where crews use
0.60-0.70.  The optimiser is under-using the legs, which is a physiological
statement the model does not yet make properly.  Kleshnev's measured power
shares -- legs 43%, trunk 33%, arms 24% -- are imposed as per-segment
budgets, but the cost of moving each segment is still a crude
mass-times-acceleration-times-rate term rather than a joint torque.

### Fixes applied, and one lesson repeated

* **Smooth power.**  ``|force x velocity|`` has a corner wherever either
  factor crosses zero -- four times a cycle per segment.  Replaced with
  ``sqrt(x^2 + eps)``, a bounded smoothing with the bound reported: at
  ``eps = 1e-4`` the error is under 0.01 W against a 300 W budget.
* **Scaling, again.**  The predictive NLP mixed O(1) geometry with O(100)
  power terms and was unscaled -- the same mistake that cost the
  trajectory solver a factor of 25 in iterations before §24 fixed it.
  Now non-dimensionalised.
* **Status reporting.**  Collapsing every non-success to "not converged"
  hid the distinction between ``Maximum_Iterations_Exceeded`` and genuine
  infeasibility for several iterations of this work.  IPOPT's own status
  is now reported.

### Next

The power model is the blocker.  [VDB11] writes the cost as generalised
force times generalised velocity from an implicit musculoskeletal
formulation, which is both smoother and physically correct; the current
term is neither.  Getting that right should give convergence and, if the
IVV agreement survives it, close §24.

## 28. Withdrawing the §27 IVV result

§27 reported that the predictive formulation, told nothing about
kinematics, produced 36.8-40.6% of intracycle velocity variation against
the 37.3% measured in §24, and called it the strongest evidence yet that
prescribing the stroke was the source of the §24 gap.

**That result is withdrawn.**  It was produced by a power model that is
wrong, and it does not survive the correction.

### The error

The per-segment power terms charged each segment only for accelerating
its own mass:

```
leg_power ~ m_leg * |s_ddot * s_dot|
```

and then applied Kleshnev's measured shares -- legs 43%, trunk 33%, arms
24% -- as budgets on those terms.  But Kleshnev's shares are shares of
the work delivered **through the handle**, which is a different quantity.
Constraining one with the other is a category error.

It also has a clear mechanical consequence.  The legs are the largest
segment mass in the body, so the formulation billed them the most and
credited them with none of the work that mass does on the blade.  The
optimiser responded exactly as it should have: it stopped using them.
That is the 0.22-0.37 m of slide travel §27 flagged as "plainly wrong",
against the 0.60-0.70 m crews actually use.

### The correction

``hand_x = s + L sin(phi) - a``, so handle velocity splits additively:

```
v_hand = s_dot + L cos(phi) phi_dot - a_dot
```

This is the decomposition Kleshnev's shares are shares *of*.  Handle work
is taken from the oar -- so it stays consistent with the blade model --
and attributed across segments by their share of handle speed, with
smoothed weights so the denominator stays bounded at the four instants
per cycle where the handle reverses.  The three shares sum to the total
by construction, so the attribution can neither invent nor destroy power.

### What the correction does

| | old (wrong) | corrected |
|---|---|---|
| slide travel, 400 W | 0.225 m | **0.430 m** |
| trunk swing | 44.3 deg | 29.4 deg |
| **IVV** | **36.8%** | **54.5%** |
| measured IVV (§24) | | **37.3%** |

Slide travel moves substantially toward the measured 0.60-0.70 m, which
is the predicted consequence of no longer starving the legs.  But the IVV
agreement evaporates: 54-64%, right back with the prescribed-kinematics
model's 54-58%.

**So §26's hypothesis is not supported.**  The §24 gap is not explained
by the stroke being prescribed from keyframes.  A formulation that solves
for the stroke reproduces the same excess fluctuation once its power
model is correct, which points the diagnosis back at the momentum
coupling or the hydrodynamics rather than at the kinematics.

### Standing caveat

Neither the withdrawn numbers nor the corrected ones come from a
converged solve -- everything here terminates on
``Maximum_Iterations_Exceeded``, and the 400 W case returns 647 W against
its budget, so that column is an infeasible point and is quoted only to
show the direction of the change.  The withdrawal does not depend on the
new numbers being right; it depends on the old ones having been generated
by a model with a demonstrable error in it.

Route C is not yet producing numbers that can be trusted either way.  The
honest status is that it has found and fixed three modelling errors
(§27) plus this fourth, and has not yet earned a result.

## 29. The §24 gap is the model, and the mechanism is segment sequencing

Asked directly whether the intracycle-velocity gap is a fault of the model
or of the target, the answer is **the model** -- and the target survives
three independent checks while the model contradicts itself.

### The target is sound

**Sampling is not the problem.**  IVV is a peak-to-peak statistic and the
DGPS trace is 10 Hz, 25 samples per 2.5 s cycle, on a signal §24 showed is
sharply peaked -- so discrete sampling should bias it low.  Sampling a
waveform tuned to the measured peakiness of 2.407 at 10 Hz with arbitrary
phase recovers 37.0% of a true 37.3%: a **0.8% loss**.  The instrument is
adequate.

| rate | samples/cycle | IVV recovered | loss |
|---|---|---|---|
| 10 Hz | 25 | 37.0% | 0.8% |
| 20 Hz | 50 | 37.2% | 0.2% |
| 50 Hz | 125 | 37.3% | 0.0% |

**Momentum balance reproduces the target from first principles.**  With no
hydrodynamics, no blade model and no keyframes -- if the crew COM moves
``L`` over a drive occupying fraction ``f`` of the stroke, the hull must
swing by the crew mass fraction times the crew's relative velocity swing:

| COM travel | drive fraction | IVV |
|---|---|---|
| 0.72 m | 33% | 38.6% |
| 0.72 m | 40% | 35.6% |

That brackets the measured 37.3%, and agrees with the published 37.5% for
elite 1x.  Three independent lines -- club 2x telemetry, published elite
1x, and momentum conservation -- all land near 37%.

**The segment masses are right.**  Checked against de Leva: head 6.9%,
trunk 43.5%, thigh 14.2% each, shank+foot 5.7% each.  No indexing error.

### The model contradicts itself

The model's own crew COM travel is **0.811 m**, against a seat travel of
0.652 m that matches the 0.60-0.70 m literature.  To produce the measured
IVV the COM travel would have to be **0.635 m**.  §24 justified 0.811 m by
arguing that trunk swing adds to seat travel rather than cancelling it.

**That is only true if the segments move in phase.**  They do not, in real
rowing -- the drive sequences legs, then trunk, then arms.

Phase of each segment's most-sternward instant, as a fraction of the
stroke:

| segment | most stern | most bow |
|---|---|---|
| head | 0.919 | 0.360 |
| upper trunk | 0.923 | 0.360 |
| lower trunk | 0.934 | 0.368 |
| thigh | 0.935 | 0.405 |
| shank+foot | 0.935 | 0.405 |
| forearm+hand | 0.992 | 0.331 |

The entire body reaches its extreme within 0.073 of a stroke.  Quantified
as the cancellation between the in-phase upper bound on COM travel and
what is actually realised:

```
COM travel if all segments in phase : 0.813 m
COM travel actually realised        : 0.811 m
sequencing cancellation             :   0.3%
required for the measured IVV       :  21.9%
```

**The model's rower moves as a rigid block.**  That inflates COM travel by
28%, which inflates the hull surge by the same factor through the momentum
coupling, and that is the §24 gap.

### This partially reinstates §23

§23 blamed the four-keyframe plus three-harmonic reconstruction and was
overruled by §24 on the grounds that the hull peakiness matched.  That
refutation was correct about peakiness and wrong to conclude "the shape is
right": the shape of the *aggregate* is right, the *relative timing
between segments* is not.  With four keyframes and ``KEYFRAME_HARMONICS =
3``, differential segment timing is close to unrepresentable, which is
exactly the failure observed.

### Consequence for the rower model

This gives a concrete, falsifiable acceptance test for any replacement
rower model, which §30 takes up:

* crew COM travel **0.635 m** against a seat travel of 0.65 m
* sequencing cancellation **~22%**, not 0.3%
* resulting IVV **37.3%**

and it means the fix belongs in the *kinematics*, not in the
hydrodynamics, the blade model or the solver.

## 30. Sequencing eliminated; the defect was interpolant overshoot

§29 proposed that the hull-fluctuation gap came from the model's segments
moving in phase, and set a target of ~22% sequencing cancellation.  That
hypothesis is **wrong**, and testing it properly found the actual defect.

### Sequencing was implemented, and it cannot close the gap

`SegmentSequencing` gives each joint driver its own timing.  Two
parametrisations were tried:

**A plain phase shift is the wrong operation.**  It moves the catch along
with everything else, so the seat is already retreating when the blade
goes in and the arms -- pinned to the handle by the oar -- cannot reach.
The failure appears at stroke phase 0.08-0.12, just after the catch, and
binds at +-0.08 having moved IVV only 47.7% -> 45.0%.

**A within-phase warp is the right operation** and it does not help
either.  The map holds the catch, finish and mid-recovery fixed and
redistributes motion between them, so keyframe postures are untouched.
Scanned over the full sign space of leg and trunk warps:

```
IVV % : rows = legs warp, cols = trunk warp     (target 37.3)
           -0.30   -0.20   -0.10    0.00    0.10    0.20    0.30
  -0.30   reach   reach   reach    65.4    63.5    62.0    61.0
  -0.10   reach   reach    50.5    49.5    48.8    48.3   reach
   0.00   reach   reach    48.1    47.7   reach   reach   reach
   0.10    50.2    51.6    52.4   reach   reach   reach   reach
   0.30    66.4   reach   reach   reach   reach   reach   reach
```

**Measured this way, the synchronous case is the minimum** -- every
perturbation makes things worse and most of the space is unreachable.

That conclusion did not survive the fix below, and the reason is worth
recording: **this scan was run through the overshooting interpolant.**
Re-run against the shape-preserving one, sequencing becomes a real lever
in the anatomically correct direction:

```
IVV % (post-PCHIP): rows = legs, cols = trunk       target 37.3
         -0.20  -0.15  -0.10  -0.05   0.00   0.05   0.10
 -0.10   reach   49.0   49.3   49.4   49.4   49.2   48.9
  0.00    42.6   43.1   43.9   44.6   45.3   45.8   46.4
  0.10    45.8   45.4   45.3   45.4   46.0  reach  reach
```

Best is **42.6%** at legs nominal and trunk lagging -- legs first, then
trunk, which is the uncontroversial description of a rowing drive.  The
overshoot had been masking the effect it was itself causing.

The magnitude is *calibrated* against the measured fluctuation rather
than predicted, so the resulting agreement is not independent evidence,
and the default stays `SYNCHRONOUS` until there is kinematic data to set
it.  What the scan establishes is that the lever exists and points the
right way.

§29's arithmetic was separately wrong in a way worth recording: **crew COM
travel is geometric.**  It is fixed by the catch and finish postures, not
by timing, so no amount of sequencing changes it.  What sequencing
changes is the velocity *shape*.  §29 set its target on travel.

The machinery is kept and defaults to `SYNCHRONOUS`.  It is a genuine
second-order lever, worth about 2.7 points of fluctuation, but it is not
the primary defect.

### The actual defect

Checking the reconstruction against the keyframes it interpolates:

| joint | keyframes span | reconstruction | inflation |
|---|---|---|---|
| trunk link | 54.7 deg | **62.4 deg** | +14% |
| shank | 76.9 deg | **85.2 deg** | +11% |

`CubicSpline(..., bc_type="periodic")` through four unevenly spaced points
**overshoots**.  Every joint excursion came out 11-14% larger than the
data, crew COM travel is a mass-weighted sum of those excursions, and the
hull surge is proportional to COM travel.

This is distinct from the traverse-shape question in
`FourierProfile.from_keyframes`, which is a genuine data-resolution
limit.  Reporting a joint swing *larger than the data being interpolated*
is not a resolution limit -- it is an artefact of the interpolant, and it
had propagated into every kinematic quantity in the model.

### The fix

PCHIP is shape-preserving: it cannot exceed the local data range.
Periodicity comes from interpolating a three-period tiling and keeping
the middle.  `shape_preserving=True` is now the default.

| | before | after | reference |
|---|---|---|---|
| trunk link swing | 62.4 deg | **54.5 deg** | Kleshnev 50.8 |
| shank swing | 85.2 deg | 76.1 deg | keyframes 76.9 |
| seat travel | 0.652 m | 0.604 m | 0.60-0.70 |
| crew COM travel | 0.811 m | **0.751 m** | |
| IVV, momentum balance | 47.7% | **45.3%** | measured 37.3% |

All 1011 fast tests pass unchanged.

### What is left, and a prediction

45.3% against 37.3% -- about a quarter of the gap closed.  But note the
sign of what remains.  45.3% is the **crew-only** figure: what the crew's
motion alone would do to a free hull.  During the drive the crew moves
bow-ward, pushing the hull sternward, while the blade thrust pushes it
bow-ward.  **These oppose.**  The measured 37.3% sitting *below* the
crew-only 45.3% is exactly what that cancellation predicts.

The full 6-DOF model reported 57.3% -- *above* its own crew-only figure.
That is the wrong side.  If thrust opposed crew reaction correctly, the
6-DOF result would have to come in below 45.3%, not above it.

So the prediction for §31 is that the remaining gap is not in the
kinematics at all but in the **relative phase of blade thrust and crew
reaction** in the surge equation, and that checking their phase against
each other over the stroke will show them adding where they should cancel.

## 31. The §30 prediction was wrong: the phasing is correct

§30 predicted that the remaining fluctuation gap was a **relative phase
error between blade thrust and crew reaction** in the surge equation --
that they were adding where they should cancel.  Checked against the full
6-DOF simulator, that is **not** what is happening.

### The phasing is textbook

Straight-line run, 1x at 24 spm, eight strokes, last two cycles:

| | phase | expected |
|---|---|---|
| hull speed **minimum** | 0.094 | just after the catch |
| hull speed **maximum** | 0.438 | early recovery |

Drive occupies 0 to 0.32.  Minimum just after the catch and maximum in
the early recovery is the standard picture, and it is what a rower
reports: the boat runs fastest on the recovery, not the drive.  There is
no sign or phase error to find.

### But the amplitude still exceeds the crew-only bound

| | IVV |
|---|---|
| full 6-DOF | **60.1%** |
| crew-only momentum bound, same kinematics | 52.2% |
| measured (§24) | 37.3% |

Sitting above the crew-only bound is still the wrong side of the line,
and §30's reasoning about that stands even though its proposed mechanism
does not.  Thrust and drag varying over the cycle *can* legitimately add
to the crew term; what they cannot do is add this much while being
correctly phased.

### A second contributor, previously missed

The same run reports a **mean speed of 3.907 m/s** for a single scull at
24 spm.  That is not itself wrong -- Kleshnev puts a 1x at that rate near
3.8-4.0 m/s, and 4.2-4.5 is *race* pace at 32-36 spm, not 24.  But it
matters directly, because IVV is a ratio:

```
IVV = swing / mean
```

The crew-only bound quoted as 45.3% in §30 was computed against an
assumed 4.5 m/s.  Against the model's own 3.907 m/s the same swing gives
52.2%.  So **roughly a seventh of the apparent gap is a normalisation
difference rather than a fluctuation excess** -- and the fault is in the
comparison, not the model.  §30's 45.3% used a mean speed the model does
not have; the honest crew-only figure for this boat at this rate is
52.2%.  Chasing that seventh in the kinematics would have been chasing a
quantity that was never wrong.

### Standing caveat on the comparison

The measured 37.3% is a club **2x**; every model figure here is a **1x**.
The crew mass fraction is nearly identical between the two (0.859 vs
0.863) so the momentum coupling transfers, but mean speed does not, and
IVV is normalised by mean speed.  This has been a weakness of the
comparison since §24 and it is now the largest single uncontrolled
difference in it.

### Where this leaves the diagnosis

Three candidates remain, in order of how much they would move the number:

1. **1x model against 2x data at different mean speeds.**  Now the
   largest uncontrolled difference, and the one that needs fixing first
   because every other comparison is normalised through it.  A 2x runs
   faster than a 1x at the same rate, so the same absolute swing reports
   a smaller IVV.  Fixable by building the 2x; the catalog has only
   `eight`, `coxed_four` and `single_scull`.
2. **Crew COM swing still too large** -- §30 took travel from 0.811 to
   0.751 m; sequencing is worth another ~2.7 points once there is data
   to set its magnitude.
3. **Residual excess over the crew-only bound** -- 60.1% against 52.2%,
   correctly phased, so whatever it is, it is not a sign error.

The phase hypothesis is closed.

## 32. Like for like at last: the 2x

§31 named the 1x-model-against-2x-data mismatch as the largest
uncontrolled difference in the fluctuation comparison.  `double_scull`
now exists, so the comparison can be run properly.

10.4 m, 27 kg (World Rowing minimum for the class), two seats at the
usual 1.22 m spacing, 0.80 m span, peak oarlock force 780 N.

### Boat class is worth 8.6 points

Same rate, same kinematics, same blade model:

| boat | mean speed | IVV | crew-only bound |
|---|---|---|---|
| 1x | 3.907 m/s | 60.1% | 52.2% |
| **2x** | **4.805 m/s** | **51.5%** | 42.5% |
| measured club 2x | | **37.3%** | |

IVV is normalised by mean speed, and a 2x runs faster than a 1x at the
same rate, so the same absolute swing reports a smaller ratio.  **8.6 of
the 22.8 points of apparent disagreement were the comparison, not the
model.**

The remaining like-for-like gap is 51.5% against 37.3%.

### Running total on the §24 gap

| | IVV | note |
|---|---|---|
| as first reported (1x vs 2x data) | 57-60% | |
| after §30 removed interpolant overshoot | -2.4 | genuine fix |
| after §32 compares 2x with 2x | **51.5%** | comparison fix |
| sequencing, if calibrated (§30) | ~-2.7 | needs data to justify |
| measured | **37.3%** | |

About a third of the original discrepancy has been accounted for, split
between one real defect and one bad comparison.  What is left is a
genuine amplitude excess that sits above the crew-only momentum bound
(51.5% against 42.5%) while being correctly phased (§31).

## 33. Getting the deterministic leg to run end to end

The first full-leg attempt stopped at 476 m of 850, block 18 going
infeasible with 8.5 m of clearance and roll spiking to 4.8 deg.  Three
separate problems, only one of which was physics.

### The channel pinches, and the horizon could not see it

Centreline clearance along the Weeks-Anderson section:

| station | 0 | 200 | 300 | **450** | 600 | 850 |
|---|---|---|---|---|---|---|
| clearance m | 38 | 65 | 60 | **30** | 51 | 38 |

The boat failed at 476 m with 8.5 m of room, meaning it entered a 30 m
channel about 22 m off the centreline.  It had drifted through the wide
reach at 200-350 m, where a *terminal* clearance reward barely competes
with progress and there is room to spare in every direction.

**Fix: a running comfort barrier.**  Zero cost while the boat is
comfortably clear, quadratic once it is not -- the usual soft treatment
of a state constraint whose hard version is already imposed.  It shapes
the approach without distorting the optimum where there is room, so the
boat is still free to be anywhere it likes; it just stops spending
clearance it will need later.  `comfort = 1.2` reads as "keep 1.2
boat-clearances in reserve", the units being non-dimensionalised the same
way the hard constraint already was.

The same term and roll penalty were added to the stochastic solver, which
has its own objective.  A stochastic run that behaved differently from
the deterministic one for reasons unrelated to uncertainty would not be
measuring uncertainty.

### A failed block ended the run

The driver broke out of its loop on the first non-converged block.  It
now escalates -- more iterations, then a shorter horizon -- before giving
up, and stops only after three consecutive blocks make no progress.

### The mean speed was a reporting bug

The first run reported **2.35 m/s** for an eight, which would be a
serious physics problem, and is not one.  It divided total progress by
`BLOCKS * STROKES_PER_BLOCK * period` using the number of blocks
*requested* rather than the number actually solved before breaking out.
Accumulating simulated time from the blocks that ran gives **4.63 m/s**,
which is right for an eight at rate 32.

Worth recording as its own item: a wrong number in a summary line looked
exactly like a modelling failure and would have sent the next day's work
into the drag model.

## 34. Rower kinematics, from the same boat and the same outing

The figshare dataset used in §24 carries a **rower-mounted** sensor as
well as the hull one, and it had never been touched.  For the club
session it is `Pelvis2x-20180420T085631` -- the same 2x, the same outing
that gave the 37.3% intracycle velocity variation.  Rower kinematics and
hull fluctuation therefore come from one crew on one row, which is what
makes this a like-for-like check rather than another cross-dataset
comparison.

50.7 Hz, 45 minutes, with device orientation, so **the trunk rotation is
measured directly** -- no integration, no drift to fight.

### Two pipeline faults worth recording

**The log is not in time order.**  `log_time` steps between −322 s and
+326 s between adjacent rows: the file interleaves streams.  Unsorted, it
gave a pelvis swing of 0.2-1.7 deg, which reads as "the rower barely
moves" and is entirely an artefact.  Sorting and de-duplicating first is
mandatory.

**Detrending can eat the signal.**  A 6 s moving-average high-pass is
only 2.4 stroke periods and attenuates the fundamental it is supposed to
preserve, giving a median swing of 16.8 deg with a 3.2-28.5 deg spread --
the scatter being the tell.  At 20 s (~8 strokes) the same window gives
**37.1 deg with an inter-quartile range of 36.5-38.1**.  A measurement
that tight is the real one; the earlier scatter was the method.

### The trunk kinematics are probably fine

| | value |
|---|---|
| measured pelvis roll swing, club 2x @ 22.3 spm | **37.1 deg** (IQR 36.5-38.1) |
| model trunk **link** swing (post-§30) | 54.5 deg |
| Kleshnev trunk swing | 50.8 deg |

These are not the same quantity: a pelvis-mounted sensor measures pelvic
tilt, while the trunk angle Kleshnev reports is the hip-to-shoulder line,
which is pelvic tilt **plus** lumbar and thoracic flexion.  37 deg of
pelvis with the usual 10-20 deg of spinal contribution lands at roughly
50 deg of trunk -- consistent with both Kleshnev and the model.

**So the measurement does not indict the trunk kinematics.**  Which
matters for what to do next, because it removes the most obvious
remaining kinematic suspect.

### What this means for "do we need more data"

The remaining like-for-like gap (§32) is 51.5% against 37.3%, and it
splits at the crew-only momentum bound of 42.5%:

* **51.5 -> 42.5, about 9 points.**  The model exceeds what its own crew
  motion could do to a free hull, while being correctly phased (§31).
  This is the force path -- thrust, drag, added mass -- and **no amount
  of kinematic data addresses it.**
* **42.5 -> 37.3, about 5 points.**  This is inside the crew motion, and
  denser kinematics is the right instrument for it.

Densely sampled kinematics is therefore worth having but is **not the
main lever**: it can reach at most about a third of what is left, and the
larger share is in the hull force balance.

## 35. The model had no added mass

Asked to make the boat's *response* trustworthy -- what a rudder input or
a pressure split actually does -- the first thing to check was the
inertia the boat turns with.  Searching the codebase for added mass
returned nothing. There was none.

A hull accelerating through water accelerates water with it.  For a
rowing shell the entrained water is not a correction:

| boat | added sway / boat mass | added yaw / physical yaw |
|---|---|---|
| 1x | 1.63 | **9.2x** |
| 2x | 1.26 | **5.9x** |
| **8+** | 0.71 | **1.22x** |

The eight was turning with ~9500 kg m^2 of yaw inertia where it should
have ~21000.  The small boats were out by most of an order of magnitude.
**Every steering result the model has produced was computed on a boat
several times too easy to turn.**

### What was implemented

`coxswain.hydro.addedmass`, classical strip theory.  Sections here are
semi-elliptical, and the 2D added mass of an ellipse is classical: a full
ellipse with semi-axes ``b`` horizontal and ``T`` vertical entrains
``rho pi T^2`` per unit length moving horizontally and ``rho pi b^2``
moving vertically [Lamb 1932 art. 71].  A surface-piercing half section
is treated by images with the free surface as a rigid wall, halving those
values [Newman 1977 sec. 4.13; Korotkin 2009].  Integrating along the
length gives sway, heave, roll, pitch and yaw, plus the sway-yaw and
heave-pitch coupling that makes a hull turn and heel together.

Surge is separate: a slender body moving along its own axis entrains
almost nothing, and the right figure is Lamb's ``k1`` for a prolate
spheroid.  The implementation is checked against Lamb's published table
-- 0.500 for a sphere, 0.209 at 2:1, 0.059 at 5:1.

### What it changes

Steady turn rate barely moves, and should not: steady state is rudder
moment against yaw damping, with no inertia in it.  The transient is the
whole effect, and a coxswain's corrections live in the transient.

Eight, +-5 deg oscillating rudder:

| steering period | yaw amplitude, with added mass |
|---|---|
| 4 s | **62%** of before |
| 8 s | **79%** of before |

Frequency-dependent, as inertia must be.  For quick corrections the model
was overstating the effect of the rudder by about 60%.

### What it does not change

**Surge added mass is tiny** -- 3.3 kg on an eight, 0.4% of displacement
-- so this does *not* close the intracycle velocity gap of §32.  That
hypothesis, floated in the §31 candidate list, is wrong: a slender hull
slipping along its own axis entrains almost nothing.  The asymmetry
between surge and sway is the physics, not an artefact.

### A related negative result: flatness

§30's shape-preserving interpolant made `flatness` worth retesting, since
the reason it was abandoned -- Fourier ringing around corners -- was
partly the overshooting spline.  On the kinematics alone it looked like a
fix: flatness 0.8 at 14 harmonics brings crew COM velocity swing down to
where the measured fluctuation implies, with keyframe error of 1.4 deg
against a dataset whose own measurement SD is 5.5 to 11.1 deg.

Run through the full 6-DOF it makes things **worse** -- IVV 51.5% ->
54.5%, and crew acceleration peak-to-peak nearly doubles, 11.7 -> 22.2
m/s^2.

The reason is worth keeping: **the hull responds to crew acceleration,
not crew velocity.**  A flattened traverse has corners at the keyframes,
and corners are acceleration spikes.  Judging the fix on a velocity
metric hid that completely.  `flatness` is now plumbed through the rower
for experiment but stays off by default, and the original docstring's
verdict stands.

### Measured hull dynamics, for the record

From the club 2x boat phone, axis identified by stroke-band power and
validated by integrating to velocity (which reproduces the DGPS 37.3%
independently, at 35.3 / 37.8 / 35.8 / 35.7% across four clean windows):

| | measured | model 2x |
|---|---|---|
| hull surge accel, peak-to-peak | ~9.2 m/s^2 | 11.7 |
| hull surge velocity, peak-to-peak | ~1.57 m/s | 2.47 |

### Known limitation: no Munk moment

Added mass also produces *velocity-dependent* forces, not just the mass
matrix.  The most important for a slender hull is the Munk moment, which
is destabilising in yaw and is what makes a bare hull want to broach.
The mass-matrix terms are implemented here; the Munk moment is not, and
until it is, directional stability is set by the skeg and rudder alone.
This is now the leading known gap in the steering model.

## 36. Manoeuvring literature, and calibrating the Munk moment

§35 implemented added mass and left the Munk moment off, on the grounds
that at full strength it broaches an eight and "a model that broaches is
worse than one that lacks a term".  **That was wrong**, and a coxswain's
report is what showed it.

### The evidence that settles it

A racing shell that loses its skeg becomes uncontrollable.  The rudder is
**mounted on the skeg**, so both go together -- a detail the first test
got wrong by stripping the skeg and keeping the rudder, which is not a
thing that can happen.  Reported behaviour: violent slewing, crews
squaring blades on one side or dragging a hand to steer, and at ACRAs an
eight that lost its skeg crossed into the adjacent lane and hit another
crew inside 20-30 seconds.

Run against that, `munk_factor = 0` fails outright:

| factor | skeg + rudder fitted | both gone |
|---|---|---|
| 0.00 | holds (-2.6 deg) | **holds (-7 deg)** |
| 0.25 | holds | slews, 32 deg / 25 s |
| **0.35** | holds | **46 deg / 25 s, past 20 deg at 11.9 s** |
| 1.00 | broaches (+61) | broaches (+79) |

At zero the model says losing the skeg barely matters.  That is not a
conservative choice, it is a wrong one, and switching the term off to
avoid a bad behaviour hid a worse one.

### Steering authority, from the crew

Quantitative anchors for an eight at 24-30 spm, from the coxswain:

* full rudder: heading takes 1-2 strokes to shift meaningfully, then
  roughly **15 deg in 5 s, so about 3 deg/s at most**
* a typical steering input: about **1 deg/s**
* turns are planned well in advance; a full turn consumes a lot of water

At `munk_factor = 0.35` the model gives 2.15 deg/s at full (25 deg)
rudder and 1.2-1.4 deg/s at typical deflections -- the right order, and
the bare-hull divergence matches the ACRAs account.

**Known discrepancy, not resolved.**  The modelled turn rate is too
*insensitive* to rudder angle: 1.23 deg/s at 5 deg against 2.15 at 25
deg, a factor of 1.7 for five times the rudder, where the report implies
about three.  Rudder authority is under-modelled relative to the
instability, and the honest reading is that `munk_factor` is currently
absorbing an error that belongs to the rudder.  Fixing the rudder would
change the calibrated factor.

### That the sign is negative is now a test

`test_weathervane_is_stabilising` asserted ``yaw_from_sway > 0`` and now
asserts the opposite.  The old assertion held only because the model had
no Munk moment: a slender body in potential flow is *destabilised* by
drift, the entrained water giving a moment that turns the hull broadside.
The assembled boat is directionally unstable and is held straight by its
appendages -- which is precisely why losing them is catastrophic, and why
a coxswain steers continuously.

### The literature this rests on

**Added mass and slender-body theory**

1. Munk, M. (1924) *The aerodynamic forces on airship hulls*, NACA
   Rep. 184 -- the original destabilising moment.
2. Lamb, H. (1932) *Hydrodynamics*, 6th ed., art. 71 -- ellipsoid added
   mass; the source of the ``k1`` used for surge.
3. Jones, R.T. (1946) *Properties of low-aspect-ratio pointed wings* --
   the lift a slender hull generates in drift.
4. Newman, J.N. (1977) *Marine Hydrodynamics*, sec. 4.13 -- strip theory
   and the rigid-wall image.
5. Korotkin, A.I. (2009) *Added Masses of Ship Structures*, Springer --
   the standard compendium of sectional coefficients.
6. Fossen, T.I. (2011) *Handbook of Marine Craft Hydrodynamics and Motion
   Control*, Wiley, secs. 6.3-6.4 -- the added-mass Coriolis matrix used
   here, and Hoerner cross-flow drag.
7. Hoerner, S.F. (1965) *Fluid-Dynamic Drag*, ch. 3 -- cross-flow drag
   coefficients.
8. Lighthill, M.J. (1960) *Note on the swimming of slender fish* --
   slender-body momentum reasoning, extended to moderate aspect ratio by
   Candelier, Porez & Boyer (2011).
9. Fuwa, T. et al. (1973) -- combines small-aspect-ratio wing theory with
   slender-body theory including shed vorticity; the standard route to
   **viscous correction of the potential Munk moment**, which is exactly
   what a factor below one represents.
10. Skejic, R. & Faltinsen, O.M. (2008) -- nonlinear slender-body
    manoeuvring generalised to account for heel.

**Skegs, rudders and course stability**

11. MARIN, *Influence of Skeg on Ship Manoeuvrability at High and Low
    Speeds* -- the classic result that directional stability trades
    against manoeuvrability, and that if stability must be raised it is
    better bought with **movable rudder area than fixed skeg area**.
    Directly relevant to whether this model's skeg or rudder is mis-sized.
12. Yasukawa & Yoshimura, *The influence of skegs on course stability of
    a barge* (Ocean Eng., 2015).

**Rowing-specific dynamics**

13. Formaggia, L., Miglio, E., Mola, A. & Montano, A. (2009) *A model for
    the dynamics of rowing boats* -- the 6-DOF formulation this simulator
    is built on.
14. Formaggia, L., Mola, A., Parolini, N. & Pischiutta, M. (2010) *A
    three-dimensional model for the dynamics and hydrodynamics of rowing
    boats*, Proc. IMechE Part P.
15. Mola, A., Del Grosso, L., Formaggia, L. & Miglio, E. (2006)
    *Performance prediction of olympic rowing boats accounting for full
    dynamics*, Comm. SIMAI Congress 1.  **Structurally confirms the
    approach taken here**: the hull load is split into "a component
    proportional to the acceleration vector -- the mass matrix M -- and a
    component proportional to the velocity vector -- the damping matrix
    S", both from a potential-flow solve.  Same decomposition, reached by
    potential flow rather than strip theory.  It is planar (surge, heave,
    pitch) so it does not calibrate yaw.
16. Day, A.H., Campbell, I., Clelland, D. & Cichowicz, J. -- experimental
    unsteady hydrodynamics of a single scull; acceleration measurably
    affects viscous drag.  Also the source of the observation that racing
    boat velocity **fluctuates by roughly 20% about the mean**, which is
    a third independent corroboration of the 37.3% peak-to-peak of §24.
17. Robinson, M. et al. -- experimental drag coefficient of a men's 8+
    racing shell (SpringerPlus 3:512, 2014).
18. Sliasas, A. & Tullis, S. -- shell-velocity-coupled blade
    hydrodynamics.
19. Knarr, C., Kwoun, H. & Kleshnev, V. (2024/2026) *Using IMU sensors to
    compare rowing ergometers with rowing on the water*, Proc. IMechE
    Part P.
20. Kleshnev, V. *Biomechanics of Rowing* -- segment power shares, trunk
    swing, and the rate-dependence of boat velocity fluctuation.

**A calibration datum still wanted.** Reported within-stroke attitude
variation for racing shells is under 1 deg in pitch and under 5 deg in
roll and yaw.  The model can be checked against that directly, and it is
the cheapest remaining validation of the steering model.

## 37. Two dynamics implementations, one boat

The trajectory optimiser does not use `RowingSimulator`.  It has its own
CasADi dynamics in `coxswain/river/sixdof.py`, because the optimiser
needs expressions it can differentiate rather than a stepper.  That is a
reasonable design and it carries an obvious hazard, which duly happened:
§35 and §36 added entrained water, distributed cross-flow drag and the
Munk moment to the **simulator only**.

For an eight that left the two models disagreeing by a factor of 1.2 in
yaw inertia; for a single, by a factor of 9.  The optimiser was planning
lines for a boat several times more manoeuvrable than the one the
simulator would have flown, and every trajectory result in §33 was
computed on the old physics.

`SixDofModel` now carries the same `AddedMass`, the same
`CrossFlowHull` station table and the same `munk_factor` as the
simulator, and a test asserts they match rather than trusting that they
do.  A mismatch here is invisible in every other test: both models are
individually self-consistent, both converge, and only the comparison
catches it.

### It made the solve easier, not harder

The concern was that a boat which is harder to turn and directionally
unstable would be harder to steer down a narrowing channel.  The opposite
happened: blocks converge in 99-129 iterations against 155-168 before.

That is worth a sentence of interpretation.  Added mass raises the yaw
inertia, which lengthens the boat's yaw time constant, which **smooths
the map from rudder to path**.  A boat that responds instantly to rudder
gives the optimiser a stiff, twitchy control problem; one with realistic
inertia gives it a better-conditioned one.  The physics being right and
the numerics being easier are not a coincidence here.

## 38. Chasing the fluctuation: one fix, one withdrawal, one failure

### Kleshnev's force curve, and why it belongs

`OarForceProfile` applied a **symmetric half-sine** over the drive, which
peaks at 50% of the drive length.  Real force curves are front-loaded.
Two of Kleshnev's published figures pin the shape:

* peak force at **40% of the drive length**
* force already down to **74% of peak at 60%** of the drive, where a
  half-sine is still at 95%

Fitting ``u**a (1-u)**b`` to both gives ``a = 1.4852, b = 2.2278`` and
reproduces each to three figures.  The old shape is retained as
``shape="half_sine"`` because every catalogue speed calibration predates
the change.

The new shape carries 0.5385 of its peak as its mean against the
half-sine's ``2/pi``, so peak forces were rescaled by 1.182 to hold mean
thrust -- and hence every calibrated speed -- fixed.  Speeds land within
1.5% of where they were.

**It is a small effect on the fluctuation**: 2x IVV 57.9% -> 56.1%.  It
is in the model because it is right, not because it fixed anything.

### A diagnosis I am withdrawing

Differencing the boat-mounted and rower-mounted IMUs from the club 2x
session appeared to give the crew's acceleration *relative to the hull*
directly, and it produced a clean result across eight gated windows:

| | measured | model |
|---|---|---|
| hull accel ptp | 8.88 m/s^2 | 11.70 |
| crew-relative accel ptp | 11.31 | 9.90 |
| hull / crew-relative | **0.785** | **1.18** |

Read at face value that is a striking finding -- the model's crew moves
*less* than the real crew while its hull moves more, so the coupling
would have to be wrong, with thrust amplifying the crew reaction where it
should partly cancel it.

**It does not survive scrutiny.**  The pelvis phone rotates with the
rower, whose trunk swings some 50 degrees through the stroke, so a
device-frame axis is not a fixed direction in the boat frame and the
"crew-relative" number is contaminated by that rotation.  Rotating both
signals into the world frame with the logged attitude gives a hull
acceleration of 1.28 m/s^2 -- against 8.88 in the device frame, and the
device-frame figure is the one that is *independently validated*, since
integrating it recovers the DGPS intracycle variation (35.3 / 37.8 / 35.8
/ 35.7% across four windows against 37.3% from position).

So the attitude correction is wrong somewhere, and until it reproduces
the validated hull figure nothing built on the rotated data can be
trusted.  **The measured hull acceleration stands; the crew-relative
number does not, and the conclusion drawn from it is withdrawn.**

What remains solid is narrower and worth stating plainly: the model's
hull acceleration is 10.6-11.7 m/s^2 against a measured 8.88, and its
intracycle velocity variation is 56% against a measured 37-41%.  *Why*
is still open.

### The leg run failed on the corrected physics

With added mass, cross-flow damping and the Munk moment in the
optimiser's dynamics, the receding-horizon leg **stalled at about 409 m**
-- worse than the 476 m it reached on the old physics.  Blocks 19 and 23
made negative progress, block 21 went infeasible with 6 m of clearance,
and block 22 ran for over eight hours.

This is not a surprise and it should not be papered over.  The boat is
now correctly harder to turn and correctly directionally unstable, so the
station-450 pinch is a genuinely harder control problem than the old
over-manoeuvrable boat faced.  A three-stroke horizon that could
previously bluff its way through a bend no longer can.

The fix is horizon length and warm starting, not physics: the dynamics
are better than they were, and the optimiser has to be made equal to
them.

## 39. The fluctuation gap: what it is not

The intracycle velocity variation is the one physical quantity the model
still gets wrong: **56% against a measured 37-41%**, and hull surge
acceleration 10.6 m/s^2 against a measured 8.88.  This section records
what has been eliminated, because the eliminations are most of the value.

### The coupling is implemented correctly

The momentum identity

    a_hull = (F_ext - m_crew a_rel) / m_total

holds in the assembled simulator to 0.24 m/s^2, correlation 0.9993, over
a full cycle.  There is no bug in how crew motion drives the hull.  The
decomposition:

| term | peak-to-peak |
|---|---|
| crew reaction, ``-m_crew a_rel / m_total`` | 8.48 m/s^2 |
| external, ``F_ext / m_total`` | 3.15 m/s^2 |
| correlation between them | **+0.136** |
| resulting hull acceleration | 10.58 m/s^2 |

The crew term dominates by nearly three to one.  Anything that fixes this
has to act on crew motion or on the phase of thrust against it.

### Six candidates, eliminated

1. **Segment sequencing** (§29-30).  Implemented as a within-phase warp;
   worth about 2.7 points at best and only with a magnitude fitted to the
   answer.
2. **Traverse flatness** (§35).  Looked right on kinematics and makes the
   6-DOF *worse* -- IVV 51.5 -> 54.5%, crew acceleration nearly doubling.
   The hull responds to crew **acceleration**, and a flattened traverse
   has corners.
3. **Surge added mass** (§35).  3.3 kg on an eight, 0.4% of displacement.
   A slender hull moving along its own axis entrains almost nothing.
4. **Drive force-curve shape** (§38).  Moving the peak from 50% to
   Kleshnev's 40% is correct and is kept, but it is worth 1.8 points.
5. **Recovery retiming.**  ``recovery_arrival`` implements "slow into the
   catch" after Kleshnev.  Scanned from 1.0 to 0.40: IVV moves 56.0 ->
   56.3%, and the phase-averaged shape agreement gets *worse*, 0.325 ->
   0.216.  It is not the missing piece.
6. **An implementation bug in the coupling.**  Ruled out above.

Individually, every crew kinematic quantity validates: trunk swing 54.5
deg against Kleshnev's 50.8 and a pelvis IMU implying about 50; seat
travel 0.597 m against a literature 0.60-0.70; segment masses matching de
Leva exactly.

### The one live lead, and why it is not yet evidence

Phase-averaging the measured hull surge acceleration and the model's, each
aligned on its own trough (the catch), gives a shape correlation of only
**+0.325**:

| | peak at phase | trough at phase |
|---|---|---|
| measured | **0.74** (late recovery) | 0.01 |
| model | **0.18** (early drive) | 0.01 |

If that is real it says the model gains its speed during the drive where
the real boat gains it during the recovery, which would be a phasing
error rather than an amplitude one, and would explain why a 19% error in
acceleration amplitude produces a 40% error in velocity variation.

**It is not yet evidence.**  The measured profile is averaged across
eight windows with independently detected periods, and any period error
smears the average -- preferentially destroying the sharp drive-phase
features and flattening exactly the part of the curve the comparison
turns on.  Establishing it needs per-stroke catch detection rather than
spectral period estimation, so that strokes are aligned on an event
rather than on an assumed frequency.

That is the next thing to do, and until it is done the phasing claim
stays a hypothesis.

## 40. The drive is mistimed, and Gibbs was never the problem

### Catch-aligned measurement

§39 could not use the phase comparison because the measured profile was
averaged over windows with spectrally-estimated periods, and period error
smears exactly the features the comparison rests on.  Replacing that with
**per-stroke catch detection** -- find the sharpest deceleration in each
cycle, resample between successive catches -- gives 162 catch-aligned
cycles and a clean profile.

It validates: hull velocity minimum at phase 0.14 (just after the catch)
and maximum at 0.80 (on the recovery), which is the textbook behaviour
and was not imposed.

| | measured | model |
|---|---|---|
| velocity min / max phase | 0.14 / 0.80 | 0.09 / 0.45 |
| velocity peak-to-peak | 1.31 m/s | 2.08 |
| **mean abs hull acceleration through the drive** | **0.71 m/s^2** | **3.47** |

**The real boat is nearly in equilibrium during the drive.**  Thrust and
crew reaction cancel to within 0.71 m/s^2.  The model's boat is driven
forward five times harder, and then coasts.

### The cause: the crew stops accelerating too early

The model's crew centre of mass reaches peak speed at **37% of the
drive**; Kleshnev reports peak handle velocity at **60%**.  After the
model's crew passes its peak it is decelerating, and its reaction then
*pushes the hull forward* -- adding to thrust rather than opposing it.
That is the +0.136 correlation of §39 and the 3.47 m/s^2.

Retiming the whole body with the §30 within-phase warp moves the peak to
52% and takes IVV from 56.0% to **49.4%**, the largest single improvement
so far.  It binds at -0.20 on reach, because the body is retimed while
the oar sweep is not, and the hands are pinned to the handle.

### Two bugs in the traverse warp

`uniform_traverse` was abandoned in §25 for truncating the stroke.  Two
separate mechanisms were doing that.

1. **The composition was not onto.**  `np.maximum.accumulate` on a
   clipped warp flattens any decreasing stretch into a plateau and pins
   everything past an overshoot at the endpoint, so the warp stopped
   spanning its interval and the extreme postures were never sampled.
   Fixed by renormalising the drive onto ``[0, f]`` and the recovery onto
   ``[f, 1]`` after enforcing monotonicity.  A reparameterisation
   traverses the same path, so travel must be preserved exactly.
2. **The refit truncated it.**  A warped three-harmonic signal is no
   longer three-harmonic, and refitting it at ``KEYFRAME_HARMONICS = 3``
   discarded the difference.  Crew CoM travel against harmonic count, at
   blend 0.9: 0.539 (3), 0.606 (6), 0.676 (14), 0.696 (20), against 0.744
   unwarped.

### Gibbs was a wrong diagnosis

With both fixed, the warp preserves travel and IVV gets **worse** -- 62.7%
at 20 harmonics against 56.0% at 3.  This was attributed to Gibbs ringing
around the corners the warp introduces.  **That was wrong**, and the
check is direct.  Representing one warped joint-angle profile:

| basis | range deg | d2/dt2 ptp | travel kept |
|---|---|---|---|
| Fourier, 3 harmonics | 71.79 | 12.7 | 94.4% |
| Fourier, 8 | 75.78 | 27.1 | 99.6% |
| Fourier, 20 | 76.07 | **38.2** | 100.0% |
| dense periodic spline | 76.07 | **307.9** | 100.0% |
| **spline, knots at catch and finish** | **75.66** | **34.7** | **99.5%** |
| **raw warped signal** | **76.07** | **38.7** | **100%** |

The raw signal's own second derivative is 38.7.  Fourier at 20 harmonics
gives 38.2 -- that is *faithful*, not ringing.  Three harmonics were
damping the accelerations threefold, and raising the count merely stopped
hiding them.

So the warped motion genuinely demands those accelerations, and a better
basis would represent them more honestly and make the hull fluctuate
more.  This is the same lesson as §35: **flattening crew velocity
concentrates crew acceleration into the reversals**, and the hull feels
the acceleration.

### On the basis question anyway

A cubic spline knotted at the catch and the finish is nonetheless the
right representation for this signal -- 99.5% of travel with a
well-behaved second derivative -- and a *dense* spline is catastrophic,
307.9, because it interpolates resampling noise.  Putting the corners on
knots is what matters; it is the Gauss-Lobatto instinct applied where it
bites, the stroke being two smooth phases joined at near-corners rather
than one smooth periodic signal.  Worth doing, but it is a representation
improvement, not the fluctuation fix.

### Where this leaves it

The live hypothesis is now specific and literature-anchored: **the
model's drive is mistimed**, the crew reaching peak speed at 37% of the
drive where Kleshnev measures 60%, so the crew reaction stops opposing
thrust a third of the way in.  Fixing it properly needs the body and the
oar sweep retimed together, since retiming the body alone breaks the
reach constraint -- which is exactly what the densely sampled kinematics
requested in `DATA_REQUESTS.md` would settle.

## 41. Drive timing falsified; the force loop is the live suspect

### The §40 hypothesis, tested properly and rejected

§40 proposed that the fluctuation gap came from the model's crew reaching
peak centre-of-mass speed at 37% of the drive where Kleshnev measures
60%.  Retiming the body alone appeared to support it -- IVV 56.0% ->
49.4%.  But retiming the body while leaving the oar sweep prescribed
moves the shoulders out from under the handle, and the improvement came
from distorting the arms rather than from the timing.

`drive_timing_warp` now applies the same warp to the **oar sweep and the
joint angles together**, which is what keeps the hands on the handle.
Done consistently:

| lag | peak, % of drive | CoM travel | IVV | drive abs acc |
|---|---|---|---|---|
| 0.00 | 37 | 0.744 | 56.0% | 3.46 |
| 0.10 | 47 | 0.747 | 54.1% | 3.23 |
| **0.20** | **58** | 0.756 | **57.2%** | 3.22 |
| 0.30 | 65 | 0.772 | 64.9% | 3.71 |

Travel is preserved, as a reparameterisation must, and Kleshnev's 60% is
reached at lag 0.20.  **The fluctuation does not improve.**  The
hypothesis is rejected; the warp is kept because it is correct machinery
and defaults to zero.

### What is left: the crew and the blade are prescribed independently

Through the drive the model has the crew reaction and the blade thrust
pushing the hull the same way in **62% of samples**:

| phase | crew reaction | oar | same sign |
|---|---|---|---|
| 0.000 | -831 N | +2 N | no |
| 0.096 | -341 | +414 | no |
| 0.144 | +417 | +519 | **yes** |
| 0.192 | +715 | +403 | **yes** |
| 0.288 | +392 | +26 | **yes** |

A positive crew reaction means the crew is *decelerating* -- their
reaction pushes the hull forward.  So through the second half of the
drive the model has a rower who is being decelerated **and** pulling hard
on the handle at the same time, both driving the boat forward.

**On the water, what decelerates the rower is the handle.**  A large
handle force and a decelerating body are the same event seen from two
ends, not two independent events that happen to coincide.  In this model
they *are* independent: `OarForceProfile` prescribes the blade loading
from a fitted curve, and the crew reaction comes from prescribed
kinematics, and nothing enforces the rower's own Newton equation between
them.

That is consistent with everything measured.  On the water the two very
nearly cancel through the drive -- mean absolute hull acceleration 0.71
m/s^2 against the model's 3.46 -- which is what a closed force loop
through one body produces.  It is also consistent with the failure of
every kinematic remedy tried so far (§30, §35, §38, §40): if the defect
is that two prescribed quantities are mutually inconsistent, no amount of
retiming one of them will fix it.

### Why this is not yet a fix

Closing the loop means the handle force and the crew's acceleration have
to be solved together rather than both imposed -- which is exactly what
the predictive formulation of §27 was for, and why it is worth returning
to now that its power model is right.  It is a structural change to how
the crew drives the boat, not a parameter, and it should not be attempted
on a hunch.

The measurement that would confirm or kill it is the same one
`DATA_REQUESTS.md` asks for: synchronised handle force and rower
kinematics through the stroke.  With both, the rower's force balance can
be checked directly instead of inferred from the hull.

## 42. Correcting §41: there is no double count

§41 proposed that the crew reaction and the blade thrust were both being
charged to the hull with the handle path counted twice, and called it the
live suspect.  **That is wrong.**  Worked through properly it does not
hold, and the model's bookkeeping is correct.

### The derivation

Take the oar as massless, handle at ``r_h`` from the pin and blade at
``L - r_h`` on the other side.

*Oar, moments about the pin:* ``F_h r_h = F_b (L - r_h)``.

*Oar, force balance:* the pin carries ``F_pin = F_h + F_b``, so

    F_pin = F_b [(L - r_h)/r_h + 1] = F_b L / r_h
    =>  F_b = (r_h / L) F_pin

*Crew:* ``m_c a_c = F_stretcher + F_handle_on_crew``.

*Force on the boat*, adding the stretcher path and the pin path and using
the crew equation to eliminate the stretcher term, the handle force
cancels identically and what is left is

    F_boat = -m_c a_c + F_b

**The handle force does not appear.**  It is internal to the loop
crew -> handle -> oar -> pin -> boat -> stretcher -> crew, and it cancels
against itself.  The hull needs only the total crew reaction and the
blade force; how that total splits between stretcher and handle is
invisible to it.

### The model implements exactly this

`hull_load` computes ``net_force = gearing * F_o`` with
``gearing = r_h / L``, which by the line above **is the blade force**.
Added to the moving-mass reaction ``-m_c a_c`` from
`moving_mass_reaction`, the hull receives ``-m_c a_c + F_b``.  That is
the derivation, and it is Formaggia eq. (14a).  No term is counted twice.

The 62% same-sign statistic in §41 is real but means nothing on its own:
crew reaction and blade force *may* push the same way without any
inconsistency, because they are genuinely separate contributions.

### What actually remains

The two prescriptions are independent **in provenance**, which is a
different and smaller claim than the one §41 made.  Crew kinematics come
from Caplan & Gardner's four keyframes; the oarlock force peak is
calibrated to make each boat hit its known cruising speed.  Nothing makes
them describe the same crew on the same outing, and the measurement says
real rowing couples them tightly -- mean absolute hull acceleration
through the drive is 0.71 m/s^2, against 3.46 in the model.

So the model is not wrong in its physics; it is being driven by two
inputs that were never measured together.  That is not something a
derivation can fix, and it is precisely what the synchronised handle
force and rower kinematics in `DATA_REQUESTS.md` would supply.

### The standing lesson

§41 was written after a plausible mechanism and a suggestive statistic,
without doing the algebra.  The algebra takes ten lines and says the
opposite.

## 43. It is the timing, not the amplitude

### Inverting the hull equation

The hull equation ``a_hull = (F_b - m_c a_rel) / m_total`` can be solved
for the crew's acceleration given a measured ``a_hull``.  The blade force
is known to have the right *mean* -- it is calibrated so the boat hits the
measured 3.82 m/s -- so this gives a crew acceleration profile inferred
from the hull, without touching the pelvis channel whose device-frame
rotation §38 could not correct.

Against the model, on the catch-aligned grid of §40:

| phase | implied | model |
|---|---|---|
| 0.01 | +3.32 | +5.30 |
| **0.14** | **+3.00** | **-1.86** |
| **0.20** | **+1.88** | **-4.28** |
| 0.26 | +0.05 | -2.90 |
| 0.45 | -0.74 | -0.79 |
| **0.70** | **-4.43** | -0.48 |
| **0.76** | **-5.00** | -1.17 |
| 0.95 | +2.13 | +3.85 |

| | implied | model | ratio |
|---|---|---|---|
| peak-to-peak | 9.22 | 9.83 | **1.07** |
| mean abs through the drive | 2.15 | 3.39 | 1.58 |

**The amplitude is right to 7%.**  Through the mid-drive the two profiles
have **opposite signs**, and in the late recovery the model has almost
nothing where the real crew has its largest excursion.

### Why every previous remedy failed

Sequencing (§30), traverse flatness (§35), the drive force curve (§38),
the uniform-traverse warp (§40) and harmonic count all adjust *how much*
the crew moves, or how sharply.  The amount was never wrong.  Six
attempts moved a quantity that was already correct to 7%, which is why
none of them closed more than a couple of points and several made it
worse.

### Why the §41 test came out null

`drive_timing_warp` retimes the joint angles **and the oar sweep
together**, because retiming the body alone breaks the reach constraint.
But moving both preserves the phase relationship between crew motion and
blade force -- and that relationship is the thing that is wrong.  The
test was null because it could not perturb the quantity in question.

What is needed is the crew's motion retimed *relative to* the blade
force, which is precisely the case that fails on reach.

### The reach failure is the finding, not the obstacle

Taken at face value, the model's rig-and-arm geometry **cannot
accommodate the measured timing**: asking the body to keep accelerating
through the mid-drive, as the inference says it does, moves the shoulders
out from under a handle whose path is fixed by the oar sweep.

Two things could give.  The **arm posture table**, which §29 already
flagged as the un-measured input in the chain -- `DEFAULT_ARM_POSTURE`
fixes how much elbow bend the rower carries, and a different posture
changes how far the shoulders may travel while the hands stay on the
handle.  Or the **oar sweep**, if the blade's angular history through the
drive is not what `OarAngleSweep` prescribes.

Both are testable against the same synchronised measurement, and the ask
in `DATA_REQUESTS.md` should be stated in those terms: not "rower
kinematics" in general, but **handle position and body position sampled
together through the drive**, which is what fixes the relative phase.

## 44. The causal chain is inverted

§43 named two candidates for the reach failure that blocks retiming the
crew relative to the blade: the arm posture table, or the oar sweep.

### The arm posture is not it, and is not even active

`DEFAULT_ARM_POSTURE` is consulted only when ``hand_targets is None``.
Every boat built from the catalogue has a rig, so ``hand_targets`` is
set, the arms are solved by inverse kinematics to reach the handle, and
the posture table is bypassed entirely.  Three quite different postures
-- default, early break, fully compliant -- give **bit-identical**
results at every retiming amplitude tested.

The reach failure is therefore pure geometry: retiming the body moves the
shoulders further from the handle than the arm is long.  No posture
choice can absorb it.

### Which leaves the sweep, and the direction it points

The model prescribes the oar sweep and requires the arms to reach
whatever handle position that implies.  **Physically it is the other way
round.**  The rower's legs, trunk and arms put their hands somewhere; the
handle is in their hands; the oar angle is whatever that hand position
makes it.  The hands do not chase the oar -- the oar follows the hands.

That inversion is exactly what blocks the fix.  Retiming the body while
the sweep is prescribed asks the arms to bridge a gap that has grown;
retiming both together (§41) preserves the very phase relationship that
§43 shows is wrong.  There is no third option while the sweep is an
input, which is why the reach constraint appears unavoidable.  It is not
unavoidable; it is an artefact of the arrow pointing the wrong way.

With the oar following the hands, retiming the body retimes the blade
automatically and consistently, reach cannot fail by construction, and
the blade force follows from the blade's actual motion through the water
rather than from a prescribed profile that was calibrated separately.

### This makes Route C the main line, not a side branch

`coxswain.crew.predictive` already does this -- "the oar angle *follows*
from where those put the hands" -- and it was built as an experiment in
solving for the stroke rather than prescribing it.  §43 says the
prescribed chain cannot represent the measured timing at all, so the
inversion is not an alternative formulation to compare against.  It is
the correction.

The work that remains is to carry the inversion into the main kinematic
model rather than only the predictive one: hands from the body, oar angle
from the hands, blade force from the oar's motion.  That is a structural
change to `JointDrivenRower` and `OarAngleSweep`, and it should be
planned rather than attempted piecemeal.

### What is now known, end to end

1. Crew acceleration **amplitude** is right to 7% (§43).
2. Its **phase** relative to the blade force is wrong -- opposite signs
   through the mid-drive.
3. Six amplitude-based remedies failed because they addressed a quantity
   that was already correct (§30, §35, §38, §40, §41, harmonic count).
4. The phase cannot be corrected while the sweep is prescribed, because
   the arms cannot bridge the gap and the posture table is inert.
5. The fix is to invert the chain so the oar follows the hands.

The measurement that would confirm it before the rewrite is the one
`DATA_REQUESTS.md` asks for, now stated precisely: **handle position and
body position sampled together through the drive.**

## 45. The inversion is not a drop-in

§44 concluded that the causal chain runs backwards -- the model
prescribes the oar sweep and makes the arms reach for it, where
physically the oar follows the hands -- and that inverting it would let
the crew be retimed relative to the blade without the reach constraint
failing.

### The geometry works

Released from `hand_targets`, the arms are placed by the (previously
inert) posture table, and the body can be retimed all the way to a lead
of -0.40 with **no reach failure** and travel preserved:

| lead | CoM travel | CoM speed peak, % of drive | implied oar sweep |
|---|---|---|---|
| 0.00 | 0.757 | 36 | 110.7 deg |
| -0.20 | 0.755 | 61 | 109.5 |
| -0.40 | 0.753 | 73 | 109.2 |

Reach cannot fail, by construction: the oar goes where the hands are.
The implied sweep is a stable ~110 degrees, against the 90 that
`OarAngleSweep` prescribes (catch 55, finish -35) -- worth noting on its
own.

### The dynamics do not

Deriving the oar angle from the hands and feeding it to the existing
force model makes the fluctuation **worse**:

| lead | IVV | drive abs acc | speed |
|---|---|---|---|
| 0.00 | 62.9% | 3.58 | 3.499 |
| -0.20 | 91.1% | 3.64 | 2.763 |
| -0.40 | 172.2% | 5.32 | 2.134 |

against 56.0% and 3.76 m/s for the prescribed chain.

**Because only half the chain was inverted.**  The oar angle now follows
the hands, but `OarForceProfile` still prescribes the blade loading as a
function of *stroke phase*.  Retiming the body therefore moves the oar's
actual motion while the force keeps its original clock, and the two are
more inconsistent than before, not less.  The speed collapse is the same
fault seen from the other side: thrust arriving when the blade is no
longer where the profile assumes.

### What the full change requires

The chain has to be inverted end to end:

    body -> hands -> oar angle -> blade slip -> blade force

The last link is the one missing.  The blade force must come from the
blade's motion through the water -- which `BladeModel` already computes,
slip-based after Cabrera & Ruina, and which the simulator currently uses
only to *rescale* a prescribed profile via `_blade_efficiency`.  Making
it the source rather than a correction means `OarForceProfile` stops
being an input, and with it goes the per-class peak-force calibration
that currently makes each boat hit its known speed.

That is a substantial rewrite touching `JointDrivenRower`,
`OarAngleSweep`, `OarForceProfile` and the catalogue's calibration, and
it removes the mechanism by which boat speed is currently made correct.
`coxswain.crew.predictive` already implements this chain, which is why
§44 called it the main line rather than a branch.

### The honest state

The diagnosis is complete and consistent: crew acceleration amplitude
right to 7%, its phase relative to the blade wrong, unfixable while
either the sweep or the force is prescribed by phase.  The fix is
structural and is now scoped.  It should be done as a deliberate piece of
work against the synchronised handle-and-body measurement that
`DATA_REQUESTS.md` asks for, not by further probing -- half-inversions
make things worse, and this section is the evidence.

## 46. The rewrite is not justified yet

§45 scoped the end-to-end inversion -- body, hands, oar angle, blade
slip, blade force -- and noted it removes the per-class peak-force
calibration, so boat speed stops being fitted and becomes predicted.
That is better science, and it is also a test the current model has never
had to pass.

Rather than rewrite four modules to find out, the chain was assembled in
a **one-degree-of-freedom surge model**: the same crew kinematics, the
oar angle inverted from the hands, the blade force from slip after
Cabrera & Ruina, hull drag from the same resistance model, and no
prescribed force profile anywhere.

| lead | predicted speed | IVV |
|---|---|---|
| 0.00 | **5.333 m/s** | 63.9% |
| -0.10 | 5.504 | 51.0% |
| -0.20 | 6.031 | 55.6% |
| -0.40 | 7.732 | 84.9% |

**The crew actually rowed 3.82 m/s.**  The inverted chain over-predicts
by 40%, and the fluctuation does not improve.

### What that means

The rewrite would exchange a model whose speed is right because it was
calibrated and whose fluctuation is wrong, for one whose speed is wrong
by 40% *and* whose fluctuation is still wrong.  On this evidence it is
not justified, and running it would have cost four modules and the
catalogue calibration to arrive somewhere worse.

It also says something the calibrated model was hiding.  Speed is
currently correct by construction -- `PEAK_OARLOCK_FORCE` is fitted per
class to make it so -- and that fitting has been absorbing an error
somewhere else.  Given the same kinematics and a physical blade model,
the boat goes far too fast, so either the blade is too effective at the
slips these kinematics produce, or the kinematics move the handle too
quickly.

The second would be consistent with §43 in a way worth checking: the
crew's *acceleration* amplitude is right to 7%, but that was measured
against the constrained-arm rower.  With the arms released to the posture
table the hand path is different, and hand speed is what sets slip.

### Standing position

Three structural candidates have now been tested and none is the fix:
the force-loop double count (§42, does not exist), the drive retiming
(§41, null when done consistently), and the causal inversion (§45-46,
worse).  The diagnosis of §43 stands and is narrow -- amplitude right,
phase wrong -- but every mechanism proposed for the phase error has
failed on test.

The next thing is not another mechanism.  It is the measurement:
handle position and body position sampled together through the drive,
which fixes the phase directly instead of inferring it from the hull and
guessing at what would produce it.

## 47. Correcting §46: the blade model is sound, the arm table is not

§46 ran the inverted chain in one degree of freedom, predicted 5.33 m/s
against a measured 3.82, and concluded the rewrite was not justified.
The 40% error was real but **misattributed**.

Repeating the experiment with the model's **own prescribed oar sweep**
instead of one derived from free arms, everything else identical:

| c2 scale | predicted speed | IVV |
|---|---|---|
| **1.00** | **4.102 m/s** | 64.6% |
| 0.60 | 4.039 | 57.2% |
| 0.40 | 3.882 | 55.2% |

**4.10 m/s against a measured 3.82 -- 7% high, at full blade
coefficient.**  That is a good first-principles prediction, and it is one
the model has never before had to make: boat speed is normally correct by
construction because `PEAK_OARLOCK_FORCE` is fitted per class to make it
so.  Given the crew's kinematics, the oar's geometry, the slip-based
blade of Cabrera & Ruina and the hull's own drag, the boat goes very
nearly the right speed with nothing fitted.

**The blade model passes.  The 40% was the hand path.**

### What actually breaks the inversion

| | sweep | hand / handle travel |
|---|---|---|
| prescribed `OarAngleSweep` | 90 deg | 1.226 m |
| derived from free arms | 110 deg | 1.440 m |

Released from the handle constraint, the arms carry the hands 17%
further and correspondingly faster, which inflates blade slip and with it
thrust.  The arm posture table is the cause -- and it is the input §29
flagged as never having been measured, `DEFAULT_ARM_POSTURE`, which has
sat inert in the code because every rigged boat overrides it with
`hand_targets`.

So §45's inversion did not fail because the causal chain is wrong.  It
failed because inverting the chain makes the model depend, for the first
time, on an arm posture that was never validated -- and it is wrong by
17% in hand travel.

### Where this leaves the rewrite

The position of §46 is withdrawn.  The inversion is not disqualified; it
is blocked on one specific unmeasured input, and the blade model behind
it is now independently validated to 7% on a quantity it was not fitted
to.

That also sharpens the data request to something much smaller than
"rower kinematics".  What is needed is **the hand path through the
drive** -- where the handle is relative to the body, sampled through the
stroke.  With that, `DEFAULT_ARM_POSTURE` becomes a measurement, the
inversion becomes testable, and the phase question of §43 can be
addressed rather than guessed at.

## 48. Sculling boats were rowing a sweep arc

### How it surfaced

§47 established that the inverted chain fails on `DEFAULT_ARM_POSTURE`,
the never-measured arm table.  Rather than ask for that measurement, it
can be **extracted**: the constrained rower's hands lie on the handle by
construction, so its arm configuration through the drive is exactly the
posture the rig implies.

Measured off the 2x:

| % of drive | extension | prescribed table |
|---|---|---|
| **0 (catch)** | **0.562** | 0.970 |
| 25 | 0.742 | -- |
| 38 | 0.795 | -- |
| 50 | 0.752 | 0.960 |
| 100 (finish) | 0.370 | 0.460 |

**Arm extension of 0.56 at the catch means the arms are 44% bent.**  A
rower at the catch has straight arms; it is the first thing anyone is
taught.  Worse, the extension then *rises* to 0.795 by mid-drive -- the
arms straightening as the legs go down, which nobody does.

The arms had been acting as a silent slack variable, bending to absorb a
geometric inconsistency between the body, the rig and the oar arc.

### The bug

`OarAngleSweep` defaults to catch 55 deg, finish -35 deg: a **90 degree
arc**.  The catalogue never overrode it, so every boat used it.  But
sweep and sculling arcs are not the same thing:

| rig | catch | finish | total |
|---|---|---|---|
| sweep | 53-58 deg | 32-35 | 87-90 |
| **sculling** | **60-66** | **42-50** | **104-110** |

A sculler holds two handles and swings them through a far wider arc than
a sweep athlete swings one.  The 1x and 2x carry `SCULLING_OAR` and were
rowing a sweep arc, so the handle never reached far enough at the catch
and the arms bent to make up the difference.

**§45 had already found the right number without knowing it.**  Releasing
the arms and letting the oar follow the hands produced a 110 degree arc,
which was read as evidence the free arms were wrong.  They were right;
the prescribed sweep was wrong.

### The fix

`SWEEP_ARC` (56 / -34) and `SCULLING_ARC` (65 / -45) in the catalogue,
assigned by rig type.  Arm extension through the 2x drive becomes:

| % of drive | before | after |
|---|---|---|
| 0 | 0.562 | **0.736** |
| 25 | 0.742 | **0.933** |
| 75 | 0.487 | 0.575 |
| 100 | 0.370 | 0.479 |

The shape is now right -- long through the leg drive, drawing to the
finish -- where before it was inverted.  Residual bend at the catch
(0.736 against a straight-armed 0.95) says something else is still short
by a few centimetres: the rig span, the inboard, or how far the body
reaches at the catch.

### What it does and does not fix

2x speed 3.762 -> 3.665 m/s against a measured 3.82, and IVV 56.0% ->
58.4%.  **The fluctuation is not fixed.**  This is a geometry correction,
made because it is right, and it does not close §43's phase gap.

It does remove a confound that had been sitting under every experiment in
§39-47: those were all run on a sculling boat whose oar arc was 18% too
narrow and whose rower's arms were bent at the catch.
