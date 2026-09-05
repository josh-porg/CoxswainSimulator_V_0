# Damping in six degrees of freedom

Every damping term in this model used to be **quadratic**, and three of
the six degrees of freedom had no hull damping at all. This page records
what was wrong, what replaced it, where each number comes from, and how
far any of it should be trusted.

---

## The failure that found it

A women's veteran coxed four, rate 30, rowing a leg of Tail of the Lake.
The 3-D render showed a boat that was not sitting flat on flat water.
Measured over the run:

| | heave | pitch rms | max pitch |
|---|---|---|---|
| coxed four @ r30 | +0.56 m mean, **1.56 m peak** | **9.7°** | **±25.5°** |
| coxed four @ r24 | 0.044 m | 0.28° | 0.6° |
| eight @ r28 | 0.032 m | 0.94° | 2.6° |

A racing shell pitches a fraction of a degree. Riding 1.5 m clear of its
own waterline is not a modelling inaccuracy, it is a different boat.

Three things ruled out immediately:

* **Not the integrator.** Identical at `dt` = 0.005, 0.01 and 0.02.
* **Not the initial condition.** The static trim is 0.043 m of heave and
  −0.36° of pitch, and the motion *grows* from there — pitch rms 1.3° at
  the start of a 90 s run, 11.5° at the end.
* **Not resonance with the stroke.** The response is at **0.867 Hz**.
  The stroke at rate 30 is 0.500 Hz and its harmonics are 1.0, 1.5, 2.0.
  0.867 is none of them; it is the boat's own coupled heave/pitch mode.

A response at the system's own frequency, growing, is **self-excitation**
— net negative damping — not forcing. And it was sharp: rates 30 and 32
diverged while 18, 22, 26, 28 and 36 did not. A resonance band that
narrow is itself the diagnosis, because a real hull's pitch response is
broad precisely because it is well damped.

---

## Two separate defects

### 1. A lumped force cannot make a moment

`hull_resistance` returned **forces only**, and the vertical term was a
single resultant applied at the origin:

```
Z = -½ ρ C_d A_plan · w|w|
```

A force at the origin exerts no moment about the origin. The hull
therefore had **no pitch damping whatsoever**; what little existed came
from the skeg and rudder, which are small and near the centreline.

This is the identical defect `coxswain/hydro/crossflow.py` had already
fixed for yaw — the docstring there says "the hull's yaw moment was set
to zero outright" — applied to the plane it was left out of. The fix is
the same Hoerner argument: local vertical velocity at station *x* is
`w − x q`, so a pitching hull drives its ends through the water even with
no heave, and integrating the sectional load against *x* gives the
moment.

**It is a strict extension, not a re-tuning.** At zero pitch rate the
integral of waterline beam along the length *is* the plan area, so it
reduces exactly to the previous lumped force.

### 2. Quadratic damping does not damp small motions

`v|v|` vanishes **faster than the energy going in** as amplitude falls.
A small oscillation is therefore effectively undamped and grows until the
quadratic term finally catches it — which is exactly the observed slow
build to a violent limit cycle.

Real damping at these amplitudes is **linear**, and it is wave radiation.

---

## What replaced it

### Radiation damping, from potential flow

A hull oscillating at a free surface radiates waves; the energy carried
away is a force linear in velocity. This is a potential-flow result — no
viscosity, no separation — and for a slender hull it comes from strip
theory (Salvesen, Tuck & Faltinsen 1970). For a section oscillating at
frequency ω, the damping per unit length follows from the radiated energy
(Newman 1977):

$$b(\omega) = \frac{\rho g^{2}\bar{A}^{2}}{\omega^{3}}$$

with $\bar{A}$ the radiated wave amplitude per unit motion amplitude.
Strip theory then assembles the coupled matrix, and **the coupling is the
point** — a hull that heaves also pitches, because the sectional forces
act at a lever arm:

$$B_{33}=\int b_{33}\,dx,\quad B_{55}=\int x^{2}b_{33}\,dx,\quad B_{35}=B_{53}=-\int x\,b_{33}\,dx$$

and likewise $B_{22}, B_{66}, B_{26}$ from the sway sectional damping.

### Roll, where potential flow will not do

A shell is 0.155 m deep and 0.50 m wide. A section that shallow rolling
about its own waterline displaces almost no fluid vertically, so it
radiates almost no waves — **potential-flow roll damping is negligible**.
Computing only that would leave roll nearly undamped, which would be
wrong in exactly the way the pitch bug was wrong.

Roll damping is viscous and lift-driven, and the standard treatment is
Ikeda's component method, which the ITTC recommends. Of its five
components:

| Component | Treated | Why |
|---|---|---|
| Lift | **yes** | Dominant at 4–5 m/s; linear in *U* |
| Friction (Kato) | **yes** | Small, but the only term surviving at zero speed |
| Eddy-making | no | Falls away rapidly with forward speed; a round-bilge shell sheds little |
| Wave | no | Negligible at this draft, per above |
| Bilge keel | n/a | A shell has none |

Leaving eddy-making out makes this a **floor** on roll damping, not a
best estimate. That is the safe direction: it under-damps rather than
flattering the boat.

### Frequencies and inertias, derived not assumed

Radiation damping goes as $\omega^{-3}$, so evaluating every mode at one
frequency is a large error, not a small one. Each mode is evaluated at
its own natural frequency, computed from the model's own geometry —
restoring stiffness from the waterplane, inertia from the boat plus the
strip-theory added mass already in `coxswain/hydro/addedmass.py`:

$$C_{33}=\rho g A_{wp},\qquad C_{55}=\rho g\!\int\! x^{2}b(x)\,dx,\qquad C_{44}=\rho g\nabla\overline{GM}$$

Roll is the exception and is returned as undefined: a shell's metacentric
height is set by **the crew**, not the hull — four people with their mass
a foot above the waterline — so it is not a property this geometry can
derive, and guessing it would be worse than saying so. Roll damping does
not need it, because Ikeda's lift and friction terms are
frequency-independent.

The generalised inertia matters as much as the frequency. Reported
against the bare hull, pitch damping came out at **1.37 of critical** —
over-damped, which it emphatically is not. The hull of a four is 51 kg;
what actually resists pitching is four rowers at their stations plus a
coxswain 4.3 m up the bow.

---

## The numbers, and how far to trust them

Damping as a fraction of critical, at race speed:

| | heave | pitch | roll | sway | yaw |
|---|---|---|---|---|---|
| coxed four 4+ | 0.065 | 0.041 | 0.043 | 0.001 | 0.004 |
| eight 8+ | 0.071 | 0.044 | 0.037 | 0.001 | 0.004 |

Natural frequencies: four — heave 1.008 Hz, pitch 1.166 Hz; eight —
heave 0.905 Hz, pitch 1.048 Hz.

**Against published values.** Slender ships run roughly 0.1–0.4 in heave
and pitch and 0.02–0.10 in roll. Roll lands inside its band. Heave and
pitch come out **below** the ship band, which is the expected direction —
a rowing shell has a length-to-beam ratio near 27 against 6–8 for a ship,
and a hull that slender radiates far less per unit displacement — but
being in the expected direction is not the same as being verified.

**The weak number** is $\bar{A}$, the radiated wave amplitude ratio.
Published curves for a wall-sided section at the reduced frequency a
shell sits at ($\omega^{2}B/2g \approx 0.4$) run about 0.4–0.7;
`HEAVE_AMPLITUDE_RATIO = 0.55` is the middle. This project has **no
measurement of it for a racing shell**, so it is a parameter, not a
result. Quote the band, not the number.

**What would actually verify this** is measured shell motion — an IMU in
a four, giving pitch amplitude and the decay rate after a disturbance.
Kleshnev's group has run exactly that instrumentation for two decades.
Until such a comparison exists, this page describes a model that is
*physically derived and internally consistent*, which is a weaker claim
than *validated*, and the difference should not be glossed.

---

## Effect

Rate sweep for the coxed four, 60 s, no steering, pitch rms first fifth →
last fifth:

| rate | before | after | max pitch after |
|---|---|---|---|
| 18 | 0.28 → 0.28 | 0.34 → 0.33 | 0.52° |
| 24 | — | 0.35 → 0.35 | 0.55° |
| 28 | 0.33 → 0.34 | 0.35 → 0.35 | 0.55° |
| **30** | **0.72 → 4.99** | **0.38 → 0.38** | **0.66°** |
| **32** | **0.89 → 3.44** | **0.40 → 0.42** | **0.79°** |
| 36 | 0.36 → 0.34 | 0.37 → 0.37 | 0.68° |

The instability is gone, **the rates that were already stable are
unchanged**, and peak pitch across the whole range is 0.5–0.8° — which is
what a shell does. Roll peaks at 0.38°.

The most telling number is what it took. The first attempt at a fix used
a crude estimate giving ζ_pitch ≈ 0.25; the properly derived value is
**0.041**, six times smaller, and it stabilises the boat just as
completely. The defect was never that the damping was too *weak*. It was
that in pitch there was none at all, and that what existed elsewhere was
quadratic and therefore absent at small amplitude.

---

## Related: the 4+ is a bow-loader

Found while fixing the above. `catalog.coxed_four` placed the coxswain at
station −4.00 m, behind the stern seat, like an eight. Nearly every 4+
raced today is a **bow-loader**: the coxswain lies supine ahead of the bow
seat, now +4.30 m.

That is 8.3 m — most of a quarter of the hull — for a 55–90 kg mass. The
crew centre of mass moves from −1.063 m to +0.597 m and the static trim
flips from −0.364° (stern down) to +0.202° (bow down). It also changes
the pitch inertia, which is one of the terms above.

It changes the only viewpoint that matters, too: the coxswain's eye drops
from 0.70 m seated upright to **0.25 m** lying down, and the crew is
behind them rather than in front. `Rig.coxswain_eye_height` carries this,
and the 3-D scene reads it from the boat instead of assuming.

Pass `bow_loaded=False` for the older stern-coxed layout.
