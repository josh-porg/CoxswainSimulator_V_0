# Plan: crew synchronisation, and blades on the water

Two pieces of physics currently outside the model, planned together
because they are the same problem seen twice. Both are ways that *timing
error* becomes *boat behaviour*, and both couple into the roll margin
established in `SOURCES.md` §15–16.

The organising fact from §15 is that the criterion is **bounded growth,
not stability**: roll must not grow too much between the finish and the
next catch. At rate 32 an eight amplifies heel by ×180 across the
recovery and can arrest 2.27°, so the tolerance at the finish is about
**0.013°**. Everything below is either a source of that finish-line error
or a consequence of exceeding it.

---

## Part A — Crew synchronisation (§18)

### A0. The modelling commitment

Every rower currently shares one stroke phase. That is not a small
idealisation: it sets the disturbance input to a marginal plant to exactly
zero.

Synchronisation is driven by **two coupling channels**, and the plan
models both rather than choosing:

1. **Mechanical coupling through the hull.** Rower *i*'s motion moves the
   shell; every other rower is rigidly attached to it and feels that as
   seat and stretcher motion. This is involuntary, fast, and already
   computed by the simulator — the hull acceleration is a state.
2. **Sensory coupling — visual and auditory.** Rowers watch the blade or
   the back in front of them and hear the catch. This is voluntary,
   delayed by reaction time, and directional (you can only see forward
   toward the stern).

**Which dominates is not established**, and the plan is built so the
question is answerable rather than assumed: a single mixing parameter
weights the two, and the model is run across its range.

### A1. Two phases, not one

Early blade extraction changes the vertical timing without changing the
horizontal sweep, and the two affect the boat through different terms.
So each rower carries **two phase variables**:

| | symbol | governs | enters the dynamics through |
|---|---|---|---|
| horizontal | φᵢ | oar sweep angle, handle force | surge, yaw |
| vertical | ψᵢ | blade immersion and extraction | roll, blade forces, added mass |

A rower who washes out early has ψᵢ leading φᵢ. That is a real and common
fault, it is invisible to a single-phase model, and it is a roll
disturbance precisely when roll authority is lowest.

**Deliverable A1**: extend `Boat.crew_field` and the oar kinematics to
accept per-rower `(φᵢ, ψᵢ)` offsets. Default all zero, so every existing
result is unchanged and the 977-test suite stays green.

### A2. The coupling law

Kuramoto-type phase dynamics with two channels:

```
dφᵢ/dt = ω + K_hull · h_i(hull state) + K_sensory · Σ_j A_ij sin(φ_j(t − τ_r) − φᵢ) + σ dW
```

* `A_ij` — the sensory coupling topology. **Directed**, not all-to-all: a
  rower sees the crew in front of them (toward the stern), so the natural
  graph is a directed chain from stroke seat to bow with the stroke as
  reference. This is the structural claim to test.
* `τ_r` — reaction delay, ~150–250 ms from human sensorimotor literature.
  Comparable to the roll e-folding time of 0.218 s, which is itself a
  finding worth stating: **sensory correction is not fast enough to catch
  a roll excursion**, only to prevent one.
* `h_i` — mechanical entrainment from hull motion, computed from the state
  the simulator already carries.
* `σ` — stroke-to-stroke timing noise. This is the timing half of the
  stochastic-control goal, the power half being separate.

### A3. Skill as a parameter

An unskilled crew does not synchronise; a skilled one does, and only after
many hours together. The plan represents skill by three numbers, not one:

| parameter | unskilled | skilled | meaning |
|---|---|---|---|
| σ | large | small | intrinsic timing scatter |
| K_sensory | small | large | how well they hold to the stroke |
| ψ–φ offset | large, variable | small | blade-height discipline |

This gives a *crew quality* axis that is physical rather than a fudge
factor, and it makes "hours of drilling" a trajectory through parameter
space rather than a mood.

### A4. What it is for

Phase spread is not merely realism, it is the missing disturbance input:

* **port/starboard phase asymmetry is a roll moment** — it injects error
  at the finish, against a 0.013° tolerance;
* **port/starboard phase asymmetry is also a yaw moment** — the standing
  steering bias whose correction `studies/steering_strategy.py` already
  compares;
* it supplies the timing half of the stochastic optimal control problem.

### A5. Literature

* **Leonard, N. E.** (Princeton, MAE / PACM) — phase models of coupled
  oscillators and collective motion, including the improvisational-dance
  collaboration on in-the-moment collective decision making.
  <https://naomi.princeton.edu/publications/>
  The connection to argue in writing: rowing supplies a coupling channel
  most human-synchronisation systems lack — a **shared rigid body**, so
  the mean-field term is not a modelling convenience but an actual
  mechanical path with a computable transfer function. That is a genuine
  contribution back toward her framework, not merely an application of it.
* **Kuramoto** — the canonical phase-oscillator model and its stability.
* On-water biomechanics scoping review (PMC11436553) — reports that crew
  synchronisation quality relates to lateral stability, which is the
  qualitative version of the link this plan makes quantitative.
* Kleshnev — measured crew timing scatter, for calibrating σ.

### A6. Validation targets

1. Zero phase spread must reproduce the current results exactly.
2. Predicted roll disturbance vs measured heel variance against crew level.
3. The rate-scaling prediction of §15 — heel variance rising sharply as
   rating falls — must survive the addition of a realistic disturbance.
4. Directed-chain versus all-to-all coupling should be distinguishable in
   the phase-lag pattern down the boat, which telemetry can measure.

---

## Part B — Blades on the water (§17)

### B0. Why it is worth doing at all

The user's own crews race the Charles with the blades clear on nearly
every stroke, so this is not the normal operating regime. It matters
because it is the **failure mode that gives roll error a cost in
seconds**. Without it, poor balance in this model costs nothing but roll
angle, which makes the whole §15–16 analysis unfalsifiable against a
stopwatch.

### B1. Three separate effects, in order of value

1. **Blade–surface contact drag.** A feathered blade skipping the surface
   decelerates the boat. Modelled as a contact that switches on when the
   blade's vertical position crosses the surface, with the immersion depth
   from the existing geometry.
2. **Contact as a balance aid.** The same contact *unweights the rigger*,
   and [D96] describes this as a powerful stabiliser — "by exact hand
   control you can scull a boat dead flat this way." The same event is a
   speed loss and a stability gain, which is why crews do it when unset
   even though it is slow. Modelled as a vertical force at the blade,
   feeding the rigger through the same lever as §15's drive term.
3. **Truncated drive.** Once the blade is in, the rower must go with it:
   an unset boat forces an early catch and less than full extension. This
   is a length-and-timing penalty, and it is the direct coupling from
   ψᵢ (Part A) into propulsion.

### B2. The smoothing problem

Contact is a switch, and IPOPT needs a derivative. The licensed approach
is a **bounded smoothing with a measured bound**, as used for the
phase-dependent authority window: a logistic in blade-to-surface clearance
whose width is set by something physical — surface roughness and wave
height — rather than chosen for numerical convenience, with the departure
from the hard switch reported.

### B3. Sequencing

B1.2 first. It is the smallest change, it uses machinery §15 already
built, and it is the one that closes a loop: it lets the model exhibit the
real trade a crew makes when the boat is unset — accept drag to buy back
roll authority. B1.1 and B1.3 follow.

### B4. Scope boundary, stated

If B is not implemented, the model's boundary is: **blades clear
throughout the recovery, and roll error carries no speed penalty.** That
is written into `SOURCES.md` §17 so that no result is quoted past it.

---

## Ordering against the rest of the programme

Part A comes before Part B. A supplies the disturbance that makes B's
failure mode occur at all, and A1 (per-rower phases) is a prerequisite for
B1.3 (truncated drives keyed to ψᵢ). Both come after the wind field,
which is a larger missing force and is needed by the steering studies
already queued.

Revised order: **wind → per-rower phase offsets (A1) → stochastic stroke
power and timing (A2–A3) → plots and study matrix → blades on water (B)**.
