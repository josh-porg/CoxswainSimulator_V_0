# Project status

Rowing shell simulator: full 6-DOF rigid-body dynamics, phase-locked
collocation, trajectory optimisation over a real reach of the Charles.
Written against the ultimate goal of a **stochastic** optimal control
problem -- uncertain stream, crew, friction and temperature -- with the
deterministic path as the prerequisite.

Detailed derivations, sources and the record of what was tried and
rejected live in [SOURCES.md](SOURCES.md), numbered by section; this file
points at them rather than repeating them.

Test suite: **1014 passing** in the fast lane (`pytest -m "not slow"`,
~4 min); full suite ~1160 tests, ~25 min.

---

## 1. What works

**Physics.** 6-DOF rigid-body dynamics after Formaggia et al. (2009);
hull hydrostatics from an exact mesh, wrapped in a b-spline surrogate
over (heave, pitch, roll); slip-based blade model after Cabrera & Ruina;
feathered vs squared blade drag; wind with a log profile at WMO
anemometer height; de Leva segment inertias driving a full joint chain.

**Crew.** Phase-dependent balance authority (drive vs recovery), learned
stroke-to-stroke trim by iterative learning control, coupled-oscillator
crew synchronisation with anticipatory coupling, blades-on-water contact
for an unset boat.

**Numerics.** The NLP is non-dimensionalised -- variables, objective and
constraints -- which bought a factor of 25 in iterations (§24). The
reference-frame bug class is closed structurally by 31 invariance tests.

**River.** Charles channel raster, bridges including BU and Grand
Junction, clearance field, progress field along a centreline.

**Validation.** Real telemetry: DGPS hull traces and, as of §34,
rower-mounted kinematics from the same boat and outing.

---

## 2. Open bugs, in priority order

### 2.1 Hull surge exceeds the crew-only momentum bound  *(highest)*

The model reports 51.5% of intracycle velocity variation for a 2x where
the measured figure is 37.3%. The gap splits at the **crew-only momentum
bound** of 42.5% -- what the crew's own motion could do to a free hull:

* **51.5 -> 42.5 (~9 points): the model exceeds its own free-body
  bound**, while being correctly phased (§31: minimum just after the
  catch, maximum in early recovery). Blade thrust and crew reaction
  *oppose* during the drive, so the total must come in *below* the
  crew-only figure. It does not. Something in the surge force balance --
  thrust magnitude, drag variation, or missing surge added mass -- is
  wrong.
* **42.5 -> 37.3 (~5 points):** inside the crew motion; a kinematics
  question.

This is the single largest known discrepancy in the model and the first
thing to fix.

### 2.2 The receding-horizon leg does not yet reach 850 m

Best run to date: 476 m before §33's fixes. The current run (comfort
barrier, retry escalation) is past that and negotiating the station-450
pinch where the channel narrows to 30 m. Not yet closed.

### 2.3 Route C (predictive stroke) does not converge

`coxswain/crew/predictive.py` solves for the rower's motion rather than
prescribing it. It terminates on `Maximum_Iterations_Exceeded`, and its
slide travel (0.43 m at 400 W) is short of the measured 0.60-0.70. Four
modelling errors were found and fixed through it (§27-28); the power
attribution is now correct but the solve is not converged, so **none of
its numbers are claimable**.

### 2.4 Model boats are 1x-derived; only the 2x is validated

`double_scull` was added in §32 specifically to make the fluctuation
comparison like-for-like. The eight -- the boat that actually matters for
the Head of the Charles -- has no equivalent validation data.

---

## 3. Open avenues of development

**Close the surge force balance (2.1).** Decompose the surge equation
term by term over a stroke: crew reaction, blade thrust, hull drag,
added mass. The phasing is right, so the fault is in a magnitude. Surge
added mass is currently absent and is the obvious first suspect --
including it moves the hull's response to crew motion in the right
direction.

**Finish the deterministic leg, then the stochastic one.** The stochastic
machinery exists, is scenario-sampled from Kleshnev power scatter and
Cuijpers timing scatter, uses a mean-standard-deviation objective, and
now shares the deterministic solver's comfort and roll terms so the two
differ only by uncertainty. It solves (E[progress] 19.19 m, sd 0.022) but
runs ~375 s per block against ~90 s deterministic, so a full leg is a
multi-hour job.

**Sequencing, calibrated properly.** `SegmentSequencing` works and is
worth ~2.7 points of fluctuation in the anatomically correct direction
(legs lead, trunk lags). Its magnitude is currently *fitted* to the very
quantity it would predict, so it stays `SYNCHRONOUS` by default until
there is kinematic data to set it independently.

**Denser rower kinematics.** Worth having, but §34 bounds the prize: at
most about a third of the remaining gap, and the trunk kinematics are
probably already fine.

---

## 4. Wanted, but not a priority

* **A 2x/4x/8+ family with per-class validation data.** Only the 2x is
  validated; the eight is the boat that matters for the race.
* **Steering study conclusions.** The oscillate-vs-correct question that
  started the rudder work has machinery but no written answer.
* **Bridge piers as constraints.** Bridges are landmarks; the arches and
  their piers are not yet obstacles, and at the Head of the Charles the
  Weeks and Anderson arches are exactly where boats lose time.
* **Stream field.** The Charles discharge data is loaded but the current
  is uniform; a spatially varying field is the physically right thing and
  matters for line choice around bends.
* **Crew fatigue over 4.8 km.** The reserve state exists; a
  physiologically grounded depletion model does not.
* **Visualisation.** VTK output exists; nothing animates a whole leg.
* **Route C as a frozen rower model.** Solve the stroke once against
  real data, freeze it, use it in the trajectory NLP -- which would also
  shrink that NLP. Blocked on 2.3.
