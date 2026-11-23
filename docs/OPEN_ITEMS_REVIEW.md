# Open items: literature review, tests and answers (2026-11-23, SOURCES §260–263)

The user, 2026-11-23 (/goal): address the open items with a literature review, then implement and validate solutions;
send the Concept2 request; explain the failures to reproduce papers; say how far the model is from right; and progress
from kinematics to dynamics following the biomechanics literature. Each section: the question, what the literature
says, what was run, and the decision. Hypotheses go to docs/HYPOTHESIS_REGISTER.md (H41–H45).

## 1. "Reproducing a paper should not fail if we copy its method"

Two different things have been called "reproducing" in this project:

* **Replication**: run the paper's *own model and method* and recover the numbers it prints. Here a failure means a
  copying error, and every one found was fixed. Replications that pass: [CR06]'s finish turn-round from her data
  (0.999 s against 1.025 s), the trunk rungs 3–5 rebuilt and reproduced to 0.03°, Michell against Lazauskas's Wigley
  curve, [L14]'s shaft stiffness against Breitschädel & Løset (0.3 %), Pitto's elbow law (1.3° rms), the [K05] travels.
* **Validation**: run *our* model and compare it with the paper's *measurements*. A difference here is the model
  being wrong or its inputs differing from the experiment's, not a copying error, and it is how missing physics is
  found. Every listed "failure to reproduce" is of this kind: [H10]'s parallel force at entry and release, [GTW21]'s
  lagging tangential load, [G19]'s entrained water and Holt's singles are measurements our quasi-steady blade or
  inputs do not produce; C1 and B2s are our model against measured force curves.

So the reading "we must be copying their methods wrong" does not apply: none of the open items is a replication. They
are places where the model's physics or inputs differ from the experiment's. Two were traced this round to *inputs*
rather than physics (blade area, oar mass distribution; §2, §5).

## 2. Do we need [H10]'s entry/release parallel force, [GTW21]'s unsteady tangential lag, [G19]'s entrained water?

**Literature.** On an impulsively started plate, potential-flow added mass supplies part of the force at once and the
rest builds as the wake and the leading-edge vortex form (Wagner; for a sharp-edged plate such as an oar blade a
leading-edge vortex forms from the start). Post-stall plates rise faster than attached ones. [G19] measured the
entrained water growing with time; [GTW21] measured the tangential load lagging the sliding; [H10] measured on water
that the parallel force costs ~15 % of blade losses, mostly at entry and release.

**Size of each, in this model.**

| effect | what it changes | measured size in our closed loop | direction for the open failures |
|---|---|---|---|
| [H10] parallel force at entry/release | blade losses at shallow immersion | at [H10]'s on-water level: −2.7 % / −0.3 % speed (§200) | none on C1 or B2s (drive-phase force level and shape) |
| [GTW21] tangential lag | phase of the tangential load | the tangential load is ~2–3 % of the normal on a stalled blade (§199) | a lag *delays* force: it would deepen B2s |
| [G19] entrainment growth | added mass rising after entry | κ ≤ 0.1 × Patton already bounded by two measurements (§173, §228); growth rates known at model scale only | adds force after entry, but every tested κ failed to close B2s (§245) |

**Decision: not now.** None of the three can be built without a full-scale coefficient we do not have, each is
smaller than the gaps it might touch, and the one with a direction on B2s points the wrong way. Recorded as H41.

## 3. How far is the model from right?

| quantity | model against measurement | the spread between rowers | verdict |
|---|---|---|---|
| boat speed at measured power | within ±3 % on both athletes (D1); population within 1 % | — | inside |
| speed swing (IVV) | +0.9 to +3.5 points | band ±6.7 / ±7.5 (0.75 of Leão 2025's between-rower spread) | inside |
| oar timing (rung 2) | 0.02–0.04 of drive progress rms | — | good |
| handle-force curve, all-boat population (B2) | 10.2 points rms | individual curves differ by 10–15 points | inside |
| handle-force curve, sculling like for like (B2s) | 21.3 points rms (rigid oar 12.4) | as above | **outside**: the flexible shaft loads too slowly early |
| [CR06]'s force level (C1) | 18.2 % rms at her paper's blade; **11.6 % at the photographed Big Blade** (§261) | band 15 % | depends on one input (§8) |
| energy and momentum books, round trips | close to < 0.1 % | — | the code does what it says |

**Nothing points to a fundamental error.** The books close, independent code paths agree to 1–3 N, and every
integrated quantity sits inside the between-rower spread. The two real gaps are the early-drive force time course
on a flexible shaft (B2s) and the body's sequencing (the kinematic body loads 0.1–0.3 s early, §255). Both are the
kind of local structure a validation programme expects to find.

## 4. Will the corrected blade areas fix tier 2's open issues?

No for the shape issues, yes for some level issues. Area scales every force component equally, so it cannot fix tier
2's catch overload (from its small-angle lift) or its tangential sign. Those are why tier 2 was superseded by the
cross-flow normal law (31 Oct). Area does move the *level*: on [CR06]'s measured motion the photographed area brings
speed, power and the force curve all closer (§8). On [BR24] the smaller blade brings his speed from +2.9 % to +1.9 %.

## 5. The oar's weight moment from the feathered recovery

Yes, it gives the oar's centre of mass. Feathered, the force sensor reads the vertical plane, so mid-recovery the
hands hold the oar's weight about the pin (§260, `research/biorow/recovery_oar_moment.py`):
`F_h l_h = m g x_cg cos ψ + I_v ψ''`. [BR24]: static moment **m x_cg = 0.92 kg m** (the two oars 0.88 / 0.96).
With Concept2's 1.375 kg (their Ultralight/Skinny scull), **x_cg = 0.67 m outboard** of the pin (1.54 m from the
handle end) against the uniform rod's 0.57 m. The inertia about the pin becomes ~1.56 kg m² against the model's 1.22.
The `Oar` docstring's claim that the rod is an upper bound is contradicted. `Oar.centre_of_mass` is now an optional
measured input (default unchanged), and the matched-torque cache key now carries blade area and centre of mass (both
were missing; STRUCTURAL_ISSUES S1).

## 6. τ(θ) "failing"

That ledger row is the **old research profile's prescribed torque shape**. Its middle matches Holt's and Kleshnev's
peaks, and its ends are contradicted by Holt's measured slips. In the validated configuration it is gone: the handle
force is an output of the hands (rung 1) or of the muscles (rung 2). It still matters wherever the project runs
`PROFILES["research"]` (the route optimiser, the trainer's research mode). The fix is not to tune its ends but
to move the rung-2 crew into the profile (R1, the user's decision). Until then the cr06-fitted shape option halves
the end error.

## 7. A measured oar deceleration law

Yes, partly. Kleshnev's sculling averages give the handle speed through the whole recovery at rates 32, 36 and 40,
including the deceleration into the catch. It is built as `--pop-recovery sculling` (§242), where it fitted [BR24]'s
recovery better and [CR06]'s worse with no material effect, and was kept as an option. Under the adopt rule it is a
sourced, physically real change, so it was run through the ladder (§263): the same rung-2 results within 0.2 points; adoption waits on its half-step run. The finish's deceleration (the
turn-round) was reproduced from [CR06]'s own data (the ledger's finish row). No open source gives the oar's angular deceleration
directly; the handle-speed curve is the measured form of it.

## 8. The expected failures

**C1 ([CR06]'s force level).** Her paper states 0.0903 m²; Concept2's photographs put the Big Blade, the scull blade
of her era, at 0.068–0.074 m². At 0.071 m² her rung-1 handle force passes (11.6 % against 18.2 %; peak 4.5° from
hers), her imposed-motion speed falls from 4.365 to 4.201 m/s (measured 4.191) and her power from 294 to 273 W
(measured 261). Three quantities move toward her measurements together. But two sources conflict, and choosing the
one that passes would be a fit. **Not adopted:** C1 stays an expected failure with this as its leading explanation,
decided by Concept2's figures (asked, §259) or her blade's type.

**B2s (the early loading).** Literature and data together close three candidates this round: the angle sensor's
position (refuted, §258), the shaft spec (confirmed as modelled), and the oar's mass distribution (Concept2's mass
with the measured centre of mass: 22.3 points rms against 22.4, §261). What is left: the blade's early-drive
loading itself on a scull path, which the published tank data ([GTW21], [RLBQV19]) measure only on sweep motions.
B2s stays an expected failure; a measured scull-path blade load (asked of Barré & Kobus earlier) or the Sensors 2023
authors' data would test it.

**The stronger athlete's oar, adopted.** Smoothie2 0.074 m², 1.375 kg, centre of mass 0.665 m: the ladder passes
27 / 0 / 2, his speed from +2.97 % to +1.9 % (§263). Now the rung-2 research default.

## 9. The antagonists

**Literature.** Static optimisation is known to underestimate antagonist activity (e.g. at the shoulder, Kian et
al. 2019). The remedy in the field is EMG-driven or EMG-assisted modelling (Lloyd & Besier 2003; Sartori et al.
2012, CEINMS), in which measured EMG fixes each muscle's timing and a calibration fixes its level against the joint
moments from inverse dynamics. In rowing, normalising EMG to isometric maximal contractions has repeatedly **failed**
(rowers exceed their isolated MVCs during rowing; Journal of Sports Science and Medicine 2005; an ergometer–on-water
comparison), which is why on-water EMG is published peak-normalised. So an absolute antagonist level cannot come
from rowing EMG; it has to come from a calibration.

**What the flaw in our model was.** H37 gave every peak-normalised module the same amplitude, which assumes every
muscle peaks at the same fraction of its strength. The literature's approach replaces that with a per-group gain
fitted to the joint moments.

**Built** (`research/crew/rung2_modules.py --calibrate`, `rung2_trunk_modules.py`; §261–262). The calibration
gives the hip a fair fit (r² 0.5–0.6), with the trunk flexors at ~0.75–0.9 of their strength at peak. But the knee
and elbow fits fail (r² < 0.1), and the trunk extensors need 1.24× their maximal strength. **The calibration
target, the kinematic body's inverse dynamics, is itself wrong in timing** (the body loads 0.1–0.3 s early, §255). An
EMG-timed muscle cannot reproduce a demand the real body never makes. The calibration is circular until the body's
kinematics come out of its muscles (§10) or the target is measured kinematics.

## 10. From kinematics to dynamics

**How the field does it** (gait, cycling, upper limb):

1. *Inverse pipeline.* Inverse kinematics, then inverse dynamics, then static optimisation or EMG-assisted muscle
   forces (OpenSim's standard workflow).
2. *Computed muscle control* (Thelen & Anderson 2006). A forward simulation whose excitations are recomputed each
   step to track the measured kinematics (feedforward plus PD feedback), with the activation dynamics and delays
   explicit. The kinematics are still prescribed, as a target.
3. *Tracking optimal control by direct collocation* (Ackermann & van den Bogert; Lin & Pandy; OpenSim Moco, Dembia
   et al. 2020). The whole stroke is solved at once with tracking terms in the cost; cycling has open models and code
   for 16 riders (Sci. Rep. 2023).
4. *Predictive simulation.* Drop the tracking terms and optimise a physiological objective. The kinematics become
   an output.

**In rowing**, Hase et al. 2002 (Biomechanics of Rowing II) built a control model for simulation (PD control, a
simple neural model, inverse dynamics as feedback, optimised control parameters), and Hase et al. 2004 ran the
inverse pipeline on ergometer rowing. A rowing optimisation-simulation exists as well. No open on-water predictive
musculoskeletal rowing model was found.

**The path here** follows steps 2 → 4 segment by segment, with the existing layers:

* the trunk first, as computed muscle control: its feedforward from the trunk extensor and flexor modules, its
  feedback the layers already built (`rung2_trunk_modules.py`, §262). **Done, and it fails its gate for a reason the
  literature predicts:** CMC tracks a *target*, and our target is the kinematic trunk clock, which its EMG-timed
  muscles cannot produce (wander 27° at no delay, 64° at 100 ms; calibrated levels above maximal strength). The
  structure works; the target must change. Two next steps, both in the literature's order: track the athletes'
  *measured* trunk motion (does a real trunk fit its muscles?), then lower the tracking so the timing is an output;
* then the legs (knee extensors, the seat as a state) and the arms (the elbow held by the triceps' stiffness, drawn
  by the pullers), each with the same structure;
* then reduce the tracking gains toward zero: the segments' timing becomes an output, which is the gate of step 4
  (sequencing emerges);
* then the predictive version (collocation over one stroke) when every segment is a state.
