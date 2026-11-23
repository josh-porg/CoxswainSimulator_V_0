# The multi-clock body: build plan, what exists, what blocks it, and the placeholders

The user, 2026-11-19: "multi clock bodies and noting where we are blocked ... lets take a look at what we can do safely
now with things in place for what we are currently missing and make as much progress as possible." The architecture
is docs/NEUROMUSCULAR_CONTROL_REVIEW.md §12 (approved 2026-11-01, SOURCES §208). Each step has a gate; a step proceeds
on a **placeholder** where data are missing, and every placeholder is named here and carried in the code's docstrings,
so that a result built on one is never reported as sourced.

## Why it matters

The body is the largest remaining error that bears on the coxing questions: the stronger athlete's catch approach takes
~0.2 m/s too much out of the hull (§242), the single sculler's swing is +3.5 points (§241), and in rung 2 the body moves
on clocks (the population's or the athlete's channels) rather than responding to the boat (MODELLING_ASSUMPTIONS A1,
priority H). A rudder hit or a call can only change the crew's timing in a model whose timing is an output.

## Steps, state, and gates

| step | what | exists | gate | state |
|---|---|---|---|---|
| 1 | actuators: hip and knee (Anderson et al. 2007), elbow (Frey-Law et al. 2012), activation dynamics (Zajac / Thelen) | `coxswain/crew/joint_muscles.py`; the capacity check `research/crew/muscle_capacity.py` (§208, §216) | the required activations ≤ 1 at the athletes' strokes | **fails only outside the data's angles** at the catch (knee 138–145° flexion; the data stop at 100°) — blocker B1 |
| 2 | zero-delay layer: activation-scaled intrinsic stiffness and damping | trunk: `TrunkImpedance` (Moorhouse & Granata 2007) | trunk static stiffness in their range | **trunk done**; legs and arms not started |
| 3 | spinal layer: spindle / tendon-organ feedback at 20–45 ms | trunk: `ReflexState` (40 ms) | the identified trunk admittance | **trunk done** (a velocity reflex); legs and arms not started |
| 4 | excitations from modules on a phase oscillator, entrained to sensed events | **started (§253)**: `coxswain/crew/excitation.py` — on-water modules from Pitto et al. 2024 (trunk extensors, arm pullers, trunk flexors, arm return, arm extensors) and a phase-resetting oscillator with period adaptation (tests `test_excitation.py`) | the legs → trunk → arms sequence and the hands' time law emerge; rung 2's IVV improves on the shared clock; nothing fitted | building — see "next" |
| 5 | long-latency layer: delay-aware estimator (forward model + efference copy) | trunk: rung 5's predictor in `rung2_trunk_layers.py` (§213: wander 10–13°, not a few degrees) | stable at sourced delays; trunk error < 5° | trunk partial |
| 6 | options: LQG to derive amplitudes and gains; collocation for the optimum stroke; fatigue | — | — | later |

**The sequencing check, today's body against on-water EMG** (`research/crew/sequencing_vs_emg.py` on
`muscle_capacity.py --rung2` re-run 2026-11-19, `data/local/crew/muscle_capacity_rung2_s253.json`; §253): the model's
hip-extensor demand peaks at 28–29 % of the drive and its elbow-flexor demand at 25–26 %; rowers' trunk extensors fire at
40 % and their arm pullers at 56 % (triceps, holding the arms straight, at 35 %). A muscle's force follows its EMG by
~50–100 ms, so the model's body loads the trunk ~150–200 ms and the arms ~300–350 ms **too early** — the sequencing gap
of §205, now measured against on-water muscle timing. Step 4's modules are the remedy (the arm pullers' excitation is
near zero until mid-drive); this is step 4's gate baseline. (On an older model state the hip demand peaked late; the
stored file of that run, `muscle_capacity_rung2.json`, is superseded.)

## Blockers and the placeholders used meanwhile

| | blocker | what would remove it | placeholder (named in code) | effect on claims |
|---|---|---|---|---|
| B1 | hip and knee strength beyond the tested angles (the catch flexes the knee 138–145°; data stop at ~100°; hip at 75°) | strength at deep flexion (a data request; Kukić et al. 2022 give 62 % of peak at 100° only) | hold the angle factor at its value at the last tested angle (no extrapolation to zero) | step 1's gate is reported "inside the tested angles"; catch activations flagged extrapolated |
| B2 | no on-water knee-extensor EMG (Pitto et al. recorded none) | **resolved with a flag (§255)**: Vieira et al. 2020 (CC BY), vasti on an indoor rower | the leg module is the ergometer vasti, drive and recovery mapped onto the on-water cycle; flagged ERGOMETER in `Modules.sources` | leg-timing results carry "ergometer EMG" |
| B3 | shoulder strength and scapular (girdle) kinematics unsourced | a rowing scapular-kinematics series; shoulder extension strength by angle | the arm pullers are driven at the elbow (Frey-Law), the shoulder taking the rest kinematically | arm forces are elbow-limited |
| B4 | the module amplitudes (how hard each group is driven), including the antagonists' co-contraction level (§257) | step 5's estimator, or LQG (step 6); for the antagonists a rowing %MVC source | matched to the athlete's measured power, as rung 2 scales its muscles (one scale per athlete, not a fit of shape); antagonists off by default | the same status as rung 2's scale; H37's level open |
| B5 | the auditory delay (for coxswain calls) | a sourced auditory-motor delay in rowing | not needed until calls drive the crew | none yet |
| B6 | the data author's answers (oars, angle sensor, original recordings) | **answered 2026-11-22 (§258)**: sensor inboard (H33 refuted), shafts confirmed, [CR06]'s originals lost | — | C1 has no data route; B2s's cause open; blade areas now from Concept2 (§259, request #18 drafted) |

## Step 4's gate, fully specified (§255)

With legs, trunk and arms excited by the modules on the phase oscillator, each group's demand peak must follow its EMG
peak by ~50–100 ms (excitation to force), nothing fitted. **Today's body: legs −55, trunk −80, arms −220 ms** (the
demands peak at 25–31 % of the drive against EMG at 33 / 40 / 56 %): every group early, the sequence compressed toward the
catch. The straight-arm catch spike is the shoulder girdle's missing compliance (B3), not the elbow's (a straight arm
carries no elbow moment); the arm-chain stretch placeholder halves it.

## Step 4's first closed loop (§256)

`research/crew/rung2_modules.py`: rung 2's clock driven by the knee, hip and elbow muscles excited on the modules
(placeholders: d 0.075 s, one amplitude per athlete, B1's angle clamp). Stable on both athletes; against the borrowed
population force on the same (model) body: oar timing better for both (rms 0.035 / 0.016 against 0.039 / 0.019), speed
better for [BR24], IVV better for [CR06], worse for [BR24]. The segments still move on their clocks along τ; making each
a degree of freedom is next.

## The late-drive spike and the antagonists (§257)

The clock's rate spike (1.52 / 1.77) is in the last 5 % of the drive's progress, and it comes from the module force,
not the single clock: the population force on the same body stays ≤ 1.15. At the finish the agonist modules are low and
the trunk flexors are at peak (Pitto et al. 2024; their eccentric braking of the layback is the documented role). With
the trunk flexors and triceps added at the same amplitude (`--antagonists`), [CR06]'s spike is gone and [BR24]'s is
reduced, but the IVV worsens (+4.8 / +1.6 points). The mechanism is confirmed but the level is not (H37). The
antagonists' level joins B4. Power matching (`--iterate-power 2`) reaches the measured power exactly; [BR24]'s IVV gap
(+7 points) is not the amplitude.

## The muscles' levels and the trunk on its muscles (§261–262)

The literature's remedy for antagonist levels is EMG-assisted calibration (Lloyd & Besier 2003; Sartori et al.
2012): EMG fixes each muscle's timing, a fit to the inverse-dynamics joint moments fixes its level (rowing EMG cannot
be MVC-normalised). Built (`rung2_modules.py --calibrate`) and as computed muscle control on the trunk
(`rung2_trunk_modules.py`, Thelen & Anderson 2006). **Both fail for one reason:** the calibration target is the
kinematic body, whose demands come 0.1–0.3 s before rowers' muscles fire. Knee and elbow fits r² < 0.1, and the trunk
needs 1.24–1.46× maximal strength; the muscle-driven trunk wanders 27° at no delay. The CMC structure works and the
target is wrong. **Per the literature's order: track measured trunk motion (the athletes' own channels) to test
feasibility, then lower the tracking so the timing becomes an output.** B4 is now "the levels, after the target is
right".

## Next (in order)

1. Re-run the sequencing check on today's rung 2 (`muscle_capacity.py --rung2` on the current defaults) — the baseline
   for step 4's gate.
2. The arms on steps 2 and 4: the elbow as a degree of freedom, held near straight by the arm-extensor module's
   activation and intrinsic stiffness and drawn by the arm pullers (activation dynamics, Frey-Law's elbow), replacing
   the kinematic arm clock; gates: the elbow angle against Pitto et al.'s measured flexion
   (`data/literature/pitto2024_elbow_*.csv`) and the arm demand's timing against the EMG — **baseline (§254): the default
   bent-arm body needs the elbow flexors 215–220 ms before rowers fire them (best correlation 0.89 / 0.95); the rigid
   straight-arm body spikes at the catch.** Nothing fitted.
3. The trunk on step 4: the trunk-extensor and flexor modules as the trunk's feedforward in `rung2_trunk_layers.py`
   (replacing the reference run's hip moment), with steps 2, 3 and 5 as built; gate: wander < 5°, IVV on the shared
   clock.
4. The legs: blocked on B2; until then, their sourced clock.

## Safe progress elsewhere in the project now (not blocked on replies)

- **Charles trajectory optimisation:** the Cambridge-side corridor (decided 2026-10-21, not built), then the stochastic
  problem with the uncertainty models available today (current: discharge loaded, field uniform — a placeholder; crew
  variability: `coxswain/crew/variability.py`).
- **The research profile (R1 in HYPOTHESIS_REGISTER):** moving the validated, uncontested pieces into
  `PROFILES["research"]` — the user's decision.
- **Back-filling the hypothesis register** for SOURCES §1–§155.
