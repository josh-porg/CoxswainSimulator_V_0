# Sprint board

*Sprint 1 · 2026-09-28 → 2026-10-11 (two weeks, ending before the Head of the Charles,
16–18 October). Update the board as items move; move finished items to TRACKING's Done.*

## Sprint goal

**Build past the blocks.** Until now blocked items waited for data. This sprint builds them
anyway: where a number is missing it becomes an **explicit, named, labelled parameter**
(`chosen` in the ledger), swept for sensitivity, and replaceable in one line when the data
arrives. The build itself often shows which data matter — findings here may unblock items
downstream. Everything lands in the `research` profile or `research/`; the shipped trainer
stays frozen.

**Rules for groundwork on a blocked item**
1. The placeholder value is named, documented with why it was chosen, and marked `chosen` in
   the ledger — never presented as sourced.
2. A sensitivity sweep over its plausible range is part of done: the result says how much
   the missing datum matters.
3. Tests pin the mechanics (conservation, limits, round trips), not the placeholder number.
4. The DATA_REQUESTS entry that would replace it is linked.
5. **No single-athlete fits enter the physics** (2026-09-27). [CR06], [BR24] and Holt validate;
   they do not tune. A fit to one athlete is reported as a diagnostic with its confounders.
6. **Validated physics is not dropped because a one-athlete fit disagrees with it** — the fit is
   the suspect. Sourced ranges (e.g. Coppel's corrections) are swept, not fitted.

## Backlog, in priority order

| # | item | why now | blocked on | groundwork approach | size | done when |
|---|---|---|---|---|---|---|
| 1 | **Hands on the handle (4.3)** — a constraint force between the hands and the handle, so the rower's effort splits between body and blade | the catch deficit (~6–7 IVV points) is the balance's split (SOURCES §161); also the finish turn-round and the stretcher shortfall | the torque-driven body | build the constraint on the prescribed body first: the handle force becomes the constraint reaction, body motion stays prescribed; then relax the body | L | the [BR24] and [CR06] catch dips reproduced at the right power, momentum books closed, a test that fails without the constraint |
| 2 | **Blade depth from a measured vertical oar angle**, with Grift's immersion curve | [BR24] records the vertical oar angle (V1, V2) — the first sourced blade depth; the curve is in `data/literature/` | nothing (was: a curve and a depth) | vertical angle → blade cover along the stroke → `immersion(cover)` from the Grift table on the dynamic oar, as a study flag | M | the [BR24] run with and without depth; the entry/exit force shaped by cover; ledger immersion row moves off `failing` |
| 3 | **On-water crew driver** — drive length and recovery body swing from what two on-water scullers share | worth ~2 IVV points (SOURCES §160); removes the ergometer's systematic part | nothing | a research-profile kinematics option: drive fraction and a recovery warp set from [K05], [BR24], [CR06] | M | the like-for-like rerun on both athletes; the eight/four scorecard rerun; the frozen game unchanged (profile test) |
| 4 | **Knee and ankle moments** with the foot-force direction as a parameter | phase 4.3's joint chain needs them | McGregor (DATA_REQUESTS §11c) | inverse dynamics with the direction as a named parameter, swept over the plausible range | M | moments computed with a stated sensitivity to the direction; books closed; the sweep says whether the datum matters |
| 5 | **Entrainment added mass** (Grift eq. 2.15) as a blade option | the catch is an accelerating plate; Labbé and Grift both say added mass dominates the first ~0.1 m | a full-scale scaling law | dimensional scaling (ρ·A·V or ρ·A^{3/2}·a) as a named choice, both variants swept | M | the option runs; the sweep on [BR24] brackets its effect on the catch dip |
| 6 | **`handle_power` as handle force × handle velocity** in the research profile | `mean_handle_power` reads ~1.46× a measured handle power (SOURCES §158) | promotion only (shipped stays) | a correct definition behind the profile, with a test comparing it to [BR24]'s 432 W | S | the research scorecard reports the corrected power; shipped untouched |
| 7 | **Tier 3 infrastructure** (phase 5): vectorised environment skeleton, stroke-phase tokenisation, per-channel sensory delays | planned, not blocked; long lead time | nothing | the environment API and a throughput measurement on the current physics | L | steps/s per core measured against the 10⁶–10⁷ training needs |
| 8 | **Cox-box / CoxOrb export importer** for the call model | the pipeline takes new data as folders; an importer lets crews drop exports in | data from crews | parse the common export formats into `boat.csv` sessions | S | a sample export round-trips into a session; README updated |
| 9 | **Charles stream field** — spatially varying current | the stochastic line optimisation needs it; discharge data are loaded | a validated flow field | discharge / cross-section → depth-averaged velocity per cell, a named roughness choice | M | a field on the course grid; the route evaluator reads it; sensitivity to the roughness choice |
| 10 | **Bridge piers as constraints** | bridges are landmarks only; the Weeks and Anderson arches are where boats lose time | nothing | pier footprints from the obstruction data as keep-out regions in the route optimiser | M | the optimised line passes through arches; clearance reported |

## Board

| to do | in progress | done this sprint |
|---|---|---|
| 4 Knee/ankle · 7 Tier 3 infra · 9 Stream field · 10 Piers | 5 Sourced blade (moved up) — [CG07] lift/drag + [LB19] added mass + strip integration + Grift immersion, nothing fitted; validated on [CR06], [BR24], Holt (§162, corrected) · 1 Hands on the handle — kinematic-drive reference built (`kinematic_drive.py`) and found blade-limited (§162); the constraint goes on the force-driven model | 8 NK LiNK importer — `ingest.py session --export`, reproduces race 1's parse exactly, keeps Empower fields; test on a synthetic snippet · 6 Handle power — `mean_handle_power(definition="handle")`, (1 − r_h/L) of the oarlock figure; default unchanged; 2 tests · 2 Blade depth — `coxswain/crew/blade_depth.py` (`BladeDepth`, Grift curve, `zero_offset` and `reference` named and swept), `DynamicOarSimulator.blade_depth`; worth ≤0.2 IVV points on [BR24] (SOURCES §162); 8 tests · 3 On-water driver — `OnWaterTiming` ([K05], predicts both athletes to 0.006), `Boat(sequencing=)`, `research/biorow/onwater_driver.py`; the shared features move IVV <1 point, not the ~2 hoped (§162); 5 tests |

## Findings that may unblock downstream items

*(Record here as they happen, with the item they affect.)*

- 2026-09-27, before the sprint: the catch deficit is the oar balance (SOURCES §161) → makes #1
  the lever for the catch, the finish turn-round and the stretcher shortfall at once.
- 2026-09-27: [BR24] carries a vertical oar angle → unblocks blade depth (#2) and with it the
  immersion refit that PHYSICS_PROGRAMME had blocked on data.
- 2026-09-27, #2: blade depth from his vertical oar angle changes IVV by ≤0.2 points and speed
  by ≤0.4% across every zero offset and normalisation → depth is not the catch; the immersion
  refit is done for the dynamic oar and needs no more data for this purpose (SOURCES §162).
- 2026-09-27, #3: the drive fraction and recovery swing the two on-water scullers share move
  the model's own IVV by under a point (60.6 → 60.0–62.4%) → §160's 2 transferable points are
  in the *drive's* shape her curves carry, not in the shared timing; #3's driver stays a
  research option and the crew-timing share goes with #1 (SOURCES §162).
- 2026-09-27, #1 reference: with his oar angle, blade depth and body all prescribed, no blade
  law reaches him — slip 4.37 m/s / 350 W / 59%, C2 ×2 4.52 / 388 W / 57%, tier 2 4.46 / 341 W /
  57%, against 4.64 / 432 W / 49%; a deeper blade is slower. The body is worth 2 IVV points,
  nothing in speed (SOURCES §162).
- 2026-09-27, the blade at his kinematics (`blade_law_check.py`): the slip law gives 37 N·s of
  his 105 N·s drive impulse per oar; the best constant C2 at any centre of pressure (1.795–2.01 m)
  73 of 94, half his force at 0.1 s and none after 0.75 s. His C_N(α) sits below tier 2 at the
  catch and far above a steady plate mid-drive → **#5 (unsteady blade) moves ahead of #1**;
  [BR24] is the measured load-and-velocity pair the tier 2 coefficient item was blocked on
  (PHYSICS_PROGRAMME); the late-loading blade and the model oar's 11.5° lead are one fact.
- 2026-09-27, #5 first tests (corrected the same day, rules 5–6): single-athlete fits (added
  mass negative, lift refitted to 0.40–0.76, a tip centre of pressure) are diagnostics only and
  enter nothing. Robust: driven by his measured blade force, the model's hull and his body swing
  50.8% against 49.1% — hull and body are right; the blade's force time course is the gap.
  Derived: strip integration moves the centre of pressure to 1.86–1.93 m. Open: his handle-force
  scale (question 3 in the Kleshnev draft). #5 proceeds as a blade built only from sourced physics.
