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
| 1 Hands on the handle · 3 On-water driver · 4 Knee/ankle · 5 Entrainment · 7 Tier 3 infra · 8 Importer · 9 Stream field · 10 Piers | 2 Blade depth | 6 Handle power — `mean_handle_power(definition="handle")`, (1 − r_h/L) of the oarlock figure; default unchanged; 2 tests |

## Findings that may unblock downstream items

*(Record here as they happen, with the item they affect.)*

- 2026-09-27, before the sprint: the catch deficit is the oar balance (SOURCES §161) → makes #1
  the lever for the catch, the finish turn-round and the stretcher shortfall at once.
- 2026-09-27: [BR24] carries a vertical oar angle → unblocks blade depth (#2) and with it the
  immersion refit that PHYSICS_PROGRAMME had blocked on data.
