# Validation ladder: the rung-2 rower with a flexible oar

*Created 2026-11-09 (SOURCES §228). User: "see whats worth following up on to get this model validated. get me a
validation ladder and make it pass". Evaluator: `research/validation/ladder.py` (reads the runs' outputs, prints the
table, writes `data/local/validation/ladder.json`). Memory: `oar-bending-withdrawn` (its "how to apply" names this
ladder).*

## The model under test

`DynamicOarSimulator(crew="handle")` driven by rung 2's stroke clock (`research/crew/rung2_population.py`): the
athlete's own body timing, the population's muscles handed over split (own body part + population handle part, scaled
to the athlete's oar timing), the on-water handle force–velocity (Nordez et al. 2025, V0 4.53 m/s) on the rower's own
pace, the research cross-flow blade (normal load), the buried catch, the second-half release; the oar is [L14]'s
measured scull shaft as a dynamic hinge (`BendDynamics`: blade mass, measured structural damping) carrying the blade's
added mass (κ 0.1 of Patton's since §228). The athletes' body channels are GCV-smoothed (§232). **The population whose
muscles and planned hand motion are borrowed is RBN 2023/12's reference curves since 2026-11-13** (`POP_DRIVE =
"rbn2023"`, SOURCES §237; [K05] Fig. 1 before). Each rung names any setting it changes.

## Rules

1. **Bands are fixed before the run they judge** and are not moved after it. A band that turns out to be wrong is
   changed only with a new source, recorded here with the date and the reason, and every rung is re-run.
2. **Nothing is fitted to an athlete** (memory `no-single-athlete-fits`). A failing rung is a reason to find missing or
   wrong physics; a fix must be sourced or structural, and then the *whole* ladder is re-run.
3. A band is marked **sourced** (from a measurement's own scatter or a published comparison) or **chosen** (a
   judgement, stated before the run, with the reason). Chosen bands are tightened when a source appears.
4. A rung passes only if every criterion in it passes, for every athlete it names.

## The rungs

| rung | what | criterion | band | kind |
|---|---|---|---|---|
| **A1** | shaft stiffness and modes: beam FE with [L14]'s EI against Breitschädel & Løset 2010's eight clamped sculls (`research/blade/oar_modes.py`) | mean predicted / measured first frequency | 0.95–1.05 | chosen (their SE is 0.01 Hz; 5 % allows the uniform-beam idealisation) |
| **A2** | the dynamic shaft's stiff limit (`test_bend_dynamics.py::test_a_stiff_dynamic_shaft_converges...`) | speed gap to the rigid oar with the same water at K = 3e5, and falling with K | < 0.5 %, halving | self |
| **A3** | step convergence of rung 2 (half the step) | speed / IVV change | < 0.5 % / < 0.5 points | self |
| **A4** | rigid invariance of the release and blade-track fixes (`test_release_after_...`, `test_the_blade_track_...`) | rigid run unchanged | 1e-9 | self |
| **B1** | the shaft and its water in a tank, motion imposed: [RLBQV19]'s full-size trial (`research/blade/bench_dynamic_shaft.py`) at [RLBQV19]'s own face-on regime (1.3, SOURCES §191) | bend rms against the measured bend; bend peak's angle; M_z impulse ratio | ≤ 1.5° (20 % of the 7.5° peak); within 10° of 85°; 0.85–1.15 | chosen |
| **B2** | the population's own stroke (rung 1 on RBN 2023/12's handle velocity and body timing) against the population's own force curve, per 10 % of drive time (`research/validation/population_kinematics.py`) | force rms; the force peak's time | flexible < rigid; peak within the reference's 40–50 % | comparative — added 2026-11-13 after its first run was seen, so no tuned band |
| **C1** | in the boat, motion imposed (rung 1), [CR06]'s own body: handle force against oar angle (`research/blade/bending_handle_force.py`) | rms over the drive as a fraction of her peak; peak's angle | ≤ 15 %; within 10° of hers | chosen (the best earlier model, §219, was 12.5 %; the digitised figure's channel offsets dominate below ~10 %, §191) |
| **C2** | the blade and shaft alone on each athlete's measured oar angle and boat speed, open loop (`research/validation/open_loop_shaft.py`) | handle-force rms as a fraction of the measured peak | flexible < rigid, each athlete | comparative — added 2026-11-13 after its first run was seen |
| **D1** | rung 2, both athletes: mean boat speed | model / measured − 1 | within ±3 % | chosen ([CR06]'s own model predicts her speed to 1.9 %; wind and water on the day are unknown) |
| **D2** | rung 2, both athletes: IVV (max − min) / mean of the last stroke | model − measured | **within 0.75 × the between-rower relative SD × the measured IVV: ±6.7 points ([BR24]), ±7.5 ([CR06])** — was ±3 until 2026-11-10 | sourced: Leão et al. 2025 (*Sensors* 25:4696; 23 single scullers, 2000 m races): IVV 5.78 ± 1.05 km/h (men), 5.50 ± 1.11 (women), relative SD 18.2 / 20.2 %; 0.75 keeps the first band's stringency (3 points over Loschner & Smith's ~4-point SD). Changed after the first run at the user's instruction ("source the band"); see History |
| **D2b** | rung 2, both athletes: the velocity coefficient of variation, SD / mean (BioRow's own measure, per the data's author) | model − measured | 0.75 × relative SD × measured: ±1.74 points ([BR24]), ±2.03 ([CR06]) | sourced: Leão et al. 2025, CVV 14.13 ± 2.02 % (men), 11.64 ± 1.93 % (women); added 2026-11-10 with D2's change |
| **D3** | rung 2, both athletes: handle power | model / measured − 1 | within ±10 % | chosen (power is an output; gate power meters read to a few %) |
| **D4** | rung 2, [BR24]: handle-force shape against his measured cycle (`research/crew/handle_force_shape.py`) | force centre (fraction of the drive); peak force; least force 10–80 % of the drive | within ±0.05 of 0.49; within ±15 % of 424 N; > 0 | chosen. Since 2026-11-10 the model's force is compared on BioRow's lever: the measured handle force is the inboard bending torque expressed 0.06 m from the handle's top (sculls; the data's author), the model's torque over the full inboard — ×0.875/0.815 (S12) |
| **D5** | rung 2, both athletes: oar progress against measured (the timing the muscles are scaled to) | rms | ≤ 0.05 | chosen (§201's pass level) |
| **E1** | regression: the research profile's scorecard (rigid oar; nothing here changes it) | every scorecard target | as `coxswain/validation/targets.py` | sourced there |

[CR06] has no measured handle-force *cycle in time* beside her angle trace, so D4 is [BR24]'s; C1 is her force against
angle. Both athletes are held-out tests: nothing in the model was fitted to either.

## History

- **2026-11-09, ladder set** (bands above, before the first run). First run: see SOURCES §228.
- **2026-11-09, first full run: 21 of 23 pass** (`data/local/validation/ladder_flexible_kappa0.1.log`). The model as
  above with the blade's water at κ = 0.1 of Patton's (B1 and §173), after three fixes found on the way (the own body's
  trunk clock, the bent blade's wetting, §227's release and load direction). **D2 fails for both athletes: IVV +4.4 /
  +4.1 points.** Cause traced to the catch's build-up (the flexible shaft's quasi-steady blade loads too slowly in the
  first ~0.15 s; SOURCES §228 section 5). Bands unchanged. The rigid oar under the same settings passes D2 (+1.3 /
  +1.5) and fails C1 (22.7 %) and D4 (force centre 0.32).
- **2026-11-09, user decision:** research the catch; D2 stays failing until sourced physics closes it, then the whole
  ladder is re-run. First source: Kobayashi & Kinoshita 2003 (SOURCES §229). Its rising/falling asymmetry failed the
  tank test (§230: worse on two of three measured loads); not adopted, D2 unchanged.
- **2026-11-09, a tank-profiled lift slope (k 1.3, §231)**: B1 fails, D2 moves 0.2–0.4 points. Not adopted. With the
  force matched (C1, rung 1, her own body and oar motion) IVV is still +3.5–4 points: D2's residual is in the hull and
  body dynamics, not the blade. Ladder stands at 21 / 23.
- **2026-11-09, the athletes' channels smoothed by GCV before differentiation (§232)**, a processing fix chosen before
  the run: [BR24] IVV 53.1 → 52.3 % (+3.2), [CR06] 53.9 % (+4.5). D2 still fails; 21 / 23. The excess push sits where
  the model's posture geometry carries the channels to the thighs and arms.
- **2026-11-10, user decision: "known limitation, draft data request, source the band and continue".**
  - **Known limitation (kept on record):** at the first band (±3 points) D2 fails for both athletes, +3.2 / +4.4.
    Cause traced (SOURCES §228–232): the own-body model's legs and arms slow the hull ~20–30 % harder at the catch
    approach than the athletes' accelerometer allows, though their seat and trunk channels are reproduced faithfully.
  - **Band sourced** (rule stated before the evaluation, SOURCES §233): D2's band is 0.75 × the between-rower relative SD
    of IVV in Leão et al. 2025 (23 single scullers) × the measured IVV, sex-matched: ±6.7 / ±7.5 points. D2b (SD/mean,
    the data author's own measure) added on the same rule. D4 compared on BioRow's force lever (the data author's
    definition). The other bands are unchanged.
  - **Result: 25 of 25 pass** (`data/local/validation/ladder_2026-11-10.log`). Narrow: [BR24]'s D2b (+1.73 against
    ±1.74) and D4's force centre (0.54, the band's edge). The ladder passes on a band set by the population's spread,
    not on the tighter first band; any narrowing of the band (a within-athlete variability source) re-opens D2.
  - Data requests drafted for approval (the data's author: other rates, per-stroke data; an on-water segment-kinematics
    group), not sent.
- **2026-11-13, the population re-sourced; 24 of 26** (SOURCES §237; `data/local/validation/ladder_2026-11-13.log`).
  RBN 2023/12's reference curves were read as numbers from its vector figure (and on the right axis: drive *time* —
  §234–236 had compared on drive length). Its handle velocity fits both test athletes' oar timing 2.5× closer than
  [K05]'s and, imposed on the population with its body timing, the flexible shaft reproduces the population's own
  force curve (10.2 points rms; rigid 23.2), so it is the default population from here. Rungs B2 and C2 added
  (comparative, no tuned band; both pass). Bands unchanged. **Two rungs now fail:**
  - **C1**, [CR06]'s rung-1 handle force: 18.1 % (band 15 %; 10.4 % before). With hands that follow hers the model gives
    12 % more power than she measured; the earlier pass leaned on [K05]'s slow early hands.
  - **D1** for [BR24]: speed +3.1 % (band ±3; +2.8 before).
  Both are force-level discrepancies of opposite sign in the two athletes (open loop: [CR06] +17 %, [BR24] −29 % of
  impulse), so no common change to the blade removes them; they stay failing. Improved: D4 centre 0.52 (0.54), D2b
  +0.86 / +1.32 points (+1.47 / +1.73), D2 +3.5 / +3.4.
