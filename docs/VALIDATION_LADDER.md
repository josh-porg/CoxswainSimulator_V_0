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
added mass. Each rung names any setting it changes.

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
| **C1** | in the boat, motion imposed (rung 1), [CR06]'s own body: handle force against oar angle (`research/blade/bending_handle_force.py`) | rms over the drive as a fraction of her peak; peak's angle | ≤ 15 %; within 10° of hers | chosen (the best earlier model, §219, was 12.5 %; the digitised figure's channel offsets dominate below ~10 %, §191) |
| **D1** | rung 2, both athletes: mean boat speed | model / measured − 1 | within ±3 % | chosen ([CR06]'s own model predicts her speed to 1.9 %; wind and water on the day are unknown) |
| **D2** | rung 2, both athletes: IVV (max − min) / mean of the last stroke | model − measured | within ±3 points | chosen (below one population SD: three elite women scullers 43 / 46 / 51 %, Loschner & Smith 2000; the muscles are the population's) |
| **D3** | rung 2, both athletes: handle power | model / measured − 1 | within ±10 % | chosen (power is an output; gate power meters read to a few %) |
| **D4** | rung 2, [BR24]: handle-force shape against his measured cycle (`research/crew/handle_force_shape.py`) | force centre (fraction of the drive); peak force; least force 10–80 % of the drive | within ±0.05 of 0.49; within ±15 % of 424 N; > 0 | chosen |
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
