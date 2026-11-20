# Hypothesis register

Every physical hypothesis the model has been tested against: what it claimed, the source that motivated it, the test,
the outcome, the decision, and where it is recorded. The audit trail for the model (user, 2026-11-19: "do we keep
documentation of the hypothesised things that might impact stuff and then the outcome of our test on them? because
those are going to be important for our code to be PhD auditable").

**Rule since 2026-11-19 (§252; memory `adopt-physical-improvements`):** a change that is physically motivated, sourced,
and does not fail validation goes into the **research** default (never the shipped game's), at the conservative end of
its sourced range. An option is kept only for genuinely competing or unsourced alternatives, and a physically real change
that makes a rung worse is a finding to investigate, not a reason to drop it silently. The **Re-review** column flags
past decisions that do not meet this rule.

Scope: this first edition covers the physics hypotheses of the validation era (SOURCES §156–§251) in full, and lists
the earlier withdrawn or rejected results by section for back-filling (bottom). Status keys: **adopted** (in the
research configuration), **option** (available, off), **rejected** (tested, failed), **withdrawn** (a claim taken back),
**finding** (a result, not a model change), **pending** (waits on data or a decision).

## Where the "research default" lives — an audit finding (2026-11-19)

The validated configuration is **not** the research physics profile. `coxswain/physics.py` `PROFILES["research"]` still
has a prescribed crew and a rigid oar, and none of: the blade's added mass (Patton × κ 0.1), the blade's immersion /
buried-catch handover, `release_after`, the measured on-water handle force–velocity, the flexible shaft with bend
dynamics, the grip damping, the RBN 2023/12 population drive. These are switched on by the research scripts
(`research/crew/rung2_population.py`; the ladder's command lines) and validated there (docs/VALIDATION_LADDER.md:
27 criteria, 0 fail, 2 expected fail). Anything that uses the profile — the scorecard (E1), route studies — runs the
older physics. Several of the items are tied to the open flexible-versus-rigid question (B2s; §244–§248), so moving
them into the profile is **pending the user's decision** (listed as R1 below).

## Register

| ID | Hypothesis | Motivation / source | Test | Outcome | Decision | Status | Re-review | SOURCES |
|---|---|---|---|---|---|---|---|---|
| H01 | The blade's load follows the fins' cross-flow law (Helmbold slope, face-on 2.2), normal only | Whicker & Fehlner; [CG07], [H10] | three tank loads open loop; rung 2 both athletes | fits; gate passes, nothing fitted | user 2026-10-31 | adopted (profile) | — | 188–202 |
| H02 | The tangential blade load is a construction artefact | [CG07] points resolve to −0.035 bias | decomposition | confirmed | normal-only | adopted | — | 201 |
| H03 | The catch is dynamic only once the blade is buried (BioRow 6° slip) | BioRow catch slip | rung 2 entry spike | spike gone | takeover at buried | adopted (scripts) | R1 | 203 |
| H04 | The trunk follows the on-water population's timing | Pitto et al. 2024 | athletes' back channels | 0.051 / 0.074 vs keyframes 0.122 / 0.089 | user 2026-11-02 | adopted (profile) | — | 211, 213 |
| H05 | The legs' and trunk's travel come from the population | [K05], Telfer 2023 | rung 2 | mixed | defaults unchanged | option | **yes**: decision reason not recorded beyond "travel off" | 212 |
| H06 | The legs follow [K05]'s population clock | [K05] | athletes' seat channels | no open on-water seat trace to decide | keyframes kept | option | no (no source decides) | 214 |
| H07 | Recovery-only leg timing | — | landmarks | positions without accelerations | option only | option | no (unsourced) | 215 |
| H08 | Arm-chain stretch (1.12 arm lengths) | force-proportional stretch | — | the value is a fit | off | option | no (fitted value) | 210 |
| H09 | Oar bending, quasi-static hinge, [L14] scull | [L14] thesis | rung 2 | rang on the crew (undamped) | withdrawn same day | withdrawn | — | 217–218 |
| H10 | The crew's pull has a force–velocity (on-water, V0 4.53) | Nordez et al. 2025 | rung 2 ringing; rigid unchanged | damps the mode, rigid results unchanged | used in the ladder | adopted (scripts) | R1 | 218, 222 |
| H11 | The bend is a dynamic state with the blade's mass and water | Breitschädel & Løset 2010; §224 modes | step convergence; tank bend | converges; B1 1.19° | ladder configuration | adopted (scripts) | R1 | 224 |
| H12 | The handle crew's release must not fire near the catch (release_after 0.5) | stall diagnosis | rung 2 | stall gone; rigid unchanged | user | adopted (scripts) | R1 | 226 |
| H13 | The blade's water near the surface is κ ≈ 0.1 of Patton's | §173 bound; [RLBQV19] bend | B1 at κ 0.1 / 0.5 / 1 | 1.19 / 2.20 / 3.62° | κ 0.1 | adopted (scripts) | R1 | 228 |
| H14 | A rotating plate's asymmetric angle law ([KK03]) adds early force | Kobayashi & Kinoshita 2003 | three tank datasets | better on one, worse on two | not adopted | rejected | no (fails validation) | 230 |
| H15 | A tank-profiled lift slope (×1.3) | three tanks agree 1.2–1.4 | ladder | B1 fails (bend 1.60–1.80°, M_z 1.26); D2 barely moves | not adopted | rejected | no (fails B1); open trade noted | 231 |
| H16 | Channels must be smoothed before differentiation (GCV) | Woltring 1986 | rung 2 | IVV −0.8 ([BR24]) | adopted, superseded by H26 | superseded | — | 232 |
| H17 | Stroke averaging explains the IVV gap | RBN 2017/12 | the author's own check | peaks lowered ~1 % only | closed | rejected | — | 234 |
| H18 | The blade comes out over BioRow's release slip (16.6°, half out at 90 %) | RBN 2017/12, 2023/12 | ladder | late force 20 → 17 %; D2b tipped by 0.01 then (§235); re-tested §252: 0 fail, D2 [BR24] +1.3 → +0.9, C2 [CR06] 13.7 → 13.5, B2s 21.5 → 22.4 (already an expected fail) | option (§235) → **default** (§252) | adopted (scripts) | done §252 | 235, 252 |
| H19 | The force peak is late because of the shaft | [L14] Table 12 | stiffness sensitivity on population | model +3.8 vs rowers +0.2 | finding | finding | — | 236 |
| H20 | RBN 2023/12 Fig. 1 hands for the population | Kleshnev, >50 000 strokes | athletes' oar timing | 2.5× closer | default | adopted (scripts) | R1 | 237 |
| H21 | The bend reset at each catch | — (bug) | test | fixed | — | fixed (S15) | — | 238 |
| H22 | Scullers' own hands (Fig. 2) for the population | RBN 2023/12 Fig. 2 | athletes, population | worse for [CR06]; no better for population | option | option | no (mixed, competing) | 238 |
| H23 | [BR24]'s early drive is reachable by some blade law | no-slip bound | open loop | not on the assumed shaft | inputs question | finding / pending | — | 238 |
| H24 | [BR24]'s in-air force is the oar's inertia (2.1–2.3 kg m²) | — | ratio over time | not constant | — | withdrawn | — | 238, 239 |
| H25 | The closed-loop bend coupling is right | — | open-loop path on the same motion | 1–3 N | — | finding (verified) | — | 240 |
| H26 | Channels low-passed at the rowing standard 6 Hz | Urbanczyk 2021 (Buckeridge 2015, Pollock 2009) | accelerometer rules out > 6 Hz | IVV −0.2 | default | adopted (scripts) | R1 | 240 |
| H27 | [BR24]'s legs on his own travel (S16) | his seat channel | rung 2 | IVV 52.3 → 50.4 % | default | adopted (scripts) | R1 | 240 |
| H28 | The population force test is ill-conditioned | §240 sensitivity | Jensen test | averaging gap 1–4 vs 9 points | withdrawn | withdrawn | — | 240, 244 |
| H29 | A sculling-averages recovery hand law | the author's averages | athletes' recovery hands | better for [BR24], worse for [CR06]; no material effect | option | option | no (mixed, competing) | 242 |
| H30 | Rung 2's timing drift is a separate muscle-transfer defect | — | clock rates vs force-level ratios | explained by the force-level mismatch | — | finding | — | 243 |
| H31 | B2 judged on the like-for-like sculling reference | sculling averages | ladder | flexible fails (21.3 vs 12.4) | user: keep both, follow up | expected fail (B2s) | pending | 244, 249 |
| H32 | κ, lift slope, catch angle or sourced stiffness close the early-drive deficit | — | population candidates | none closes it | not adopted | rejected | no (unsourced or fail) | 245 |
| H33 | The angle sensor captures part of the bend (double-counted compliance) | [CR06]'s bracket-on-shaft telemetry | f(d); what-if ladder | needs d 0.18–0.33 m; IVV better, two shape criteria fail | pending the sensor's position | pending | — | 246, 248, 251 |
| H34 | The blade wets from the measured vertical angle | sculling averages' vertical angle | timeline | minor, early only; conflicts with the entry rule | not implemented | pending | **yes**: a sourced entry rule should be designed, not left | 247 |
| H35 | A real grip damps the post-release ringing | Dong et al. 2007 hand-arm model | coupled modes | ζ 0.14–0.28 vs 0.034 | option (§250) → **default 0.14** (§252) | adopted (scripts) | done §252 | 250, 252 |
| H36 | The hands' timing can come from the body's own joint muscles fired on measured muscle timing (step 4) | review §12; Pitto 2024, Vieira 2020; Anderson 2007, Frey-Law 2012 | rung 2 with a module-driven force, against the population-driven force on the same body | stable; oar timing better for both athletes; speed better ([BR24]), IVV better ([CR06]) and worse ([BR24]) | first closed loop; next: segments as degrees of freedom | finding (option, research) | under development | 256 |

**R1 — the research profile.** H03, H10–H13, H18 (if adopted), H20, H26, H27 and H35 are validated in the ladder's
configuration but are not in `PROFILES["research"]`. Moving the uncontested ones into the profile, and deciding the
flexible oar (with which several were validated), is the user's decision; until then the profile and the ladder run
different physics, and every study on the profile says so.

## Earlier results withdrawn or rejected (to back-fill into the table)

From SOURCES, by section (each records its own reason): §7 ([H20] comparison), §28 (the §27 IVV result), §35 (added
mass off by default, later used at κ 0.1), §38, §41 (drive timing falsified), §47 (§46's position), §49–50 (Telfer's drive
fraction rejected, then reversed), §52–53, §55–57, §99–104 (Lake Union geometry, withdrawn until recomputed), §121, §156
(the double's IMU diagnosis), §163 (blade parts off by default, later the research blade), §168, §179, §184 (tier 1
shape rejected on a measured load), §194–195 (synchronisation options off by default: `period_adaptation` and others —
**re-review under the rule**).
