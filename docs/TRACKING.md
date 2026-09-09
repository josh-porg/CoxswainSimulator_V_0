# Bugs, gaps and things being tracked

What is known to be wrong, known to be missing, or known to be a
placeholder. Kept separate from `SOURCES.md`, which records what was
built and why; this file records what has *not* been settled.

Anything fixed moves to the bottom with the evidence that fixed it, so
the file is also a record of what kind of thing goes wrong here.

---

## Open — correctness

### The released build is far behind the code
**Impact: high.** `v0.6` predates every physics change since: Michell
wave drag, wind, crew skill, fatigue, balance authority, blade contact,
crew timing, and the water-seam fix. Anyone who downloads the link is
running none of it.

*Next:* cut a release once the physics settles.

### No model of a blade squared and immersed before the catch
**Impact: medium.** The model has two clean regimes — a feathered skim
on the recovery, and a normal drive — and names the third explicitly as
out of scope ("a squared blade catching is a crab, a different
regime"). The transition is not represented: a blade that squares while
still in the water must grip, and that involuntary early catch is
approximated by scaling the drive impulse from the *instantaneous* heel
rather than latching the heel at square-up.

Cannot be latched without state, because `breakdown()` is stateless for
RK4. Would need the same `on_step` treatment the crew timing got.

### The crew do not lean to correct the boat
**Impact: medium.** `skeleton(t)` and `segment_state(t)` take only time
— neither has any input for the boat's heel. So the crew never shifts
mass laterally in response to being unset, in the physics or the
picture.

They *do* swing across the boat as part of the stroke (a sweep rower is
wound round toward the handle; measured lateral travel is up to 0.50 m),
and that is consistent between the drawn joints and the modelled mass.
What is missing is the *reflex*: `trunk_lean_authority` counts leaning
as an available balance moment, and `PhaseAuthority` includes it in the
recovery limit, but no actual lateral mass movement happens. The moment
arrives entirely through handle heights at the riggers.

Correcting it means giving the kinematics a lean input driven by the
balance command, which changes the crew mass field and therefore the
trim — not a drawing change.

### Crew timing is wired in the trainer but not in `run()`
**Impact: medium.** `CoupledCrew` advances through `FixedStepLoop.on_step`,
which only the trainer uses. Every analysis script calling
`RowingSimulator.run` still gets fixed phase offsets, so published
figures do not include the resync dynamics.

---

### The two speed models disagree by a factor of 2.3 in efficiency
**Impact: high — it decides every published target.**

`CoursePacing` (which produces the targets in `scripts/targets.py`)
solves `R(v)v = 0.80 x gate power x rowers`: a flat blade efficiency of
0.80. The 6-DOF simulator, driven through `power_scales`, implies an
efficiency that RISES with power and is far lower:

| gate W/rower | simulator speed | implied efficiency |
|---|---|---|
| 131 | 2.71 m/s | 0.34 |
| 188 | 3.29 m/s | 0.40 |
| 250 | 3.89 m/s | 0.49 |
| 486 | 5.60 m/s | 0.70 |

Some rise is real — a slower boat slips its blades more — but 0.34 is
implausibly low, and the two halves cannot both be right. On the same
crew they predict 23:49 and 29:43 for the same course.

The field says `CoursePacing` is closer: the slowest crew in six years
of results is 25:33, and the simulator would need 188 W a rower just to
row that, which is a strong 60+ erg. But this is inference, not
measurement.

**Localised to the oar-to-hull conversion.** Over one cycle at scale
1.0, the mean forward force the oars put into the hull is 248.7 N, and
the crew's handle power is 1945 W. At the speed the simulator settles
at, that is a propulsive-to-handle ratio of **0.346** -- and 0.346 is
exactly `blade_efficiency x inboard/outboard` (0.78 x 0.445).

The FORCE relation looks right: for the boat-and-crew system the
external propulsive force is the blade's reaction, `F_handle x
inboard/outboard`. What does not reconcile is the POWER. For the
efficiency to reach 0.78 the boat would have to move about 2.25 times
the handle speed; the geometry here gives nearer 1.2-1.6, and the
shortfall is the whole gap.

So the open question is narrow and worth stating exactly: whether the
handle kinematics (and therefore `mean_handle_power`) are consistent
with the force path, not whether the hull or the gearing is wrong.

**Why no existing test caught it.** The Holt validation SCALES oar force
until the delivered power matches a target, so it never exercises the
handle-power-to-force relation at all -- it is calibrated around
whatever that relation happens to be. The 6-DOF path is validated for
force to speed and unvalidated for watts to force, and the trainer uses
the second.

**Partly settled, and it goes against the simulator.** Sammamish rowed
this category in 2024 and finished 12th of 20 in 22:07.9. Asked what
that took:

* `CoursePacing` says **164 W a rower** -- a 5 km erg of 21:27.9, which
  is an ordinary masters women's score and lands mid-field exactly where
  the crew finished;
* the 6-DOF path says **223 W** -- a 5 km of 19:22.6, which for a 60+
  woman is close to national level, for a crew that came 12th of 20.
  That is not believable.

So the flat 0.80 is closer to reality than the simulator's power chain,
and the published targets stand. What is still open is WHY the 6-DOF
path is so lossy -- whether the fault is in `power_scales`, in the
blade model, or in the unsteady losses.

### The balance controller has nothing to do
Found while making the crew's port arms port (the kinematics signature
omitted the side; see Fixed).  With that bias gone, the 6-DOF eight is
**roll-stable at race pace on its own**: from an 8 deg initial roll it
returns to 0.04 deg within 6 s with no balance controller at all, and
every authority — full, oars-only, constant, none — gives the same swing
(0.047 deg matched, 1.331 deg at a 60 ms split).

The terms at 2 deg of heel, 4.6 m/s, eight: buoyancy −36.7 N m, gravity
+75.7 (net +39, destabilising, as `roll_divergence_time`'s surrogate
says) — and **appendage −209.4 N m**, at zero sideslip and zero roll
rate.  A fin restoring roll five times harder than the hull destabilises
it is either a real heel–sideforce coupling or a frame slip in how the
appendage moment is taken about the hull origin; nobody has checked
which.  Until that is settled, "the boat has to be sat" is a surrogate
claim the simulation contradicts, and the two balance tests that assumed
it are expected failures pointing here.

### `mean_handle_power` ignores `power_scales`
Its docstring says it reports power "at the boat's current scale". It
does not — it integrates `oar_force` without the scale, so it always
returns the scale-1.0 figure (486 W for the catalogue four). Callers
that use it to *calibrate* a scale are fine, since power is linear in
scale; callers that use it to *check* one silently get the wrong answer.

## Open — numbers nobody has measured

These are placeholders. Each is labelled in the code as such; this is
the index of them.

| what | value | where | basis |
|---|---|---|---|
| novice power scatter | 0.110 | `crew/variability.py` | extrapolated along the elite→junior slope |
| novice timing scatter | 0.075 | `crew/variability.py` | same |
| square-up fraction | 0.25 of recovery | `crew/blade_contact.py` | "roughly where a crew squares"; put in a constant so it can be moved |
| balance falloff with inexperience | 60→150 N, 0.7→2.0° | `sim/control.py` | judgement; only the *ideal* end is calibrated |
| bias as a fraction of scatter | 0.45 | `crew/variability.py` | judgement |
| blade clearance on the recovery | 0.08 m | `crew/blade_contact.py` | typical, not measured |
| roster rower stature | 1.63 m W, 1.75 m M | `crew/roster.py` | NHANES population mean; the squad sheet logs weight and erg scores but never a height |

The roster stature is the one placeholder that is flagged in the
product as well as here: every rower built from `data/squad_roster.csv`
carries `stature_estimated=True` and the rig editor prints the height
with a leading `~`. Their *power* is measured; their *body geometry* is
a guess, and stature drives every link length in the kinematics, so it
moves the crew's centre-of-mass travel and with it the hull's speed
fluctuation. The two should never be quoted with the same confidence.

Also: `CP = 302.7 W` and `W' = 11.4 kJ` are literature means, not this
crew. They set the *shape* of every pacing answer and should be
measured before any number is quoted to an athlete.

---

## Open — validation gaps

### Michell is still 7–9% slow on singles
Against Holt it is excellent on doubles (+0.6%, +0.5%) and still low on
singles (−6.7%, −9.1%). Better than the constant coefficient on all four,
but the singles gap is real and unexplained.

### Surge swing is 10–31% too large
Model against Holt: ratios 1.31, 1.31, 1.10, 1.18. `scripts/unsteady.py`
squares this quantity, so the error is four times worse there.

### The eight is validated only by inference
Holt measured singles and pairs. The boat this project cares about most
has no measured counterpart in the comparison.

---

## Open — presentation

### Headless `--frames` is slow
About 50 s for 122 frames, because it now draws every step so that
catch-driven effects (puddles, splash) can appear at all. Fine for
screenshots; painful for long sequences. CI is unaffected — it passes no
`--frames`.

### Windows CI cannot render
The runner is a headless service session with no usable OpenGL, so the
render check warns instead of failing there and the payload is verified
by file presence. The Windows artefact is built and run locally before
release.

---

## Performance — measured, on this desktop

A rower reported the trainer "unusably sluggish" on an old gaming
laptop.  Profiled headless, 180 frames of an eight on Head of the Lake
at Standard, with nothing else running.  The numbers are this machine's;
the ratios are what matter.

| what | before | after | how |
|---|---|---|---|
| physics per frame | 31–34 ms, every tier | **2.9–3.6 ms** at 50–60 Hz (p50); 4.7 ms at High's 100 Hz | crew kinematics and oar force tabulated once per stroke (`Boat.tabulate_crew`, trainer only); 60 Hz is the same boat to 0.2 mm |
| `seattle._inside` at start | 14.2 s | 0.96 s | matplotlib's C crossing test with a bounding-box pre-filter; identical answers |
| world build at start | 33–37 s | 10–16 s | walls emitted per building in numpy; GC off for the build |
| crew tables at start | — | 0.3–1.1 s | one chain solve per sample; shared by kinematics signature; warmed after the timing scatter |

Per tier, 150 headless frames of an eight on Head of the Lake, medians
after ten warm-up frames, **on this machine's Intel UHD Graphics** — an
integrated part, which is the case the rower reported. `--bench` prints
these; the per-pass GPU timers are what found the water.

| tier | before | after | water pass | world pass |
|---|---|---|---|---|
| Ultra minimal | — (new) | **17.1 ms, 59 fps** | 5.3 ms | 3.8 ms |
| Minimal | 41.1 ms, 24 fps | **16.5 ms, 61 fps** | 3.4 ms (was 7.9) | 3.8 ms |
| Standard | 62.3 ms, 16 fps | **24.2 ms, 41 fps** | 8.5 ms | 5.8 ms |
| High | 56.0 ms, 18 fps | ~27 ms | 8.8 ms | 6.5 ms |

Two things the numbers overturned:

* **Render scale hurt.** Drawing at 0.75 into an off-screen buffer and
  stretching cost 44.7 ms against 27.8 at full size on this GPU: the
  extra pass and copy outweigh the pixels saved on a part whose
  bottleneck is not fill. Every tier is at 1.0; `--render-scale` stays
  as a lever for a GPU where the balance differs.
* **"Flat" water was flat geometry with full per-pixel work.** Inside
  `exact_within` every fragment summed eight waves, sixteen puddles, two
  textures and the wake three times over for a gradient, nearest the
  eye where pixels are densest. The low tiers now take their normal
  from the vertex-baked slope (`water_flat`) and no exact band at all.

Peak working set, 120 headless frames, the interpreter itself:
**648 MB Ultra, 671 Minimal, 868 Standard, 887 High**, against
700–1,040 MB before the CPU mesh copies were dropped after upload.

Found on the way, and fixed: the kinematics signature omitted the side
of the boat, so a matched crew shared the stroke seat's chain and every
port rower wore starboard arms (0.15 m out on eight arm masses).  A
yaw bias, steering the boat.  The golden trajectory was re-recorded for
it — the second re-record, with the reason and the numbers in
`tests/test_stepwise.py`.

What the physics is NOT moved to: the GPU.  It is a small stiff 6-DOF
system on tiny arrays; a GPU's launch latency loses to the CPU JIT that
already runs the hull sweep.  There is no tensor workload here for a TPU.

---

## Done — running on the machine you have

### Measured, this round (Intel UHD, 1180x680, hazy, eight; `--bench 150`, no timers)

| tier | physics | draw+GPU | frame p50 | fps | v0.9 frame |
|---|---|---|---|---|---|
| Ultra minimal | 1.7 ms (Heun, 50 Hz) | 7.9 | **9.7** | 104 | 16.6 |
| Minimal | 2.9 (Heun, 60) | 14.3 | **17.2** | 58 | 18.7 |
| Standard | 3.8 (Heun, 60) | 18.0 | **21.9** | 46 | 25.9 |
| High | 6.8 (RK4, 100) | 22.4 | **29.3** | 34 | 42.8 |

Per pass (`--bench-passes`): water 3.9 / 7.3 / 7.6 / 11.3 ms, world 3.0 /
4.3 / 5.8 / 5.6, sky 0.2–0.3, boat 0.1. The water is still the largest
pass at every tier; the next lever is its fragment cost at Minimal and
above, not triangles. The `.exe` is within 0.3 ms of the script.

| what | where | pinned by |
|---|---|---|
| **Four graphics tiers, each cheaper than the last.** `ultra` is the floor: flat water, no SSR/refraction, one shadow tap from a 2048² map, plain distance fog, no skyline, every tree an impostor, 400 m reach at 12 m step, no MSAA, 50 Hz physics. Render scale is a flag, not a tier default — measured, it hurt on an iGPU. Every knob is a `Tier` field; a typed flag always beats the tier. | `coxswain/viz/menu.py` `QUALITY_TIERS`; `scripts/fpv.py` tier block; shader uniforms `shadow_taps`, `fog_simple`, `reflect_steps` | `test_the_tiers_do_not_go_below_fifty_hertz`; per-tier `--bench` |
| **The tier is chosen for you.** `--quality auto` (the default) reads the adapter list before the build and the live GL renderer after the context exists; a dedicated GPU that is not the one drawing is reported with the fix, and `--prefer-dedicated-gpu` writes the per-user Windows preference — only when asked. | `coxswain/viz/hardware.py` | `tests/test_hardware_telemetry.py` |
| **The `.exe` is not slower than the script.** v0.9 frozen build against v0.9 source on the same iGPU, same tier: Minimal 19.3 vs 19.1 ms, Standard 23.4 vs 23.6, physics 2.9 ms in both — numba is live in the frozen build. Re-check after any build-tooling change with `Coxswain.exe --bench 150`. | `tools/build_exe.py`; `--bench` in the exe | measured 2026-09-09 |
| **Timer queries cost 2.3 ms a frame themselves** on an integrated part (11.8 → 9.5 ms at ultra). `--bench` takes its headline WITHOUT them; `--bench-passes` adds the per-pass breakdown, which is the tool that found the water pass and the sky noise. | `PassTimer` in `scripts/fpv.py` | `--bench` vs `--bench --bench-passes` |
| **The loop cannot spiral.** `max_frame` alone allowed fifteen steps in one late frame at 60 Hz; the loop now takes `max_steps` (4) and drops the rest, counting it (`dropped`) into the diagnostics and the report. The boat runs briefly slow; the program does not stop. | `coxswain/sim/realtime.py` | `tests/test_any_machine.py`, `tests/test_stepwise.py` |
| **The HUD is composed and uploaded only when it changes** — its text and the rudder knob's pixel. A full-window RGBA upload plus font rendering every frame was ~3 ms on an iGPU. | `scripts/fpv.py` HUD block | `test_the_hud_is_uploaded_only_when_it_changes` |
| **Low tiers stop paying for sky noise in the water** (`sky_detail`) and use the vertex-baked slope as the water normal (`water_detail`). Sky pass 1.1 → 0.3 ms, water 4.9 → 4.1 at ultra. | shader uniforms `sky_detail`, `water_detail`; `Tier.sky_detail` | `tests/test_any_machine.py` |
| **Heun below High.** Two derivative evaluations a step instead of four; at 60 Hz it is RK4 at 100 Hz to 3 cm over 367 m (0.0002 m/s, same roll). High keeps RK4; the studies and the golden never see Heun. | `coxswain/core/integrators.py`, `Tier.physics_scheme`, `--physics-scheme` | `tests/unit/test_heun.py` |
| **Performance reports come home.** Off until switched on in the menu (remembered); at close, a JSON of numbers and product names — never a name or path, the sender refuses — goes to a URL the program is told (`--report-url`, `COXSWAIN_REPORT_URL`, `report_url.txt` beside the exe), intended for a Google Apps Script that appends to a sheet; with no URL it is written beside the logs to paste. | `coxswain/viz/phonehome.py`, `settings.py`, `packaging/phonehome/` | `tests/test_phonehome.py` |
| **A diagnostics file.** Build, hardware, settings, world timings, ten-second frame summaries with the physics/draw/present split, stalls over 100 ms with context, tracebacks. Local only, in the OS log folder, printed at start-up; `--no-diagnostics` turns it off. | `coxswain/viz/telemetry.py` | `tests/test_hardware_telemetry.py` |
| **`--bench N`** runs N headless frames with `ctx.finish()` and prints the split, so a regression is a number. | `scripts/fpv.py` | — |
| **Startup**: the wall builder emits each building in one numpy block instead of two Python lists per edge; CPU mesh copies are dropped after the GPU upload (140 MB); the cyclic GC is off during the build and the world is frozen after. | `coxswain/viz/worldmesh.py` `building_walls`; `scripts/fpv.py` upload loop | — |
| **The release tag is in the build**, so the diagnostics file names it. | `tools/build_exe.py` → `packaging/VERSION`; `release.yml` | — |

---

## Done

**Read this before touching anything a summary, a hook, or a memory
says is missing.** Everything below is in the code. Each row says
where it lives and what stops it silently going away. If a request
matches a row here, the answer is a pointer to the row, not a second
implementation — a second splash system, a second wind bed, a second
sky-noise term would *break* the constraints the first one was built
to.

### Scene and audio

| what | where | pinned by |
|---|---|---|
| **Shadows cast by everything, onto everything.** `static` and `shadow_casters` are built in the same loop over every mesh part, so buildings and trees shadow each other and themselves — not just the ground. 4096² map over 3965 m, 1.0 m a texel, baked once because the sun does not cross the sky during a 5 km race. | `scripts/fpv.py` `shadow_casters` | `test_everything_in_the_world_casts_a_shadow` |
| **Shadows received on the surface's own normal.** The world shader passes the fragment normal, because the lookup is pushed a texel along it to clear its own cell. A fixed up-vector would offset every wall as though it were flat ground. | `scripts/fpv.py` `sun_visibility(v_world, n, …)` | `test_the_world_receives_shadows_on_its_own_normals` |
| **Bow disturbance and the hull breaking the water.** `waterline_foam` paints white where the hull meets the surface, tied to the hull frame and to speed, so a stopped boat has none. | `scripts/fpv.py` `waterline_foam` | `test_the_bow_foam_is_tied_to_the_hull_and_the_speed` |
| **Catch splash at blade entry.** A small pooled particle burst, deliberately not dramatic. Off at standard graphics, on at high; `--no-particles` forces it off for A/B. | `scripts/fpv.py` `SplashSystem` | `test_the_splash_pool_exists_and_empties` |
| **Splash off a blade dragging on the recovery.** Every frame, each oar on its own seat clock asks `BladeContact.immersion` at the hull's roll — the same immersion the skim drag uses — and a blade riding below the surface trickles three drops every 80 ms. A clear blade throws nothing; a blade on the drive is left alone. | `scripts/fpv.py` `DRAG_SPLASH_*` | `test_a_dragging_blade_trickles_and_a_catch_bursts`, `test_the_drag_splash_is_gated_on_the_physics_not_on_the_picture` |
| **Ambient world audio** — wind gusting on a slow envelope and brightening with wind speed, water working at the bank, the odd gull. About a fifth of the stroke's level so it never competes with the boat; follows the wind setting live. | `coxswain/viz/ambient.py` | — |
| **Low-frequency variation in the grey sky.** Two octaves of simplex over the exponential gradient, amplitude 0.045, gated on overcast, so a clear day takes almost none and the gradient still dominates. | `scripts/fpv.py` sky shader | `test_the_sky_noise_stays_under_the_gradient` (≤ 0.08) |
| **Diagonal line on the water** — the detailed patch met the far plane at a hard edge. Fixed. | `763e57b` | — |
| **Charles arch bridges: piers under the arches.** Three faults: solid piers drawn at even fractions while the openings followed the measured stations (Western Avenue 17.3 m out); Weeks skipped by a gate on an NBI length a footbridge does not have (13.4 m out, 24 m of arch on dry land); the structure centred on the deck line while the comment said channel (Western 26 m up the Boston bank). Piers now stand at the edges the arches are cut between; every arch bridge is trimmed to the water. | `coxswain/viz/worldmesh.py` `arch_bridge`, `bridge_solids` | `test_every_arch_bridge_has_its_piers_where_navigation_has_them`, `test_no_arch_bridge_stands_on_dry_land_past_its_abutments`, `test_arch_piers_stand_under_the_arch_springings` |
| **Grand Junction trestle** drawn from the surveyed piers, not at thirds. | `truss_bridge` | `test_the_surveyed_piers_are_where_the_survey_puts_them` |

### The crew, as drawn

| what | where | pinned by |
|---|---|---|
| **Per-seat timing in the picture.** Every rower and oar is posed at its own `t − phase·period`, the same offset the physics applies. | `coxswain/viz/worldmesh.py` `oar_pose`, `crew_solids` | `tests/unit/test_viz.py` timing tests |
| **Blade enters and leaves over a window**, not in one frame (was a 190 mm step). | `BLADE_ENTRY`, `BLADE_EXIT`, `_smoothstep` | `test_viz` |
| **Blades square on the drive, feathered on the recovery.** | `oar_pose` | `test_viz` |
| **Bucket rig drawn as a bucket.** S-P-P-S from a starboard stroke: riggers, handles, blades *and* the direction each trunk winds all flip with the side. Measured on the HOCR four. | `coxswain/boats/rig.py`, `crew_solids` | `test_the_plan_puts_riggers_on_the_side_the_rower_rows` |
| **Each rower their own size.** Link lengths from their own de Leva segments (already); girth now ∝ √(mass/stature), so a 70 kg bow is no longer drawn as slight as a 54 kg stroke. Default rower returns exactly 1.0. | `_build_factor` | `test_a_heavier_rower_is_drawn_heavier`, `test_the_drawn_crew_are_not_all_the_same_size` |
| **Riggers on every shell.** Three struts from the gunwale to the pin per oarlock, on the oarlock's own side, read off the hull outline at that station; merged into the hull mesh so they ride with it. There were none before -- the loom pivoted in mid-air. | `coxswain/viz/worldmesh.py` `rigger_solids` | `tests/test_riggers.py` |
| **Body parts where the physics puts them.** Hands on the handle, elbows split the lift, trunk rotation from the kinematics — verified joint by joint. | `crew_solids` | `test_viz` |

### Physics wired into the trainer

| what | where | pinned by |
|---|---|---|
| **Michell wave resistance** in the physics, tabulated once and interpolated. | `coxswain/boats/boat.py` `USE_MICHELL` | Holt calibration tests |
| **Wind as the excess over still air** — no double count. 12.0 % vs the 12.2 % measured. | `coxswain/hydro/wind.py` `excess_loads` | `0111af8` tests |
| **Blade contact**: skim drag, righting moment, lost sweep, early/late catch bounded by `square_up_fraction`. | `coxswain/crew/blade_contact.py` | `0240616` tests |
| **Crew variability** (skill slider), **fatigue reserve** (CP / W′) with coxswain power calls. | `coxswain/crew/variability.py`, `exertion.py` | `dc93060` tests |
| **Balance by experience** (ILC trim, phase authority), handle heights and lean matching. | `coxswain/sim/control.py` `balance_for_experience` | `5dffacb` tests |
| **Coupled crew timing** — the hull and the crew's clocks close the loop. | `coxswain/crew/coupled.py` | `2be526d` tests |
| **Recovery from a perturbation** measured in strokes. | `scripts/recovery.py` | — |

### The boat, the crew, the menu

| what | where | pinned by |
|---|---|---|
| **The HOCR four**, rower by rower, at its own weight and power; report and time predictions. | `scripts/my_crew.py`, `scripts/hocr_crew.py` | — |
| **Rig editor**: top-down plan, riggers on the rowing side, stat boxes linked to each seat, presets, shell and rig change, switch side. | `coxswain/viz/rigview.py`, `fpv.run_setup_menu` | `tests/test_rigview.py` |
| **Text entry** — type a rower in, field by field; bad input refused, not zeroed; a typed height is a measurement. | `rigview.FIELDS`, `parse_field`, `Lineup.commit_edit` | `test_a_rower_can_be_typed_in_from_the_menu` |
| **Anonymous squad roster** — 82 rowers, 305 pieces, names removed, weight to 5 lb, time to the second, order shuffled; the sheet itself never committed. Squad presets built from it. | `scripts/build_squad_roster.py`, `data/squad_roster.csv`, `coxswain/crew/roster.py` | `tests/test_roster.py` (no-identities test) |
| **Calibration against a real result** — 2024 Sammamish W Vet 4+ (12th, 22:07.9). | `97caa8f`, `4fe7e7e` | — |

### Distribution

| what | where |
|---|---|
| **Windows, macOS (arm64) and Linux builds** on a three-platform matrix, ad-hoc codesigned `.app`, glibc-compatible Linux, `gh` upload with retry. v0.7 live on all three. | `.github/workflows/release.yml` |
| Release notes taken from the repository. | `packaging/RELEASE_NOTES.md` |

---

## Fixed

| what it was | how it was found |
|---|---|
| **Diagonal line on the water.** The 110 m detailed water patch met the flat far plane at a hard edge; world-axis-aligned and following the boat, so it read as a moving diagonal. | Appeared in an overhead shot taken for something else, after three failed attempts to reproduce it from the seat — where the edge is past the horizon. |
| **The drawn crew ignored the timing.** Physics offset every seat by `phases[i] * period`; the renderer posed every rower and oar at the same `t`, so the crew was drawn in perfect time whatever the model did. | Reported as "it didn't really look like it". |
| **Seven models written, tested, and never called.** `CrewVariability`, both halves of `BladeContact`, `PhaseAuthority`, `StrokeTrim`, `CoupledCrew`, the Michell table, the wind model — each reachable behind a default of `None` or an unset attribute. | Asked what was actually wired, rather than what existed. |
| **The catalog rowed at ~470 W/rower**, above world class, sustainable for 68 s. Nothing noticed because nothing related the force *scale* to watts. | Tracking the reserve made it immediate. |
| **Wind would have double-counted.** The flat wave coefficient was absorbing the unmodelled still-air drag. Fixed by applying only the *excess* over still air. | Adding the full aero load put an eight 9% above its own calibration reference. |
| **The blade teleported in and out of the water.** Depth was taken straight from the drive flag, so it stepped 190 mm at the catch and again at the finish, inside one frame. | Asked to confirm oar heights through both phases. |
| **Blades were a quarter-turn out of phase**, flat through the drive and on edge through the recovery. | Looking at it. |
| **Windows CI job hung for 52 minutes**, silently, blocking the whole release. | The macOS job did the same work in 5. |
| **`macos-13` runner retired**, so the Mac job queued forever. | Asking why it was slower than the Linux build had been. |
| **Every port rower wore starboard arms.** The kinematics signature that shares one chain across identical rowers omitted the side, so a matched crew was one group led by the stroke seat. 0.15 m on eight arm masses; a yaw bias. | Tabulating the crew made the exact path's error measurable against a per-seat table. |
| **Startup spent 14 s in a pure-Python point-in-polygon.** | The first profile, sorted by internal time. |
| **The physics ran at 100 Hz for no reason.** 60 Hz is the same boat to 0.2 mm over 24 s. | Counting derivative evaluations per frame. |
