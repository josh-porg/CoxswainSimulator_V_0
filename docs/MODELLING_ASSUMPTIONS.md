# Modelling assumptions, by class

When one instance of a modelling assumption is found to matter, the whole model is searched for the same class (user,
2026-11-19: "when we find something like the rigid hands thing have we checked to see if the general class of modelling
assumption exists in other parts of the model ... we should be systemically doing that"). docs/STRUCTURAL_ISSUES.md does
this for code-bug classes; this file does it for physics assumptions. Each instance: where it is, what reality has
instead, the evidence on whether it matters, and its status. Priority: **H** could move a validation rung or a coxing
result; **M** could move a secondary quantity; **L** unlikely to matter at the model's resolution.

## A1. A prescribed (infinitely stiff) constraint where reality has a finite impedance

Found by: the post-release ringing (§240, §250) — the hands imposed as a rigid motion leave the oar's in-air mode damped
only structurally; a sourced grip (Dong et al. 2007) damps it 4–8× faster.

| instance | where | reality | evidence | priority | status |
|---|---|---|---|---|---|
| hands on the handle, recovery | `DynamicOarSimulator(crew="handle")`, the planned sweep | the grip is a spring-damper (Dong 2007) | ringing ×4–8 too persistent | M | **addressed**: `BendDynamics.grip_damping_ratio` 0.14 default (§252) |
| hands on the handle, drive (rung 1) | rung 1 imposes the hands | the rower's force–velocity yields | rung 2 replaces the imposed hands by muscles with F–V (Nordez 2025) | H | addressed in rung 2; rung 1 is a diagnostic by design |
| the body's segments relative to the hull | the clock crew / own-body field (`crew_field`) moves the body on a clock, not in response to hull acceleration | the body is a compliant, controlled multibody | the catch-approach push excess ([BR24] −0.196 m/s, §242); rung-2 trunk control studies (§179, `rung2_trunk.py`) | **H** | **open** — the multi-clock body (review §12) is the planned remedy |
| the arms in the straight-arm posture | a rigid link (arm clock, `BodyCoupledDriveSweep`) | the elbow held by the triceps' activation-scaled stiffness, the girdle compliant | catch spike 638 / 1242 N, knee activation 15 (§254); the default bent-arm body instead loads the arms ~0.22 s early | **H** | **open** — step 2 for the arms (MULTI_CLOCK_BODY_PLAN next 2) |
| the feet on the stretcher | rigid contact | shoe and heel compliance | no source found yet | L | open (search) |
| the oarlock / pin / gate | rigid pin, no play | gate play ~1–2 mm, pin compliance | no source found yet; series compliance would make the oar *softer*, the opposite of what the fits want (§245) | L | open (search) |
| the rigger | rigid | riggers deflect under 500–1000 N | in series with the shaft: softer, not stiffer; magnitude unknown | M | open — a source for rigger stiffness would bound it |
| the seat on its rails | rigid, Kato roll friction | wheel and rail compliance | friction sourced (Kato) | L | adequate |
| the hull | a rigid body | hull flexes slightly | negligible at stroke frequencies for a single | L | adequate |
| the blade face | rigid | the blade flexes and twists a little | no source | L | open |
| the inboard shaft | rigid (the bend is outboard of the pin only) | the inboard bends under the handle force (three-point bending about the pin) | with the angle sensor 0.40 m inboard, the real oar is ×0.84–0.94 as stiff on a measured angle (H38, §258); inboard EI unmeasured | M | open — needs an inboard EI; it would deepen the early-loading deficit, not close it |
| the shaft, vertical bending and torsion | only horizontal bending | the shaft also bends vertically and twists | the vertical load is small; torsion sets the blade's pitch | L–M | open |

## A2. A measured quantity compared with a model quantity defined differently (STRUCTURAL_ISSUES S12)

| instance | status |
|---|---|
| IVV (max−min)/mean against SD/mean | fixed (§233: both reported) |
| handle force on BioRow's lever (0.06 m from the top) | fixed (§233) |
| blade wetting on the gate's travel, not the blade's | fixed (§228) |
| the catch: the model's first wet sample against the data's turning point | handled in the speed-trace overlay (§241) |
| **the oar angle: where the sensor reads the shaft** | **answered (§258)**: clamped ~0.40 m inboard of the pin (Kleshnev); the outboard-capture hypothesis H33 refuted |
| the handle force before the catch and after the finish | **answered (§258)**: the shaft's force sensor turns with the feathering, so the recorded force there is not the horizontal force (Kleshnev); any use of it (§238's in-air inertia check) is void |
| **the "hip" angle: the model's trunk–thigh angle (to 180° at the catch) against anatomical hip flexion in the strength and passive data** | **found (§267) and handled (§268)**: mapped onto the anatomical hip from Telfer et al. 2023's open data (41→121°); with the passive hip the trunk muscles are feasible. Still lumped in the body model itself (one rigid trunk) |
| the stroke-averaged measured cycle against one model stroke | examined: ~1 % of peaks (§234) |

## A3. A measured input applied to part of the model and not the rest (STRUCTURAL_ISSUES S16)

| instance | status |
|---|---|
| [BR24]'s trunk travel scaled, legs not | fixed (§240) |
| the arms follow the model's hand law, not the athlete's measured hands | open — a deliberate rung design; recorded (§242: not the excess) |
| [CR06]'s boat mass: Table 1's 19.7 kg against the paper's 15.8 kg rigged boat | checked: 19.7 = 15.8 rigged + 3.9 kg of telemetry (`validate_blade.CR06_HULL`; SOURCES line ~210) — consistent |
| the implied-impulse check's system mass leaves out the oars | recorded (§242); asked — Kleshnev did not measure them (§258); Concept2 asked (#18) |
| [BR24]'s blade: the catalogue's generic 0.083 m², though his blade type is now known (Smoothie2) | **open (§259, H39)**: Concept2's outline and photographs give 0.071–0.077 m²; ladder run pending |
| [CR06]'s blade: the 2006 paper's 0.0903 m², against 0.068–0.074 m² for the Big Blade by Concept2's photographs | **conflict recorded (§259, H40)**; not resolved by fitting |

## A4. A synthetic reference built from mixed sources

| instance | status |
|---|---|
| the population stroke: [CR06]'s rig, [K05]'s body, Kleshnev's hands, a fixed rate | recorded (§240, §244); its rigs' arc split matters for the rigid oar (§245) |

## How to use this file

When a new modelling assumption is found to matter, add its class (or a row to an existing class), search the model for
the same pattern (`grep` the constraint, the input, the comparison), and give every instance a status and a priority.
Close an instance only with a test or a source.
