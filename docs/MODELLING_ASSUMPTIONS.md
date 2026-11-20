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
| the shaft, vertical bending and torsion | only horizontal bending | the shaft also bends vertically and twists | the vertical load is small; torsion sets the blade's pitch | L–M | open |

## A2. A measured quantity compared with a model quantity defined differently (STRUCTURAL_ISSUES S12)

| instance | status |
|---|---|
| IVV (max−min)/mean against SD/mean | fixed (§233: both reported) |
| handle force on BioRow's lever (0.06 m from the top) | fixed (§233) |
| blade wetting on the gate's travel, not the blade's | fixed (§228) |
| the catch: the model's first wet sample against the data's turning point | handled in the speed-trace overlay (§241) |
| **the oar angle: where the sensor reads the shaft** | **pending** (§246, §251; asked of the data's author) |
| the stroke-averaged measured cycle against one model stroke | examined: ~1 % of peaks (§234) |

## A3. A measured input applied to part of the model and not the rest (STRUCTURAL_ISSUES S16)

| instance | status |
|---|---|
| [BR24]'s trunk travel scaled, legs not | fixed (§240) |
| the arms follow the model's hand law, not the athlete's measured hands | open — a deliberate rung design; recorded (§242: not the excess) |
| [CR06]'s boat mass: Table 1's 19.7 kg against the paper's 15.8 kg rigged boat | checked: 19.7 = 15.8 rigged + 3.9 kg of telemetry (`validate_blade.CR06_HULL`; SOURCES line ~210) — consistent |
| the implied-impulse check's system mass leaves out the oars | recorded (§242); asked |

## A4. A synthetic reference built from mixed sources

| instance | status |
|---|---|
| the population stroke: [CR06]'s rig, [K05]'s body, Kleshnev's hands, a fixed rate | recorded (§240, §244); its rigs' arc split matters for the rigid oar (§245) |

## How to use this file

When a new modelling assumption is found to matter, add its class (or a row to an existing class), search the model for
the same pattern (`grep` the constraint, the input, the comparison), and give every instance a status and a priority.
Close an instance only with a test or a source.
