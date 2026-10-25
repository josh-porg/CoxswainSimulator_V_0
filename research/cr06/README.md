# research/cr06 — one athlete, every input hers: [CR06]'s woman and phase 4.3

Cabrera, Ruina & Kleshnev (2006), *A simple 1+ dimensional model of rowing mimics observed forces and
motions*, Human Movement Science 25(2):192–220, publish one women's single-scull stroke (Fig. 3,
T = 1.94 s): boat velocity, handle force, oar angle, leg and back displacement. This folder holds every
study that uses that stroke as its only athlete-specific input. Phase 4.3 of the physics programme
(docs/PHYSICS_PROGRAMME.md, "Phase 4") builds the forward-dynamic rower on it rung by rung.

Run everything from the repository root with the research environment:

```
.venv/Scripts/python.exe research/cr06/<script>.py
```

Outputs go to `data/local/cr06_study/` (gitignored; backed up in the private data repository).
The environment that produced the numbers below is in `research/environment/`.

## Inputs, and where each comes from

| input | file | provenance |
|---|---|---|
| her stroke: boat velocity, handle force (both handles), oar angle, leg displacement x_B/F, back displacement x_S/B | `data/literature/cr06_fig3_measured.csv` | digitised from the open preprint's vector Fig. 3 by `extract_fig3.py` (axes calibrated on the printed ticks) |
| body: segment masses, lengths, centres of mass, radii of gyration | `coxswain/crew/anthropometry.py` (`DE_LEVA_FEMALE`) | de Leva (1996) Table 4, women's column |
| her stature | 1.758 m | the package rower's leg travel equals her 0.581 m at this stature (`female_segments_cr06.py`) |
| the package rower's posture (seat height, segment lengths, trunk link, mass placement) | `coxswain/crew/kinematics.py`, built in `her_stroke.her_rower()` | the simulator's own body, women's table |
| women's foot-force direction | `data/literature/buckeridge2015_foot_force.csv` | [BU15] Buckeridge, Bull & McGregor (2015), Table 4: RF = 0.81 ± 0.07 at maximum handle force, 20 elite women, ergometer |
| footplate centre of pressure | `data/literature/vieira2017_cop.csv` | [VI17] Vieira et al. (2017), Figs 3–4 (CC-BY), digitised by `extract_vieira2017.py` |

All rung scripts read the stroke through **`her_stroke.py`**, the single definition of the series, the
smoothing (periodic smoothing splines; weight 1e-5, band 3e-6 to 3e-5), the time grid (777 samples), the
catch landmark (minimum boat acceleration, after [LE26]) and the maximum-handle-force instant.

## The rungs

Status: **original** = the script that produced the result survives; **rebuilt** = the original was
lost (it lived in a session scratchpad, cleaned without warning) and was rebuilt on 2026-10-24 from the
record in docs/TRACKING.md, printing the recorded numbers beside its own.

| rung | question | script | status |
|---|---|---|---|
| 1 | the stretcher force, from her momentum balance | `stretcher_cr06.py` (via `her_stroke.Stroke.stretcher_force`) | rebuilt |
| 2 | the hip moment top-down from the handle (no foot data) | `hip_topdown_cr06.py` | rebuilt |
| 3 | her trunk driven forward by the hip torque, open loop | `trunk_forward_cr06.py` | rebuilt |
| 4 | the trunk under delayed feedback | `trunk_forward_cr06.py` | rebuilt |
| 5 | the controlled trunk inside the simulator, two-way coupled | (`cr06/coupled_trunk_cr06.py`, lost) | **not yet rebuilt**; its linear check is rebuilt in `trunk_forward_cr06.py` |
| 4.3 legs | knee and ankle with the sourced foot-force direction and centre of pressure; the hip from below | `knee_ankle_cr06.py` | original (2026-10-24) |

### Rebuilt against recorded

The record is docs/TRACKING.md, "Step 4.3, first rung" to "fifth rung" (2026-09-14). Agreement is
reported as found; a difference is explained, not tuned away.

| quantity | rebuilt | recorded | note |
|---|---|---|---|
| **Rung 1** stretcher at the catch landmark | 392–447 N | 408–448 N | smoothing band |
| her absolute acceleration there | +4.9 to +5.6 m/s² (boat −8.5, legs +12.0, 0.4×back +1.8) | +5.2 to +5.7 (−8.8, +12.0, +1.8) | |
| stretcher peak | 670–723 N | 675–726 N | |
| gate force at the landmark; its peak | 70 N; 813 N | 34–36 N; 792 N | **not reproduced**: the record's gate-force definition is not recoverable from it; this rebuild uses the oar's lever balance F_gate = F_handle (1 + 0.83/1.805). So the stretcher's lead over the gate (0.15 s here, 0.23 s recorded) is not confirmed |
| **Rung 2** trunk angle, catch / maximum handle force | 128.1° / 110.3° | 127° / 110° | her back channel on the rower's catch value, K = 0.771 recovered |
| hip extension, maximum handle force | 3.58 N·m/kg (3.52–3.66) | 3.45 | +4% |
| — its handle / gravity / inertia terms | 132 / 60 / 75 N·m | 125 / 70 / 64 N·m | handle lever: oarlock height 0.243 m vs 0.228 recorded |
| hip extension, catch | 3.38 N·m/kg (3.27–3.38) | 2.74 | +23%: this rebuild drives the body with her measured hull; the record used the model's, which brakes ~1.9 m/s² harder at the catch, lowering the inertia term by about that much |
| drive peak; recovery minimum | 3.64; −0.85 N·m/kg | 3.56; −0.80 | |
| **Rung 3** effective inertia | 9.12 kg·m² | 8.8–8.9 (slope 8.67–9.15) | the package's head placement (0.743 m) and arms at the shoulder |
| round trip (RK4, 1000 and 2000 steps) | < 0.0001° | 0.0028° | needs a smooth reference (below) |
| open loop, 0.1° start error after one stroke | 118° (×1181, e-folding 0.274 s) | 72° (×720, 0.29 s) | follows the inertia |
| **Rung 4** delay table (5 and 10 rad/s × 0–117 ms) | holds / edge / fails at the same cells; 10 rad/s 100 ms: 7.9 / 3.5°, 240 N·m | 8.0 / 6.3°, 235 N·m | row by row in the script's output |
| **Rung 5's linear check**: stability boundary | 130 ms (5 rad/s), 72 ms (10 rad/s) | 132 ms, 72 ms | growth rates match (e.g. +0.47 /s at 75 ms, recorded +0.45) |

**A finding from the rebuild (rung 3).** The round trip first failed (0.17–0.30°, not converging with the
step). The cause: her back channel was a cubic smoothing spline, whose third derivative jumps at each knot;
the open-loop trunk amplifies the integrator's error at those jumps about a thousandfold per stroke. A
15-harmonic Fourier series of the same channel (0.03 mm rms from the spline) has every derivative smooth
and gives a round trip below 10⁻⁴°. Any forward-dynamic use of a measured time-law here needs a reference
that is smooth beyond its second derivative.

### Assumptions each script makes (in its own docstring too)

- **Frictionless seat**, so the stretcher's horizontal force is the body's momentum balance (rung 1).
- **[CR06] eq. 5's centre of mass** (75 kg on the seat plus 0.4 × the back channel) for the stretcher;
  the segmental body lowers the catch value to ~310 N (TRACKING, rung 1). The moment scripts use the
  segmental body.
- **Rigid trunk link** hip-to-shoulder, 0.570 m; arms as one point mass at the shoulder (rung 2).
- **Handle force horizontal at the oarlock height** (0.32 m in the hull frame).
- **Foot-force direction held at its maximum-handle-force value through the drive** ([BU15] measures it
  only there); footplate at 42° from the horizontal; ankle joint 0.07 m above the sole and 0.05 m ahead of
  the heel.
- **One-way coupling** in rungs 3–4 (the hull does not feel the trunk); rung 5 removes it.

## The legs, and the hip two ways (`knee_ankle_cr06.py`, SOURCES §177)

The support moment (knee extensor + ankle plantarflexor) does not depend on the footplate point; the
split does, and the measured point settles it. The hip from below (feet, shanks, thighs) and the hip from
above (rung 2) are independent:

| instant | hip from below | hip from above (rebuilt, same hull) | recorded rung 2 |
|---|---|---|---|
| maximum handle force | 3.41–3.56 N·m/kg | 3.52–3.66 | 3.45 |
| catch | ~0.3 | 3.27–3.38 | 2.74 |

They meet at maximum handle force and part at the catch; no plausible foot-force direction closes the
catch (SOURCES §177). Open.

## Reproducing everything here

```
.venv/Scripts/python.exe research/cr06/extract_fig3.py          # her stroke, from the preprint (data/literature)
.venv/Scripts/python.exe research/cr06/extract_vieira2017.py    # the centre of pressure (needs the CC-BY figures in data/local/literature/vieira2017)
.venv/Scripts/python.exe research/cr06/stretcher_cr06.py         # rung 1
.venv/Scripts/python.exe research/cr06/hip_topdown_cr06.py       # rung 2
.venv/Scripts/python.exe research/cr06/trunk_forward_cr06.py     # rungs 3, 4 and rung 5's linear check (~10 min)
.venv/Scripts/python.exe research/cr06/knee_ankle_cr06.py        # the legs and the hip from below
```

Each prints its numbers next to the recorded ones and writes a JSON of them to `data/local/cr06_study/`.
