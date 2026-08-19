# Coxswain

A validated 6-DOF dynamics model of a rowing shell.

The dynamics follow Formaggia, Miglio, Mola & Montano, *A model for the
dynamics of rowing boats* (Int. J. Numer. Meth. Fluids **61**, 2009,
119–143), generalised from the paper's symmetry-plane (surge/heave/pitch)
formulation to all six degrees of freedom. Rower kinematics are driven by
published motion capture rather than invented. Every empirical number is
traced in **[docs/SOURCES.md](docs/SOURCES.md)**, which also records what
each source does *not* settle.

Long-term aim: trajectory optimisation over the Charles, using real depth
and centreline data — which is why the shallow-water term matters.

## Quick start

```bash
pip install numpy scipy matplotlib pyvista
```

```python
from coxswain.boats import catalog
from coxswain.sim.simulator import RowingSimulator

boat = catalog.eight(rate=32.0)          # or "4+", "1x"
result = RowingSimulator(boat).run(duration=20.0, surge_speed=4.8, dt=0.006)
print(result.summary())
```

Diagnostics, 2-D and 3-D, in one command:

```bash
python -m coxswain.viz --boat 8+ --rate 32 --show-3d
```

## Layout

| package | what it holds |
|---|---|
| `coxswain/core` | frames, state, rigid-body dynamics, integrators, 2-jet autodiff |
| `coxswain/crew` | anthropometry, stroke timing, rower kinematics, oar forces |
| `coxswain/hydro` | hull mesh and hydrostatics, resistance, shallow water, appendages |
| `coxswain/boats` | hull offsets, rigs, and the boat catalog |
| `coxswain/sim` | the simulator, steering, results |
| `coxswain/viz` | 2-D diagnostic charts and the 3-D scene |
| `coxswain/river` | placeholder for Charles course data |
| `legacy/` | the original 3-DOF and 6-DOF scripts, kept for reference |

Swapping boats is the point of the `boats` package: an eight and a coxed
four differ only in the `Boat` handed to the simulator, and hulls differ by
their offsets table rather than by a change of coefficient.

## Conventions that bite

Two frames, following the paper's §3:

- **absolute** — `X` along the course, `Z` up, `Y = Z × X`;
- **hull** — origin at the *hull* centre of mass `G_h` (not the combined
  centre of mass, which moves), `x` stern→bow, `z` up, `y` to port.

Every rotation names its direction (`hull_to_abs`, `abs_to_hull`); there is
deliberately no function called `rotation_matrix`. Attitude is stored
`[roll, pitch, yaw]` and the index constants `ROLL`, `PITCH`, `YAW` should
always be used — the legacy code disagreed with itself about whether index
0 meant roll or yaw, which silently swapped two of the three rate
equations.

The dynamics are written in the **absolute** frame, as the paper's are.
Hydrodynamic forces are naturally hull-frame and are rotated once on the
way out.

## Tests

```bash
python -m pytest              # ~9 minutes
python -m pytest tests/unit   # ~15 seconds
```

- `tests/unit` — one module per source module, pure and fast
- `tests/integration` — the assembled dynamics: conservation, coupling,
  control, and the hands-on-oar constraint
- `tests/regression` — values pinned against the source papers

Regression tests carry the citation for the number they check, so a
failure tells you which measurement you have contradicted.

## Validation status

Reproduced, against data the model was not fitted to:

| quantity | model | measured |
|---|---|---|
| drive duration at 31.5 spm | 747 ms | 752 ms (Hill & Fahrig 2009) |
| seat travel | 0.63 m | 0.60–0.70 m |
| critical speed in 3.0 m of water | 5.42 m/s | "around 5.4" (Day et al. 2011) |
| mean speed, 8+ at 32 spm | 5.3 m/s | 5.0–5.6 |
| pitch amplitude | <0.5° | within ±1.15° (Formaggia Fig. 13) |
| hands on the oar handle | 3–5 mm mean | constraint, must hold |

Known gaps are listed in [docs/SOURCES.md §8](docs/SOURCES.md#8-open-questions).
The largest: boat speed fluctuation comes out around 1.7× the measured
value, traced to crew centre-of-mass velocity amplitude but not yet
explained.
