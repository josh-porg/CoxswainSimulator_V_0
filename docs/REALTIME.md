# Driving the boat in real time

The interactive trainer and the batch studies run **the same physics**. A
trainer that disagreed with the analysis would teach the wrong boat, so
the two differ in exactly two places, and nowhere else:

| | studies | trainer |
|---|---|---|
| who owns the clock | `RowingSimulator.run()` | `FixedStepLoop` |
| where steering comes from | `PathFollower`, `PathMPC` | a hand on a tiller |

Everything below the seam — hull, crew, water, wind, appendages — is
untouched and unaware.

## The two seams

**Time.** `RowingSimulator.step(state, t, dt) -> state` advances one
fixed RK4 step and keeps nothing. `run()` integrates a whole trajectory
and hands back a `SimulationResult`, which is what the studies want and
is useless to a game loop. Both call
`coxswain.core.integrators.rk4_step` on the same `derivative`, so they
agree **to the bit** — `tests/test_stepwise.py` asserts a hand-stepped
trajectory equals the stored golden one element for element.

`step` is stateless on purpose. The loop keeps two states, the one it is
integrating and the one it is drawing, and interpolates between them; if
`step` wrote through its argument the drawn pose would be the new state
blended with itself, and the boat would judder.

**Steering.** `Coxswain` already took a `rudder_override`, a callable
`(t, state) -> float`, because that is how the path follower and the MPC
drive the boat. A person is another one of those. `LiveControl` is a
mailbox: the input device writes a `ControlInput`, the force path reads
it, neither knows about the other.

```python
live = LiveControl()
cox  = Coxswain(rudder_override=live.rudder, pressure_split=live.split)
sim  = RowingSimulator(boat, coxswain=cox)

loop = FixedStepLoop(sim, rate=100.0)
loop.start(sim.initial_state(surge_speed=4.0))

while running:
    live.set(read_gamepad())      # or read_keyboard(), or a replay
    loop.advance()                # physics at 100 Hz, however slow the frame
    renderer.draw(loop.pose())    # interpolated, so motion is smooth
```

## Why fixed steps

Integrating by the frame time makes the boat behave differently on a
fast machine than a slow one, and makes results irreproducible — which
would cost the thing the analysis is for. So: a fixed rate, an
accumulator, and an interpolated pose for the renderer.

`max_frame` clamps how much time one frame may hand the accumulator.
Without it a single stall — a garbage collection, a chunk load, the
laptop throttling — asks for a hundred catch-up steps, which takes
longer still, which asks for more. Dropping simulated time is the right
answer; falling further behind for ever is not.

## Recording is the command stream, not the trajectory

The state is a deterministic function of the initial condition and the
commands, so `Recording` stores the commands. A full race is a few
kilobytes, replays exactly, and replays *through the analysis*: a human
run becomes a `SimulationResult` like any other. That gives ghost
racing, coaching review, and a regression test of the physics with a
person in the loop, from one small class.

## Speed

The derivative, not the integrator, is the cost. Two rounds of work,
both measured A/B **inside one process** — absolute timings drift with
the thermal state of a 15 W laptop, so ratios are the number to trust.

### Round one: numpy overhead, bit-identical

| change | cost if removed |
|---|---|
| Fourier scalar path | +1.80 ms/step |
| `cross3` | +1.01 ms/step |
| scalar `clip` | +0.49 ms/step |

**1.63×.** All three are the same mistake in three places: numpy's
generic dispatch on a scalar or a 3-vector costs a microsecond to do
what the CPU does in nanoseconds. `np.cross` spent 0.53 s of a 4.6 s run
inside `moveaxis`; `np.clip` was called 98,024 times per second of
rowing. None of it changed the model, so the golden trajectory is
reproduced **exactly**.

### Round two: the compiled kernel

`HullMesh.submerged` — the wetted-surface sweep over an 880-panel mesh —
was 27% of a derivative evaluation on its own. It is not slow because of
arithmetic: it is fifteen numpy operations, each allocating an `(880,)`
or `(880, 4)` temporary and each walking the array again. Fused into one
Numba loop:

| | |
|---|---|
| `submerged` alone | 210 µs → **19.6 µs (10.7×)** |
| end to end | 3.74 → **3.14 ms/step (1.19×)** |

Amdahl, exactly as advertised: a tenfold win on a quarter of the runtime
buys about a fifth overall.

**Physics at 100 Hz now costs 31% of one core**, which is affordable for
a game loop on this laptop.

### Why the kernel is opt-in

Numba is not bit-identical to numpy and cannot be: `ndarray.sum` uses
pairwise summation, a loop accumulates left to right. The two agree to
round-off and differ in the last bits.

So the numpy path stays the default and stays exactly reproducible, and
`RowingSimulator(..., fast=True)` asks for the compiled one. Measured
over a 12 s trajectory the two agree to **1.8e-15 relative** and do not
drift; `tests/test_kernels.py` asserts that, per quantity, across heeled
and yawed poses, and end to end. Numba is an optional dependency —
without it the kernel is a plain-Python reference and everything still
runs.

### What is left, and what it would cost

The profile is now **flat**: no hotspot above 5%, and about 7,000 Python
calls per step spread across hundreds of small functions — `Jet2`
arithmetic in the crew kinematics, per-rower oarlock callables, frame
rotations. Targeted kernels have run out of targets.

Two options remain, in increasing order of cost and risk:

1. **Numba the whole force path.** Needs the RHS reduced to plain arrays
   — no dataclasses, no callables, no `Jet2`. That is the big refactor,
   and it would likely be worth another 5-10×.
2. **Tabulate the crew kinematics.** They are a function of *time alone*
   and exactly periodic, so one table per stroke plus interpolation would
   remove most of what is left — crew field and hand positions were 35%
   of the run before any of this. But it is an **approximation**, not a
   round-off difference, so it belongs behind a fidelity tier and needs
   its own error budget against the golden trajectory. Not started, and
   not something to switch on quietly.

## What has not been built

The renderer. `RiverScene` (PyVista) stays as the backend for stills and
the report — it rebuilds meshes per frame, which is right for a figure
and wrong for 60 fps. The realtime backend is a separate adapter behind
the same scene description, so the two can coexist and either can be
swapped for the other.
