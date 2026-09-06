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

## The plan-view trainer

`scripts/trainer.py` is the first thing you can actually steer:

```bash
python scripts/trainer.py                 # Head of the Charles
python scripts/trainer.py --race hotl     # Head of the Lake
python scripts/trainer.py --profile       # per-phase frame budget on exit
```

**The steering is a stick, not a switch.** A Hudson four is steered by a
toggle on the rudder cables: push it to port and the bow goes to port,
push it to starboard and it goes to starboard, and it stays where you
put it. So the rudder is positional and does not spring back;
`--control mouse` puts it on the pointer, which is much closer to
holding a stick than tapping a key. C centres it, W/E is the pressure
split, Tab swaps heading-up for north-up, `-`/`=` zoom, R restarts,
space pauses.

**You have to steer it.** A sweep four does not run straight with the
rudder centred — the staggered oarlocks leave a standing couple worth
about **−1.7 °/s to starboard** (SOURCES §60 and §125). Nothing here
cancels that for you: a trainer that quietly straightened the boat would
teach a boat that does not exist. Finding and holding the trim is the
skill, and §125 records the awkward part — over some of the rudder range
there is no fixed angle that holds a course at all.

`coxswain/viz/planscene.py` holds the geometry and imports no window
library: world-frame metres in numpy arrays, layered back to front. The
pygame trainer and a later ModernGL renderer consume the same scene, so
swapping one for the other cannot change what is on the course.

### The frame budget, and what it cost to find

Measured on the i7-1355U at 1180x700 on Head of the Lake — 81,022
buildings, 88 water polygons, 660 docks:

| phase | before | after |
|---|---|---|
| draw | 17.23 ms | **4.72 ms** |
| physics (100 Hz) | 7.0 ms | 5.0 ms |
| HUD + flip | 0.17 ms | 0.17 ms |

Three guesses were wrong before a measurement was right, which is the
only reason this is written down:

1. *"Culling 81,022 boxes every frame must be the cost."* A uniform 250 m
   grid took the cull from 0.6 ms to 0.13 ms and the frame did not move.
2. *"Then it must be the buildings."* 95 of them a frame, 1.45 ms.
3. It was **one polygon**. `draw:water` was **14.0 ms for 1.3 polygons** —
   Lake Washington's shoreline is a single ring of 2,345 vertices, and a
   scanline fill sorts every edge on every row: 1.6 million edge tests a
   frame for one lake. Clipping it to the window first (Sutherland-
   Hodgman, `clip_polygon`) makes it a dozen vertices and the same
   picture: **14.0 ms to 1.7 ms**.

Then the same fix applied everywhere made things *worse* — clipping all
95 buildings turned a 1.5 ms layer into 6 ms, because the clip costs
more than it saves on a polygon that was already cheap. The
discriminator is vertex count, not whether it overflows: clip above 64
vertices, leave the rest to pygame.

### Known limits

The stroke rate is fixed for a session. Rate lives in the boat's stroke
timing and changing the period mid-stroke jumps the crew's phase, which
puts a step in the force; `--rate` sets it at the start.

The HUD's physics number is a smoothed estimate and reads high for the
first seconds because the Numba kernel compiles on its first call.
`--profile` prints the honest per-phase average on exit.

## The seat view

`scripts/fpv.py` is the coxswain's own view — 0.55 m off the water in the
bow of a four, looking forward over the foredeck.

```bash
python scripts/fpv.py --race charles --control mouse
python scripts/fpv.py --shot out/fpv/seat.png --frames 270 --autopilot
```

Same stick, same physics, same untrimmed yaw as the plan trainer. pygame
owns the window and the input; **moderngl** owns the drawing, against
OpenGL 3.3 — which is what the Intel UHD in this laptop reports, so that
is the target, not 4.x.

`coxswain/viz/worldmesh.py` builds the course as triangles and imports no
GL, the same way `planscene.py` imports no pygame. The world is static
and the boat moves, so the whole course goes into a handful of buffers at
load time — 582k triangles for the Charles, built in 2.7 s — and each
frame is one matrix and a few draw calls. `RiverScene` rebuilds meshes
per frame, which is right for a figure and hopeless at 60 Hz; this is a
second backend, not a replacement, and both read the same course data.

### Four things the pictures caught

1. **Fog was per-vertex.** The water is one quad whose four corners are
   all three kilometres away, so every fragment of it — including the
   water under the bow — interpolated to full haze and the river
   rendered as sky. Fog is computed per fragment from the fragment's own
   world position now.
2. **A racing line on the water is a wall.** In perspective you stand on
   the near end of it, so however narrow the ribbon it fills the bottom
   of the screen. It is marker posts every 30 m instead — which is also
   what a regatta actually sets, and leaves the water visible.
3. **The bow was missing**, so the horizon swung with nothing to swing
   against and the view felt like a camera on a stick. The hull is drawn
   from the same outline the plan trainer uses.
4. **Decking the whole shell put a surface under the eye**, 0.27 m below
   it, filling a third of the frame with the inside of the boat. A cox
   sits in a cockpit and looks *over* the foredeck, so the decking starts
   0.45 m ahead of the seat.

### Not verified

**The windowed path has never run.** This machine has no display: every
picture above was rendered through `moderngl.create_standalone_context()`
to an offscreen buffer, which does use the real GPU but does not touch
`pygame.display.set_mode(OPENGL)`, the swap chain, vsync or input.
Everything up to context creation is exercised; the interactive loop is
not. `tests/test_worldmesh.py` pins what can be pinned without a screen —
winding, the waterline cut, the marker spacing, and that the camera sits
where a coxswain's head is and rolls with the hull.

Known rough edges: the water is a flat colour, there is no texture on
anything, and building the Head of the Lake world takes 30 s against the
Charles' 3 because it walks 81,022 footprints in Python.

## What has not been built

Texture, water motion, and streaming. `RiverScene` (PyVista) stays as the
backend for stills and the report — it rebuilds meshes per frame, which is right for a figure
and wrong for 60 fps. The realtime backend is a separate adapter behind
the same scene description, so the two can coexist and either can be
swapped for the other.
