# The trainer

A coxswain's seat in a rowing shell, on a real course, driven by the same
physics the rest of this project uses for its analysis. The point is
feedback: it is meant to be handed to people who cox, so they can say
what is wrong with it.

## Running it

Double-click `Coxswain.exe` — it needs nothing installed. Or from a
checkout:

```bash
python scripts/fpv.py
```

Either way you get a menu: pick a boat, pick a course, set the rate and
the wind, and push off. To skip the menu and go straight in:

```bash
python scripts/fpv.py --race hotl --boat 4+ --rate 32 --no-menu
```

## Steering

A Hudson four is steered by a toggle on the rudder cables: **push it to
port and the bow goes to port**, and it stays where you put it. So the
rudder here is positional — it holds its angle until you move it, and
there is no spring back to centre.

| | |
|---|---|
| mouse | the stick, with `--control mouse`: left is port |
| left / right, A / D | the stick, by key |
| **M** | hand the stick between mouse and keys |
| **C** | centre the stick |
| W / E | pressure split — "more port", "more starboard" |
| **Escape** | pause menu |
| Space | freeze |
| R | back to the start line |
| Q | quit |

**You have to steer it.** A sweep four does not run straight with the
rudder centred: the staggered oarlocks make a couple that settles at
about −1.7 deg/s to starboard (SOURCES sec. 60). Nothing cancels that
for you, because cancelling it would be teaching a boat that does not
exist. Finding and holding the trim is the skill.

## The pause menu

Escape. Rate and wind change live — rate rebuilds the crew, so the
stroke phase jumps once, which is a real transient and not hidden.
Changing boat or course rebuilds the world, so those send you back to
the setup menu.

## What you are looking at

Everything in the frame is measured, not drawn from imagination:

- **the water** is USGS elevation and OpenStreetMap shoreline, with
  depth from NOAA charts and USACE multibeam surveys;
- **the buildings** are lidar-measured heights on OSM footprints, with
  `building:part` massing where it exists — which is what makes the
  Space Needle a needle;
- **the chop** comes from the JONSWAP relations at the wind and fetch
  you set, so it is the same sea the conditions analysis reports;
- **the wake** is the free-surface Green's function — the same
  thin-ship theory Michell's wave resistance comes from — so the
  19.47° wedge and the `2πU²/g` wavelength are results, not settings;
- **the sound** is measured off real recordings by phase-locked DMD,
  pooled over 14 outings for a four and 3 for an eight.

## What is not right yet

- The **third stroke-audio event** is unstable across recordings
  (phases 0.50 to 0.98) and should be treated as a placeholder.
- The **wind shelter** around the hull — the calm to leeward, the pile
  to windward — is a plausible model with fitted constants, not a
  derivation like the wake.
- Only about **1% of buildings** have real massing; the rest are a
  footprint swept to one height, because that is all the data holds.
- The boat's speed peaks at phase 0.61 of the stroke where the
  literature puts it later, and there is a second peak at 0.84 that may
  be an artefact.

## Building the executable

```bash
python tools/build_exe.py --onedir     # a folder; starts fast
python tools/build_exe.py              # one file; slower to start
```

It refuses to build if any of the 36 MB of scenery and course data is
missing, rather than producing an executable that fails on somebody
else's machine.
