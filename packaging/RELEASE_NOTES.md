A coxswain's-eye view of a real course, on a real physics model.

Pick your download from **Assets** below. Nothing to install on any of
them, and no Python needed.

### Windows

`Coxswain-windows.zip` — unzip, then double-click `Coxswain.exe` inside
the folder. Windows will warn about an unknown publisher: click **More
info**, then **Run anyway**. Keep the folder together.

### Mac — Apple Silicon only

`Coxswain-mac.zip` — unzip, then **right-click `Coxswain.app` and choose
Open**, then Open again. Do not just double-click the first time: macOS
offers only Cancel. It is signed, but not by a developer Apple knows, so
it says the developer "cannot be verified". One extra click, once.

**Needs a Mac with an Apple chip (M1 or later, so 2020 onwards).** It
cannot run on an Intel Mac.

### Linux

`Coxswain-linux.tar.gz` — `tar -xzf Coxswain-linux.tar.gz`, then
`cd Coxswain && ./Coxswain`. Needs glibc 2.35 or newer: Ubuntu 22.04+,
Debian 12+, current Fedora, Arch and Mint.

## New in this one

This one is about running on the machine you have.

- **Four graphics tiers, and it picks one for you.** *Ultra minimal*
  is new and is built for an integrated GPU: flat water, no reflections,
  a single hard shadow, plain fog, no distant skyline, every tree a
  sprite, a shorter view. On the integrated Intel graphics this was
  built on, *Ultra* and *Minimal* hold about 60 frames a second and
  *Standard* about 40, where *Minimal* managed 24 before. *Minimal* is a
  step up; *Standard* and *High* are what they were, with the reflection
  search shortened at Standard. On first start the game reads your
  graphics card and starts on the tier it should; change it under
  Graphics and sound.
- **The physics costs half what it did**, at every tier, and it is the
  same boat: the crew's motion and the oar force are solved once per
  stroke and read back, held to the full solve to a fifth of a
  millimetre.
- **If your laptop has two graphics chips**, the game tells you when it
  has been given the slow one, and how to fix it. `--prefer-dedicated-gpu`
  asks Windows for you.
- **A diagnostics file.** Every run writes a small text log --
  machine, graphics card, tier, load times, frame times, anything that
  stalled and why, any error -- and prints where it put it at start-up.
  If it runs badly, send me that file. It stays on your machine
  unless you send it.
- The **Charles arch bridges** have their piers under their arches, on
  all four of them; the shells have **riggers**; a blade dragging on the
  recovery throws **spray**; the rig editor lets you **type a rower in**,
  and there are **anonymous squad boats** built from real erg scores to
  race against.

Fixed on the way: every port rower had been given starboard arms, which
was quietly steering every boat; a coxed four now covers the same water
0.16% faster and drifts half as far. And with that gone the model says
an eight at race pace sits itself -- which is a claim about the model,
noted as open, not a claim about your boat.

## What is in it

- Three courses: Head of the Charles, Tail of the Lake, Head of the Lake
- Four shells: coxed four, eight, double, single
- A Hudson-style rudder stick that stays where you put it
- Stroke sound sampled from real practice recordings, per shell
- Weather from clear to river fog, with wind driving the chop and the
  ripple — at nothing, the water is glass
- Baked sun shadows, a wake from the Green's function, and the hull
  breaking waves at the bow
- Four graphics tiers from integrated laptop to gaming desktop, chosen
  for you at first start; a diagnostics log for when it runs badly

Setup menu on launch. **Escape** opens the menu; **F1** flies a free
camera. The README beside the program has the full controls.

## What I want back

What feels wrong. Especially the steering, the sense of speed, and where
the sound sits in the stroke.
