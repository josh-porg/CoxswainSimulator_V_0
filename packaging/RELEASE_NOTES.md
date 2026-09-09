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

Faster, on every machine, and it can now tell me how it ran on yours.

- **Half the physics again.** Below High the boat is stepped with
  Heun's method -- two evaluations a step instead of four -- which is
  the same boat to 3 cm over a 367 m piece.  The physics is now 3 ms a
  frame on a laptop CPU at Minimal.
- **The low tiers stop paying for what they do not draw.** No sky noise
  in the water's reflection, the water normal straight from the mesh,
  and the HUD is only redrawn when it changes.  On an Intel UHD:
  Ultra minimal 9.7 ms a frame (104 fps), Minimal 17.2 (58), Standard
  21.9 (46), High 29.3 (34) -- from 16.6 / 18.7 / 25.9 / 42.8 in
  v0.9, headless, physics included.
- **It cannot lock up on a slow machine.** A frame that falls far
  behind takes four physics steps and lets the rest go: the boat runs
  briefly slow instead of the program stopping.
- **The `.exe` is exactly as fast as the source** -- measured, same
  machine, same tier, within 0.3 ms.
- **"Ultra minimal" is the lowest tier**, below Minimal, and its label
  now says so.

### Performance reports (please turn this on)

Under **Graphics and sound -> Send performance reports**.  At the end
of a session it sends me frame times, your GPU's name, the tier and the
settings -- numbers and product names only, never your name, a file
path or an e-mail; the program refuses to send if it finds one.  It is
off until you switch it on, remembered once you do, and the diagnostics
log beside the program records exactly what was sent.

If you would rather not, the same summary is written next to the log
as `report-<date>.json`; paste it to me and it does the same job.

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
