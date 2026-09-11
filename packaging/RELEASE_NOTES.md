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

## Fixed in v0.12.1

**v0.12 crashed on startup — this replaces it.** If you downloaded
v0.12, take this one instead. The minimap looked up the boat's position
under a name the main loop does not use, so the trainer stopped before
its first frame. Three more of exactly that kind went with it: changing
the minimap, update-check or report setting on the setup menu crashed;
so did opening the pause menu.

Everything below is v0.12's, and all of it is in here.

## New in this one

- **Your boat really races now.** v0.12 said it did, and it did not:
  pick a bucket-rigged four and a standard four rowed. The editor's
  lineup never left the menu -- not the rig, not the weights, not the
  ergs. It does now, and a test drives the real menu into the editor,
  changes the rig, and checks the boat on the water is that rig.
- **Age, skill and experience for each rower**, in the rig editor.
  Leave them blank and that seat follows the crew sliders as before.
  - **Age** scales the anaerobic reserve -- the sprint left in the
    tank -- not the power: the erg score already is the power.
  - **The reserve is your crew's now**, from their own ergs. It used to
    be a literature athlete's: 303 W of critical power for every crew.
    The HOCR four's is about 126 W, so they can finally run out.
  - **Skill** per seat: a steadier rower scatters less than the crew.
  - **Experience** per seat: the crew's balance comes from the seats
    that have one.
- **Full screen.** *Graphics and sound -> Full screen*, or **F11**
  mid-race, or `--fullscreen`. If the screen refuses, you get a window
  and a note instead of a crash.
- **Fixed:** v0.12 crashed on start for everyone; v0.12.1 fixed that,
  and the setup and pause menus, which could also crash. Every build is
  now started from the actual download before it is released.

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
- A rig editor: your own crew, rig and coxswain, saved as presets
- A minimap, and Q asks before it quits

Setup menu on launch. **Escape** opens the menu; **F1** flies a free
camera. The README beside the program has the full controls.

## What I want back

What feels wrong. Especially the steering, the sense of speed, and where
the sound sits in the stroke.
