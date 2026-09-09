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

Things people asked for, and one thing nobody was told about.

- **Your boat races.** The rig editor's lineup -- these rowers at these
  weights and heights, this rig, this coxswain, these ergs -- is now the
  boat on the water. It never was before: the editor drew a plan and the
  default crew rowed. Each erg sets that seat's share of the power; Up
  and Down still move the crew as a whole.
- **Save your own boats.** *Save as preset* in the rig editor, a name,
  done. They come back under *Load preset* after the built-ins, and
  they live in `presets.json` beside your settings.
- **Genevieve's Pink Ribbon** is a built-in preset, beside the HOCR four.
- **Q asks first.** Q and Quit both open a yes/no with the cursor on No.
- **A minimap** in the top-right corner: the course, the buoys, you.
  North up. Off under *Graphics and sound -> Minimap*.
- There is also something to type on the setup menu. It is five letters
  and it is what a coxswain calls for.

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
