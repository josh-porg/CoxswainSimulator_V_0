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

Faster again, on the machines that were slow, and a way to find out
about the next one.

- **Every tier draws less.** The world is packed to 20 bytes a vertex
  and only the tiles in front of the camera are drawn; the sky is
  painted last, where nothing else was; the low tiers use a water grid
  a quarter the size, which the far water does not show and the
  measurement did. On an integrated Intel GPU: Minimal 17.4 -> 11.7 ms
  a frame (86 fps), Standard 22.2 -> 16.3 (61 fps).
- **The physics is 0.77 ms per evaluation** -- the per-oar loop is one
  array operation now -- so a slow CPU has that much more room.
- **It tells you when there is a newer release.** One line on the setup
  menu, with the link you were sent. Nothing is downloaded. Off in
  *Graphics and sound -> Check for updates*, or `--no-update-check`.
- **Windows signing is wired, waiting on an account.** When the SignPath
  open-source project exists the exe ships signed and SmartScreen stops
  warning; until then nothing changes. `docs/SIGNING.md` has the steps.

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
