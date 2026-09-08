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

The physics changed a good deal, and most of it you feel rather than see.

- **The crew get tired.** They start at the pace that spends their
  reserve exactly at the finish, and **Up / Down** calls for more or
  less. They give what you ask until it is gone, and then they cannot,
  whatever you call.
- **Crew skill and balance** are sliders under Rowers, novice to ideal.
  A ragged crew rolls the boat, and a rolling boat drags blades: an
  unset eight loses about 13% of its speed without anybody catching a
  crab.
- **The boat has to be sat.** The crew can barely correct balance on the
  recovery -- the blades are the only thing to push against -- so it is
  something to hold rather than something that holds itself.
- **Wind actually does something**, and by the right amount: a 5 m/s
  headwind costs 12% of boat speed.
- **Wave drag from the hull's own shape** rather than a fixed number,
  which is measurably closer to instrumented race data.
- **The crew is drawn at the timing the model uses**, so you can see
  when they are not together.

Fixed: the diagonal line on the water, and the blade snapping in and out
at the catch.

## What is in it

- Three courses: Head of the Charles, Tail of the Lake, Head of the Lake
- Four shells: coxed four, eight, double, single
- A Hudson-style rudder stick that stays where you put it
- Stroke sound sampled from real practice recordings, per shell
- Weather from clear to river fog, with wind driving the chop and the
  ripple — at nothing, the water is glass
- Baked sun shadows, a wake from the Green's function, and the hull
  breaking waves at the bow

Setup menu on launch. **Escape** opens the menu; **F1** flies a free
camera. The README beside the program has the full controls.

## What I want back

What feels wrong. Especially the steering, the sense of speed, and where
the sound sits in the stroke.
