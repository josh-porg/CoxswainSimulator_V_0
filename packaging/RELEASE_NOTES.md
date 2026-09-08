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

Mostly the boat itself, and the people in it.

- **Your boat, rower by rower.** Under **Rig and crew** there is a
  top-down plan of the shell with each seat's weight, height and 5k
  erg beside it, the riggers drawn on the side each rower rows, and a
  preset for the HOCR four -- bucket rigged, starboard stroke. Switch
  a seat's side, change shell or rig, or **type a new rower in**:
  Enter on a seat, Tab through the fields. Bad input is refused, not
  turned into zero.
- **Riggers on the shells.** The frame from the gunwale to the pin is
  drawn on every boat, on the oarlock's own side. On a bucket four the
  middle pair really do rig together now.
- **The crew are the size they are.** Each rower is built from their
  own height and weight, so a 70 kg bow is no longer drawn as slight as
  a 54 kg stroke, and the stroke is the height of the stroke.
- **Squad boats to race against**, built from a real masters squad's
  erg scores with the names removed: Women 60+ four, Women 50s four,
  open women's and men's eights. They pull what people that age and
  weight actually pull -- including the spread.
- **The Charles bridges are right.** Every arch bridge's piers now
  stand under its arches, at the measured stations, and Weeks -- which
  was drawn as three even arches across dry land -- sits on the water
  where it belongs. On the bridge you line an arch up on from four
  hundred metres out, that was up to 17 m of error.
- **A blade that drags throws water.** On an unset boat the low side's
  blades skim the recovery, and now you see it: a thread of spray off
  the blade whenever the physics says it is touching, and nothing when
  it is clear.

Fixed: rowers all drawn the same width; the Charles arch bridges; the
fix log itself, which now records what is done.

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
