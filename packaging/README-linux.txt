COXSWAIN -- a rowing simulator from the coxswain's seat
=======================================================

Thanks for trying this. It is a coxswain's-eye view of a real course,
run on a real physics model, and what I want back is what feels wrong
about it.


HOW TO RUN IT
-------------

Unpack it and run the binary:

    tar -xzf Coxswain-linux.tar.gz
    cd Coxswain
    ./Coxswain

Nothing to install, no Python needed. Keep the whole folder together --
the binary needs the _internal folder beside it.

If it will not start, you are probably missing the system OpenGL and
SDL libraries. On Debian or Ubuntu:

    sudo apt install libgl1 libegl1 libsdl2-2.0-0 libasound2

On Fedora:

    sudo dnf install mesa-libGL mesa-libEGL SDL2 alsa-lib

Built on Ubuntu 22.04, so it needs glibc 2.35 or newer. That covers
Ubuntu 22.04 and later, Debian 12 and later, and current Fedora, Arch
and Mint. Anything older will say something about GLIBC_2.35 not being
found, and there is no way around that short of a rebuild.

Wayland works through XWayland. If you get an SDL error about the video
driver, try:

    SDL_VIDEODRIVER=x11 ./Coxswain


FIRST SCREEN
------------

Pick a boat, a course, a stroke rate, and the weather.
Arrow keys move and change, Enter selects. Choose "Push off".

  Head of the Charles   4.8 km, Boston. Six bridges and the Weeks turn.
  Tail of the Lake      4.0 km, Lake Union, Seattle.
  Head of the Lake      4.8 km. Portage Bay, the Montlake Cut, the Big
                        Turn.

It takes about half a minute to build the scenery when you start. That
is a million triangles of real elevation, buildings and trees being put
together, and it happens once per run.


STEERING
--------

The rudder is a stick that STAYS WHERE YOU PUT IT, like a Hudson's.
Push it toward the side you want to turn to and it stays there until
you move it back. It is not sprung and it does not centre itself.

  mouse left/right    the stick (this is the default)
  A and D             the stick, if you prefer keys
  M                   switch between mouse and keys
  W and E             lighten and press the crew
  V                   look over your shoulder
  F1                  free camera, to fly around and look at the course
  Escape              menu -- weather, graphics, sound, restart
  Space               freeze


WHAT I WANT TO KNOW
-------------------

What feels wrong. Especially:

  - does it steer like a boat, and does the stick feel right
  - the sense of speed, and whether the run between strokes is right
  - where the sound sits in the stroke
  - anything on the course that is in the wrong place


KNOWN, AND NOT WORTH REPORTING
------------------------------

  - The crew rows a metronomic stroke and never catches a crab.
  - Buildings are extruded footprints, so they are the right shape in
    plan and flat-topped where a real roof is not.

IF IT RUNS BADLY
----------------
Open Graphics and sound from the menu and pick a lower tier; "Ultra
minimal" is built for laptops with integrated graphics.  The game
picks a tier on its own the first time, from your graphics card.

If your laptop has two graphics chips, the start-up text will say when
the game has been given the slow one, and how to change that.

Every run writes a small diagnostics file and prints where it went at
start-up (on Windows: %LOCALAPPDATA%\Coxswain\logs).  It has the
machine, the graphics card, the tier, load times, frame times, and any
error.  If something is wrong, send that file.  It stays on your
machine unless you send it.
