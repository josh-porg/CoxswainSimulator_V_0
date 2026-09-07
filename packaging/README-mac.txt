COXSWAIN -- a rowing simulator from the coxswain's seat
=======================================================

Thanks for trying this. It is a coxswain's-eye view of a real course,
run on a real physics model, and what I want back is what feels wrong
about it.


HOW TO RUN IT -- PLEASE READ, THE FIRST TIME IS AWKWARD
-------------------------------------------------------

RIGHT-CLICK on Coxswain.app and choose "Open". Then click "Open" again
in the box that appears.

Do NOT just double-click it the first time. macOS will say the app "is
damaged and can't be opened" or "cannot be opened because the developer
cannot be verified", and offer you only a Cancel button. The app is not
damaged. macOS quarantines anything downloaded from the internet that
has not been signed by a developer registered with Apple, and I have
not paid Apple for that.

Right-click then Open is the standard way past it, and you only have to
do it once. After that, double-clicking works normally.

If macOS still refuses, open Terminal, type this, and press Return:

    xattr -dr com.apple.quarantine

then drag Coxswain.app onto the Terminal window (that types its
location for you) and press Return again. Then open the app normally.

Keep Coxswain.app and this README together, or don't -- the app is
self-contained. Nothing to install, no Python needed.


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

  - It is not signed, hence the song and dance above.
  - The crew rows a metronomic stroke and never catches a crab.
  - Buildings are extruded footprints, so they are the right shape in
    plan and flat-topped where a real roof is not.
