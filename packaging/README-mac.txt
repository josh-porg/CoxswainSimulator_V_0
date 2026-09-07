COXSWAIN -- a rowing simulator from the coxswain's seat
=======================================================

Thanks for trying this. It is a coxswain's-eye view of a real course,
run on a real physics model, and what I want back is what feels wrong
about it.


HOW TO RUN IT -- ONE EXTRA CLICK THE FIRST TIME
-----------------------------------------------

RIGHT-CLICK on Coxswain.app and choose "Open". Then click "Open" again
in the box that appears.

That is it. Afterwards, double-clicking works normally.

The reason: macOS quarantines anything downloaded from the internet
unless it is signed by a developer registered with Apple, which costs
99 dollars a year and I have not paid it. So the first launch asks
whether you are sure. Right-click then Open is how you say yes; a plain
double-click only offers you Cancel.

The app IS signed, just not by anyone Apple knows, so the message you
get should say the developer "cannot be verified" -- which is true. If
you instead see the word "damaged", the download did not survive the
trip; delete it and download it again rather than fighting it.

If macOS still will not budge, open Terminal, type this with a space
after it, and do NOT press Return yet:

    xattr -dr com.apple.quarantine

then drag Coxswain.app onto the Terminal window -- that types its
location for you -- and press Return. Then open the app normally.

Keep Coxswain.app and this README wherever you like; the app is
self-contained. Nothing to install, no Python needed.


APPLE SILICON ONLY -- CHECK BEFORE YOU DOWNLOAD
------------------------------------------------

This needs a Mac with an Apple chip: an M1, M2, M3, M4 or later, which
means any Mac bought from late 2020 onwards.

It will NOT run on an Intel Mac. If yours is from 2019 or earlier, or
if the Apple menu > About This Mac says "Intel", this build cannot
start on it and there is nothing you can do at your end.

I would build an Intel version if I could. The free Mac build machines
that GitHub provides are all Apple chips now, and the Intel ones are a
paid tier. If you are on an Intel Mac and want to try it, tell me and
I will look at what it would take.


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
