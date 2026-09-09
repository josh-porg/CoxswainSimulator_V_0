COXSWAIN -- a rowing simulator from the coxswain's seat
=======================================================

Thanks for trying this. It is a coxswain's-eye view of a real course,
run on a real physics model, and what I want back is what feels wrong
about it.


HOW TO RUN IT
-------------

Double-click Coxswain.exe, in this folder.

Nothing to install. Keep the whole folder together -- the .exe needs the
_internal folder beside it, so do not drag the .exe out on its own.

Windows will probably warn you it is from an unknown publisher. That is
because I have not paid to sign it, not because anything is wrong.
Click "More info", then "Run anyway".

Windows only, I am afraid.


FIRST SCREEN
------------

Pick a boat, a course and a stroke rate. Arrow keys move and change,
Enter selects. Choose "Push off".

"Weather" on the same screen sets the sky -- clear, hazy, overcast or
fog -- and the wind, which sets the chop, the ripple on the water and
the shelter behind the hull. Fog takes the far bank out. "Graphics and
sound" trades water detail for frame rate and picks the stroke sound.

  Head of the Charles   4.8 km, Boston. Six bridges and the Weeks turn.
  Tail of the Lake      4.0 km, Lake Union, Seattle.
  Head of the Lake      4.8 km. Portage Bay, the Montlake Cut, the Big
                        Turn.

It takes about half a minute to build the scenery when you start. That
is a million triangles of real elevation, buildings and trees being put
together, and it happens once per run.

Where you sit depends on the boat, as it does on the water. In the four
you are bow-loaded: lying down in the bow, a hand's breadth off the
water, with the crew behind you. In the eight you are up in the stern,
sitting, looking down the boat at the crew.


STEERING
--------

The stick works like a Hudson's: push it toward the side you want to go
and IT STAYS WHERE YOU PUT IT. It does not spring back, and nothing
re-centres it for you. Holding a straight course is your job.

  mouse           the stick, if you switch to mouse steering with M
  left / right    the stick, by key
  A / D           the same
  M               hand the stick between the mouse and the keys
  C               centre the stick
  W / E           pressure split -- more work on one side than the other
  V               look over your shoulder

A double or a single has no rudder at all. In those you steer entirely
on pressure, with W and E, and you sit facing the stern like the rower
you are -- V looks over your shoulder to see where you are going.

A sweep four turns toward the stroke side with the rudder centred. That
is not a bug; it is the thing you spend a race correcting.


WHILE YOU ARE ROWING
--------------------

  Esc     the menu -- change stroke rate, the weather and wind, restart,
          read the controls, or go back and change boat or course.
          There is a reminder along the bottom of the screen.
  F1      a free camera, to fly around and look at the course: WASD
          moves, Q and E go down and up, shift is faster, F1 again
          puts you back in the boat.
  Space   freeze it
  R       restart the course
  Q       quit


WHAT I WOULD LIKE TO KNOW
-------------------------

Does it steer like a boat? That is the whole question. Specifically:

  - how the stick responds, and how long the boat takes to answer it
  - whether the sense of speed is right at a given rate
  - whether the stroke sound sits where it should in the cycle
  - anywhere the course or the scenery is plainly wrong

The sound is recorded from real practices, not synthesised, so if the
timing feels off to you it probably is.


CREDITS
-------

Menu music is "Romantic 03" by Mixkit, used under the Mixkit Free
License. Everything else -- the courses, the scenery, the physics and
the stroke sound -- is built from public survey data and from
recordings of real outings.

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

If it runs badly, please turn on Graphics and sound -> Send performance
reports in the menu: at the end of a session it sends the frame times,
your GPU's name and the tier -- numbers and product names only, never a
name, a path or an e-mail.  That is how it gets faster on your machine.
