r"""Steer the boat yourself, from above.

    python scripts/trainer.py                    # Head of the Charles
    python scripts/trainer.py --race hotl        # Head of the Lake
    python scripts/trainer.py --race totl --rate 28

A plan-view trainer on the real courses, driven by the real physics.  The
6-DOF boat, the surveyed bed, the docks and the bridges are the ones the
studies use; the only things that change are **who owns the clock** ad
**where the steering comes from** (see ``docs/REALTIME.md``).

Why plan view first
-------------------
Most of what a head race demands is a plan-view skill: hold the line,
take the right arch, keep the buoys on the right side, do not let the
bend push you wide, and know how long the boat takes to answer the
rudder.  None of that needs a 3-D scene, and all of it needs the physics
to be right -- which it now is, at about 3 ms a step.  So this is the
trainer that can exist today, and the seat view can follow.

The steering is a stick, not a switch
-------------------------------------
A Hudson four is steered by a toggle on the rudder cables: **push it to
port and the bow goes to port, push it to starboard and it goes to
starboard**, and it stays where you put it.  So the rudder here is
positional -- it holds its angle until you move it, and there is no
spring back to centre.  ``--control mouse`` puts it on the mouse, which
is much closer to holding a stick than tapping a key is.

**You have to steer it.**  A sweep four does not run straight with the
rudder centred: the staggered oarlocks make a couple whose cycle mean is
-39 N m, which settles at about -1.7 deg/s to starboard (SOURCES sec. 60,
and the same effect gives the eight -0.97 deg/s).  Nothing here cancels
that for you, because cancelling it would be teaching a boat that does
not exist.  Finding and holding the trim is the skill.

Controls
--------
====================  ====================================================
mouse                 the stick, with ``--control mouse``: left is port
left / right, A / D   the stick, by key; it stays where you put it
C                     centre the stick
W / E                 pressure split -- "more port", "more starboard"
Tab                   heading-up or north-up
- / =                 zoom
R                     restart at the start line
Space                 pause
Escape                quit
====================  ====================================================

What it does not do yet
-----------------------
The stroke rate is fixed for a session.  Rate lives in the boat's stroke
timing, and changing the period mid-stroke jumps the crew's phase, which
puts a step in the force and is a real piece of work to do properly.
``--rate`` sets it at the start.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.sim.control import Coxswain                   # noqa: E402
from coxswain.sim.realtime import (ControlInput,            # noqa: E402
                                   FixedStepLoop, LiveControl)
from coxswain.sim.simulator import RowingSimulator          # noqa: E402
from coxswain.viz.planscene import (boat_outline,           # noqa: E402
                                    build_scene, clip_polygon, oar_lines)

#: Rudder travel, radians, and how fast a key moves the stick.
#:
#: **The stick holds its position.**  It used to spring back to centre,
#: which is what a game controller does and not what a rudder toggle
#: does -- and on a boat with a standing yaw bias it meant letting go
#: put you back into the turn.  A cox sets a trim and steers around it.
RUDDER_LIMIT = 0.20
RUDDER_RATE = 0.55
#: Fraction of the window width the mouse sweeps to cover full travel.
MOUSE_SPAN = 0.55
#: Pressure split travel and rates, in the same units as ControlInput.
SPLIT_LIMIT = 1.0
SPLIT_RATE = 1.8
SPLIT_RETURN = 1.4

INK = (232, 238, 242)
DIM = (128, 142, 152)
LINE = (255, 146, 72)
GHOST = (125, 143, 156)
HULL = (238, 240, 236)
OAR_AIR = (196, 200, 190)
OAR_WATER = (120, 200, 170)
BACKGROUND = (18, 24, 29)
#: Clip a filled polygon to the window only above this many vertices.
CLIP_ABOVE = 64


class Camera:
    """World metres to screen pixels, optionally heading-up."""

    def __init__(self, size, metres_across=260.0):
        self.width, self.height = size
        self.metres_across = float(metres_across)
        self.heading_up = True
        self.centre = np.zeros(2)
        self.heading = 0.0

    @property
    def scale(self) -> float:
        return self.width / self.metres_across

    def look_at(self, centre, heading):
        self.centre = np.asarray(centre, dtype=float)
        self.heading = float(heading)

    def rotation(self):
        # Heading-up puts the bow at the top of the screen, which is how
        # a coxswain thinks; north-up is for reading the course.
        angle = (np.pi / 2.0 - self.heading) if self.heading_up else 0.0
        cos, sin = np.cos(angle), np.sin(angle)
        return np.array([[cos, -sin], [sin, cos]])

    def to_screen(self, points):
        """``(n, 2)`` world metres to ``(n, 2)`` pixels."""
        local = (np.asarray(points, dtype=float) - self.centre) @ \
            self.rotation().T
        return np.column_stack([
            self.width * 0.5 + local[:, 0] * self.scale,
            self.height * 0.5 - local[:, 1] * self.scale])

    def view_box(self):
        """A generous world-frame box covering the screen, for culling."""
        reach = 0.5 * self.metres_across * \
            (1.0 + self.height / max(self.width, 1)) + 40.0
        return (self.centre[0] - reach, self.centre[1] - reach,
                self.centre[0] + reach, self.centre[1] + reach)


def nearest_station(course, point):
    """``(index, cross_track)`` of the course point nearest ``point``.

    Cross-track is signed: positive to port of the course direction, so
    the sign matches the side a buoy would be on.
    """
    gap = course - point
    index = int(np.argmin(np.einsum("ij,ij->i", gap, gap)))
    ahead = course[min(index + 1, len(course) - 1)] - \
        course[max(index - 1, 0)]
    norm = float(np.hypot(*ahead))
    if norm < 1e-9:
        return index, 0.0
    normal = np.array([-ahead[1], ahead[0]]) / norm
    return index, float(np.dot(point - course[index], normal))


def split_label(speed: float) -> str:
    """Metres per second as a 500 m split, which is what crews read."""
    if speed < 0.2:
        return "--:--"
    seconds = 500.0 / speed
    return "%d:%04.1f" % (int(seconds // 60), seconds % 60)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default="charles",
                        choices=("charles", "totl", "hotl"))
    parser.add_argument("--rate", type=float, default=30.0,
                        help="stroke rate, fixed for the session")
    parser.add_argument("--physics", type=float, default=100.0,
                        help="physics rate in hertz")
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--no-buildings", action="store_true")
    parser.add_argument("--frames", type=int, default=0,
                        help="run this many frames and exit; with "
                             "SDL_VIDEODRIVER=dummy this is the smoke test")
    parser.add_argument("--control", default="keys",
                        choices=("keys", "mouse"),
                        help="mouse puts the stick on the pointer, which is "
                             "closer to holding a Hudson's toggle")
    parser.add_argument("--profile", action="store_true",
                        help="print a per-phase frame budget on exit")
    parser.add_argument("--shot", default=None,
                        help="save the last frame here and exit")
    parser.add_argument("--autopilot", action="store_true",
                        help="steer down the course by proportional "
                             "cross-track feedback, for testing")
    args = parser.parse_args(argv)

    import pygame

    print("building %s ..." % args.race)
    scene = build_scene(args.race)
    if args.no_buildings and scene.layer("buildings") is not None:
        scene.layers = [l for l in scene.layers if l.name != "buildings"]
    print("   %s: %s" % (scene.name,
                         ", ".join("%s %d" % (l.name, len(l.polylines))
                                   for l in scene.layers)))

    boat = catalog.coxed_four(rate=args.rate, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    live = LiveControl()
    cox = Coxswain(rudder_override=live.rudder, pressure_split=live.split)
    simulator = RowingSimulator(boat, coxswain=cox, fast=True)
    hull = boat_outline(boat)

    def fresh_state():
        state = simulator.initial_state(surge_speed=3.6)
        state[0], state[1] = scene.start
        state[5] = scene.start_heading
        state[6] = 3.6 * np.cos(scene.start_heading)
        state[7] = 3.6 * np.sin(scene.start_heading)
        return state

    loop = FixedStepLoop(simulator, rate=args.physics)
    loop.start(fresh_state())

    pygame.init()
    pygame.display.set_caption("%s -- coxswain trainer" % scene.name)
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17)
    big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 26)
    camera = Camera((args.width, args.height))
    if args.control == "mouse":
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos((args.width // 2, args.height // 2))

    from collections import defaultdict
    cost = defaultdict(float)
    rudder = 0.0
    split = 0.0
    paused = False
    running = True
    frames = 0
    physics_ms = 0.0

    while running:
        frame = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    camera.heading_up = not camera.heading_up
                elif event.key == pygame.K_c:
                    rudder = 0.0
                elif event.key == pygame.K_r:
                    loop.start(fresh_state())
                    rudder = split = 0.0
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    camera.metres_across = min(camera.metres_across * 1.4,
                                               4000.0)
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    camera.metres_across = max(camera.metres_across / 1.4,
                                               60.0)

        if args.frames and frames >= args.frames:
            running = False

        keys = pygame.key.get_pressed()
        # The stick holds its position.  Push it to starboard and the bow
        # goes to starboard; let go and it stays there, because that is
        # what a rudder toggle does.
        if args.control == "mouse":
            offset = (pygame.mouse.get_pos()[0] - args.width * 0.5) \
                / (args.width * MOUSE_SPAN * 0.5)
            rudder = float(np.clip(offset, -1.0, 1.0)) * RUDDER_LIMIT
        else:
            turn = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - \
                   (keys[pygame.K_LEFT] or keys[pygame.K_a])
            if turn:
                rudder = float(np.clip(rudder + turn * RUDDER_RATE * frame,
                                       -RUDDER_LIMIT, RUDDER_LIMIT))
        press = keys[pygame.K_e] - keys[pygame.K_w]
        if press:
            split = float(np.clip(split + press * SPLIT_RATE * frame,
                                  -SPLIT_LIMIT, SPLIT_LIMIT))
        else:
            split -= np.sign(split) * min(abs(split), SPLIT_RETURN * frame)
        if args.autopilot and scene.course is not None:
            # Crude proportional steering, only so an unattended run
            # stays on the course long enough to be worth timing.
            index, cross = nearest_station(scene.course, loop.state[0:2])
            ahead = scene.course[min(index + 12, len(scene.course) - 1)]                 - loop.state[0:2]
            want = float(np.arctan2(ahead[1], ahead[0]))
            error = (want - float(loop.state[5]) + np.pi) % (2 * np.pi) - np.pi
            rudder = float(np.clip(-1.2 * error + 0.02 * cross,
                                   -RUDDER_LIMIT, RUDDER_LIMIT))
        live.set(ControlInput(rudder=rudder, pressure_split=split))

        if not paused:
            start = time.perf_counter()
            loop.advance(frame)
            physics_ms = 0.9 * physics_ms + 0.1 * 1000 * (time.perf_counter()
                                                          - start)
        cost["physics"] += physics_ms

        mark = time.perf_counter()
        pose = loop.pose()
        position, heading = pose[0:2], float(pose[5])
        camera.look_at(position, heading)
        box = camera.view_box()

        _m = time.perf_counter()
        screen.fill(BACKGROUND)
        cost["fill"] += 1000 * (time.perf_counter() - _m)
        for layer in scene.layers:
            _m = time.perf_counter()
            colour, width = layer.colour, max(int(layer.width), 1)
            found = layer.visible(*box, scale=camera.scale,
                                  centre=camera.centre)
            cost["cull:" + layer.name] += 1000 * (time.perf_counter() - _m)
            _m = time.perf_counter()
            cost["n:" + layer.name] += len(found)
            for index in found:
                pixels = camera.to_screen(layer.polylines[index])
                if len(pixels) < 2:
                    continue
                if layer.fill and len(pixels) >= 3:
                    # Clip the big rings only.  A scanline fill costs
                    # vertices times scanlines, so a 2,345-vertex
                    # shoreline is worth 12 ms and a ten-vertex building
                    # is free -- clipping everything turned a 1.5 ms
                    # layer into a 6 ms one.
                    if len(pixels) > CLIP_ABOVE:
                        pixels = clip_polygon(pixels, -2.0, -2.0,
                                              args.width + 2.0,
                                              args.height + 2.0)
                    if len(pixels) >= 3:
                        pygame.draw.polygon(screen, colour, pixels)
                else:
                    pygame.draw.lines(screen, colour, layer.closed,
                                      pixels, width)
            cost["draw:" + layer.name] += 1000 * (time.perf_counter() - _m)

        if scene.course is not None:
            pygame.draw.lines(screen, GHOST, False,
                              camera.to_screen(scene.course), 2)
        if scene.buoys is not None and len(scene.buoys):
            for keep_to_port, bx, by in scene.buoys:
                if not (box[0] <= bx <= box[2] and box[1] <= by <= box[3]):
                    continue
                spot = camera.to_screen(np.array([[bx, by]]))[0]
                pygame.draw.circle(
                    screen, (255, 146, 72) if keep_to_port else (255, 214, 10),
                    spot, max(3, int(0.9 * camera.scale)))

        # -- the boat ------------------------------------------------------
        cos, sin = np.cos(heading), np.sin(heading)
        rotate = np.array([[cos, -sin], [sin, cos]])
        pygame.draw.polygon(screen, HULL,
                            camera.to_screen(hull @ rotate.T + position))
        lines, drive = oar_lines(boat, loop.t)
        for oar in lines:
            pygame.draw.lines(screen, OAR_WATER if drive else OAR_AIR, False,
                              camera.to_screen(oar @ rotate.T + position),
                              max(2, int(0.18 * camera.scale)))

        cost["draw"] += 1000 * (time.perf_counter() - mark)
        mark = time.perf_counter()

        # -- the numbers ---------------------------------------------------
        speed = float(np.hypot(pose[6], pose[7]))
        readout = [
            ("%s" % split_label(speed), big, INK),
            ("%.2f m/s   rate %.0f" % (speed, boat.timing.rate), font, DIM),
        ]
        if scene.course is not None:
            index, cross = nearest_station(scene.course, position)
            station = float(np.hypot(*np.diff(scene.course[:index + 1],
                                              axis=0).T).sum()) \
                if index else 0.0
            side = "port" if cross > 0 else "starboard"
            readout.append(("%.0f m along" % station, font, DIM))
            readout.append(("%.1f m %s of the line" % (abs(cross), side),
                            font, LINE if abs(cross) > 15 else DIM))
        readout.append(("stick %+.0f%%  (%+.1f deg)   split %+.0f%%"
                        % (100 * rudder / RUDDER_LIMIT,
                           np.degrees(rudder), 100 * split),
                        font, DIM))
        readout.append(("yaw %+.2f deg/s" % np.degrees(pose[11]),
                        font, LINE if abs(np.degrees(pose[11])) > 1.0
                        else DIM))
        readout.append(("%.0f fps   physics %.1f ms/frame   %s"
                        % (clock.get_fps(), physics_ms,
                           "heading-up" if camera.heading_up else "north-up"),
                        font, DIM))
        if paused:
            readout.append(("PAUSED -- space to row", font, LINE))

        # The stick, drawn where a hand would be.  With a standing yaw
        # bias and no spring, seeing the trim you are holding is most of
        # what makes the boat steerable.
        bar_w = int(args.width * MOUSE_SPAN)
        bar_x = (args.width - bar_w) // 2
        bar_y = args.height - 42
        backing = pygame.Surface((bar_w + 130, 54), pygame.SRCALPHA)
        backing.fill((12, 17, 21, 190))
        screen.blit(backing, (bar_x - 62, bar_y - 20))
        pygame.draw.line(screen, (70, 82, 92), (bar_x, bar_y),
                         (bar_x + bar_w, bar_y), 3)
        pygame.draw.line(screen, (110, 124, 136),
                         (args.width // 2, bar_y - 9),
                         (args.width // 2, bar_y + 9), 2)
        knob = bar_x + int(bar_w * (0.5 + 0.5 * rudder / RUDDER_LIMIT))
        pygame.draw.circle(screen, LINE, (knob, bar_y), 9)
        pygame.draw.circle(screen, BACKGROUND, (knob, bar_y), 5)
        for label, at in (("port", bar_x - 52),
                          ("stbd", bar_x + bar_w + 14)):
            screen.blit(font.render(label, True, DIM), (at, bar_y - 9))

        y = 12
        for text, face, colour in readout:
            screen.blit(face.render(text, True, colour), (14, y))
            y += face.get_height() + 2

        cost["hud"] += 1000 * (time.perf_counter() - mark)
        mark = time.perf_counter()
        pygame.display.flip()
        cost["flip"] += 1000 * (time.perf_counter() - mark)
        frames += 1

    if args.shot:
        os.makedirs(os.path.dirname(args.shot) or ".", exist_ok=True)
        pygame.image.save(screen, args.shot)
        print("wrote %s" % args.shot)
    pygame.quit()
    print("%d frames, %d physics steps" % (frames, loop.steps))
    if frames and args.profile:
        print("  ms/frame, averaged:")
        for key in sorted(cost):
            value = cost[key] / frames
            print("  %-22s %8.2f%s" % (key, value,
                                       "" if not key.startswith("n:")
                                       else "  (count)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
