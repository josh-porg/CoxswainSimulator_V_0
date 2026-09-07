r"""The coxswain's seat, in real time.

    python scripts/fpv.py                          # Head of the Charles
    python scripts/fpv.py --race hotl --control mouse
    python scripts/fpv.py --shot out/fpv/seat.png  # one frame, no window

0.55 m off the water in the bow of a four, looking forward over four
backs.  The **only** view from which "does this steer like a boat" is a
question with an answer: from up here the bank swinging across the bow is
the cue a coxswain actually uses, and no plan view reproduces it.

How it is put together
----------------------
The world is static and the boat moves, so the whole course is uploaded
once as a few hundred thousand triangles
(:mod:`coxswain.viz.worldmesh`) and each frame is one matrix and three
draw calls.  ``RiverScene`` rebuilds meshes per frame, which is right for
a figure and hopeless at 60 Hz -- this is a second backend, not a
replacement, and both read the same course data.

pygame owns the window and the input, moderngl owns the drawing.  That
keeps every control identical to ``scripts/trainer.py`` and means the GL
code is only the GL code.

Steering is the same stick as the plan trainer, and it is still **not**
trimmed for you: a sweep four turns to starboard with the rudder centred
(SOURCES sec. 60 and 125) and holding that off is the skill.

Controls
--------
====================  ====================================================
mouse                 the stick, with ``--control mouse``; left is port
left / right, A / D   the stick, by key; it stays where you put it
M                     hand the stick between mouse and keys
C                     centre the stick
W / E                 pressure split
R                     restart          Space  pause          Esc  quit
====================  ====================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.core.frames import hull_to_abs                # noqa: E402
from coxswain.sim.control import Coxswain                   # noqa: E402
from coxswain.sim.realtime import (ControlInput,            # noqa: E402
                                   FixedStepLoop, LiveControl)
from coxswain.sim.simulator import RowingSimulator          # noqa: E402
from coxswain.viz.planscene import oar_lines                # noqa: E402
from coxswain.viz.water import (KELVIN_HALF_ANGLE,          # noqa: E402
                                PuddleTrail, kelvin_wavelength, sea_for,
                                wake_table)
from coxswain.viz.worldmesh import _face_normals            # noqa: E402
from coxswain.viz.worldmesh import build_world, hull_solid  # noqa: E402

RUDDER_LIMIT = 0.20
RUDDER_RATE = 0.55
MOUSE_SPAN = 0.55
SPLIT_LIMIT, SPLIT_RATE, SPLIT_RETURN = 1.0, 1.8, 1.4

#: How far you can see before everything is sky.  Fog closes the gap so
#: the world does not end at a hard edge.
FAR = 2600.0
SKY = (0.52, 0.60, 0.68)

VERTEX_SHADER = """#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
out vec3 v_colour;
out vec3 v_normal;
out vec3 v_world;
uniform mat4 mvp;
void main() {
    v_colour = in_colour;
    v_normal = in_normal;
    // The world position, so the fragment can measure its OWN distance.
    // Interpolating a per-vertex distance across a big triangle is wrong
    // by construction: the water is one quad whose four corners are all
    // three kilometres away, so every fragment of it -- including the
    // water under the bow -- came out fully hazed and the river rendered
    // as sky.
    v_world = in_pos;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """#version 330
in vec3 v_colour;
in vec3 v_normal;
in vec3 v_world;
out vec4 f_colour;
uniform vec3 sun;
uniform vec3 sky;
uniform vec3 eye;
uniform float far;
void main() {
    float v_depth = length(v_world - eye);
    // A single directional light with a generous ambient: this is an
    // overcast New England morning, not a stage.
    float lambert = max(dot(normalize(v_normal), sun), 0.0);
    vec3 lit = v_colour * (0.55 + 0.45 * lambert);
    // Distance fog, which is what stops the far bank reading as near.
    float haze = clamp(v_depth / far, 0.0, 1.0);
    haze = haze * haze;
    f_colour = vec4(mix(lit, sky, haze), 1.0);
}
"""


#: The moving water patch: how far it reaches, and how many divisions
#: across it.
#:
#: The grid is **graded**, not uniform.  Wind chop on a river has a peak
#: wavelength of two or three metres, and a uniform grid fine enough to
#: resolve that -- eight samples a wave, so 0.3 m -- would need 640,000
#: triangles to reach a hundred metres.  The first attempt used 0.9 m
#: uniform, which is 2.5 samples a wave, and the water came out as flat
#: angular shards: that is aliasing, not chop.
#:
#: Spacing instead grows with the square of the distance from the boat,
#: which puts 0.06 m cells under the bow where a wave subtends a real
#: angle and metre cells at the edge where it does not.
WATER_REACH = 110.0
WATER_DIVISIONS = 300

WATER_VERTEX = """#version 330
in vec2 in_grid;
out vec3 v_world;
out vec3 v_normal;
out float v_foam;
uniform mat4 mvp;
uniform vec2 centre;          // the patch follows the boat
uniform vec4 waves[8];        // amplitude, wavenumber, direction, phase
uniform vec4 puddles[16];     // east, north, strength, unused
uniform vec3 boat;            // east, north, heading
uniform float speed;
uniform float wake_k;         // 2 pi g / V^2, the transverse wavenumber
uniform float wake_amp;
uniform float hull_length;    // bow-to-stern source separation
uniform float tan_wedge;      // tan(19.47 deg)
uniform float time;

const float G = 9.80665;

// The sea: a sum of components, each a solution of the linearised free
// surface, so their sum is one too.  This is the same field
// coxswain.viz.water evaluates on the CPU for the tests.
float sea(vec2 p, float t) {
    float h = 0.0;
    for (int i = 0; i < 8; ++i) {
        float a = waves[i].x;
        if (a <= 0.0) continue;
        float k = waves[i].y;
        float d = waves[i].z;
        float w = sqrt(G * k);
        h += a * cos(k * (p.x * cos(d) + p.y * sin(d)) - w * t + waves[i].w);
    }
    return h;
}

// One Kelvin system, from a disturbance at `offset` along the hull.
//
// Zero outside the 19.47-degree wedge, and a(x) = A/sqrt(x) inside it,
// which is what spreading the wave-resistance energy across the
// widening wedge gives.
float wake_from(vec2 p, float offset, float amp) {
    float c = cos(-boat.z), s = sin(-boat.z);
    vec2 d = p - boat.xy;
    float along = -(d.x * c - d.y * s) + offset;   // positive astern
    float across = d.x * s + d.y * c;
    if (along < 0.15) return 0.0;
    float limit = tan_wedge * along;
    if (abs(across) > limit) return 0.0;
    float fade = exp(-along / 120.0) / sqrt(max(along, 0.4));
    float edge = abs(across) / max(limit, 1e-6);
    // Riding up toward the cusp lines is where a real wake is steepest,
    // and it is what makes the divergent V read from the stem.
    float crest = 0.35 + 0.65 * edge * edge * edge;
    return amp * fade * crest * cos(wake_k * along);
}

// The hull as TWO disturbances, not one.
//
// A single system centred on the boat has no V leaving the stem: the
// wedge just begins under the boat, which is not what a coxswain sees.
// The bow throws a pair of crests that run slightly wider than the hull
// and open out behind it.  Havelock models a ship as a pressure source
// at the bow and a sink at the stern, a waterline length apart, and
// superposing their two Kelvin systems gives both that bow V and the
// bow-stern interference -- the same interference that puts the humps
// in a wave-resistance curve.  The stern system is weaker and opposite
// in sign, because it is a sink.
float wake(vec2 p) {
    if (speed < 0.4) return 0.0;
    float half_len = 0.5 * hull_length;
    return wake_from(p, -half_len, wake_amp)
         - wake_from(p, half_len, 0.55 * wake_amp);
}

// Puddles: a decaying dimple where a blade went in.
float puddle(vec2 p, out float foam) {
    float h = 0.0;
    foam = 0.0;
    for (int i = 0; i < 16; ++i) {
        float strength = puddles[i].z;
        if (strength <= 0.0) continue;
        float r = length(p - puddles[i].xy);
        if (r > 4.0) continue;
        float ring = exp(-r * r / 2.2) * cos(r * 3.4);
        h += 0.055 * strength * ring;
        foam = max(foam, strength * exp(-r * r / 1.4));
    }
    return h;
}

float surface(vec2 p, out float foam) {
    return sea(p, time) + wake(p) + puddle(p, foam);
}

void main() {
    vec2 p = in_grid + centre;
    float foam;
    float h = surface(p, foam);
    v_foam = foam;
    // Normal by finite difference: two extra evaluations a vertex, and
    // without it the water is flat-shaded and the chop is invisible.
    float e = 0.6, junk;
    float hx = surface(p + vec2(e, 0.0), junk);
    float hy = surface(p + vec2(0.0, e), junk);
    v_normal = normalize(vec3((h - hx) / e, (h - hy) / e, 1.0));
    v_world = vec3(p, h);
    gl_Position = mvp * vec4(p, h, 1.0);
}
"""

WATER_FRAGMENT = """#version 330
in vec3 v_world;
in vec3 v_normal;
in float v_foam;
out vec4 f_colour;
uniform vec3 sun;
uniform vec3 sky;
uniform vec3 eye;
uniform float far;
uniform vec3 deep;
void main() {
    vec3 n = normalize(v_normal);
    vec3 to_eye = normalize(eye - v_world);
    // Water is mostly a mirror at grazing angles and mostly dark looking
    // straight down, which is the whole reason chop reads as chop: the
    // Fresnel term turns a slope into a brightness.
    float fresnel = pow(1.0 - max(dot(n, to_eye), 0.0), 3.0);
    vec3 base = mix(deep, sky, clamp(0.08 + 0.55 * fresnel, 0.0, 1.0));
    float spec = pow(max(dot(reflect(-sun, n), to_eye), 0.0), 60.0);
    vec3 lit = base + vec3(0.9) * spec * 0.5 + vec3(0.75) * v_foam * 0.55;
    float haze = clamp(length(v_world - eye) / far, 0.0, 1.0);
    f_colour = vec4(mix(lit, sky, haze * haze), 1.0);
}
"""


def water_grid(reach: float = WATER_REACH,
               divisions: int = WATER_DIVISIONS):
    """A graded grid of triangles, in boat-relative coordinates.

    Built once; the shader moves it with the boat and lifts it onto the
    surface.  The grading is a squared warp of a uniform parameter, so
    cells near the boat are small and cells at the edge are large --
    which is where the resolution is needed and where it is not.
    """
    u = np.linspace(-1.0, 1.0, divisions + 1)
    line = (reach * np.sign(u) * u * u).astype("f4")
    gx, gy = np.meshgrid(line, line)
    a = np.stack([gx[:-1, :-1], gy[:-1, :-1]], axis=-1)
    b = np.stack([gx[:-1, 1:], gy[:-1, 1:]], axis=-1)
    c = np.stack([gx[1:, 1:], gy[1:, 1:]], axis=-1)
    d = np.stack([gx[1:, :-1], gy[1:, :-1]], axis=-1)
    quads = np.concatenate([
        np.stack([a, b, c], axis=2).reshape(-1, 3, 2),
        np.stack([a, c, d], axis=2).reshape(-1, 3, 2)])
    return quads.reshape(-1, 2).astype("f4")


def perspective(fov_y: float, aspect: float, near: float, far: float):
    f = 1.0 / math.tan(math.radians(fov_y) * 0.5)
    matrix = np.zeros((4, 4), dtype="f4")
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye, target, up):
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    side = np.cross(forward, up)
    side /= max(np.linalg.norm(side), 1e-9)
    true_up = np.cross(side, forward)
    matrix = np.eye(4, dtype="f4")
    matrix[0, :3], matrix[1, :3], matrix[2, :3] = side, true_up, -forward
    matrix[:3, 3] = -matrix[:3, :3] @ np.asarray(eye, dtype=float)
    return matrix


def seat_camera(state, boat, sway: float = 1.0):
    """``(eye, target, up)`` for the coxswain's head.

    A bow-loader puts the cox in the bow, lying back: the seat is at
    ``rig.coxswain_position`` and the eye ``rig.coxswain_eye_height``
    above it.  Roll and pitch are carried through, because the horizon
    tipping with the boat is most of what tells you the boat is alive.
    """
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    seat = np.asarray(boat.rig.coxswain_position, dtype=float).copy()
    seat[2] += float(boat.rig.coxswain_eye_height)
    position = np.asarray(state[0:3], dtype=float)
    eye = position + rotation @ (seat * np.array([1.0, sway, 1.0]))
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    up = rotation @ np.array([0.0, 0.0, 1.0])
    return eye, eye + forward, up


def boat_geometry(boat, hull, t, state):
    """The shell and its oars, in world space, for this frame.

    Both ride the hull, so both are built in hull coordinates and carried
    through the same rotation the camera uses.  That is what makes the
    bow sit still in the frame while the world swings behind it -- which
    is the whole cue a coxswain steers on.
    """
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    position = np.asarray(state[0:3], dtype=float)
    pieces = [(hull.vertices @ rotation.T + position, hull.colours)]
    oars, colours = oar_geometry(boat, t, state)
    if oars is not None:
        pieces.append((oars, colours))
    vertices = np.concatenate([p[0] for p in pieces]).astype("f4")
    shades = np.concatenate([p[1] for p in pieces]).astype("f4")
    return vertices, shades


def oar_geometry(boat, t, state):
    """Oars as thin world-space quads, for the near field."""
    lines, drive = oar_lines(boat, t)
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    position = np.asarray(state[0:3], dtype=float)
    quads, colours = [], []
    shaft = np.array([0.28, 0.75, 0.30]) if drive \
        else np.array([0.80, 0.82, 0.78])
    for oar in lines:
        # The plan view gives (handle, lock, blade) in the hull plane;
        # lift them to loom height and give them a little width.
        nodes = np.column_stack([oar, np.full(len(oar), 0.32)])
        world = nodes @ rotation.T + position
        for a, b in zip(world[:-1], world[1:]):
            along = b - a
            side = np.cross(along, [0.0, 0.0, 1.0])
            side /= max(np.linalg.norm(side), 1e-9)
            side *= 0.05
            quads.append([a - side, b - side, b + side])
            quads.append([a - side, b + side, a + side])
            colours.extend([shaft] * 6)
    if not quads:
        return None, None
    return (np.asarray(quads, dtype="f4").reshape(-1, 3),
            np.asarray(colours, dtype="f4"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default="charles",
                        choices=("charles", "totl", "hotl"))
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--physics", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=680)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--reach", type=float, default=900.0,
                        help="how far either side of the course to build")
    parser.add_argument("--step", type=float, default=8.0,
                        help="ground mesh cell, metres")
    parser.add_argument("--no-buildings", action="store_true")
    parser.add_argument("--no-trees", action="store_true")
    parser.add_argument("--wind", type=float, default=6.0,
                        help="wind at 10 m, m/s -- sets the chop through "
                             "the same JONSWAP relations the conditions "
                             "analysis uses")
    parser.add_argument("--fetch", type=float, default=900.0, help="fetch, m")
    parser.add_argument("--wind-from", type=float, default=200.0,
                        help="bearing the wind blows from, degrees")
    parser.add_argument("--audio", default="events",
                        choices=("events", "full"),
                        help="events: catch, release and a bed.  full: one "
                             "clip a stroke following the measured "
                             "spectral envelope through the cycle")
    parser.add_argument("--no-sound", action="store_true",
                        help="silence the stroke; it is the only cue in the "
                             "seat view that says drive from recovery")
    parser.add_argument("--no-guide", action="store_true",
                        help="drop the marker posts along the course")
    parser.add_argument("--control", default="keys",
                        choices=("keys", "mouse"))
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--autopilot", action="store_true")
    parser.add_argument("--shot", default=None,
                        help="render one frame headless and save it")
    parser.add_argument("--start", type=float, default=0.0,
                        help="metres along the course to begin at")
    args = parser.parse_args(argv)

    import moderngl

    print("building %s ..." % args.race)
    clock0 = time.perf_counter()
    # The sea first: the flat far-water quad has to be sunk below the
    # deepest trough of the near field, or it hides them.
    sea = sea_for(args.wind, args.fetch, np.radians(args.wind_from))
    trough = float(np.sum(sea.amplitude)) if len(sea.amplitude) else 0.0
    mesh, scene = build_world(args.race, reach=args.reach, step=args.step,
                              water_level=-1.25 * trough - 0.02,
                              with_buildings=not args.no_buildings,
                              guide=not args.no_guide,
                              trees=not args.no_trees)
    print("   %d triangles in %d parts, %.1f s"
          % (mesh.triangles, len(mesh.parts), time.perf_counter() - clock0))

    boat = catalog.coxed_four(rate=args.rate, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    live = LiveControl()
    cox = Coxswain(rudder_override=live.rudder, pressure_split=live.split)
    simulator = RowingSimulator(boat, coxswain=cox, fast=True)
    hull = hull_solid(boat)

    course = scene.course
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(course, axis=0).T))])
    begin = int(np.argmin(np.abs(station - args.start)))

    def fresh_state():
        state = simulator.initial_state(surge_speed=3.6)
        state[0], state[1] = course[begin]
        ahead = course[min(begin + 3, len(course) - 1)] - course[begin]
        heading = float(np.arctan2(ahead[1], ahead[0]))
        state[5] = heading
        state[6] = 3.6 * math.cos(heading)
        state[7] = 3.6 * math.sin(heading)
        return state

    loop = FixedStepLoop(simulator, rate=args.physics)
    loop.start(fresh_state())

    # The stroke, out loud.  From the bow of a four you cannot see the
    # blades go in, and without the catch there is nothing in the seat
    # view that separates drive from recovery -- which makes calling the
    # boat impossible, and calling is what this is for.
    audio = None
    if not args.no_sound and not args.shot:
        from coxswain.viz.strokeaudio import StrokeAudio
        audio = StrokeAudio(boat, mode=args.audio)
        print("   stroke audio: %s, %s"
              % (audio.mode,
                 "on" if audio.available else "no device, running silent"))

    headless = bool(args.shot)
    if headless:
        ctx = moderngl.create_standalone_context()
        target = ctx.simple_framebuffer((args.width, args.height))
        target.use()
        screen = None
    else:
        import pygame

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_caption("%s -- the seat" % scene.name)
        screen = pygame.display.set_mode((args.width, args.height),
                                         pygame.OPENGL | pygame.DOUBLEBUF)
        ctx = moderngl.create_context()
        target = ctx.screen

    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.cull_face = "back"
    program = ctx.program(vertex_shader=VERTEX_SHADER,
                          fragment_shader=FRAGMENT_SHADER)
    program["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                 / np.linalg.norm([0.42, 0.30, 0.85]))
    program["sky"].value = SKY
    program["far"].value = FAR

    static = []
    for part in mesh.parts:
        buffer = ctx.buffer(part.interleaved().tobytes())
        static.append(ctx.vertex_array(
            program, [(buffer, "3f 3f 3f", "in_pos", "in_normal",
                       "in_colour")]))

    # -- the water ------------------------------------------------------
    water_prog = ctx.program(vertex_shader=WATER_VERTEX,
                             fragment_shader=WATER_FRAGMENT)
    water_prog["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                    / np.linalg.norm([0.42, 0.30, 0.85]))
    water_prog["sky"].value = SKY
    water_prog["far"].value = FAR
    water_prog["deep"].value = (0.055, 0.115, 0.155)
    water_prog["hull_length"].value = float(boat.length)
    water_prog["tan_wedge"].value = float(np.tan(KELVIN_HALF_ANGLE))
    grid = water_grid()
    water_buffer = ctx.buffer(grid.tobytes())
    water_vao = ctx.vertex_array(water_prog,
                                 [(water_buffer, "2f", "in_grid")])
    field = sea
    water_prog["waves"].write(field.as_uniform().tobytes())
    trail = PuddleTrail()
    # Wake amplitude from the hull's own wave resistance, tabulated once.
    wake_speed, wake_scale = wake_table(boat)
    print("   wake: Michell wave resistance gives %.3f m at 10 m astern "
          "at 4.5 m/s" % (float(np.interp(4.5, wake_speed, wake_scale))
                          / np.sqrt(10.0)))
    print("   water: H_s %.3f m, T_p %.2f s at %.0f m/s over %.0f m fetch; "
          "%d triangles; far plane sunk to %.3f m"
          % (field.significant_height, field.peak_period, args.wind,
             args.fetch, len(grid) // 3, -1.25 * trough - 0.02))

    # The oars change every frame, so they get a stream buffer sized once.
    oar_buffer = ctx.buffer(reserve=256 * 1024, dynamic=True)
    oar_vao = ctx.vertex_array(
        program, [(oar_buffer, "3f 3f 3f", "in_pos", "in_normal",
                   "in_colour")])

    projection = perspective(args.fov, args.width / args.height, 0.25, FAR)
    # Set on the function so the closure can carry state without a global.
    _ = None

    def draw(state, t):
        """One frame.  ``draw.last_phase`` remembers where in the stroke
        the previous frame was, so a catch can be detected by the wrap."""
        eye, target_point, up = seat_camera(state, boat)
        view = look_at(eye, target_point, up)
        program["mvp"].write((projection @ view).T.tobytes(order="C"))
        program["eye"].value = tuple(float(v) for v in eye)
        target.clear(SKY[0], SKY[1], SKY[2], 1.0)
        for vao in static:
            vao.render()
        # The water goes on after the land, so the shore reads through it
        # at the edges and the patch does not have to be clipped.
        speed = float(np.hypot(state[6], state[7]))
        water_prog["mvp"].write((projection @ view).T.tobytes(order="C"))
        water_prog["eye"].value = tuple(float(v) for v in eye)
        water_prog["centre"].value = (float(state[0]), float(state[1]))
        water_prog["boat"].value = (float(state[0]), float(state[1]),
                                    float(state[5]))
        water_prog["speed"].value = speed
        water_prog["wake_k"].value = float(
            2.0 * np.pi / max(kelvin_wavelength(speed), 0.5))
        water_prog["wake_amp"].value = float(
            np.interp(speed, wake_speed, wake_scale))
        water_prog["time"].value = float(t)
        # Drop a pair of puddles at each catch -- the same phase that
        # fires the catch in strokeaudio, so what you hear and what you
        # see on the water are the same event.
        period = float(boat.timing.period)
        if period > 0.0:
            phase = (t % period) / period
            if phase < draw.last_phase:            # wrapped: a new catch
                # oar_lines gives (handle, lock, blade) in the HULL
                # frame and a drive flag; the blade tip has to be rotated
                # and translated like the oars themselves are.
                lines, _drive = oar_lines(boat, t)
                rot = hull_to_abs(np.asarray(state[3:6], dtype=float))
                here = np.asarray(state[0:3], dtype=float)
                for oar in lines:
                    tip = np.append(np.asarray(oar)[-1], 0.0) @ rot.T + here
                    trail.drop(float(tip[0]), float(tip[1]), t)
            draw.last_phase = phase
        water_prog["puddles"].write(trail.as_uniform(t).tobytes())
        water_vao.render()
        vertices, colours = boat_geometry(boat, hull, t, state)
        if vertices is not None and len(vertices):
            normals = _face_normals(vertices)
            blob = np.hstack([vertices, normals, colours]).astype("f4")
            if blob.nbytes <= oar_buffer.size:
                oar_buffer.write(blob.tobytes())
                oar_vao.render(vertices=len(vertices))

    draw.last_phase = 0.0

    if headless:
        for _ in range(int(args.frames or 0)):
            loop.advance(1.0 / 60.0)
        draw(loop.pose(), loop.t)
        from PIL import Image

        image = Image.frombytes("RGB", (args.width, args.height),
                                target.read(components=3))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        os.makedirs(os.path.dirname(args.shot) or ".", exist_ok=True)
        image.save(args.shot)
        print("wrote %s" % args.shot)
        return 0

    import pygame

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17)
    overlay = pygame.Surface((args.width, args.height), pygame.SRCALPHA)
    hud_texture = ctx.texture((args.width, args.height), 4)
    hud_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    rudder, split, paused, running, frames = 0.0, 0.0, False, True, 0
    if args.control == "mouse":
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos((args.width // 2, args.height // 2))

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
                elif event.key == pygame.K_c:
                    rudder = 0.0
                elif event.key == pygame.K_m:
                    args.control = "keys" if args.control == "mouse" \
                        else "mouse"
                    pygame.mouse.set_visible(args.control != "mouse")
                    if args.control == "mouse":
                        pygame.mouse.set_pos((
                            int(args.width * 0.5 + rudder / RUDDER_LIMIT
                                * args.width * MOUSE_SPAN * 0.5),
                            args.height // 2))
                elif event.key == pygame.K_r:
                    loop.start(fresh_state())
                    rudder = split = 0.0

        keys = pygame.key.get_pressed()
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
        live.set(ControlInput(rudder=rudder, pressure_split=split))

        if args.autopilot:
            gap = course - loop.state[0:2]
            index = int(np.argmin(np.einsum("ij,ij->i", gap, gap)))
            ahead = course[min(index + 12, len(course) - 1)] - loop.state[0:2]
            want = float(np.arctan2(ahead[1], ahead[0]))
            error = (want - float(loop.state[5]) + np.pi) % (2 * np.pi) - np.pi
            rudder = float(np.clip(-1.4 * error, -RUDDER_LIMIT, RUDDER_LIMIT))
            live.set(ControlInput(rudder=rudder))

        if not paused:
            loop.advance(frame)
            if audio is not None:
                audio.update(loop.t)
                # The slide rumble follows the recovery: silent through
                # the drive, rising as the seats run back up.
                period = float(boat.timing.period)
                drive = float(getattr(boat.timing, "drive_duration", 0.0)
                              ) / max(period, 1e-9) or 0.4
                phase = (loop.t % period) / period
                if phase <= drive:
                    audio.set_slide_level(0.0)
                else:
                    u = (phase - drive) / max(1.0 - drive, 1e-6)
                    audio.set_slide_level(math.sin(math.pi * u))
        pose = loop.pose()
        draw(pose, loop.t)

        speed = float(np.hypot(pose[6], pose[7]))
        seconds = 500.0 / speed if speed > 0.2 else 0.0
        lines = ["%d:%04.1f   %.2f m/s   rate %.0f"
                 % (int(seconds // 60), seconds % 60, speed, boat.timing.rate),
                 "stick %+.0f%% (%+.1f deg)   yaw %+.2f deg/s"
                 % (100 * rudder / RUDDER_LIMIT, math.degrees(rudder),
                    math.degrees(pose[11])),
                 "roll %+.1f deg   pitch %+.1f deg   %.0f fps"
                 % (math.degrees(pose[3]), math.degrees(pose[4]),
                    clock.get_fps())]
        overlay.fill((0, 0, 0, 0))
        for row, text in enumerate(lines):
            overlay.blit(font.render(text, True, (233, 240, 245)),
                         (14, 12 + row * 20))
        # The stick.  On the mouse there is nothing else to tell you where
        # the rudder is -- the pointer is hidden and the boat answers a
        # second later -- and on a boat with a standing yaw bias, seeing
        # the trim you are holding is most of what makes it steerable.
        bar_w = int(args.width * MOUSE_SPAN)
        bar_x = (args.width - bar_w) // 2
        bar_y = args.height - 40
        pygame.draw.rect(overlay, (12, 17, 21, 170),
                         (bar_x - 62, bar_y - 20, bar_w + 124, 46))
        pygame.draw.line(overlay, (70, 82, 92), (bar_x, bar_y),
                         (bar_x + bar_w, bar_y), 3)
        pygame.draw.line(overlay, (110, 124, 136), (args.width // 2,
                                                    bar_y - 9),
                         (args.width // 2, bar_y + 9), 2)
        knob = bar_x + int(bar_w * (0.5 + 0.5 * rudder / RUDDER_LIMIT))
        pygame.draw.circle(overlay, (255, 146, 72), (knob, bar_y), 9)
        pygame.draw.circle(overlay, (18, 24, 29), (knob, bar_y), 5)
        for label, at in (("port", bar_x - 52), ("stbd", bar_x + bar_w + 14)):
            overlay.blit(font.render(label, True, (128, 142, 152)),
                         (at, bar_y - 9))
        # The HUD is a texture blitted over the scene: pygame cannot draw
        # into an OpenGL window directly.
        # One texture, rewritten -- allocating a 1180x680 RGBA texture
        # every frame is 3 MB of churn for a few lines of text.
        ctx.disable(moderngl.DEPTH_TEST)
        hud_texture.write(_surface_bytes(pygame, overlay))
        hud_texture.use(0)
        _hud_blit(ctx)
        ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()
        frames += 1
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()
    print("%d frames, %d physics steps" % (frames, loop.steps))
    return 0


def _surface_bytes(pygame, surface):
    """RGBA bytes from a surface, flipped for GL's origin.

    ``tostring`` is deprecated in favour of ``tobytes`` and only one of
    them exists depending on the pygame; ask for whichever is there.
    """
    getter = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    return getter(surface, "RGBA", True)


_BLIT = {}


def _hud_blit(ctx):
    """Draw a full-screen RGBA texture over whatever is already there."""
    import moderngl

    if "vao" not in _BLIT:
        program = ctx.program(
            vertex_shader='''#version 330
            in vec2 in_pos; out vec2 uv;
            void main(){ uv = in_pos * 0.5 + 0.5;
                         gl_Position = vec4(in_pos, 0.0, 1.0); }''',
            fragment_shader='''#version 330
            in vec2 uv; out vec4 f; uniform sampler2D image;
            void main(){ f = texture(image, uv); }''')
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        buffer = ctx.buffer(quad.tobytes())
        _BLIT["vao"] = ctx.vertex_array(program, [(buffer, "2f", "in_pos")])
        _BLIT["program"] = program
    ctx.enable(moderngl.BLEND)
    _BLIT["program"]["image"].value = 0
    _BLIT["vao"].render()
    ctx.disable(moderngl.BLEND)


if __name__ == "__main__":
    raise SystemExit(main())
