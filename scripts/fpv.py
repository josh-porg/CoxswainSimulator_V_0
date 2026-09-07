r"""The coxswain's seat, in real time.

    python scripts/fpv.py                          # Head of the Charles
    python scripts/fpv.py --race hotl --control mouse
    python scripts/fpv.py --shot out/fpv/seat.png  # one frame, no window

0.55 m off the water in the bow of a four, looking forward over four
backs -- or up in the stern of an eight, where the crew face you, or in
the stroke's seat of a scull, facing astern.  The **only** view from which "does this steer like a boat" is a
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
W / E                 pressure split -- the only steering a scull has
V                     look over your shoulder
Esc                   the menu: rate, wind, restart, controls, boat
R                     restart          Space  freeze          Q  quit
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
from coxswain.viz.menu import (build_boat, chart_surface,    # noqa: E402
                               draw_controls, draw_menu,
                               handle_key, options_menu, pause_menu,
                               quality_settings, setup_menu,
                               start_music, stop_music)
from coxswain.viz.planscene import oar_lines                # noqa: E402
from coxswain.viz.strokeaudio import shell_of               # noqa: E402
from coxswain.viz.water import (KELVIN_HALF_ANGLE,          # noqa: E402
                                load_nearfield, load_wavefield,
                                PuddleTrail, sea_for)
from coxswain.viz.worldmesh import _face_normals            # noqa: E402
from coxswain.viz.worldmesh import (build_world,             # noqa: E402
                                    crew_solids, hull_solid,
                                    oar_solids, viewpoint)

RUDDER_LIMIT = 0.20
RUDDER_RATE = 0.55
MOUSE_SPAN = 0.55
SPLIT_LIMIT, SPLIT_RATE, SPLIT_RETURN = 1.0, 1.8, 1.4

#: How far you can see before everything is sky.  Fog closes the gap so
#: the world does not end at a hard edge.
FAR = 2600.0
SKY = (0.52, 0.60, 0.68)

#: Weather, as everything the air does at once.
#:
#: These are one setting because they are one thing: on an overcast day
#: the zenith loses its blue, the horizon closes in, the sun stops
#: casting a direction and the visibility drops, and changing any of
#: those without the others gives you a lit day with fog bolted on.
#:
#: ``(zenith, horizon, glow, fog_density, fog_height, scatter)``.  Fog
#: density is per metre; the reciprocal is roughly the range at which a
#: dark object is half lost, so 1/2600 is a clear morning and 1/420 is
#: the kind of river fog that has crews rowing on sound.
WEATHER = {
    "clear": ((0.30, 0.47, 0.70), (0.68, 0.76, 0.83),
              (0.40, 0.34, 0.24), 1.0 / 3400.0, 40.0, 0.55),
    "hazy": ((0.36, 0.50, 0.68), (0.71, 0.77, 0.82),
             (0.34, 0.30, 0.22), 1.0 / 2600.0, 34.0, 0.45),
    "overcast": ((0.60, 0.63, 0.66), (0.74, 0.76, 0.78),
                 (0.10, 0.10, 0.10), 1.0 / 1100.0, 26.0, 0.15),
    # 1/420 was chosen for the number and looked like nothing: the
    # Charles is 150 m across and its far bank is well inside a 290 m
    # half-loss range, so the "fog" was doing almost exactly what the
    # overcast did.  A river fog you would actually be careful in takes
    # the far bank most of the way out.
    "fog": ((0.74, 0.76, 0.77), (0.82, 0.83, 0.84),
            (0.05, 0.05, 0.05), 1.0 / 130.0, 11.0, 0.06),
}

#: The dome, as ``(zenith, horizon, glow)``.  A New England overcast:
#: the zenith holds some blue, the horizon washes out toward white, and
#: the glow is the sun's place behind the cloud rather than a disc.
SKY_ZENITH = (0.36, 0.50, 0.68)
SKY_HORIZON = (0.71, 0.77, 0.82)
SUN_GLOW = (0.34, 0.30, 0.22)

#: Fog: extinction per metre, and the height over which the haze thins.
#:
#: 1/2600 is roughly where the old linear ramp reached full strength, so
#: the far bank sits about where it did; the difference is that this one
#: never saturates abruptly and thins as you rise out of it.  The height
#: scale is deliberately low -- river haze lies on the river.
FOG_DENSITY = 1.0 / 2600.0
FOG_HEIGHT = 34.0

#: The second normal: slope amplitude, spatial frequency, and how far
#: out it is worth drawing.  Slope, not height -- see the shader.
#: Tuned down from 0.085, which read as sparkle rather than texture:
#: at that amplitude the second normal was competing with the chop
#: instead of sitting under it, which is the one thing it must not do.
RIPPLE_SLOPE = 0.038
RIPPLE_SCALE = 1.35
RIPPLE_FADE = 45.0

#: How much rougher the small scale is where the water has been stirred.
RIPPLE_WAKE_GAIN = 1.6

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

#: Sky and fog, shared by everything that used to reach for one flat
#: colour: the sky itself, the haze on the land, the haze on the water,
#: and the water's reflection.
#:
#: The old sky was a single RGB used as the clear colour and as the fog
#: colour, so the dome was a flat wall, the horizon had no glow, and the
#: fog could not agree with the sky because it *was* the sky, everywhere
#: at once.  Three changes:
#:
#: * a gradient from zenith to horizon, with a broad glow around the sun,
#:   so the light has a place it comes from;
#: * fog that is exponential rather than linear -- extinction along a
#:   ray is exp(-density * distance), which is what actually happens,
#:   and unlike a linear ramp it has no distance at which it abruptly
#:   saturates;
#: * a height term, because the haze over a river sits ON the river.
#:   The integral of an exponentially-stratified density along a ray has
#:   a closed form, so this is not a fudge factor: it is the analytic
#:   optical depth for that profile.
#:
#: The fog colour is then the sky in the direction being looked along,
#: which is what makes a far bank sit *in* the air rather than under a
#: grey wash, and the water's reflections are fogged with the same
#: function so a reflected bank fades exactly as the real one does.
SKY_FOG_GLSL = """
uniform vec3 sky_zenith;
uniform vec3 sky_horizon;
uniform vec3 sun_glow;
uniform float fog_density;
uniform float fog_height;
uniform float fog_scatter;

vec3 sky_colour(vec3 dir, vec3 sun_dir) {
    vec3 d = normalize(dir);
    // Up the dome.  The power keeps most of the change near the
    // horizon, where the air path is long, instead of spreading it
    // evenly over a quarter turn.
    float up = clamp(d.z, 0.0, 1.0);
    vec3 base = mix(sky_horizon, sky_zenith, pow(up, 0.42));
    // Horizon thickening.  Looking along the surface the ray stays in
    // the dense layer for its whole length, so the last few degrees
    // above the horizon are much paler than the gradient alone makes
    // them -- and on a river, where the far bank sits in exactly that
    // band, it is most of what tells you how far away it is.
    float band = exp(-max(up, 0.0) * 26.0);
    base = mix(base, sky_horizon * 1.04, band * 0.65);
    // A broad glow rather than a disc: the sun is behind cloud here.
    float towards = max(dot(d, normalize(sun_dir)), 0.0);
    base += sun_glow * pow(towards, 6.0) * 0.55;
    base += sun_glow * pow(towards, 2.0) * 0.10;
    return base;
}

// Optical depth through air whose density falls off exponentially with
// height, integrated along the ray.  The limit as the ray flattens out
// is handled explicitly; without it the horizon divides by zero.
float fog_depth(vec3 from, vec3 to) {
    vec3 ray = to - from;
    float distance = length(ray);
    if (distance < 1e-4) return 0.0;
    float rise = ray.z;
    float at_eye = exp(-max(from.z, 0.0) / fog_height);
    float integral;
    if (abs(rise) < 1e-3) {
        integral = at_eye * distance;
    } else {
        integral = at_eye * fog_height * distance / rise
                 * (1.0 - exp(-rise / fog_height));
    }
    return fog_density * max(integral, 0.0);
}

vec3 apply_fog(vec3 colour, vec3 from, vec3 to) {
    float depth = fog_depth(from, to);
    float keep = exp(-depth);
    return mix(sky_colour(normalize(to - from), vec3(0.0, 0.0, 1.0)),
               colour, keep);
}

vec3 apply_fog_lit(vec3 colour, vec3 from, vec3 to, vec3 sun_dir) {
    vec3 dir = normalize(to - from);
    float depth = fog_depth(from, to);
    float keep = exp(-depth);
    vec3 air = sky_colour(dir, sun_dir);
    // Directional scattering.  Haze is not a grey curtain: light coming
    // through it is scattered forward, so looking toward the sun the
    // air itself glows and looking away it does not.  The Henyey-
    // Greenstein phase function is the usual one for this; a cheap
    // forward lobe is enough at these optical depths and costs a pow.
    //
    // It scales with how much air the ray went through, because that is
    // what does the scattering -- so it appears on the far bank and not
    // on the near one, which is the whole point.
    float towards = max(dot(dir, normalize(sun_dir)), 0.0);
    float lobe = pow(towards, 5.0);
    air += sun_glow * lobe * fog_scatter * clamp(depth, 0.0, 1.5);
    return mix(air, colour, keep);
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
__SKY_FOG__
void main() {
    // A single directional light with a generous ambient: this is an
    // overcast New England morning, not a stage.
    float lambert = max(dot(normalize(v_normal), sun), 0.0);
    vec3 lit = v_colour * (0.55 + 0.45 * lambert);
    // Fogged toward the sky in the direction being looked along, so the
    // far bank sits in the air rather than under a flat wash.
    f_colour = vec4(apply_fog_lit(lit, eye, v_world, sun), 1.0);
}
"""

#: A dome drawn behind everything, so the sky has structure to fog into.
SKY_VERTEX = """#version 330
in vec2 in_pos;
out vec3 v_ray;
uniform mat4 inverse_vp;
void main() {
    // A full-screen triangle, unprojected to a direction per pixel.
    vec4 near = inverse_vp * vec4(in_pos, -1.0, 1.0);
    vec4 far_point = inverse_vp * vec4(in_pos, 1.0, 1.0);
    v_ray = far_point.xyz / far_point.w - near.xyz / near.w;
    gl_Position = vec4(in_pos, 1.0, 1.0);
}
"""

SKY_FRAGMENT = """#version 330
in vec3 v_ray;
out vec4 f_colour;
uniform vec3 sun;
__SKY_FOG__
void main() {
    vec3 dir = normalize(v_ray);
    vec3 colour = sky_colour(dir, sun);
    // Below the horizon the dome is not seen directly -- the water is
    // in the way -- but the reflection march can look there, so it is
    // held at the horizon colour rather than going dark.
    f_colour = vec4(colour, 1.0);
}
"""


#: Texture unit the hull's near-field map lives on.  Unit 0 belongs to
#: the HUD blit; sharing it was a 2.5 m wave beside the boat.
NEAR_UNIT = 1
#: And the baked wave pattern on the next one.
WAVE_UNIT = 2

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
WATER_DIVISIONS = 340

#: Share of the water grid's reach carried by the linear term; see
#: :func:`water_grid`.  Higher spreads resolution further out from the
#: hull and coarsens the horizon.
NEAR_FRACTION = 0.30

#: The sea's slope, differentiated rather than sampled.
#:
#: The water was shaded from a per-vertex normal, so the shading was
#: tied to the mesh: with cells graded from 0.19 m near the boat to
#: 1.9 m at the horizon, and a finite difference taken at a fixed 0.6 m
#: regardless, every triangle got its own normal and the grid showed
#: through the surface as a triangular overlay.
#:
#: The chop is a sum of cosines, so its slope is a sum of sines -- exact,
#: per pixel, and cheaper than the three extra surface evaluations a
#: finite difference would cost in the fragment shader.  The baked parts
#: (the wake, the near field, the puddles) vary slowly compared with the
#: cell size and are still interpolated from the vertices.
WATER_SLOPE_GLSL = """
const float SLOPE_G = 9.80665;

//: How far from the boat the surface is differentiated per pixel, m.
//:
//: Inside this the whole surface -- chop, wake, near field and puddles
//: -- is evaluated three times a fragment and differenced.  That is the
//: water a coxswain actually inspects, a few metres either side of the
//: hull, and it is where the fast-varying parts live: a puddle is under
//: a metre across and the near field is baked at 0.13 m, both far finer
//: than the 0.19 m cells there, so interpolating THEIR slope from the
//: vertices left the mesh visible however good the chop had become.
//:
//: Outside it the baked parts are smooth against a cell and the
//: interpolated slope is indistinguishable, so only the chop is done
//: exactly and the cost falls away with distance.
uniform float exact_within;

// --- micro-ripple, the second normal ---------------------------------
//
// The physics-based surface -- chop, wake, near field, puddles -- is
// everything with a length scale a boat cares about.  What it has no
// term for is the centimetre-scale texture that makes water look wet:
// the cat's paw of the wind on the surface.
//
// So a second normal is laid over the first, from simplex noise, and it
// is DELIBERATELY not allowed near the height field.  It perturbs the
// normal only; nothing displaces the geometry and nothing reaches the
// physics.  Its amplitude is a slope, not a height, and it is small --
// enough to break up a mirror, not enough to be mistaken for chop.
//
// It fades out with distance, which is not laziness: a ripple a few
// centimetres across, seen at fifty metres, is far below one pixel, and
// drawn anyway it is just aliasing.
uniform float ripple_slope_amp;
uniform float ripple_scale;
uniform float ripple_fade;

vec3 simplex_permute(vec3 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float simplex(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = simplex_permute(simplex_permute(i.y + vec3(0.0, i1.y, 1.0))
                             + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                            dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Two octaves, drifting with the wind, differenced for a slope.  The
// second octave runs the other way so the pattern does not read as one
// sheet sliding.
uniform float ripple_wake_gain;

vec2 ripple(vec2 p, float range, float stirred) {
    float fade = clamp(1.0 - range / ripple_fade, 0.0, 1.0);
    if (fade <= 0.0 || ripple_slope_amp <= 0.0) return vec2(0.0);
    // Disturbed water is rougher at the small scale than calm water.
    // A wake and a puddle are not just a shape: they are a patch of
    // surface that has been stirred, and it stays stirred after the
    // shape has flattened out.  So the second normal is amplified where
    // the first one is doing something -- which puts the extra texture
    // along the wake and in the puddles, and leaves the water either
    // side of the boat smooth.
    float rough = 1.0 + ripple_wake_gain * clamp(stirred, 0.0, 1.0);
    vec2 drift = vec2(cos(wind_to), sin(wind_to)) * time;
    float e = 0.05;
    vec2 q = p * ripple_scale;
    vec2 a = q - drift * 0.35;
    vec2 b = q * 2.7 + drift.yx * 0.20;
    float h  = simplex(a) + 0.5 * simplex(b);
    float hx = simplex(a + vec2(e, 0.0)) + 0.5 * simplex(b + vec2(e, 0.0));
    float hy = simplex(a + vec2(0.0, e)) + 0.5 * simplex(b + vec2(0.0, e));
    return vec2(hx - h, hy - h) / e
         * ripple_slope_amp * fade * fade * rough;
}

vec2 sea_slope(vec2 p) {
    vec2 g = vec2(0.0);
    for (int i = 0; i < 8; ++i) {
        float a = waves[i].x;
        if (a <= 0.0) continue;
        float k = waves[i].y;
        float d = waves[i].z;
        vec2 along = vec2(cos(d), sin(d));
        float w = sqrt(SLOPE_G * k);
        float s = sin(k * dot(p, along) - w * time + waves[i].w);
        g += -a * k * along * s;
    }
    return g;
}

vec3 water_normal(vec3 world, vec2 baked_slope, float range,
                  float foam) {
    vec2 g;
    // How stirred this patch is: the slope the boat's own disturbance
    // contributes, plus the foam the puddles carry.  Taken from the
    // wake and near field rather than from the chop, because wind waves
    // do not leave a rougher surface behind them and a wake does.
    float stirred = foam;
    if (distance(world.xy, boat.xy) < exact_within) {
        // Differenced at a step finer than the cells, so the answer is
        // the surface's slope and not the mesh's.
        float e = 0.05, junk;
        float h  = surface(world.xy, junk);
        float hx = surface(world.xy + vec2(e, 0.0), junk);
        float hy = surface(world.xy + vec2(0.0, e), junk);
        g = vec2((hx - h) / e, (hy - h) / e);
        stirred += 6.0 * abs(h - sea(world.xy, time));
    } else {
        g = sea_slope(world.xy) + baked_slope;
        stirred += 3.0 * length(baked_slope);
    }
    g += ripple(world.xy, range, stirred);
    return normalize(vec3(-g.x, -g.y, 1.0));
}
"""


#: Uniforms describing the surface, shared by both shader stages.
#: Declared once so the vertex and the fragment cannot disagree about
#: what the water is.
WATER_UNIFORMS_GLSL = """
uniform mat4 mvp;
uniform vec2 centre;          // the patch follows the boat
uniform vec4 waves[8];        // amplitude, wavenumber, direction, phase
uniform vec4 puddles[16];     // east, north, strength, unused
uniform vec3 boat;            // east, north, heading
uniform float speed;
uniform sampler3D wave_map;   // baked wave pattern, slices in speed
uniform vec2 wave_lo;         // its box in the boat frame
uniform vec2 wave_hi;
uniform vec3 wave_size;       // samples along, across, in speed
uniform float wave_speed_lo;  // speeds the slices span
uniform float wave_speed_hi;
uniform float hull_length;    // bow-to-stern source separation
uniform sampler2D near_map;   // baked near-field shape, F(x, y)
uniform vec2 near_lo;         // its box in the boat frame
uniform vec2 near_hi;
uniform vec2 near_size;       // samples in the baked grid
uniform float wind_to;        // bearing the wind blows toward
uniform float time;
"""

#: The surface itself.  Spliced into the vertex shader, which displaces
#: the grid by it, and into the fragment shaders, which differentiate it
#: per pixel near the boat.
WATER_SURFACE_GLSL = """
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

// The wave pattern the hull radiates, from the free-surface Green's
// function -- baked by coxswain.hydro.havelock and sampled here.
//
// This replaces the hand-built Kelvin wedge.  The wedge had the right
// angle and the right transverse wavelength and guessed everything
// else; this is the same thin-ship source sheet and the same Green's
// function Michell's wave resistance comes from, so the divergent and
// transverse systems, the bow-stern interference and the way the
// pattern reshapes with speed all come out of one integral, and the
// amplitude is pinned to the wave resistance by an energy closure.
//
// The wavelengths scale with U^2, so unlike the near field this cannot
// be one texture times a speed factor: it is a stack of slices in speed
// and the third texture coordinate interpolates between them.
float wake(vec2 p) {
    if (speed < wave_speed_lo) return 0.0;
    float c = cos(-boat.z), s = sin(-boat.z);
    vec2 d = p - boat.xy;
    float along = d.x * c - d.y * s;          // positive toward the bow
    float across = d.x * s + d.y * c;
    vec3 uv = vec3((vec2(along, across) - wave_lo) / (wave_hi - wave_lo),
                   (min(speed, wave_speed_hi) - wave_speed_lo)
                   / (wave_speed_hi - wave_speed_lo));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    // Half-texel correction in every axis, as for the near field.
    uv = uv * (wave_size - 1.0) / wave_size + 0.5 / wave_size;
    vec2 e = min(uv.xy, 1.0 - uv.xy);
    float edge = clamp(min(e.x, e.y) / 0.08, 0.0, 1.0);
    return texture(wave_map, uv).r * edge;
}

// Puddles: a vortex pair that spreads and flattens as it ages.
//
// A crew drops four within two and a half metres of each other, so
// fixed-width rings summed constructively into a peak beside the hull
// every stroke, and a ring that never widens eventually outruns the
// grid and aliases.  A real puddle does neither: it spreads as it
// decays, its structure coarsening, and is gone in a few seconds.
float puddle(vec2 p, out float foam) {
    float h = 0.0;
    foam = 0.0;
    for (int i = 0; i < 16; ++i) {
        float strength = puddles[i].z;
        if (strength <= 0.0) continue;
        float r = length(p - puddles[i].xy);
        float age = 1.0 - strength;               // 0 fresh, 1 spent
        float spread = 1.5 + 3.0 * age;           // metres
        if (r > 2.5 * spread) continue;
        float k = 3.4 / (1.0 + 2.2 * age);        // ring coarsens
        float ring = exp(-r * r / (spread * spread)) * cos(r * k);
        h += 0.030 * strength * strength * ring;
        foam = max(foam, pow(strength, 3.0) * exp(-r * r / 1.4));
    }
    return h;
}

// The water the hull itself pushes about, baked.
//
// eta = (U^2 / g) * F(x, y) with F a function of the hull's shape alone,
// so the thin-ship source sum is done once offline and this is a texture
// lookup.  It is the near field -- stagnation at the stem, acceleration
// along the midbody -- and it is what the Kelvin construction, which is
// a far-field description, has nothing to say about.
float near_field(vec2 p) {
    float c = cos(-boat.z), s = sin(-boat.z);
    vec2 d = p - boat.xy;
    float along = d.x * c - d.y * s;      // positive toward the bow
    float across = d.x * s + d.y * c;
    vec2 uv = (vec2(along, across) - near_lo) / (near_hi - near_lo);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    // Half-texel correction: the baked grid holds N samples spanning the
    // box inclusively, but a texture's samples sit at texel CENTRES, so
    // a naive 0..1 mapping shifts the whole field by half a cell -- four
    // centimetres across a hull half a metre wide, and one-sided.
    uv = uv * (near_size - 1.0) / near_size + 0.5 / near_size;
    // Fade at the edges of the baked box so it cannot leave a seam where
    // it hands over to the radiated wake.
    vec2 e = min(uv, 1.0 - uv);
    float edge = clamp(min(e.x, e.y) / 0.12, 0.0, 1.0);
    return texture(near_map, uv).r * speed * speed / 9.80665 * edge;
}

// The boat's wind shadow, and the pile against its windward side.
//
// Not diffraction: at half a metre of beam against a two-metre wave the
// hull is nearly transparent to the sea, so scattering is not what makes
// the lee calm.  Blocking the WIND is.  Short wind waves are held up by
// the air working on them continuously; take the air away and they go
// within metres, which is the smooth patch that sits downwind of a
// boat.  The strip it shelters is as wide as the hull's projection
// across the wind, so a crew lying beam-on shades a long band and one
// pointing into it shades almost nothing.
float shelter(vec2 p) {
    vec2 blow = vec2(cos(wind_to), sin(wind_to));
    vec2 axis = vec2(cos(boat.z), sin(boat.z));
    float across_wind = abs(axis.x * blow.y - axis.y * blow.x);
    float along_wind = abs(dot(axis, blow));
    float half_width = 0.5 * (hull_length * across_wind + 0.5 * along_wind);
    vec2 d = p - boat.xy;
    float downwind = dot(d, blow);
    float lateral = abs(-d.x * blow.y + d.y * blow.x);
    float inside = clamp(1.0 - (lateral - half_width) / 1.5, 0.0, 1.0);
    float lee = downwind > 0.0 ? 0.55 * exp(-downwind / 14.0) : 0.0;
    float wind = downwind < 0.0 ? 0.35 * exp(downwind / 1.6) : 0.0;
    return 1.0 - inside * lee + inside * wind;
}

float surface(vec2 p, out float foam) {
    return sea(p, time) * shelter(p)
         + wake(p) + near_field(p) + puddle(p, foam);
}
"""

WATER_VERTEX = """#version 330
in vec2 in_grid;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_slope;
out float v_foam;
""" + WATER_UNIFORMS_GLSL + WATER_SURFACE_GLSL + """
void main() {
    vec2 p = in_grid + centre;
    float foam;
    float h = surface(p, foam);
    v_foam = foam;
    // The slope of everything EXCEPT the sea, by finite difference.
    // The sea's own slope is done exactly, per pixel, in the fragment
    // shader -- it is the fast-varying part and the part that was
    // making the mesh visible.  What is left here is the wake, the near
    // field and the puddles, all of which change slowly across a cell,
    // so interpolating them from the vertices costs nothing visible.
    float e = 0.6, junk;
    float base = h - sea(p, time);
    float bx = surface(p + vec2(e, 0.0), junk) - sea(p + vec2(e, 0.0), time);
    float by = surface(p + vec2(0.0, e), junk) - sea(p + vec2(0.0, e), time);
    v_slope = vec2((bx - base) / e, (by - base) / e);
    // Kept for anything still reading it; the fragment shaders build
    // their own from v_slope and the analytic sea.
    v_normal = normalize(vec3(-v_slope.x, -v_slope.y, 1.0));
    v_world = vec3(p, h);
    gl_Position = mvp * vec4(p, h, 1.0);
}
"""

WATER_FRAGMENT = """#version 330
in vec3 v_world;
in vec3 v_normal;
in vec2 v_slope;
in float v_foam;
__WATER_SHARED__
out vec4 f_colour;
uniform vec3 sun;
uniform vec3 sky;
uniform vec3 eye;
uniform float far;
uniform vec3 deep;
__SKY_FOG__
__WATER_SLOPE__
void main() {
    float range = length(v_world - eye);
    vec3 n = water_normal(v_world, v_slope, range, v_foam);
    vec3 to_eye = normalize(eye - v_world);
    // Water is mostly a mirror at grazing angles and mostly dark looking
    // straight down, which is the whole reason chop reads as chop: the
    // Fresnel term turns a slope into a brightness.
    float fresnel = pow(1.0 - max(dot(n, to_eye), 0.0), 3.0);
    vec3 mirror = sky_colour(reflect(-to_eye, n), sun);
    vec3 base = mix(deep, mirror, clamp(0.08 + 0.55 * fresnel, 0.0, 1.0));
    float spec = pow(max(dot(reflect(-sun, n), to_eye), 0.0), 60.0);
    vec3 lit = base + vec3(0.9) * spec * 0.5 + vec3(0.75) * v_foam * 0.55;
    f_colour = vec4(apply_fog_lit(lit, eye, v_world, sun), 1.0);
}
"""


#: The richer water shader: screen-space refraction, depth absorption
#: and a wave-distorted reflection.
#:
#: What the minimal one does is mix a deep colour toward the sky by a
#: Fresnel term.  That gets chop reading as chop -- a slope becomes a
#: brightness -- and it is all it does: the water is opaque, it reflects
#: a single flat sky colour, and nothing behind it is visible through
#: it.  Real water is translucent, absorbs by depth, and reflects a
#: distorted picture of what is above it.
#:
#: This wants the opaque scene already rendered, so it runs in a second
#: pass over a colour and depth texture of the world.
WATER_FRAGMENT_RICH = """#version 330
in vec3 v_world;
in vec3 v_normal;
in vec2 v_slope;
in float v_foam;
__WATER_SHARED__
out vec4 f_colour;
uniform vec3 sun;
uniform vec3 sky;
uniform vec3 eye;
uniform float far;
uniform vec3 deep;
uniform sampler2D scene;
uniform sampler2D scene_depth;
uniform vec2 viewport;
uniform float near_plane;
uniform float far_plane;
uniform float refract_scale;
uniform int reflect_steps;
//: Per metre of water.  Red goes first in both, which is what makes a
//: metre of river green and ten metres of it nearly black.
const vec3 WATER_ABSORB = vec3(0.66, 0.34, 0.22);
const vec3 WATER_SCATTER = vec3(0.11, 0.22, 0.28);
__SKY_FOG__
__WATER_SLOPE__

float linear_depth(float raw) {
    float ndc = raw * 2.0 - 1.0;
    return (2.0 * near_plane * far_plane)
         / (far_plane + near_plane - ndc * (far_plane - near_plane));
}

void main() {
    float range_to_eye = length(v_world - eye);
    vec3 n = water_normal(v_world, v_slope, range_to_eye, v_foam);
    vec3 to_eye = normalize(eye - v_world);
    vec2 uv = gl_FragCoord.xy / viewport;

    // --- refraction ----------------------------------------------------
    // The surface bends what is behind it.  Offsetting the lookup by the
    // slope is the whole of it; the offset shrinks with distance so the
    // far water does not smear, and it is clamped so a steep wave cannot
    // pull in a sample from off the far side of the screen.
    float range = max(length(v_world - eye), 1.0);
    vec2 bend = n.xy * refract_scale / range;
    bend = clamp(bend, vec2(-0.06), vec2(0.06));
    vec2 under_uv = clamp(uv + bend, vec2(0.001), vec2(0.999));

    float here = linear_depth(gl_FragCoord.z);
    float behind = linear_depth(texture(scene_depth, under_uv).r);
    // A refracted sample that turns out to be IN FRONT of the water is
    // something sticking up out of it -- a hull, a pier -- and dragging
    // that into the water is the classic artefact.  Fall back to the
    // undisplaced sample there.
    if (behind < here) {
        under_uv = uv;
        behind = linear_depth(texture(scene_depth, uv).r);
    }
    float thickness = max(behind - here, 0.0);

    vec3 under = texture(scene, under_uv).rgb;
    // Absorption and scattering are different things and were being
    // done as one.  What comes back out of water is what survived the
    // trip -- Beer-Lambert, longer wavelengths first, which is why
    // depth reads blue-green -- PLUS what the column itself scattered
    // back on the way, which is why deep water is not black.
    //
    // Written separately they behave properly at both ends: a hand's
    // depth over a pale bottom stays pale, and a channel goes to the
    // scattering colour rather than to an arbitrary "deep".
    vec3 transmitted = under * exp(-thickness * WATER_ABSORB);
    vec3 scattered = deep * (1.0 - exp(-thickness * WATER_SCATTER));
    vec3 refracted = transmitted + scattered;

    // --- reflection ----------------------------------------------------
    // A short screen-space march along the reflected ray.  It picks up
    // the bank, the bridges and the crew; where it finds nothing it
    // falls back to the sky, which is what most of a reflected ray hits
    // from 0.6 m off the water anyway.
    vec3 ray = reflect(-to_eye, n);
    // The sky as the surface sees it: brighter overhead, and darker
    // toward the horizon where a wave face turns away from it.  This is
    // where the wave-dependent distortion lives -- the reflected ray
    // swings with the surface normal, so a chop pattern becomes a
    // pattern of light, without any search at all.
    vec3 mirror = sky_colour(ray, sun);
    float hit = 0.0;
    // A reflection off a water surface goes UP.  Without this the march
    // happily finds the drowned part of the bank -- the apron carried
    // below the waterline -- and paints it on the surface as blocks.
    if (reflect_steps > 0 && ray.z > 0.02) {
        vec3 march = v_world;
        float step_len = 0.6 + range * 0.05;
        for (int i = 0; i < reflect_steps; ++i) {
            march += ray * step_len;
            vec4 clip = mvp * vec4(march, 1.0);
            if (clip.w <= 0.0) break;
            vec3 ndc = clip.xyz / clip.w;
            vec2 probe = ndc.xy * 0.5 + 0.5;
            if (probe.x < 0.0 || probe.x > 1.0
             || probe.y < 0.0 || probe.y > 1.0) break;
            float scene_z = linear_depth(texture(scene_depth, probe).r);
            float ray_z = linear_depth(ndc.z * 0.5 + 0.5);
            // Reject anything below the surface as well: only what
            // stands above the water can be reflected in it.
            if (march.z > v_world.z
             && ray_z > scene_z && ray_z - scene_z < step_len * 3.0) {
                // Fogged as the real thing is, over the path the
                // reflected ray actually travelled, so a reflected far
                // bank fades into the air exactly as the bank does.
                mirror = apply_fog_lit(texture(scene, probe).rgb,
                                       v_world, march, sun);
                // Fade the hit out at the edges of the screen, where a
                // screen-space reflection has no information and a hard
                // stop is more obvious than no reflection at all.
                vec2 edge = smoothstep(vec2(0.0), vec2(0.12), probe)
                          * smoothstep(vec2(0.0), vec2(0.12), 1.0 - probe);
                // Taken at full strength.  It was softened first, on
                // the theory that the hard patches were the march
                // overreaching -- they were not; they were the mesh
                // showing through a per-vertex normal, and softening a
                // real reflection to hide a shading bug only made the
                // reflection worse.  With the normal now per pixel the
                // march can be trusted, and only the last steps are
                // faded, where it genuinely is running out of screen.
                float trust = 1.0 - float(i) / float(reflect_steps);
                hit = edge.x * edge.y * clamp(0.55 + trust, 0.0, 1.0);
                break;
            }
            step_len *= 1.35;
        }
    }
    mirror = mix(sky_colour(ray, sun), mirror, hit);

    // --- put them together ---------------------------------------------
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, to_eye), 0.0), 5.0);
    vec3 lit = mix(refracted, mirror, clamp(fresnel, 0.0, 1.0));
    float spec = pow(max(dot(reflect(-sun, n), to_eye), 0.0), 90.0);
    lit += vec3(1.0, 0.98, 0.92) * spec * 0.65;
    lit += vec3(0.75) * v_foam * 0.55;
    f_colour = vec4(apply_fog_lit(lit, eye, v_world, sun), 1.0);
}
"""


# The shared slope code goes into both fragment shaders.  Done here
# rather than by hand in each so the two cannot drift apart.
for _name in ("FRAGMENT_SHADER", "SKY_FRAGMENT",
              "WATER_FRAGMENT", "WATER_FRAGMENT_RICH"):
    globals()[_name] = globals()[_name].replace("__SKY_FOG__", SKY_FOG_GLSL)

for _name in ("WATER_FRAGMENT", "WATER_FRAGMENT_RICH"):
    _text = globals()[_name]
    _text = _text.replace("__WATER_SHARED__",
                          WATER_UNIFORMS_GLSL + WATER_SURFACE_GLSL)
    globals()[_name] = _text.replace("__WATER_SLOPE__", WATER_SLOPE_GLSL)
del _name, _text


def _optional(program, **values) -> None:
    """Set uniforms that the compiled shader may or may not still have.

    A uniform the shader stops reading is optimised out of the program,
    and assigning to it raises.  Everything here is genuinely optional
    -- a leftover from the flat-sky days -- so a missing one is not an
    error worth stopping for.
    """
    for name, value in values.items():
        if name in program:
            program[name].value = value


def water_grid(reach: float = WATER_REACH,
               divisions: int = WATER_DIVISIONS):
    """A graded grid of triangles, in boat-relative coordinates.

    Built once; the shader moves it with the boat and lifts it onto the
    surface.  The grading is a squared warp of a uniform parameter, so
    cells near the boat are small and cells at the edge are large --
    which is where the resolution is needed and where it is not.
    """
    u = np.linspace(-1.0, 1.0, divisions + 1)
    # A blend of a linear term and a quartic, not a plain square.
    #
    # The square puts almost all of its resolution in the first few
    # metres and has spent it by ten: cells were 0.45 m at 10 m and
    # 0.55 m at 15 m, while the near-field texture behind them is baked
    # at 0.13 m -- so the grid was undersampling the wake it exists to
    # show by three or four times, right where the puddles from the last
    # few strokes sit and where a coxswain is actually looking.
    #
    # NEAR_FRACTION is how much of the reach the linear term carries.
    # It buys an almost uniform 0.19-0.31 m out to about 15 m, ahead and
    # to the sides, and pays for it in the far field, where the cells go
    # from 1.4 m to 1.9 m at a hundred metres and nothing is resolved at
    # that range anyway.
    warp = (NEAR_FRACTION * u
            + (1.0 - NEAR_FRACTION) * np.sign(u) * np.abs(u) ** 4)
    line = (reach * warp).astype("f4")
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


class FreeCamera:
    """A camera flown by hand, for looking at the course rather than rowing.

    Checking whether a bridge's piers are where the survey says, or
    whether the terrain reaches the water, means getting an eye to that
    spot -- and from the seat that costs a row up the course at racing
    speed with the crew in the way.  This detaches the eye from the boat
    entirely.

    Yaw and pitch in radians, position in world metres.  Deliberately a
    plain fly-camera with no collision and no gravity: it is an
    inspection tool, and every constraint it grew would be one more
    thing between the eye and the thing being looked at.
    """

    #: Metres per second, and what the shift key multiplies it by.
    SPEED = 26.0
    SPRINT = 6.0
    LOOK = 0.0022

    def __init__(self, eye, yaw: float = 0.0, pitch: float = 0.0):
        self.eye = np.asarray(eye, dtype=float).copy()
        self.yaw = float(yaw)
        self.pitch = float(pitch)

    def forward(self):
        return np.array([np.cos(self.pitch) * np.cos(self.yaw),
                         np.cos(self.pitch) * np.sin(self.yaw),
                         np.sin(self.pitch)])

    def turn(self, dx: float, dy: float) -> None:
        self.yaw -= dx * self.LOOK
        self.pitch = float(np.clip(self.pitch - dy * self.LOOK,
                                   -1.55, 1.55))

    def move(self, keys, dt: float, pygame) -> None:
        ahead = self.forward()
        flat = np.array([ahead[0], ahead[1], 0.0])
        norm = float(np.linalg.norm(flat))
        flat = flat / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        left = np.array([-flat[1], flat[0], 0.0])

        step = self.SPEED * dt
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            step *= self.SPRINT
        if keys[pygame.K_w]:
            self.eye += ahead * step
        if keys[pygame.K_s]:
            self.eye -= ahead * step
        if keys[pygame.K_a]:
            self.eye += left * step
        if keys[pygame.K_d]:
            self.eye -= left * step
        if keys[pygame.K_e]:
            self.eye[2] += step
        if keys[pygame.K_q]:
            self.eye[2] -= step

    def view(self):
        """``(eye, target, up)``, the same shape :func:`seat_camera` gives."""
        return (self.eye.copy(), self.eye + self.forward(),
                np.array([0.0, 0.0, 1.0]))


def seat_camera(state, boat, sway: float = 1.0, look=None):
    """``(eye, target, up)`` for whoever is looking out of the boat.

    ``look`` overrides which way they face: +1 toward the bow, -1
    astern.  A sculler sits facing astern and glances over a shoulder,
    which is what that override is for.

    A bow-loader puts the cox in the bow, lying back: the seat is at
    ``rig.coxswain_position`` and the eye ``rig.coxswain_eye_height``
    above it.  Roll and pitch are carried through, because the horizon
    tipping with the boat is most of what tells you the boat is alive.
    """
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    seat, eye_height, facing = viewpoint(boat)
    seat = np.asarray(seat, dtype=float).copy()
    seat[2] += float(eye_height)
    position = np.asarray(state[0:3], dtype=float)
    eye = position + rotation @ (seat * np.array([1.0, sway, 1.0]))
    if look is not None:
        facing = float(look)
    forward = rotation @ np.array([float(facing), 0.0, 0.0])
    up = rotation @ np.array([0.0, 0.0, 1.0])
    return eye, eye + forward, up


def crew_poses(boat, samples: int = 48):
    """Crew and oars through one stroke cycle, baked once.

    Solving the joint chain for a whole crew costs about 12 ms, which is
    most of a frame at 60 Hz and would show as stutter.  The motion is
    periodic, so it is solved at ``samples`` phases up front and indexed
    per frame instead -- the only per-frame cost is the same rotation
    the hull already pays.

    Returns ``(poses, colours)`` or ``None`` if the crew cannot be drawn
    with a constant vertex count, in which case the caller simply leaves
    them out rather than uploading ragged buffers.
    """
    period = float(boat.timing.period)
    if period <= 0.0 or not getattr(boat, "crew", None):
        return None
    poses, colours = [], None
    for index in range(int(samples)):
        when = period * index / float(samples)
        pieces = [part for part in (crew_solids(boat, when),
                                    oar_solids(boat, when))
                  if part is not None]
        if not pieces:
            return None
        poses.append(np.concatenate([part.vertices for part in pieces]))
        colours = np.concatenate([part.colours for part in pieces])
    if len({len(pose) for pose in poses}) != 1:
        return None
    return np.asarray(poses, dtype="f4"), colours


def boat_geometry(boat, hull, t, state, crew=None):
    """The shell and its oars, in world space, for this frame.

    Both ride the hull, so both are built in hull coordinates and carried
    through the same rotation the camera uses.  That is what makes the
    bow sit still in the frame while the world swings behind it -- which
    is the whole cue a coxswain steers on.
    """
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    position = np.asarray(state[0:3], dtype=float)
    pieces = [(hull.vertices @ rotation.T + position, hull.colours)]
    if crew is not None:
        poses, crew_colours = crew
        period = float(boat.timing.period)
        # Nearest baked phase.  At 48 samples and rate 30 that is 40 ms
        # of stroke per pose, which is below what the eye resolves on a
        # body moving this slowly.
        index = int((t % period) / period * len(poses)) % len(poses)
        pieces.append((poses[index] @ rotation.T + position, crew_colours))
    vertices = np.concatenate([p[0] for p in pieces]).astype("f4")
    shades = np.concatenate([p[1] for p in pieces]).astype("f4")
    return vertices, shades


def oar_geometry(boat, t, state):
    """Oars as thin horizontal ribbons -- SUPERSEDED, kept for the plan view.

    This drew each oar as a flat quad at a constant 0.32 m, which is
    invisible from the seat: a horizontal ribbon seen from a horizontal
    eye is a line, and the blades never touched the water.  The first
    person view now uses :func:`coxswain.viz.worldmesh.oar_solids`.
    """
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


def run_setup_menu(screen, args):
    """Modal setup menu on a plain pygame window.

    Returns the chosen settings, or ``None`` if the window was closed.
    This runs *before* the GL context exists, because the course it
    picks decides what world to build.
    """
    import pygame

    clock = pygame.time.Clock()
    if not getattr(args, "no_sound", False):
        start_music()
    font = pygame.font.SysFont("dejavusans,arial", 22)
    small = pygame.font.SysFont("dejavusans,arial", 15)
    menu = setup_menu(boat=args.boat, course=args.race, rate=args.rate,
                      wind=args.wind)
    chosen = menu.settings()
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    showing_controls = False
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_music()
                return None
            if event.type == pygame.KEYDOWN:
                if showing_controls:
                    showing_controls = False       # any key goes back
                    continue
                if event.key == pygame.K_ESCAPE:
                    stop_music()
                    return None
                action = handle_key(menu, event.key)
                if action == "controls":
                    showing_controls = True
                    continue
                if action == "options":
                    chosen = menu.settings()
                    menu = options_menu(audio=args.audio,
                                        quality=args.quality,
                                        weather=args.weather)
                    continue
                if action == "back":
                    picked = menu.settings()
                    args.audio = picked["audio"]
                    args.quality = picked["quality"]
                    args.weather = picked["weather"]
                    menu = setup_menu(boat=chosen["boat"],
                                      course=chosen["race"],
                                      rate=chosen["rate"],
                                      wind=chosen["wind"])
                    continue
                if action == "start":
                    # NOT stopped here: the world takes half a minute to
                    # build after this, and silence landing the instant
                    # you press go is the loudest possible signal that
                    # something has died.  It plays over the loading
                    # screen and fades when the boat appears.
                    return menu.settings()
                if action == "quit":
                    stop_music()
                    return None
        # Behind the menu: the soundings for whichever course is
        # highlighted, so the backdrop changes as you choose and is a
        # chart of somewhere real rather than a flat colour.
        showing = menu.settings()
        if "race" in showing:
            chosen = showing
        chart = chart_surface(chosen.get("race"), screen.get_size())
        if chart is not None:
            screen.blit(chart, (0, 0))
        else:
            screen.fill((18, 24, 29))
        if showing_controls:
            draw_controls(overlay, font, small, screen.get_size())
        else:
            draw_menu(overlay, menu, font, small, screen.get_size())
        screen.blit(overlay, (0, 0))
        pygame.display.flip()


LOADING_TIPS = (
    "The stick stays where you put it.  Nothing re-centres it for you.",
    "A sweep four turns toward the stroke side with the rudder centred.",
    "Press M to hand steering between the mouse and the arrow keys.",
    "Escape pauses: stroke rate, wind, restart, or change boat.",
    "The stroke sound is recorded from real outings, not synthesised.",
    "In an eight you sit in the stern and the crew face you.",
)


def run_loading(screen, label, work):
    """Run ``work()`` on a worker thread behind an animated screen.

    Building a course is half a minute of terrain, buildings and trees.
    Done on the main thread the window stops repainting, Windows paints
    it grey and titles it "not responding", and it reads exactly like a
    crash -- which, for someone who was handed an unsigned exe and told
    to click through a security warning, is the moment they give up.

    So the build goes to a thread and this draws while it runs.  The
    work is pure computation handing back arrays; it touches no GL
    context, which is what makes it safe to move off the main thread.
    """
    import threading

    import pygame          # module-level pygame does not exist here

    done, failed = {}, {}

    def run():
        try:
            done["value"] = work()
        except BaseException as exc:                  # re-raised below
            failed["value"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    clock = pygame.time.Clock()
    title = pygame.font.SysFont("dejavusans,arial", 26)
    small = pygame.font.SysFont("dejavusans,arial", 15)
    width, height = screen.get_size()
    started = time.perf_counter()
    tip = LOADING_TIPS[int(started) % len(LOADING_TIPS)]

    while thread.is_alive():
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass                # the build finishes; quitting is later
        elapsed = time.perf_counter() - started
        screen.fill((18, 24, 29))
        text = title.render(label, True, (233, 240, 245))
        screen.blit(text, ((width - text.get_width()) // 2, height // 2 - 60))

        # An indeterminate bar: a band sliding inside a track.  Honest,
        # because there is no progress number to report -- a fake
        # percentage that stalls at 90 is worse than none.
        track = pygame.Rect((width - 380) // 2, height // 2, 380, 6)
        pygame.draw.rect(screen, (44, 54, 62), track)
        span = 120
        travel = (track.width + span) * ((elapsed * 0.45) % 1.0) - span
        band = pygame.Rect(track.x + max(0, int(travel)), track.y,
                           int(min(span, min(track.width - travel, travel + span))),
                           track.height)
        if band.width > 0:
            pygame.draw.rect(screen, (255, 146, 72), band)

        for row, line in ((0, "%.0f seconds" % elapsed), (1, tip)):
            text = small.render(line, True, (150, 164, 176))
            screen.blit(text, ((width - text.get_width()) // 2,
                               height // 2 + 28 + row * 26))
        pygame.display.flip()

    thread.join()
    if "value" in failed:
        raise failed["value"]
    return done.get("value")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default="charles",
                        choices=("charles", "totl", "hotl"))
    parser.add_argument("--boat", default="4+",
                        choices=("4+", "8+", "2x", "1x"))
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--weather", default="hazy",
                        choices=tuple(WEATHER),
                        help="what the air is doing")
    parser.add_argument("--quality", default="standard",
                        choices=("minimal", "standard", "high"),
                        help="water detail against frame rate")
    parser.add_argument("--samples", type=int, default=4,
                        help="multisample anti-aliasing; 0 turns it off")
    parser.add_argument("--freecam", action="store_true",
                        help="fly the camera around the course instead of "
                             "sitting in the boat")
    parser.add_argument("--no-menu", action="store_true",
                        help="skip the setup menu and use the flags")
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
    parser.add_argument("--audio", default="full",
                        choices=("events", "full", "off"),
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

    # Defined before the branch, not inside it: the loading screen below
    # tests ``screen is not None`` and --no-menu skips this block
    # entirely, so leaving it to be assigned here is a NameError on
    # exactly the path the packaging tests use.
    screen = None
    if not args.shot and not args.no_menu:
        # A plain window first, so a coxswain can choose a boat and a
        # course without knowing what a command-line flag is.  The GL
        # context comes afterwards, over the same window.
        #
        # Imported here and again below rather than at the top of the
        # file: a ``--shot`` run never opens a window, and the import
        # is what pulls in SDL.  Both sites need their own statement --
        # the one further down is inside an ``else``, so it does not run
        # for this block, and without this line ``pygame`` is a local
        # that is read here before it is ever assigned.
        import pygame

        pygame.init()
        pygame.display.set_caption("Coxswain")
        screen = pygame.display.set_mode((args.width, args.height))
        picked = run_setup_menu(screen, args)
        if picked is None:
            pygame.quit()
            return 0
        args.boat, args.race = picked["boat"], picked["race"]
        args.rate, args.wind = picked["rate"], picked["wind"]

    print("building %s ..." % args.race)
    clock0 = time.perf_counter()

    def build_everything():
        """All the heavy work, and none of it touching GL."""
        # The sea first: the flat far-water quad has to be sunk below
        # the deepest trough of the near field, or it hides them.
        sea = sea_for(args.wind, args.fetch, np.radians(args.wind_from))
        trough = float(np.sum(sea.amplitude)) if len(sea.amplitude) else 0.0
        mesh, scene = build_world(args.race, reach=args.reach, step=args.step,
                                  water_level=-1.25 * trough - 0.02,
                                  with_buildings=not args.no_buildings,
                                  guide=not args.no_guide,
                                  trees=not args.no_trees)
        return sea, trough, mesh, scene, build_boat(args.boat, args.rate)

    divisions, keep_trees, rich_water = quality_settings(args.quality)
    if not keep_trees:
        args.no_trees = True

    label = "Building %s" % dict(
        charles="the Charles", totl="Tail of the Lake",
        hotl="Head of the Lake").get(args.race, args.race)
    if screen is not None:
        sea, trough, mesh, scene, (boat, made) = run_loading(
            screen, label, build_everything)
    else:
        sea, trough, mesh, scene, (boat, made) = build_everything()
    print("   %d triangles in %d parts, %.1f s"
          % (mesh.triangles, len(mesh.parts), time.perf_counter() - clock0))
    # The menu music has been playing over the loading screen; fade it
    # out now the world exists, so the stroke is the first thing heard.
    if screen is not None:
        stop_music(900)

    if made != args.boat:
        print("   (no %s in the catalog; rowing a %s)" % (args.boat, made))
    args.boat = made
    live = LiveControl()
    cox = Coxswain(rudder_override=live.rudder, pressure_split=live.split)
    simulator = RowingSimulator(boat, coxswain=cox, fast=True)
    hull = hull_solid(boat)
    crew = crew_poses(boat)
    # Set here rather than beside the rest of the loop state: ``draw``
    # closes over both and is called once before the loop starts, so
    # assigning them later is an unbound free variable on the first
    # frame -- which is the same mistake, in the same file, as the
    # pygame import that broke every launch.
    steers_with_rudder = boat.rig.has_coxswain
    looking_ahead = False
    showing_controls = False
    freecam = None

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
    if args.freecam:
        # Started here rather than with the rest of the loop state:
        # fresh_state is defined just above, and reaching for it earlier
        # is an unbound local -- the third time that exact shape of
        # mistake has been made in this file.
        _eye, _target, _up = seat_camera(fresh_state(), boat)
        freecam = FreeCamera(_eye + np.array([0.0, 0.0, 6.0]), pitch=-0.15)

    # The stroke, out loud.  From the bow of a four you cannot see the
    # blades go in, and without the catch there is nothing in the seat
    # view that separates drive from recovery -- which makes calling the
    # boat impossible, and calling is what this is for.
    audio = None
    # "off" is a mode in the menu and a way of saying no sound at all.
    if args.audio == "off":
        args.no_sound = True
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
        # Multisampling.  Nearly everything in this scene is a long
        # near-horizontal edge -- the gunwale, the oar looms, the far
        # bank, the bridge chords -- and those are the worst case for
        # aliasing: a one-pixel-wide edge crawling as the boat yaws.
        # MSAA is asked for on the framebuffer rather than done in a
        # shader because the driver does it for free on the edges, which
        # is exactly where the problem is.  Requested, not required: if
        # the Intel UHD part will not give 4 samples, the context still
        # comes up (see the fallback below) rather than failing to start
        # on the one machine this has to run on.
        if args.samples > 0:
            pygame.display.gl_set_attribute(
                pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(
                pygame.GL_MULTISAMPLESAMPLES, int(args.samples))
        pygame.display.set_caption("%s -- the seat" % scene.name)
        try:
            screen = pygame.display.set_mode((args.width, args.height),
                                             pygame.OPENGL | pygame.DOUBLEBUF)
        except pygame.error:
            # No multisample visual: drop it and take the jaggies rather
            # than not starting.
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
            screen = pygame.display.set_mode((args.width, args.height),
                                             pygame.OPENGL | pygame.DOUBLEBUF)
        ctx = moderngl.create_context()
        got = pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES)
        if got:
            ctx.enable(moderngl.DEPTH_TEST)
            try:
                ctx.multisample = True
            except Exception:
                pass
        print("   %dx multisampling" % got if got
              else "   no multisampling available")
        target = ctx.screen

    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.cull_face = "back"

    # The rich water shader reads the opaque scene, so it cannot be in
    # the same pass as the thing it reads.
    #
    # Everything draws into ONE multisampled buffer.  After the world is
    # laid down, its colour and depth are resolved out into plain
    # textures for the water to sample; the water and the boat then go
    # on drawing into that same multisampled buffer, which still holds
    # the world's depth, so the water tests against the bank it is
    # lapping and composites over the shore for free.  The result is
    # resolved to the screen at the end.
    #
    # A multisampled depth buffer cannot be read with texture() in
    # GL 3.3, so the resolve was needed whatever happened; doing it this
    # way also keeps the antialiasing, which the first version of this
    # threw away by rendering the whole scene to plain textures.
    scene_fbo = resolve_fbo = None
    scene_colour = scene_depth_tex = None
    # Enabled headless too: --shot is how this gets looked at without a
    # person watching, and a water shader that cannot be screenshotted
    # cannot be checked.
    if rich_water:
        size = (args.width, args.height)
        samples = max(int(args.samples), 0)
        if samples > 0:
            try:
                scene_fbo = ctx.framebuffer(
                    color_attachments=[ctx.renderbuffer(size, 3,
                                                        samples=samples)],
                    depth_attachment=ctx.depth_renderbuffer(
                        size, samples=samples))
            except Exception as error:
                print("   no multisampled buffer (%s); water is aliased"
                      % type(error).__name__)
                samples = 0
        if samples == 0:
            scene_fbo = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer(size, 3)],
                depth_attachment=ctx.depth_renderbuffer(size))
        scene_colour = ctx.texture(size, 3)
        scene_colour.filter = (moderngl.LINEAR, moderngl.LINEAR)
        scene_colour.repeat_x = scene_colour.repeat_y = False
        scene_depth_tex = ctx.depth_texture(size)
        scene_depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        scene_depth_tex.repeat_x = scene_depth_tex.repeat_y = False
        resolve_fbo = ctx.framebuffer(color_attachments=[scene_colour],
                                      depth_attachment=scene_depth_tex)
        print("   rich water on a %s buffer"
              % ("%dx multisampled" % samples if samples else "plain"))
    program = ctx.program(vertex_shader=VERTEX_SHADER,
                          fragment_shader=FRAGMENT_SHADER)
    sky_prog = ctx.program(vertex_shader=SKY_VERTEX,
                           fragment_shader=SKY_FRAGMENT)
    sky_quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3],
                                   dtype="f4").tobytes())
    sky_vao = ctx.vertex_array(sky_prog, [(sky_quad, "2f", "in_pos")])

    def set_sky(prog, weather=None):
        """Every shader that fogs or reflects shares one sky."""
        row = WEATHER.get(weather or args.weather, WEATHER["hazy"])
        zenith, horizon, glow, density, height, scatter = row
        _optional(prog, sky_zenith=zenith, sky_horizon=horizon,
                  sun_glow=glow, fog_density=density, fog_height=height,
                  fog_scatter=scatter)
    program["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                 / np.linalg.norm([0.42, 0.30, 0.85]))
    # Set only if the shader still wants them.  Both were the flat sky
    # colour and the linear fog range; with the gradient and the
    # exponential fog in, GLSL drops whichever a shader no longer reads,
    # and assigning to a dropped uniform is a KeyError.
    _optional(program, sky=SKY, far=FAR)

    static = []
    for part in mesh.parts:
        buffer = ctx.buffer(part.interleaved().tobytes())
        static.append(ctx.vertex_array(
            program, [(buffer, "3f 3f 3f", "in_pos", "in_normal",
                       "in_colour")]))

    # -- the water ------------------------------------------------------
    water_prog = ctx.program(
        vertex_shader=WATER_VERTEX,
        fragment_shader=(WATER_FRAGMENT_RICH if scene_fbo is not None
                         else WATER_FRAGMENT))
    water_prog["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                    / np.linalg.norm([0.42, 0.30, 0.85]))
    _optional(water_prog, sky=SKY, far=FAR)
    water_prog["deep"].value = (0.055, 0.115, 0.155)
    water_prog["hull_length"].value = float(boat.length)
    for _prog in (program, sky_prog, water_prog):
        set_sky(_prog)
    # The second normal and the per-pixel surface are the two things
    # that cost real fragment work, so they are what the quality setting
    # actually buys.  Minimal keeps a short exact radius -- the water
    # right under the eye still has to be smooth, because the mesh
    # showing through it is the most obvious fault there is -- and no
    # ripple at all, which is six simplex evaluations a fragment saved.
    _ripple, _exact = {
        "minimal": (0.0, 7.0),
        "standard": (RIPPLE_SLOPE, 16.0),
        "high": (RIPPLE_SLOPE, 26.0),
    }.get(args.quality, (RIPPLE_SLOPE, 16.0))
    _optional(water_prog, ripple_slope_amp=_ripple,
              ripple_scale=RIPPLE_SCALE, ripple_fade=RIPPLE_FADE,
              exact_within=_exact, ripple_wake_gain=RIPPLE_WAKE_GAIN)
    #: Texture units 0-2 are the HUD, the near field and the wave table.
    SCENE_UNIT, DEPTH_UNIT = 3, 4
    if scene_fbo is not None:
        water_prog["scene"].value = SCENE_UNIT
        water_prog["scene_depth"].value = DEPTH_UNIT
        water_prog["viewport"].value = (float(args.width), float(args.height))
        water_prog["near_plane"].value = 0.25
        water_prog["far_plane"].value = float(FAR)
        # How hard the surface bends what is behind it, before the 1/range
        # falloff.  Set by eye against the near water: enough that a wave
        # visibly displaces the bank behind it, not so much that the
        # shoreline swims.
        water_prog["refract_scale"].value = 5.5
        water_prog["reflect_steps"].value = 16
    water_prog["wind_to"].value = float(np.radians(args.wind_from) + np.pi)
    wave = load_wavefield(shell_of(boat))
    if wave is not None:
        w_speeds, w_east, w_north, w_field = wave
        wave_tex = ctx.texture3d(
            (len(w_east), len(w_north), len(w_speeds)), 1,
            np.ascontiguousarray(w_field, dtype="f4"), dtype="f4")
        wave_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        wave_tex.repeat_x = wave_tex.repeat_y = wave_tex.repeat_z = False
        wave_tex.use(WAVE_UNIT)
        water_prog["wave_map"].value = WAVE_UNIT
        water_prog["wave_lo"].value = (float(w_east[0]), float(w_north[0]))
        water_prog["wave_hi"].value = (float(w_east[-1]), float(w_north[-1]))
        water_prog["wave_size"].value = (float(len(w_east)),
                                         float(len(w_north)),
                                         float(len(w_speeds)))
        water_prog["wave_speed_lo"].value = float(w_speeds[0])
        water_prog["wave_speed_hi"].value = float(w_speeds[-1])
        print("   wave field: %d slices %.1f..%.1f m/s, %dx%d, "
              "%+.3f..%+.3f m at 4.5 m/s -- from the Green's function"
              % (len(w_speeds), w_speeds[0], w_speeds[-1], len(w_east),
                 len(w_north),
                 w_field[np.argmin(np.abs(w_speeds - 4.5))].min(),
                 w_field[np.argmin(np.abs(w_speeds - 4.5))].max()))
    else:
        wave_tex = ctx.texture3d((2, 2, 2), 1, np.zeros(8, dtype="f4"),
                                 dtype="f4")
        wave_tex.use(WAVE_UNIT)
        water_prog["wave_map"].value = WAVE_UNIT
        water_prog["wave_lo"].value = (-1.0, -1.0)
        water_prog["wave_hi"].value = (1.0, 1.0)
        water_prog["wave_size"].value = (2.0, 2.0, 2.0)
        water_prog["wave_speed_lo"].value = 1e9
        water_prog["wave_speed_hi"].value = 2e9
        print("   wave field: not baked (run tools/bake_nearfield.py)")
    near = load_nearfield(shell_of(boat))
    if near is not None:
        n_east, n_north, n_field = near
        near_tex = ctx.texture((len(n_east), len(n_north)), 1,
                               np.ascontiguousarray(n_field, dtype="f4"),
                               dtype="f4")
        near_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        near_tex.repeat_x = near_tex.repeat_y = False
        # Texture unit ONE, not zero, and re-bound every frame below.
        #
        # This was bound to unit 0 once at setup, and the HUD overlay is
        # blitted from unit 0 every frame.  From the second frame on the
        # water shader was therefore sampling the HUD as the hull's
        # near-field elevation: a white pixel reads 1.0, times U^2/g, is
        # a 2.5 m wave -- in the boat frame, so it followed the hull, and
        # to one side, because that is where the bright pixels map into
        # the near-field box.  It never showed in a headless render
        # because headless mode never draws the HUD, which is why every
        # probe of the height field came back clean.
        near_tex.use(NEAR_UNIT)
        water_prog["near_map"].value = NEAR_UNIT
        water_prog["near_lo"].value = (float(n_east[0]), float(n_north[0]))
        water_prog["near_hi"].value = (float(n_east[-1]), float(n_north[-1]))
        water_prog["near_size"].value = (float(len(n_east)),
                                         float(len(n_north)))
        print("   near field: baked %dx%d, %+.1f..%+.1f m along the hull, "
              "peak %+.3f m at 4.5 m/s"
              % (len(n_east), len(n_north), n_east[0], n_east[-1],
                 float(np.abs(n_field).max()) * 4.5 ** 2 / 9.80665))
    else:
        near_tex = ctx.texture((2, 2), 1, np.zeros(4, dtype="f4"), dtype="f4")
        near_tex.use(NEAR_UNIT)
        water_prog["near_map"].value = NEAR_UNIT
        water_prog["near_lo"].value = (-1.0, -1.0)
        water_prog["near_hi"].value = (1.0, 1.0)
        water_prog["near_size"].value = (2.0, 2.0)
        print("   near field: not baked (run tools/bake_nearfield.py)")
    grid = water_grid(divisions=divisions)
    water_buffer = ctx.buffer(grid.tobytes())
    water_vao = ctx.vertex_array(water_prog,
                                 [(water_buffer, "2f", "in_grid")])
    field = sea
    water_prog["waves"].write(field.as_uniform().tobytes())
    trail = PuddleTrail()

    print("   water: H_s %.3f m, T_p %.2f s at %.0f m/s over %.0f m fetch; "
          "%d triangles; far plane sunk to %.3f m"
          % (field.significant_height, field.peak_period, args.wind,
             args.fetch, len(grid) // 3, -1.25 * trough - 0.02))

    # The oars change every frame, so they get a stream buffer sized once.
    # Hull, oars and eight bodies.  The draw is skipped outright if
    # the blob will not fit, so an undersized buffer here shows as
    # the boat vanishing rather than as an error.
    oar_buffer = ctx.buffer(reserve=1024 * 1024, dynamic=True)
    oar_vao = ctx.vertex_array(
        program, [(oar_buffer, "3f 3f 3f", "in_pos", "in_normal",
                   "in_colour")])

    projection = perspective(args.fov, args.width / args.height, 0.25, FAR)
    # Set on the function so the closure can carry state without a global.
    _ = None

    def draw(state, t):
        """One frame.  ``draw.last_phase`` remembers where in the stroke
        the previous frame was, so a catch can be detected by the wrap."""
        # Facing astern in a scull, unless you are looking over your
        # shoulder to see where you are going.
        if freecam is not None:
            eye, target_point, up = freecam.view()
        else:
            _seat, _height, facing = viewpoint(boat)
            look = -facing if looking_ahead else facing
            eye, target_point, up = seat_camera(state, boat, look=look)
        view = look_at(eye, target_point, up)
        program["mvp"].write((projection @ view).T.tobytes(order="C"))
        program["eye"].value = tuple(float(v) for v in eye)
        # Pass one: the opaque world, into its own buffer when the
        # water is going to read it back.
        first = scene_fbo if scene_fbo is not None else target
        first.use()
        first.clear(SKY[0], SKY[1], SKY[2], 1.0)
        # The dome first, behind everything.  Depth test off so it fills
        # the frame, depth write off so it does not occlude the world.
        inverse_vp = np.linalg.inv(projection @ view)
        sky_prog["inverse_vp"].write(inverse_vp.T.astype("f4")
                                     .tobytes(order="C"))
        sky_prog["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                      / np.linalg.norm([0.42, 0.30, 0.85]))
        ctx.disable(moderngl.DEPTH_TEST)
        sky_vao.render()
        ctx.enable(moderngl.DEPTH_TEST)
        for vao in static:
            vao.render()
        if scene_fbo is not None:
            # Carry the world forward into the buffer the water draws
            # into, so the water reads an untouched copy of what is
            # behind it rather than the surface it is drawing.
            ctx.copy_framebuffer(resolve_fbo, scene_fbo)
            scene_fbo.use()
        # The water goes on after the land, so the shore reads through it
        # at the edges and the patch does not have to be clipped.
        speed = float(np.hypot(state[6], state[7]))
        water_prog["mvp"].write((projection @ view).T.tobytes(order="C"))
        water_prog["eye"].value = tuple(float(v) for v in eye)
        water_prog["centre"].value = (float(state[0]), float(state[1]))
        water_prog["boat"].value = (float(state[0]), float(state[1]),
                                    float(state[5]))
        water_prog["speed"].value = speed
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
        near_tex.use(NEAR_UNIT)          # never trust the binding
        wave_tex.use(WAVE_UNIT)
        if scene_fbo is not None:
            scene_colour.use(SCENE_UNIT)
            scene_depth_tex.use(DEPTH_UNIT)
        water_vao.render()
        vertices, colours = boat_geometry(boat, hull, t, state, crew)
        if vertices is not None and len(vertices):
            normals = _face_normals(vertices)
            blob = np.hstack([vertices, normals, colours]).astype("f4")
            if blob.nbytes <= oar_buffer.size:
                oar_buffer.write(blob.tobytes())
                oar_vao.render(vertices=len(vertices))
        if scene_fbo is not None:
            # And out to the screen, so the HUD has something to sit on.
            ctx.copy_framebuffer(target, scene_fbo)
            target.use()

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
    menu, restart_session = None, False
    if args.control == "mouse":
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos((args.width // 2, args.height // 2))

    while running:
        frame = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if showing_controls:
                    showing_controls = False
                    continue
                if menu is not None:
                    action = handle_key(menu, event.key)
                    if event.key == pygame.K_ESCAPE:
                        action = "resume"
                    if action == "resume":
                        settings = menu.settings()
                        if abs(settings["rate"] - args.rate) > 1e-9:
                            args.rate = settings["rate"]
                            # Rebuilding the boat restarts the crew's
                            # cycle, so the phase jumps once.  That is a
                            # real transient and not worth hiding; the
                            # alternative is retiming mid-stroke, which
                            # puts a step in the force.
                            boat, _made = build_boat(args.boat, args.rate)
                            hull = hull_solid(boat)
                            crew = crew_poses(boat)
                            simulator = RowingSimulator(boat, coxswain=cox,
                                                        fast=True)
                            loop.simulator = simulator
                            if audio is not None:
                                audio.boat = boat
                        if abs(settings["wind"] - args.wind) > 1e-9:
                            args.wind = settings["wind"]
                            water_prog["waves"].write(
                                sea_for(args.wind, args.fetch,
                                        np.radians(args.wind_from))
                                .as_uniform().tobytes())
                        menu, paused = None, False
                    elif action == "restart":
                        loop.start(fresh_state())
                        rudder = split = 0.0
                        menu, paused = None, False
                    elif action == "options":
                        menu = options_menu(audio=args.audio,
                                            quality=args.quality,
                                            weather=args.weather)
                    elif action == "back":
                        picked = menu.settings()
                        if picked["audio"] != args.audio:
                            args.audio = picked["audio"]
                            # Rebuilt rather than retuned: the mode
                            # decides which envelope is loaded.
                            if args.audio == "off":
                                audio = None
                            else:
                                from coxswain.viz.strokeaudio import StrokeAudio
                                audio = StrokeAudio(boat, mode=args.audio)
                        args.quality = picked["quality"]
                        if picked["weather"] != args.weather:
                            # Weather is only uniforms, so it can change
                            # mid-outing without rebuilding anything.
                            args.weather = picked["weather"]
                            for _p in (program, sky_prog, water_prog):
                                set_sky(_p, args.weather)
                        menu = pause_menu(rate=args.rate, wind=args.wind)
                    elif action == "controls":
                        showing_controls = True
                    elif action in ("setup", "quit"):
                        restart_session = action == "setup"
                        running = False
                    continue
                if event.key == pygame.K_ESCAPE:
                    menu = pause_menu(rate=args.rate, wind=args.wind)
                    paused = True
                elif event.key == pygame.K_q and freecam is None:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    rudder = 0.0
                elif event.key == pygame.K_F1:
                    # Detach the eye from the boat, and put it back.
                    if freecam is None:
                        here, _t, _u = seat_camera(loop.pose(), boat)
                        freecam = FreeCamera(here + np.array([0.0, 0.0, 4.0]),
                                             pitch=-0.12)
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)
                    else:
                        freecam = None
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(args.control != "mouse")
                elif event.key == pygame.K_v:
                    # Over the shoulder.  In a scull this is the only
                    # way to see where you are going; in a coxed boat it
                    # turns round and shows you the crew from in front.
                    looking_ahead = not looking_ahead
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
        if freecam is not None:
            # The flying camera takes the whole keyboard and the mouse:
            # WASD share letters with the steering and the pressure
            # split, so they cannot both be live.
            freecam.move(keys, frame, pygame)
            dx, dy = pygame.mouse.get_rel()
            freecam.turn(float(dx), float(dy))
        elif args.control == "mouse":
            offset = (pygame.mouse.get_pos()[0] - args.width * 0.5) \
                / (args.width * MOUSE_SPAN * 0.5)
            rudder = float(np.clip(offset, -1.0, 1.0)) * RUDDER_LIMIT
        else:
            turn = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - \
                   (keys[pygame.K_LEFT] or keys[pygame.K_a])
            if turn:
                rudder = float(np.clip(rudder + turn * RUDDER_RATE * frame,
                                       -RUDDER_LIMIT, RUDDER_LIMIT))
        press = 0 if freecam is not None else             keys[pygame.K_e] - keys[pygame.K_w]
        if press:
            split = float(np.clip(split + press * SPLIT_RATE * frame,
                                  -SPLIT_LIMIT, SPLIT_LIMIT))
        else:
            split -= np.sign(split) * min(abs(split), SPLIT_RETURN * frame)
        if not steers_with_rudder:
            rudder = 0.0
        live.set(ControlInput(rudder=rudder, pressure_split=split))

        if args.autopilot:
            gap = course - loop.state[0:2]
            index = int(np.argmin(np.einsum("ij,ij->i", gap, gap)))
            ahead = course[min(index + 12, len(course) - 1)] - loop.state[0:2]
            want = float(np.arctan2(ahead[1], ahead[0]))
            error = (want - float(loop.state[5]) + np.pi) % (2 * np.pi) - np.pi
            rudder = float(np.clip(-1.4 * error, -RUDDER_LIMIT, RUDDER_LIMIT))
            live.set(ControlInput(rudder=rudder))

        if not paused and menu is None and not showing_controls:
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
                 ("stick %+.0f%% (%+.1f deg)   yaw %+.2f deg/s"
                  % (100 * rudder / RUDDER_LIMIT, math.degrees(rudder),
                     math.degrees(pose[11]))
                  if steers_with_rudder else
                  "no rudder -- steer on pressure (W / E)"
                  "   yaw %+.2f deg/s" % math.degrees(pose[11])),
                 "roll %+.1f deg   pitch %+.1f deg   %.0f fps"
                 % (math.degrees(pose[3]), math.degrees(pose[4]),
                    clock.get_fps()),
                 # Nobody guesses this, and without it the pause menu
                 # and everything in it may as well not exist.
                 ("F1 free camera    Esc menu    V look astern"
                  if freecam is None else
                  "FREE CAMERA  WASD move  QE down/up  shift fast  "
                  "F1 back to the boat")]
        if menu is not None or showing_controls:
            if showing_controls:
                draw_controls(overlay, font, font,
                              (args.width, args.height))
            else:
                draw_menu(overlay, menu, font, font,
                          (args.width, args.height))
            hud_texture.write(_surface_bytes(pygame, overlay))
            hud_texture.use(0)
            _blit(ctx, hud_texture)
            pygame.display.flip()
            frames += 1
            continue

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
    if restart_session:
        # "Change boat or course": the world has to be rebuilt, so the
        # session starts again from the top rather than being patched.
        return main(argv)
    return 0


def _surface_bytes(pygame, surface):
    """RGBA bytes from a surface, flipped for GL's origin.

    ``tostring`` is deprecated in favour of ``tobytes`` and only one of
    them exists depending on the pygame; ask for whichever is there.
    """
    getter = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    return getter(surface, "RGBA", True)


_BLIT = {}


def _opaque_blit(ctx, unit: int) -> None:
    """Copy a full-screen colour texture, ignoring depth entirely.

    Used to carry one framebuffer's colour into the next.  Depth test
    and depth write are both off: the depth buffer being copied *into*
    is the world's, and it has to survive so the water still tests
    against the bank.
    """
    import moderngl

    if "opaque" not in _BLIT:
        program = ctx.program(
            vertex_shader='''#version 330
            in vec2 in_pos; out vec2 uv;
            void main(){ uv = in_pos * 0.5 + 0.5;
                         gl_Position = vec4(in_pos, 0.0, 1.0); }''',
            fragment_shader='''#version 330
            in vec2 uv; out vec4 f; uniform sampler2D image;
            void main(){ f = vec4(texture(image, uv).rgb, 1.0); }''')
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        buffer = ctx.buffer(quad.tobytes())
        _BLIT["opaque"] = ctx.vertex_array(program,
                                           [(buffer, "2f", "in_pos")])
        _BLIT["opaque_program"] = program
    # DEPTH_TEST off is enough on its own: with the test disabled the
    # depth buffer is not written either, so the world's depth survives
    # the copy.  (moderngl has no depth_mask on the context.)
    ctx.disable(moderngl.DEPTH_TEST)
    _BLIT["opaque_program"]["image"].value = int(unit)
    _BLIT["opaque"].render()
    ctx.enable(moderngl.DEPTH_TEST)


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
