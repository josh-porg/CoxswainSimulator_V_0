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
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.core.frames import hull_to_abs                # noqa: E402
from coxswain.sim.control import (Coxswain,
                                  balance_for_experience)                   # noqa: E402
from coxswain.sim.realtime import (ControlInput,            # noqa: E402
                                   FixedStepLoop, LiveControl)
from coxswain.sim.simulator import RowingSimulator          # noqa: E402
from coxswain.viz.menu import (build_boat, chart_surface,    # noqa: E402
                               confirm_quit_menu,
                               draw_controls, draw_menu,
                               handle_key, options_menu, pause_menu,
                               quality_settings, setup_menu,
                               rowers_menu, start_music,
                               stop_music, weather_menu)
from coxswain.viz.rigview import (PRESETS as RIG_PRESETS,   # noqa: E402
                                  FIELDS, PANE_ROWS, RIGS, SHELLS,
                                  draw_plan, draw_side_pane, field_text)
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
#: ``(zenith, horizon, glow, fog_density, fog_height, scatter,
#: overcast)`` -- the last being how much of the sky is cloud, which
#: sets how much low-frequency texture the dome takes.  Fog
#: density is per metre; the reciprocal is roughly the range at which a
#: dark object is half lost, so 1/2600 is a clear morning and 1/420 is
#: the kind of river fog that has crews rowing on sound.
WEATHER = {
    # Even a clear day has air in it.  With no fog at all the far bank
    # sits at the same apparent distance as the near one and the reach
    # goes flat, so "clear" is a long visibility rather than none.
    "clear": ((0.30, 0.47, 0.70), (0.68, 0.76, 0.83),
              (0.40, 0.34, 0.24), 1.0 / 2200.0, 46.0, 0.55, 0.08),
    "hazy": ((0.36, 0.50, 0.68), (0.71, 0.77, 0.82),
             (0.34, 0.30, 0.22), 1.0 / 2600.0, 34.0, 0.45, 0.30),
    "overcast": ((0.60, 0.63, 0.66), (0.74, 0.76, 0.78),
                 (0.10, 0.10, 0.10), 1.0 / 1100.0, 26.0, 0.15, 0.95),
    # 1/420 was chosen for the number and looked like nothing: the
    # Charles is 150 m across and its far bank is well inside a 290 m
    # half-loss range, so the "fog" was doing almost exactly what the
    # overcast did.  A river fog you would actually be careful in takes
    # the far bank most of the way out.
    "fog": ((0.74, 0.76, 0.77), (0.82, 0.83, 0.84),
            (0.05, 0.05, 0.05), 1.0 / 130.0, 11.0, 0.06, 0.55),
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
#:
#: This is the amplitude at :data:`RIPPLE_FULL_WIND`; below that it
#: scales away, because it has to.  Micro-ripple is the wind's
#: fingerprint on the surface -- it is what the first metre per second
#: of a breeze does before there is any wave to speak of -- so with no
#: wind there is none of it, and a calm lake at dawn really is a sheet
#: of glass.  Scaling it as the square root of wind speed rather than
#: linearly keeps a light air from looking like nothing at all.
RIPPLE_SLOPE = 0.038

#: Wind speed, m/s, at which the ripple reaches full amplitude.
RIPPLE_FULL_WIND = 6.0
RIPPLE_SCALE = 1.35
RIPPLE_FADE = 45.0

#: How much rougher the small scale is where the water has been stirred.
#: Down from 1.6: at that gain the wake pattern came back through the
#: ripple as concentric rings of sparkle, which is the texture reporting
#: the shape underneath it rather than sitting on it.
RIPPLE_WAKE_GAIN = 0.8

#: How far the hull's own motion moves the water, per m/s of heave and
#: per m/s^2 of surge.  A shell heaves a couple of centimetres a second
#: and surges around half a g's worth of a g, so these put both effects
#: in the centimetres -- visible beside the hull, gone by a length away.
HEAVE_GAIN = 0.30
SURGE_GAIN = 0.014

#: The shadow map covers the course line plus this much either side, m,
#: and is sized so a texel is about this big on the ground.
SHADOW_MARGIN = 260.0
SHADOW_TEXEL = 1.2

VERTEX_SHADER = """#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_colour;
out vec3 v_colour;
out vec3 v_normal;
out vec3 v_world;
uniform mat4 mvp;
uniform float colour_scale;
void main() {
    // colour_scale is 1/255 for the packed world buffers (uint8 colour,
    // int8 normal, arriving as raw integers) and 1.0 for the float
    // buffers the boat and oars still use.  Normalising here makes the
    // int8 normal's magnitude irrelevant, and costs nothing the
    // fragment stage was not already paying.
    v_colour = in_colour * colour_scale;
    v_normal = normalize(in_normal);
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
#: 2-D simplex noise, its own block so that every program that needs it
#: gets it exactly once: the sky block carries it for the fragment
#: shaders, and the water's vertex shader -- which does not take the sky
#: block, and whose surface code now uses it for the bow's streaks --
#: takes it directly.
SIMPLEX_GLSL = """
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
"""

SKY_FOG_GLSL = """
uniform vec3 sky_zenith;
uniform vec3 sky_horizon;
uniform vec3 sun_glow;
uniform float fog_density;
uniform float fog_height;
uniform float fog_scatter;
uniform int fog_simple;           // 1: plain exp(-density * distance)
uniform float sky_overcast;
// 0: no noise on the dome.  The two octaves are evaluated for every sky
// pixel AND for every water pixel, because the water's reflection
// samples sky_colour -- on an integrated part that was a measurable
// share of the water pass, for a texture the low tiers are not there
// to show.
uniform int sky_detail;
uniform float sky_time;

__SIMPLEX__

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
    // A very little low-frequency variation on an overcast, so the
    // grey is a sky and not a wall.  Two octaves of simplex over the
    // dome, drifting slowly, scaled by how overcast it is -- a clear
    // blue barely takes any -- and held to a few percent, so it sits
    // UNDER the zenith-to-horizon gradient rather than competing with
    // it.  That gradient is the sky's shape; this is its texture.
    if (sky_overcast > 0.0 && sky_detail > 0) {
        vec2 q = d.xy / (max(d.z, 0.0) + 0.35);
        float cloud = simplex(q * 0.9 + vec2(sky_time * 0.012, 0.0))
                    + 0.5 * simplex(q * 2.1 - vec2(0.0, sky_time * 0.017));
        base *= 1.0 + 0.045 * sky_overcast * cloud;
    }
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
    // The lower tiers take a uniform atmosphere: no height integral,
    // no exp per fragment for the density at the eye.  The horizon
    // still thickens, because distance still grows.
    if (fog_simple == 1) return fog_density * distance;
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
# Two forms.  The land and sky programs take the noise inside the sky
# block.  The water programs already carry it at the top with their
# surface code -- which uses it, and sits above the sky block -- so they
# take the block WITHOUT it, or the function is defined twice.
SKY_FOG_GLSL_PLAIN = SKY_FOG_GLSL.replace("__SIMPLEX__", "")
SKY_FOG_GLSL = SKY_FOG_GLSL.replace("__SIMPLEX__", SIMPLEX_GLSL)



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
__SHADOW__
void main() {
    // A single directional light with a generous ambient: this is an
    // overcast New England morning, not a stage.
    vec3 n = normalize(v_normal);
    float lambert = max(dot(n, sun), 0.0);
    float visible = sun_visibility(v_world, n, 1.0 - abs(dot(n, sun)));
    vec3 lit = v_colour * (0.55 + 0.45 * lambert * visible);
    if (shadow_debug == 1) { f_colour = vec4(vec3(visible), 1.0); return; }
    if (shadow_debug >= 2) { f_colour = vec4(shadow_probe(v_world), 1.0); return; }
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
uniform int water_detail;         // 0: vertex-baked normal only
uniform int water_flat;           // 1: normals from the vertex slope only

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


// Two octaves, drifting with the wind, differenced for a slope.  The
// second octave runs the other way so the pattern does not read as one
// sheet sliding.
uniform float ripple_wake_gain;

vec2 ripple(vec2 p, float range, float stirred, float span) {
    float fade = clamp(1.0 - range / ripple_fade, 0.0, 1.0);
    // The same filter as the chop, at the ripple's own wavenumber: a
    // centimetre ripple under a pixel a metre wide is pure fizz.
    float qf = 6.2832 * ripple_scale * 2.7 * span * 0.35;
    fade *= exp(-0.5 * qf * qf);
    if (fade <= 0.001 || ripple_slope_amp <= 0.0) return vec2(0.0);
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

//: Angle one pixel subtends, radians: the vertical field of view over
//: the viewport height.  With it the shader knows how much water a
//: pixel covers, and that is what decides which waves it can show.
uniform float pixel_angle;

// How much of the surface one pixel covers, in metres.  Grows with
// range, and stretches along the view at grazing angles, where a pixel
// looking nearly along the water spans a long thin patch of it.
float footprint(vec3 world) {
    vec3 to_eye = eye - world;
    float range = length(to_eye);
    float grazing = max(abs(to_eye.z) / max(range, 1e-3), 0.06);
    return range * pixel_angle / grazing;
}

// The chop's slope, PRE-FILTERED to what the pixel can resolve.
//
// This is the "static" at distance.  Every wave was differentiated
// exactly at every pixel, which is right up close and wrong far off:
// once a pixel spans more water than a wavelength, an exact sample is
// an effectively random phase, and a field of random slopes is noise
// that does not even glitter -- it just fizzes.  The cure is the same
// as for any signal sampled below its bandwidth: attenuate what the
// sampling cannot carry.  Each component is weighted by a Gaussian in
// (k * footprint), so a wave several pixels long passes untouched and
// one shorter than a pixel is gone.
//
// ``lost`` reports the share of the slope that was filtered away.  It
// is not thrown away: an unresolved rough surface reflects like a
// broad one, so the caller widens the specular lobe by it.  That is
// the difference between a horizon that fizzes and one that sheens.
vec2 sea_slope(vec2 p, float span, out float lost) {
    vec2 g = vec2(0.0);
    float total = 0.0, kept = 0.0;
    for (int i = 0; i < 8; ++i) {
        float a = waves[i].x;
        if (a <= 0.0) continue;
        float k = waves[i].y;
        float d = waves[i].z;
        vec2 along = vec2(cos(d), sin(d));
        float w = sqrt(SLOPE_G * k);
        float s = sin(k * dot(p, along) - w * time + waves[i].w);
        float q = k * span * 0.35;
        float weight = exp(-0.5 * q * q);
        g += -a * k * along * s * weight;
        total += a * k;
        kept += a * k * weight;
    }
    lost = total > 0.0 ? 1.0 - kept / total : 0.0;
    return g;
}

vec3 water_normal(vec3 world, vec2 baked_slope, float range,
                  inout float foam, out float roughness, float blend) {
    vec2 g;
    float span = footprint(world);
    roughness = 0.0;
    // How stirred this patch is: the slope the boat's own disturbance
    // contributes, plus the foam the puddles carry.  Taken from the
    // wake and near field rather than from the chop, because wind waves
    // do not leave a rougher surface behind them and a wake does.
    // How stirred this patch is, computed the SAME way on both sides of
    // the exact_within boundary.  It was measured differently inside
    // and out -- from the per-pixel height there, from the interpolated
    // slope here -- which put a step in the ripple amplitude on a
    // circle around the boat, and that circle is the edge that showed
    // up in the water.  Both terms below are continuous everywhere.
    float stirred = foam + 3.0 * length(baked_slope);
    // Flat water: the normal is the one the vertex stage baked from the
    // mesh, and nothing is re-evaluated per fragment.  On an Intel UHD
    // the water pass was 7 ms at the lowest tier with the mesh already
    // at 51k triangles, because every fragment inside exact_within was
    // summing eight waves, sixteen puddles, two textures and the wake
    // three times over for a gradient, and every fragment outside it
    // was summing the eight waves once for a slope.  A tier that asked
    // for flat water was getting flat GEOMETRY and full per-pixel work.
    if (water_flat == 1) {
        return normalize(vec3(-baked_slope.x * blend,
                              -baked_slope.y * blend, 1.0));
    }
    if (distance(world.xy, boat.xy) < exact_within) {
        // Differenced at a step finer than the cells, so the answer is
        // the surface's slope and not the mesh's.
        float e = 0.05, junk, here;
        float h  = surface(world.xy, here);
        float hx = surface(world.xy + vec2(e, 0.0), junk);
        float hy = surface(world.xy + vec2(0.0, e), junk);
        g = vec2((hx - h) / e, (hy - h) / e) * blend;
        // The waterline foam is a 30 cm band; interpolated from 20 cm
        // vertices it is a smear.  Here it is known exactly.
        foam = max(foam, here);
        // Breaking.  A gravity wave cannot stand steeper than the Stokes
        // limit -- a 120 degree crest, a face slope of about 0.58 -- and
        // where the computed surface (chop, bow wave, pulse and run-up
        // together) approaches it, it breaks and goes white.  In flat
        // water a shell's bow never gets there, which is correct; in a
        // breeze the crest on top of the pile-up does, at the bow, which
        // is where a coxswain sees it.  Confined to the boat's own water
        // so a whitecap far off is the wind's business, not this.
        float steep = length(g);
        float near_boat = 1.0 - smoothstep(0.6 * hull_length,
                                           0.6 * hull_length + 6.0,
                                           distance(world.xy, boat.xy));
        float breaking = smoothstep(0.34, 0.58, steep) * near_boat;
        foam = max(foam, 0.85 * breaking);
    } else {
        // Faded with the geometry, or the surface goes flat at
        // the patch edge while the SHADING still shows chop --
        // which leaves the seam visible even once the height
        // matches.
        g = sea_slope(world.xy, span, roughness) * blend
          + baked_slope;
    }
    g += ripple(world.xy, range, stirred, span) * blend;
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
uniform float heave_rate;     // hull vertical velocity, m/s, up positive
uniform float surge_accel;    // fore-aft acceleration, m/s^2
uniform float heave_gain;
uniform float surge_gain;
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
        // Scaled down from 1.0: the foam is now BLENDED toward white
        // rather than added, which reads brighter for the same value,
        // and a puddle is a swirl with a little foam in it, not a slick.
        foam = max(foam, 0.55 * pow(strength, 3.0) * exp(-r * r / 1.4));
        // The entry itself: a short bright burst where the blade went
        // in, gone in a third of a second, with a small ring running
        // out from it.  Deliberately not dramatic -- a clean catch
        // throws very little water, and the point is that it is there.
        float since = puddles[i].w;          // seconds since the entry
        if (since < 0.8) {
            foam = max(foam, 0.75 * exp(-since / 0.20) * exp(-r * r / 0.28));
            h += 0.009 * exp(-since / 0.30) * exp(-r * r / 0.7)
               * cos(r * 9.0 - since * 16.0);
        }
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
    // THIS WAS THE DIAGONAL LINE.  The lee began at downwind = 0 with a
    // step and the windward pile-up ended there with another, so along
    // the line through the boat perpendicular to the wind the wave
    // amplitude jumped by 0.9 -- a straight seam through the water,
    // hull-length long, following the boat and aligned with the wind
    // rather than the hull, which is why it never lined up with
    // anything.  The two now cross over across the hull's own extent
    // along the wind, so the pile-up peaks at the windward face and the
    // lee begins at the leeward one, with the water between them --
    // which the hull is sitting on anyway -- blended smoothly.
    // Never narrower than the band's own 1.5 m edge: beam-on the
    // hull is a quarter of a metre along the wind, and a crossover
    // that short is a soft line rather than no line.  The riggers
    // and bodies the wind actually sees are wider than the hull.
    float half_along = max(0.5 * (hull_length * along_wind
                                  + 0.5 * across_wind), 1.5);
    float t = smoothstep(-half_along, half_along, downwind);
    float lee = 0.55 * exp(-max(downwind - half_along, 0.0) / 14.0);
    float wind = 0.35 * exp(min(downwind + half_along, 0.0) / 1.6);
    return 1.0 + inside * mix(wind, -lee, t);
}

// Heave and surge, which the water had no term for at all.
//
// The wake and the near field were both driven by SPEED, and speed
// alone describes a hull travelling steadily.  A rowing shell does not:
// inside every stroke it surges -- accelerating hard through the drive
// and coasting through the recovery -- and it heaves, rising and
// settling as the crew's mass runs up and down the slide.  Those are
// two of the six degrees of freedom the simulator integrates, and none
// of it reached the surface.
//
// What the water does about it:
//
//  * A hull dropping pushes water out from under itself and up around
//    it; rising, it draws the surface down after it.  That is a
//    displacement, so it follows heave VELOCITY, not heave position.
//  * Accelerating piles water against the bow and lets it fall away
//    aft, which is why the bow wave breathes once a stroke rather than
//    sitting at the value the mean speed would give.  Bow-biased, and
//    it follows acceleration rather than speed.
//
// Both are shaped by a smooth bump over the hull's own footprint and
// die at its edge, so they add nothing where the hull is not.
float hull_pulse(vec2 p) {
    if (heave_gain <= 0.0 && surge_gain <= 0.0) return 0.0;
    float c = cos(-boat.z), s = sin(-boat.z);
    vec2 d = p - boat.xy;
    float along = d.x * c - d.y * s;          // positive toward the bow
    float across = d.x * s + d.y * c;
    float reach_along = 0.5 * hull_length + 3.0;
    float reach_across = 3.0;
    float a = along / reach_along;
    float b = across / reach_across;
    if (abs(a) >= 1.0 || abs(b) >= 1.0) return 0.0;
    float shape = (1.0 - a * a) * (1.0 - b * b);
    shape *= shape;
    return shape * (-heave_rate * heave_gain + surge_accel * surge_gain * a);
}

// The hull IN the water, which the surface had no account of.
//
// The wake was the hull's effect on the water it had passed; the near
// field was the water it was pushing aside; the shelter was the wind
// it was blocking.  None of them was the hull itself: the chop ran
// straight through the footprint as if the boat were not there, and
// nothing went white anywhere.  What a shell does at speed is cut the
// water -- the stem throws it aside as spray, the sides run along it in
// a line of broken water, and a wave that meets the hull does not pass
// through, it breaks against it.
//
// hull_damp takes the incident chop out inside the footprint and just
// around it; waterline_foam paints the white where the two meet.  Both
// hug the outline and vanish a metre from it, so the wake and near
// field further out are untouched.
uniform float hull_beam;

void hull_frame(vec2 p, out float along, out float across) {
    float c = cos(-boat.z), s = sin(-boat.z);
    vec2 d = p - boat.xy;
    along = d.x * c - d.y * s;               // positive toward the bow
    across = d.x * s + d.y * c;
}

// How much of the hull's footprint a point is in, 0 outside to 1 on
// the centreline, feathered a metre out.  What is done with it depends
// on the wave, so this only reports the shape.
float hull_inside(vec2 p) {
    float along, across;
    hull_frame(p, along, across);
    float a = abs(along) / (0.5 * hull_length + 0.8);
    float b = abs(across) / (0.5 * hull_beam + 0.9);
    if (a >= 1.0 || b >= 1.0) return 0.0;
    return (1.0 - a * a) * (1.0 - b * b);
}

// The chop as the hull leaves it.  A hull does not stop waves; it
// scatters the ones it cannot follow.  A wave long against the hull
// lifts the whole boat and passes through it -- the boat rides it,
// which is the simulator's business, not the surface's -- while a
// wave short against the hull cannot lift it, meets it as a wall, and
// is broken up.  So each component is weighted by its wavelength
// against the hull length: chop of a metre or two is gone inside the
// footprint of an eight, a ten-metre swell is dented, twenty metres
// passes untouched.  A single, being half the length, breaks up less.
float sea_at_hull(vec2 p, float t, float inside) {
    if (inside <= 0.0) return sea(p, t);
    float h = 0.0;
    for (int i = 0; i < 8; ++i) {
        float a = waves[i].x;
        if (a <= 0.0) continue;
        float k = waves[i].y;
        float d = waves[i].z;
        float w = sqrt(G * k);
        float wavelength = 6.2831853 / k;
        float passes = smoothstep(0.35 * hull_length, 1.2 * hull_length,
                                  wavelength);
        float keep = 1.0 - 0.85 * inside * (1.0 - passes);
        h += a * cos(k * (p.x * cos(d) + p.y * sin(d)) - w * t + waves[i].w)
           * keep;
    }
    return h;
}

// The bow, from the fluid mechanics rather than from a painted box.
//
// A racing shell is slender, and its bow throws little in flat water:
// a bow wave a couple of centimetres high (the near-field bake, which
// is the double-body potential flow and already has its shape), and a
// thin sheet of water riding along the entry.  What is drawn here is
// that sheet and where the surface breaks, each from a stated model:
//
//  * The sheet.  Thin-ship theory gives the velocity the hull pushes
//    water outward with as U * b'(x) -- the ship speed times the local
//    slope of the half-breadth -- so the head driving the sheet is
//    (U b')^2 / 2g.  For a parabolic waterline b' is proportional to x,
//    so the sheet is strongest at the stem where the entry is finest,
//    dies where the hull runs parallel, and goes as speed squared.
//    That is the along-hull distribution, from the hull, not a ramp.
//
//  * Run-up.  A crest arriving at the stem is partly reflected, and a
//    reflected wave stands to (1 + R) times its height there.  A thin
//    stem reflects little; R is set below.  The hull plunging into the
//    crest (heave velocity against the surface) adds relative motion.
//
//  * Advection.  Broken water moves with the water: aft along the hull
//    at the flow speed U.  The streak texture is carried at that speed
//    in the hull frame, so the whitewater slides past the boat instead
//    of being stuck to it.
//
// Breaking -- whitewater where the computed surface is too steep -- is
// decided in water_normal, where the per-pixel slope is known.
uniform float bow_reflect;        // R, the stem's reflection coefficient

float waterline_foam(vec2 p) {
    if (speed < 0.4) return 0.0;
    float along, across;
    hull_frame(p, along, across);
    float half_len = 0.5 * hull_length;
    float half_beam = 0.5 * hull_beam;
    if (along < -0.5 * half_len || along > half_len + 1.5) return 0.0;

    // The waterline itself, b(x) = b0 (1 - (x/L)^2): the sheet sits
    // just outboard of THAT, not of a constant half-beam.  Measured
    // from the constant, the sheet was a band of full width right up to
    // the stem -- where the hull is a point -- and stopped there in a
    // flat edge.  Measured from the waterline it narrows with the hull
    // and closes to the stem: the V a real bow wave makes.  Ahead of
    // the stem the distance is taken from the stem point itself, so
    // the tuft there is round, not a lid.
    float x = clamp(along, 0.0, half_len) / half_len;
    float waterline = half_beam * (1.0 - x * x);
    float out_across;
    if (along > half_len) {
        out_across = length(vec2(along - half_len, across));
    } else {
        out_across = abs(across) - waterline;
    }
    if (out_across > 0.9 || out_across < -0.3) return 0.0;

    // Half-breadth slope of that waterline, |b'(x)| = 2 b x / L^2,
    // over the forward half of the hull where the entry is.
    float slope = 2.0 * half_beam * x / half_len;
    float head = (speed * slope) * (speed * slope) / (2.0 * 9.80665);
    // Normalised against a shell at race pace so the visual amplitude
    // is set once: 4.5 m/s and the stem slope of a 17 m eight.
    float head_ref = (4.5 * 2.0 * 0.285 / 8.65);
    head_ref = head_ref * head_ref / (2.0 * 9.80665);
    float sheet = clamp(head / head_ref, 0.0, 2.5);

    // The incident wave at the stem, partly reflected, and the plunge.
    vec2 stem = boat.xy + vec2(cos(boat.z), sin(boat.z)) * half_len;
    float eta = sea(stem, time);
    float run_up = clamp((1.0 + bow_reflect) * eta / 0.05, -0.5, 2.0);
    float plunge = clamp(-heave_rate * 10.0, 0.0, 1.2);
    float at_stem = exp(-(half_len - along) * (half_len - along) / 1.8);
    sheet *= 1.0 + at_stem * (0.8 * max(run_up, 0.0) + 0.6 * plunge);

    // The sheet hugs the side.  Its lateral reach is the run-up head
    // itself -- centimetres -- plus what the breaking throws, so it is
    // a line of broken water along the entry and a tuft at the stem.
    // The reach opens out aft of the stem as the sheet spreads, which
    // with the waterline narrowing forward is what makes the point.
    // A real sheet is centimetres; drawn at centimetres it vanishes
    // from a few metres away, and the point of it is to be seen.  So
    // it opens from a hand's width at the stem to about a quarter of a
    // metre a few metres aft, still closing to the point.
    float spread = smoothstep(0.0, 3.5, half_len - along);
    float reach = (0.10 + 0.15 * spread) + 0.10 * sheet
                + 0.25 * at_stem * (max(run_up, 0.0) + plunge);
    float band = exp(-max(out_across, 0.0) / reach);
    // And the sheet fades aft rather than being cut off.
    band *= 1.0 - smoothstep(-0.2 * half_len, -0.5 * half_len, along);

    // Streaks, carried aft at the flow speed: the whitewater slides
    // past the hull, which is the single strongest cue that it is water
    // and not paint.
    float streak = simplex(vec2((along + speed * time) * 1.6, across * 7.0))
                 + 0.5 * simplex(vec2((along + speed * time) * 3.7 + 11.0,
                                      across * 13.0));
    float texture_gain = clamp(0.55 + 0.45 * streak, 0.0, 1.3);

    return 0.55 * sheet * band * texture_gain;
}

// The reflected part of the incident wave at the stem, as a height: a
// crest meeting the stem stands up by R times itself over the last
// metre or so of the entry, and a trough drops the same.
float stem_run_up(vec2 p) {
    if (bow_reflect <= 0.0) return 0.0;
    float along, across;
    hull_frame(p, along, across);
    float half_len = 0.5 * hull_length;
    float d_along = along - half_len;
    if (abs(d_along) > 2.0 || abs(across) > 1.2) return 0.0;
    vec2 stem = boat.xy + vec2(cos(boat.z), sin(boat.z)) * half_len;
    float eta = sea(stem, time);
    float shape = exp(-(d_along * d_along) / 0.9 - across * across / 0.5);
    return bow_reflect * eta * shape;
}

float surface(vec2 p, out float foam) {
    float h = sea_at_hull(p, time, hull_inside(p)) * shelter(p)
            + wake(p) + near_field(p) + hull_pulse(p) + stem_run_up(p)
            + puddle(p, foam);
    foam = max(foam, waterline_foam(p));
    return h;
}
"""

WATER_VERTEX = """#version 330
in vec2 in_grid;
uniform float patch_reach;   // half-width of the moving water patch, m
out vec3 v_world;
out vec3 v_normal;
out vec2 v_slope;
out float v_blend;
out float v_foam;
""" + SIMPLEX_GLSL + WATER_UNIFORMS_GLSL + WATER_SURFACE_GLSL + """
void main() {
    vec2 p = in_grid + centre;
    float foam;
    float h = surface(p, foam);

    // Fade the whole surface out at the edge of the patch.
    //
    // The detailed water is a square patch that follows the boat, and
    // beyond it there is a flat plane at the still-water level.  The two
    // met at a hard edge: waves one side, glass the other.  Because the
    // patch is world-axis-aligned and moves with the boat, that edge
    // reads from any oblique angle as a straight DIAGONAL line lying on
    // the water and travelling along with you -- which is exactly the
    // artefact that kept being reported and that never showed up from
    // the seat, where the edge is over the horizon.
    //
    // Chebyshev distance, because the patch is a square: this follows
    // the actual boundary instead of inscribing a circle in it and
    // throwing away the corners.
    float edge = max(abs(in_grid.x), abs(in_grid.y)) / max(patch_reach, 1.0);
    float blend = 1.0 - smoothstep(0.80, 1.0, edge);
    h *= blend;
    foam *= blend;
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
    v_slope = vec2((bx - base) / e, (by - base) / e) * blend;
    v_blend = blend;
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
in float v_blend;
in float v_foam;
__WATER_SHARED__
out vec4 f_colour;
uniform vec3 sun;
uniform vec3 sky;
uniform vec3 eye;
uniform float far;
uniform vec3 deep;
__SKY_FOG__
__SHADOW__
__WATER_SLOPE__
void main() {
    float range = length(v_world - eye);
    float rough;
    float foam = v_foam;
    vec3 n;
    if (water_detail == 0) {
        // The slope the vertex shader baked is the whole story at the
        // low tiers: no ripple octaves, no exact band, no roughness
        // term.  This is what "flat water" means per pixel.
        n = normalize(vec3(-v_slope.x, -v_slope.y, 1.0));
        rough = 0.0;
    } else {
        n = water_normal(v_world, v_slope, range, foam, rough, v_blend);
    }
    vec3 to_eye = normalize(eye - v_world);
    // Water is mostly a mirror at grazing angles and mostly dark looking
    // straight down, which is the whole reason chop reads as chop: the
    // Fresnel term turns a slope into a brightness.
    float fresnel = pow(1.0 - max(dot(n, to_eye), 0.0), 3.0);
    vec3 mirror = sky_colour(reflect(-to_eye, n), sun);
    vec3 base = mix(deep, mirror, clamp(0.08 + 0.55 * fresnel, 0.0, 1.0));
    // A surface too fine to resolve reflects like a broad one: the lobe
    // widens and dims by the slope that was filtered out.
    float shine = mix(60.0, 8.0, rough);
    // In shadow the sun is not in the water to sparkle, and the sky the
    // surface scatters is the darker part of it.
    float visible = sun_visibility(v_world, vec3(0.0, 0.0, 1.0), 0.0)
                   * boat_shadow(v_world);
    float spec = pow(max(dot(reflect(-sun, n), to_eye), 0.0), shine)
               * mix(1.0, 0.3, rough) * visible;
    vec3 lit = (base + vec3(0.9) * spec * 0.5) * mix(0.74, 1.0, visible);
    // Broken water is a blend toward white, not white added on top: an
    // additive wash saturates into a flat block, which is exactly how
    // the bow read before.
    lit = mix(lit, vec3(0.90, 0.93, 0.95) * mix(0.8, 1.0, visible),
              clamp(foam * 0.8, 0.0, 0.85));
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
in float v_blend;
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
__SHADOW__
__WATER_SLOPE__

float linear_depth(float raw) {
    float ndc = raw * 2.0 - 1.0;
    return (2.0 * near_plane * far_plane)
         / (far_plane + near_plane - ndc * (far_plane - near_plane));
}

void main() {
    float range_to_eye = length(v_world - eye);
    float rough;
    float foam = v_foam;
    vec3 n = water_normal(v_world, v_slope, range_to_eye, foam, rough, v_blend);
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
    // Widened and dimmed by whatever slope the pixel could not resolve
    // (see sea_slope): the sparkle that was aliasing becomes a sheen.
    float shine = mix(90.0, 9.0, rough);
    // A shadow on water takes the sun out of it -- no sparkle -- and
    // darkens what the surface scatters, but leaves the reflection of
    // the sky and bank alone, which is still lit.
    float visible = sun_visibility(v_world, vec3(0.0, 0.0, 1.0), 0.0)
                   * boat_shadow(v_world);
    float spec = pow(max(dot(reflect(-sun, n), to_eye), 0.0), shine)
               * mix(1.0, 0.3, rough) * visible;
    lit = mix(refracted * mix(0.72, 1.0, visible), mirror,
              clamp(fresnel, 0.0, 1.0));
    lit += vec3(1.0, 0.98, 0.92) * spec * 0.65;
    lit = mix(lit, vec3(0.90, 0.93, 0.95) * mix(0.8, 1.0, visible),
              clamp(foam * 0.8, 0.0, 0.85));
    f_colour = vec4(apply_fog_lit(lit, eye, v_world, sun), 1.0);
}
"""


# The shared slope code goes into both fragment shaders.  Done here
# rather than by hand in each so the two cannot drift apart.
for _name in ("FRAGMENT_SHADER", "SKY_FRAGMENT"):
    globals()[_name] = globals()[_name].replace("__SKY_FOG__", SKY_FOG_GLSL)
for _name in ("WATER_FRAGMENT", "WATER_FRAGMENT_RICH"):
    globals()[_name] = globals()[_name].replace("__SKY_FOG__",
                                                SKY_FOG_GLSL_PLAIN)




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


#: Splash at the catch: a handful of ballistic droplets, not a shower.
#:
#: "Not super dramatic but there" is the brief, so this is a CPU-side
#: burst of maybe ten points per blade, not a GPU particle system --
#: there is nothing here that needs one.  Each droplet is a real
#: projectile: it leaves the entry point with a upward-and-outward
#: kick, falls under gravity, and is culled the instant it would cross
#: the water plane again, which is what keeps a lazy catch throwing
#: nothing and a hard one throwing visibly more without any separate
#: "how much spray" knob -- the entry speed already carries that.
SPLASH_VERTEX = """#version 330
in vec3 in_pos;
in float in_age;      // 0 at birth, 1 at death
uniform mat4 mvp;
out float v_age;
void main() {
    v_age = in_age;
    gl_Position = mvp * vec4(in_pos, 1.0);
    // Shrinks as it ages, so a droplet reads as settling rather than
    // just vanishing.
    gl_PointSize = mix(5.0, 1.0, in_age) ;
}
"""

SPLASH_FRAGMENT = """#version 330
in float v_age;
out vec4 f_colour;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    float fade = 1.0 - v_age;
    f_colour = vec4(vec3(0.95), fade * fade * 0.8);
}
"""


#: A blade dragging on the recovery throws this many drops, this often,
#: per oar.  Three every 80 ms is a visible thread of water off the
#: blade without turning the low side into a wake of its own.
DRAG_SPLASH_DROPLETS = 3
DRAG_SPLASH_INTERVAL = 0.08


class SplashSystem:
    """A fixed pool of ballistic droplets, spawned at the catch.

    A ring buffer over a numpy array, same shape as :class:`PuddleTrail`
    for the same reason: this is rebuilt and re-uploaded every frame, so
    the count has to be fixed and the update has to be one vectorised
    pass rather than a Python loop over live particles.
    """

    #: Per catch, per blade.  A coxed four throws roughly this many
    #: visible droplets at a firm catch; more reads as a bucket of water
    #: rather than an oar.
    PER_BLADE = 9
    LIFETIME = 0.55
    GRAVITY = 9.80665
    CAPACITY = 256

    def __init__(self):
        self.position = np.zeros((self.CAPACITY, 3), dtype="f4")
        self.velocity = np.zeros((self.CAPACITY, 3), dtype="f4")
        self.birth = np.full(self.CAPACITY, -1e9, dtype="f4")
        self._next = 0
        self._rng = np.random.default_rng(4)

    def spawn(self, tip: np.ndarray, speed: float, t: float,
              count: int = None) -> None:
        """A burst at ``tip`` (world xyz, z at the water), scaled by the
        blade's speed through the surface -- a gentle catch barely
        splashes, a rushed one throws further and higher.

        ``count`` overrides :data:`PER_BLADE`: a catch is a burst, a
        blade skimming the recovery is a trickle of a few drops at a
        time, and the difference between the two is what tells the eye
        which it is looking at."""
        kick = float(np.clip(speed, 0.0, 2.5))
        if kick <= 0.02:
            return
        n = self.PER_BLADE if count is None else max(int(count), 0)
        if n == 0:
            return
        angle = self._rng.uniform(0.0, 2.0 * np.pi, n)
        outward = (0.35 + 0.9 * kick) * self._rng.uniform(0.5, 1.0, n)
        up = (0.7 + 1.6 * kick) * self._rng.uniform(0.6, 1.0, n)
        for i in range(n):
            slot = self._next % self.CAPACITY
            self._next += 1
            self.position[slot] = tip
            self.velocity[slot] = (outward[i] * np.cos(angle[i]),
                                   outward[i] * np.sin(angle[i]), up[i])
            self.birth[slot] = t

    def as_uniform(self, t: float):
        """``(N, 4)`` of ``(x, y, z, age)`` for every droplet still
        alive, ballistic motion evaluated directly from its birth time
        rather than integrated -- so it cannot drift with the frame
        rate the way an accumulated position would."""
        age = (t - self.birth) / self.LIFETIME
        alive = (age >= 0.0) & (age < 1.0)
        if not np.any(alive):
            return np.zeros((0, 4), dtype="f4")
        dt = (t - self.birth[alive])[:, None]
        pos = (self.position[alive]
              + self.velocity[alive] * dt
              - np.array([0.0, 0.0, 0.5 * self.GRAVITY]) * dt * dt)
        # A droplet that has fallen back through the water is done,
        # even if its clock has not run out -- it rejoins the surface
        # rather than hanging visibly below it.
        above = pos[:, 2] >= self.position[alive][:, 2] - 0.05
        out = np.concatenate([pos[above], age[alive][above, None]], axis=1)
        return out.astype("f4")


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


def orthographic(half_width: float, half_height: float,
                 near: float, far: float):
    """An orthographic projection, for the sun's view of the world."""
    matrix = np.eye(4, dtype="f4")
    matrix[0, 0] = 1.0 / half_width
    matrix[1, 1] = 1.0 / half_height
    matrix[2, 2] = -2.0 / (far - near)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


#: The depth-only shader that bakes the map.
SHADOW_VERTEX = """#version 330
in vec3 in_pos;
uniform mat4 sun_vp;
void main() { gl_Position = sun_vp * vec4(in_pos, 1.0); }
"""

SHADOW_FRAGMENT = """#version 330
void main() { }
"""

#: Sampling the map: percentage-closer filtering over a 3x3 kernel.
#:
#: A shadow map is a depth image from the light's point of view, and
#: comparing against it gives a hard yes-or-no per texel -- so the edge
#: of every shadow is a staircase at the map's resolution.  PCF takes
#: the comparison at several neighbouring texels and averages the
#: ANSWERS rather than the depths (averaging depths is meaningless), and
#: nine taps is enough to turn that staircase into a soft edge.
#:
#: The map is baked ONCE.  The world and the sun are both static -- the
#: terrain, the buildings and the trees do not move and the sun does not
#: cross the sky in a five-kilometre race -- so there is nothing to
#: redraw per frame and the whole cost at run time is these nine taps.
#: The boat and crew do not cast, because they are the things that do
#: move; that is the compromise in "static", and it is why the water
#: still gets the hull's own shading from its normal rather than from a
#: shadow.
SHADOW_GLSL = """
uniform sampler2D shadow_map;
uniform mat4 sun_vp;
uniform vec2 shadow_texel;
uniform float shadow_bias;
uniform float shadow_strength;
uniform int shadow_taps;          // 16 for the 4x4 filter, 1 for a single tap
uniform int shadow_debug;
uniform float shadow_world_texel;

// For --shadow-debug: 1 paints the visibility, 2 paints the stored
// depth minus the fragment's own (grey = equal, white = caster nearer).
vec3 shadow_probe(vec3 world) {
    vec4 light_clip = sun_vp * vec4(world, 1.0);
    vec3 uv = (light_clip.xyz / light_clip.w) * 0.5 + 0.5;
    float depth = texture(shadow_map, uv.xy).r;
    // Scaled so a metre of depth difference is plainly visible: the
    // first version used x20 over a 2.9 km range and a ten-metre
    // building moved the grey by 0.08.
    if (shadow_debug == 2) return vec3(clamp((depth - uv.z) * 400.0 + 0.5, 0.0, 1.0));
    if (shadow_debug == 3) return vec3(depth);
    return vec3(uv.z);
}

// The boat's own shadow on the water, which the static map cannot hold
// because the boat moves.  Done analytically: the hull and the crew
// are a slab from the waterline to about head height, and the water
// point is walked back along the sun to see whether it passes through
// that slab -- the ellipse of the hull's footprint, tested at two
// heights and widened by their offset along the sun.  A coxswain sees
// this one beside the hull every stroke; the buildings' are on the far
// bank.
uniform vec3 shadow_boat;         // east, north, heading
uniform vec2 shadow_boat_size;    // half length, half beam
uniform vec3 shadow_sun;

float boat_shadow(vec3 world) {
    if (shadow_strength <= 0.0 || shadow_boat_size.x <= 0.0) return 1.0;
    float c = cos(-shadow_boat.z), s = sin(-shadow_boat.z);
    float cover = 0.0;
    for (int k = 0; k < 2; ++k) {
        // Where the sun ray through this point crosses height z.
        float z = (k == 0) ? 0.12 : 0.85;
        vec2 back = world.xy - shadow_sun.xy * ((z - world.z) / max(shadow_sun.z, 0.2));
        vec2 d = back - shadow_boat.xy;
        float along = (d.x * c - d.y * s) / (shadow_boat_size.x + 0.4);
        float across = (d.x * s + d.y * c) / (shadow_boat_size.y + (k == 0 ? 0.25 : 0.55));
        float r = along * along + across * across;
        cover = max(cover, 1.0 - smoothstep(0.75, 1.15, r));
    }
    return 1.0 - 0.6 * cover;
}

float sun_visibility(vec3 world, vec3 normal, float slope) {
    if (shadow_strength <= 0.0) return 1.0;
    // Normal offset.  The map is 0.7 m a texel and the ground is not
    // flat to the sun, so a texel's stored depth is the depth of ONE
    // point on a slope and the rest of that texel sits slightly deeper
    // -- which reads as self-shadow, in diagonal stripes across every
    // bank.  Pushing the lookup point a texel out along the surface
    // normal takes it clear of its own texel, and it is a far better
    // cure than a bigger bias, which just detaches the shadow from
    // whatever casts it.
    vec3 at_world = world + normalize(normal) * shadow_world_texel
                  * (0.8 + 1.4 * slope);
    vec4 light_clip = sun_vp * vec4(at_world, 1.0);
    vec3 ndc = light_clip.xyz / light_clip.w;
    vec3 uv = ndc * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0
     || uv.z > 1.0) return 1.0;
    float bias = shadow_bias * (1.0 + 2.0 * slope);
    if (shadow_taps <= 1) {
        // One tap: the pre-baked shadow as it is, hard-edged.  What
        // the lower tiers pay for a shadow at all -- a sixteenth of
        // the filter below, and on an integrated GPU the filter was
        // the single most expensive thing in the world shader.
        float depth = texture(shadow_map, uv.xy).r;
        float lit = (uv.z - bias <= depth) ? 1.0 : 0.0;
        vec2 edge = smoothstep(vec2(0.0), vec2(0.06), uv.xy)
                  * smoothstep(vec2(0.0), vec2(0.06), 1.0 - uv.xy);
        lit = mix(1.0, lit, edge.x * edge.y);
        return mix(1.0, lit, shadow_strength);
    }
    // A 3x3 box in CONTINUOUS texel space: the four corner taps carry
    // bilinear weights from where the lookup falls inside its texel, so
    // the filter slides smoothly across the map instead of snapping to
    // it.  That is what turns a staircase into a soft edge -- the
    // nearest-texel 3x3 it replaces stepped by a whole texel at a time,
    // and at 0.7 m a texel that was a visible ledge.
    vec2 p = uv.xy / shadow_texel - 0.5;
    vec2 cell = floor(p);
    vec2 f = p - cell;
    float sum = 0.0;
    for (int i = -1; i <= 2; ++i) {
        float wx = (i == -1) ? (1.0 - f.x) : ((i == 2) ? f.x : 1.0);
        for (int j = -1; j <= 2; ++j) {
            float wy = (j == -1) ? (1.0 - f.y) : ((j == 2) ? f.y : 1.0);
            vec2 at = (cell + vec2(float(i), float(j)) + 0.5) * shadow_texel;
            float depth = texture(shadow_map, at).r;
            sum += wx * wy * ((uv.z - bias <= depth) ? 1.0 : 0.0);
        }
    }
    float lit = sum / 9.0;
    vec2 edge = smoothstep(vec2(0.0), vec2(0.06), uv.xy)
              * smoothstep(vec2(0.0), vec2(0.06), 1.0 - uv.xy);
    lit = mix(1.0, lit, edge.x * edge.y);
    return mix(1.0, lit, shadow_strength);
}
"""
FRAGMENT_SHADER = FRAGMENT_SHADER.replace("__SHADOW__", SHADOW_GLSL)

# The water shaders are spliced here, AFTER SHADOW_GLSL exists: at their
# old place above it the module failed to import.
for _name in ("WATER_FRAGMENT", "WATER_FRAGMENT_RICH"):
    _text = globals()[_name].replace("__SHADOW__", SHADOW_GLSL)
    _text = _text.replace("__WATER_SHARED__",
                          SIMPLEX_GLSL + WATER_UNIFORMS_GLSL
                          + WATER_SURFACE_GLSL)
    globals()[_name] = _text.replace("__WATER_SLOPE__", WATER_SLOPE_GLSL)
del _name, _text


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


_BOAT_NORMALS = {}


def boat_geometry(boat, hull, t, state, crew=None):
    """The shell and its oars, in world space, for this frame.

    Both ride the hull, so both are built in hull coordinates and carried
    through the same rotation the camera uses.  That is what makes the
    bow sit still in the frame while the world swings behind it -- which
    is the whole cue a coxswain steers on.

    Returns ``(vertices, normals, colours)``.  The normals are baked
    once per pose in the hull frame and ROTATED here, not recomputed:
    a flat normal is a rigid property of its triangle, so rotating the
    baked one is exactly the cross product on the rotated vertices,
    without doing 5,000 cross products a frame in Python.
    """
    rotation = hull_to_abs(np.asarray(state[3:6], dtype=float))
    position = np.asarray(state[0:3], dtype=float)
    key = id(hull)
    baked = _BOAT_NORMALS.get(key)
    if baked is None or baked[0] is not hull:
        baked = (hull, _face_normals(hull.vertices), None)
        _BOAT_NORMALS[key] = baked
    pieces = [(hull.vertices @ rotation.T + position, baked[1] @ rotation.T,
               hull.colours)]
    if crew is not None:
        poses, crew_colours = crew
        period = float(boat.timing.period)
        # Nearest baked phase.  At 48 samples and rate 30 that is 40 ms
        # of stroke per pose, which is below what the eye resolves on a
        # body moving this slowly.
        index = int((t % period) / period * len(poses)) % len(poses)
        pose_key = (id(crew), index)
        pose_normals = _BOAT_NORMALS.get(pose_key)
        if pose_normals is None:
            pose_normals = _face_normals(poses[index])
            _BOAT_NORMALS[pose_key] = pose_normals
        pieces.append((poses[index] @ rotation.T + position,
                       pose_normals @ rotation.T, crew_colours))
    vertices = np.concatenate([p[0] for p in pieces]).astype("f4")
    normals = np.concatenate([p[1] for p in pieces]).astype("f4")
    shades = np.concatenate([p[2] for p in pieces]).astype("f4")
    return vertices, normals, shades


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
    # The rig editor: a plan of the boat, or None when it is not up.
    # Loaded from the preset rather than built blank, because the boat
    # this is for is a real one and typing it in again every time is
    # exactly the friction the preset exists to remove.
    lineup = None
    lineup_last = None          # the last lineup the editor showed
    from coxswain.viz.bonus import SecretTyper
    secret = SecretTyper()
    if getattr(args, "bonus_unlocked", False):
        add_bonus_row(menu, getattr(args, "bonus", "off"))
    pane_cursor = 0
    seat_cursor = -1
    # ``(seat, field, buffer)`` while a rower is being typed in.
    editing = None
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
                if lineup is not None and editing is not None:
                    # -- typing into one field of one rower.  Enter
                    # commits and stops; Tab or the arrows commit and
                    # move on to the next field, so a whole rower goes
                    # in without leaving the keyboard; Escape drops
                    # what was typed and keeps what was there.
                    seat, at, buffer = editing
                    if event.key == pygame.K_ESCAPE:
                        editing = None
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if seat < 0:
                            name = buffer.strip()
                            if name and name not in RIG_PRESETS:
                                from coxswain.viz import presets as _presets
                                lineup.name = name
                                _presets.save(lineup.to_dict())
                        else:
                            lineup.commit_edit(seat, at, buffer)
                        editing = None
                    elif event.key in (pygame.K_TAB, pygame.K_DOWN,
                                       pygame.K_UP) and seat >= 0:
                        lineup.commit_edit(seat, at, buffer)
                        step = -1 if event.key == pygame.K_UP else 1
                        at = (at + step) % len(FIELDS)
                        editing = (seat, at,
                                   field_text(lineup.rowers[seat], at))
                    elif event.key == pygame.K_BACKSPACE:
                        editing = (seat, at, buffer[:-1])
                    else:
                        char = getattr(event, "unicode", "") or ""
                        if char.isprintable() and len(buffer) < 24:
                            editing = (seat, at, buffer + char)
                    continue
                if lineup is not None:
                    # -- the rig editor owns the keyboard while it is up
                    if event.key == pygame.K_ESCAPE:
                        lineup_last, lineup = lineup, None
                        seat_cursor = -1
                        continue
                    if event.key in (pygame.K_UP, pygame.K_w):
                        pane_cursor = (pane_cursor - 1) % len(PANE_ROWS)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        pane_cursor = (pane_cursor + 1) % len(PANE_ROWS)
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT,
                                       pygame.K_a, pygame.K_d,
                                       pygame.K_RETURN, pygame.K_KP_ENTER,
                                       pygame.K_SPACE):
                        step = -1 if event.key in (pygame.K_LEFT,
                                                   pygame.K_a) else 1
                        key = PANE_ROWS[pane_cursor][0]
                        if key == "preset":
                            from coxswain.viz import presets as _presets
                            from coxswain.viz.rigview import Lineup as _Lineup
                            names = sorted(RIG_PRESETS) + sorted(
                                n for n in _presets.names()
                                if n not in RIG_PRESETS)
                            at = (names.index(lineup.name)
                                  if lineup.name in names else 0)
                            pick = names[(at + step) % len(names)]
                            if pick in RIG_PRESETS:
                                lineup = RIG_PRESETS[pick]()
                            else:
                                lineup = _Lineup.from_dict(_presets.get(pick))
                            seat_cursor = -1
                        elif key == "shell":
                            keys = list(SHELLS)
                            at = keys.index(lineup.shell)
                            lineup.set_shell(keys[(at + step) % len(keys)])
                            seat_cursor = -1
                        elif key == "rig":
                            options = RIGS[lineup.seats]
                            at = (options.index(lineup.rig)
                                  if lineup.rig in options else 0)
                            lineup.set_rig(options[(at + step)
                                                   % len(options)])
                        elif key == "switch":
                            # Walk the seats; enter flips the one shown.
                            if event.key in (pygame.K_RETURN,
                                             pygame.K_KP_ENTER,
                                             pygame.K_SPACE):
                                if seat_cursor >= 0:
                                    lineup.switch_side(seat_cursor)
                            else:
                                seat_cursor = ((seat_cursor + step)
                                               % lineup.seats)
                        elif key == "edit":
                            # Walk the seats; enter starts typing into
                            # the one shown, at its first field.
                            if event.key in (pygame.K_RETURN,
                                             pygame.K_KP_ENTER,
                                             pygame.K_SPACE):
                                if seat_cursor >= 0:
                                    editing = (seat_cursor, 0, field_text(
                                        lineup.rowers[seat_cursor], 0))
                            else:
                                seat_cursor = ((seat_cursor + step)
                                               % lineup.seats)
                        elif key == "save":
                            if event.key in (pygame.K_RETURN,
                                             pygame.K_KP_ENTER,
                                             pygame.K_SPACE):
                                # Seat -1 is the lineup's own name; the
                                # same text entry the rowers use.
                                editing = (-1, 0, "" if lineup.name in
                                           RIG_PRESETS else lineup.name)
                        elif key == "done":
                            lineup_last, lineup = lineup, None
                            seat_cursor = -1
                    continue
                if event.key == pygame.K_ESCAPE:
                    stop_music()
                    return None
                # The secret.  Printable characters typed on the menu
                # feed a five-letter window; the word unlocks the bonus
                # run, adds its row to this menu, and is remembered.
                if secret.feed(getattr(event, "unicode", "") or ""):
                    args.bonus_unlocked = True
                    args.bonus_note = True
                    _settings.update(bonus_unlocked="on")
                    add_bonus_row(menu, getattr(args, "bonus", "off"))
                action = handle_key(menu, event.key)
                if action == "controls":
                    showing_controls = True
                    continue
                if action == "crew":
                    if lineup is None:
                        lineup = lineup_last or RIG_PRESETS[sorted(RIG_PRESETS)[0]]()
                    pane_cursor = 0
                    continue
                if action in ("options", "weather", "rowers"):
                    chosen = menu.settings()
                    if action == "weather":
                        menu = weather_menu(weather=args.weather,
                                            wind=args.wind)
                    elif action == "rowers":
                        menu = rowers_menu(skill=args.skill,
                                           balance=args.balance)
                    else:
                        menu = options_menu(report=getattr(args, "report", "off"), updates=getattr(args, "updates", "on"), minimap=getattr(args, "minimap", "on"), audio=args.audio,
                                            quality=args.quality)
                    continue
                if action == "back":
                    picked = menu.settings()
                    if picked.get("minimap") not in (None, getattr(args, "minimap", "on")):
                        args.minimap = picked["minimap"]
                        _settings.update(minimap=args.minimap)
                    if picked.get("updates") not in (None, getattr(args, "updates", "on")):
                        args.updates = picked["updates"]
                        _settings.update(updates=args.updates)
                    if picked.get("report") not in (None, getattr(args, "report", "off")):
                        args.report = picked["report"]
                        _settings.update(report=args.report)
                    for key, name in (("audio", "audio"),
                                      ("quality", "quality"),
                                      ("weather", "weather"),
                                      ("wind", "wind"),
                                      ("skill", "skill"),
                                      ("balance", "balance")):
                        if key in picked:
                            setattr(args, name, picked[key])
                    if "bonus" in picked:
                        args.bonus = picked["bonus"]
                    args.lineup = picked.get("lineup")
                    if args.lineup is not None:
                        args.boat = args.lineup.shell
                    # ``chosen`` is the setup menu's own settings, and wind
                    # is no longer one of them -- it lives on the weather
                    # menu -- so it comes from args, which the weather
                    # menu has just written.  Reading it from ``chosen``
                    # was a KeyError on every "Back", which is to say the
                    # weather menu crashed the game on the way out.
                    menu = setup_menu(boat=chosen.get("boat", args.boat),
                                      course=chosen.get("race", args.race),
                                      rate=chosen.get("rate", args.rate),
                                      wind=args.wind)
                    continue
                if action == "start":
                    # NOT stopped here: the world takes half a minute to
                    # build after this, and silence landing the instant
                    # you press go is the loudest possible signal that
                    # something has died.  It plays over the loading
                    # screen and fades when the boat appears.
                    picked = menu.settings()
                    # The rig editor's boat races.  Kept on args so a
                    # restart from the pause menu rebuilds the same crew.
                    picked["lineup"] = lineup_last
                    return picked
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
        overlay.fill((0, 0, 0, 0))
        if showing_controls:
            draw_controls(overlay, font, small, screen.get_size())
        elif lineup is not None:
            draw_plan(overlay, lineup, font, small, screen.get_size(),
                      selected=seat_cursor, editing=editing)
            draw_side_pane(overlay, lineup, font, small, screen.get_size(),
                           cursor=pane_cursor)
        else:
            _update = getattr(getattr(args, "update_check", None), "result",
                              None)
            if _update is not None:
                draw_menu(overlay, menu, font, small, screen.get_size(),
                          footer="NEWER RELEASE: " + _update.line())
            elif getattr(args, "bonus_note", False):
                draw_menu(overlay, menu, font, small, screen.get_size(),
                          footer="BONUS RUN UNLOCKED  .  coins and boosts "
                                 "along the course  .  switch it on above")
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
    parser.add_argument("--no-shadows", action="store_true",
                        help="skip the baked shadow map")
    parser.add_argument("--shadow-size", type=int, default=0,
                        dest="shadow_size",
                        help="shadow map resolution, one side; 0 picks "
                             "it from the course's size")
    parser.add_argument("--shadow-bias", type=float, default=0.35,
                        dest="shadow_bias",
                        help="shadow depth bias, in METRES along the sun")
    parser.add_argument("--shadow-debug", type=int, default=0,
                        dest="shadow_debug",
                        help="1: paint visibility, 2: depth difference, "
                             "3: stored depth, 4: fragment depth")
    parser.add_argument("--dump-shadow", default=None, dest="dump_shadow",
                        help="write the baked shadow map to this PNG")
    parser.add_argument("--cam-up", type=float, default=6.0,
                        dest="cam_up",
                        help="height above the seat for --freecam")
    parser.add_argument("--cam-pitch", type=float, default=-0.15,
                        dest="cam_pitch",
                        help="freecam pitch, radians; negative looks down")
    parser.add_argument("--cam-yaw", type=float, default=0.0,
                        dest="cam_yaw", help="freecam yaw, radians")
    parser.add_argument("--cam-relative", action="store_true",
                        dest="cam_relative",
                        help="measure --cam-yaw from the boat's heading, so "
                             "0 looks along the hull whatever the course")
    parser.add_argument("--cam-back", type=float, default=0.0,
                        dest="cam_back",
                        help="metres astern of the seat for --freecam")
    parser.add_argument("--cam-at", default=None, dest="cam_at",
                        help="absolute freecam position, 'east,north,up' "
                             "in world metres; overrides the seat-relative "
                             "placement")
    parser.add_argument("--no-particles", action="store_true",
                        dest="no_particles",
                        help="force the catch splash off, whatever the "
                             "graphics setting says; for A/B comparison")
    parser.add_argument("--ripple", type=float, default=None,
                        help="micro-ripple slope amplitude; 0 turns it off")
    parser.add_argument("--exact-within", type=float, default=None,
                        dest="exact_within",
                        help="radius, m, within which the surface is "
                             "differentiated per pixel")
    parser.add_argument("--skill", type=float, default=0.55,
                        help="crew consistency, 0 novice to 1 ideal")
    parser.add_argument("--balance", type=float, default=0.55,
                        help="how well the crew sits the boat, 0 to 1")
    parser.add_argument("--weather", default="hazy",
                        choices=tuple(WEATHER),
                        help="what the air is doing")
    parser.add_argument("--quality", default="auto",
                        choices=("auto", "ultra", "minimal", "standard",
                                 "high"),
                        help="graphics tier; auto reads the machine and "
                             "picks one (see coxswain.viz.hardware)")
    parser.add_argument("--samples", type=int, default=4,
                        help="multisample anti-aliasing; 0 turns it off")
    parser.add_argument("--freecam", action="store_true",
                        help="fly the camera around the course instead of "
                             "sitting in the boat")
    parser.add_argument("--no-menu", action="store_true",
                        help="skip the setup menu and use the flags")
    parser.add_argument("--physics-scheme", choices=("rk4", "heun"),
                        default=None,
                        help="integrator; overrides the tier's choice")
    parser.add_argument("--physics", type=float, default=100.0)
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="do not write the diagnostics log")
    parser.add_argument("--prefer-dedicated-gpu", action="store_true",
                        help="ask Windows to draw this program on the "
                             "high-performance GPU (writes a per-program "
                             "preference for the current user, once)")
    parser.add_argument("--water-divisions", type=int, default=None,
                        help="water grid divisions per side; overrides "
                             "the tier, for measurement")
    parser.add_argument("--render-scale", type=float, default=None,
                        help="draw the scene at this fraction of the "
                             "window; overrides the tier")
    parser.add_argument("--report-url", default=None,
                        help="where performance reports go; see "
                             "packaging/phonehome/README.md")
    parser.add_argument("--bonus", choices=("on", "off"), default=None,
                        help="the bonus run (coins and boosts along the "
                             "course); normally reached from the setup "
                             "menu once unlocked")
    parser.add_argument("--no-minimap", action="store_true",
                        help="no course map in the corner; the remembered "
                             "setting otherwise")
    parser.add_argument("--no-update-check", action="store_true",
                        help="do not ask GitHub whether a newer release "
                             "exists; the remembered setting otherwise")
    parser.add_argument("--report", choices=("on", "off"), default=None,
                        help="send a performance report at close; the "
                             "remembered setting otherwise")
    parser.add_argument("--bench-passes", action="store_true",
                        help="with --bench: per-pass GPU timer queries too; "
                             "they are sync points that cost about 2 ms a "
                             "frame themselves, so the headline is taken "
                             "without them")
    parser.add_argument("--bench", type=int, default=0, metavar="FRAMES",
                        help="run this many headless frames and print the "
                             "physics / draw split, then exit")
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
    # The remembered report choice, before any menu can show it.
    from coxswain.viz import settings as _settings
    if args.report is None:
        args.report = ("on" if _settings.load().get("report") == "on"
                       else "off")
    # Updates: on unless remembered off or told off for this run.  Started
    # here, before the world build, so the answer is usually in by the
    # time the setup menu is drawn -- and never waited for.
    args.bonus_unlocked = _settings.load().get("bonus_unlocked") == "on"
    if args.bonus is None:
        args.bonus = "off"
    args.minimap = ("off" if (getattr(args, "no_minimap", False)
                              or _settings.load().get("minimap") == "off")
                    else "on")
    args.updates = ("off" if (args.no_update_check
                              or _settings.load().get("updates") == "off")
                    else "on")
    from coxswain.viz.telemetry import build_version as _build_version
    from coxswain.viz.updates import UpdateCheck as _UpdateCheck
    args.update_check = _UpdateCheck(_build_version(),
                                     enabled=args.updates == "on").start()

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
        args.rate = picked["rate"]

    print("building %s ..." % args.race)
    clock0 = time.perf_counter()

    def build_everything():
        """All the heavy work, and none of it touching GL."""
        # The sea first: the flat far-water quad has to be sunk below
        # the deepest trough of the near field, or it hides them.
        sea = sea_for(args.wind, args.fetch, np.radians(args.wind_from))
        trough = float(np.sum(sea.amplitude)) if len(sea.amplitude) else 0.0
        mesh, scene = build_world(args.race, reach=args.reach, step=args.step,
                                  skyline=args.tier.skyline,
                                  tree_mode=args.tier.trees,
                                  water_level=-1.25 * trough - 0.02,
                                  with_buildings=not args.no_buildings,
                                  guide=not args.no_guide,
                                  trees=not args.no_trees)
        return sea, trough, mesh, scene, build_boat(
            args.boat, args.rate, lineup=getattr(args, "lineup", None))

    from coxswain.viz.menu import tier_settings
    from coxswain.viz.telemetry import Telemetry, install_excepthook
    from coxswain.viz.hardware import probe as probe_hardware
    from coxswain.viz.hardware import request_dedicated_gpu

    # The diagnostics file opens before anything can fail, so even a
    # failed start is on record.
    telemetry = Telemetry(enabled=not args.no_diagnostics)
    if telemetry.enabled:
        print("   diagnostics: %s" % telemetry.path)
        install_excepthook(telemetry)
    if args.prefer_dedicated_gpu:
        print("   " + request_dedicated_gpu())
    if args.bench:
        # --bench is headless: give it somewhere to draw if the caller
        # did not, and always give it its frame count.
        if not args.shot:
            args.shot = os.path.join(tempfile.gettempdir(),
                                     "coxswain-bench.png")
        args.frames = int(args.bench)
        args.no_menu = True
    # The tier gates the world build, and the build comes before there
    # is a GL context to ask -- so "auto" is decided from what the OS
    # says the machine has.  The live renderer is checked once the
    # context exists, and the idle-dedicated-GPU notice comes from that.
    hardware = probe_hardware(None)
    if args.quality == "auto":
        args.quality = hardware.recommended_tier()
        print("   graphics: auto -> %s  (%s)"
              % (args.quality, "; ".join(hardware.adapters) or "no adapter list"))
    tier = tier_settings(args.quality)
    divisions, keep_trees, rich_water, want_particles = (
        tier.water_divisions, tier.trees != "off", tier.rich_water,
        tier.particles)
    if not keep_trees:
        args.no_trees = True
    # The tier's knobs, applied only where the user did not type a
    # flag: a flag is a decision, a tier is a default.
    if args.reach == 900.0:
        args.reach = float(tier.reach)
    if args.step == 8.0:
        args.step = float(tier.step)
    if args.samples == 4:
        args.samples = int(tier.samples)
    if args.shadow_size == 0 and tier.shadow_size:
        args.shadow_size = int(tier.shadow_size)
    if tier.shadow == "off":
        args.no_shadows = True
    if float(args.physics) == 100.0:
        # 100 was the old fixed rate.  60 is the same boat to 0.2 mm
        # (tests/unit/test_physics_rate.py) at 40% fewer evaluations.
        args.physics = float(tier.physics_hz)
    args.tier = tier
    print("   graphics tier: %s  (reach %.0f m, step %.0f m, trees %s, "
          "shadow %s, water %s, fog %s, skyline %s, scale %.2f)"
          % (tier.label, args.reach, args.step, tier.trees, tier.shadow,
             "rich" if rich_water else "flat", tier.fog,
             "on" if tier.skyline else "off", tier.render_scale))
    print("   physics: %.0f Hz" % float(args.physics))

    label = "Building %s" % dict(
        charles="the Charles", totl="Tail of the Lake",
        hotl="Head of the Lake").get(args.race, args.race)
    # The build allocates hundreds of thousands of small Python objects
    # and frees them again.  CPython's cyclic collector triggers on
    # allocation counts, so it runs over and over during the build --
    # walking a heap that is mostly live numpy scaffolding -- and finds
    # nothing to free.  Off for the build, one sweep after, and then
    # the long-lived world is frozen out of every later collection so
    # the frame loop's small garbage is all it ever has to look at.
    import gc
    gc.disable()
    try:
        if screen is not None:
            sea, trough, mesh, scene, (boat, made) = run_loading(
                screen, label, build_everything)
        else:
            sea, trough, mesh, scene, (boat, made) = build_everything()
    finally:
        gc.enable()
    gc.collect()
    gc.freeze()
    telemetry.build_seconds = time.perf_counter() - clock0
    telemetry.section("world", [
        "triangles: %d in %d parts" % (mesh.triangles, len(mesh.parts)),
        "build: %.1f s" % (time.perf_counter() - clock0),
        "tier: %s" % getattr(args, "quality", "?"),
    ])
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
    # The crew's own inconsistency.  Applied once per stroke at the
    # catch, which is where a rower commits: within a stroke they are
    # deterministic, because they execute the stroke they started.
    from coxswain.crew.variability import for_skill

    variability = for_skill(args.skill)

    #: What the coxswain is asking for, as a multiple of race pace.
    #: Defined here rather than with the rest of the loop state because
    #: draw() closes over it and headless runs draw() before the
    #: interactive loop ever starts.
    call = 1.0

    # The crew's balance reflex, graded by experience.  The bare default
    # is a PD loop with a flat 4000 N m of authority available at every
    # instant, which holds an eight to a quarter of a degree -- better
    # than any crew rows.  The real limit is that the blades are the only
    # thing to push against: on the recovery the authority is nearer 50
    # N m, most of it from leaning the trunk rather than from the hands.
    # Wiring that in is what makes the boat something to sit rather than
    # something that sits itself.
    cox.balance = balance_for_experience(boat, args.balance)

    # Blades touching the water when the boat is not sat.  Both halves:
    # the skim drag and roll moment, and the length the drive loses when
    # a blade goes in early.  Nothing had ever constructed one of these,
    # so both were switched off everywhere -- including in the report
    # version, which is where they were assumed to be running.
    from coxswain.crew.blade_contact import BladeContact

    # Wind, at last, and only the part the weather adds.  The still-air
    # share is already in the hull's calibrated resistance, so this
    # contributes nothing in a calm and the excess in a blow -- which is
    # what stops it being counted twice.
    from coxswain.hydro.wind import AeroModel, UniformWind

    _aero = AeroModel.calibrate(boat)
    _wind_field = UniformWind(speed=float(args.wind),
                              bearing=np.radians(float(args.wind_from)))
    simulator = RowingSimulator(boat, coxswain=cox, fast=True,
                                blade_contact=BladeContact.from_boat(boat),
                                aero=_aero, wind=_wind_field)
    # The integrator, by tier: two evaluations a step at the low tiers,
    # four at High.  A typed flag wins, as every tier knob does.
    simulator.scheme = (args.physics_scheme
                        or getattr(args.tier, "physics_scheme", "rk4"))
    print("   physics: %s at %.0f Hz" % (simulator.scheme, args.physics))
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

    # -- what the crew can actually hold -------------------------------
    #
    # The catalog's boats row at a force scale, not at a power, and that
    # scale turns out to be about 470 W per rower -- above world class,
    # and half again what anyone holds for six minutes.  Nothing noticed
    # because nothing tracked the reserve.
    #
    # So the crew starts at the pace the two-parameter model says is
    # ideal for this piece: P = CP + W'/T, which spends the reserve
    # exactly at the line.  The coxswain can call for more, and the
    # crew will give it, and then it is gone.
    from coxswain.crew.exertion import (WPrimeBalance, mean_handle_power,
                                        optimal_pace)

    reserve = WPrimeBalance()
    reference_power = mean_handle_power(boat)
    # Along the course, not the number of points in it.
    _pts = np.asarray(course, dtype=float)
    course_metres = float(np.hypot(*np.diff(_pts[:, :2], axis=0).T).sum())
    race_seconds = max(course_metres / 4.5, 60.0)
    nominal_power = optimal_pace(race_seconds)
    base_scale = nominal_power / max(reference_power, 1.0)
    print("   crew: %.0f W/rower at scale 1.0; racing at %.0f W "
          "(CP %.0f, reserve %.0f kJ over %.0f s)"
          % (reference_power, nominal_power, reserve.critical_power,
             reserve.capacity / 1000.0, race_seconds))

    def fresh_state():
        state = simulator.initial_state(surge_speed=3.6)
        state[0], state[1] = course[begin]
        ahead = course[min(begin + 3, len(course) - 1)] - course[begin]
        heading = float(np.arctan2(ahead[1], ahead[0]))
        state[5] = heading
        state[6] = 3.6 * math.cos(heading)
        state[7] = 3.6 * math.sin(heading)
        return state

    # -- the crew's own timing, and how a bad stroke breaks it ---------
    #
    # This is the loop that was missing.  A roll reaches every rower at
    # once through the hull -- involuntary, instant, no choice about
    # listening -- and knocks their catches apart.  Once the catches are
    # apart the power is no longer aligned, which rolls the boat again.
    # Gathering that back up takes strokes, and it is what a crew is
    # actually doing after a bad one.
    #
    # It advances once per physics step rather than inside derivative():
    # RK4 evaluates the derivative four times per step at three
    # different times, so anything with memory in there is not being
    # integrated, it is being scrambled.
    from coxswain.crew.synchronisation import (CoupledCrew,
                                               stroke_chain_topology)

    crew_timing = CoupledCrew(
        n_seats=boat.n_seats,
        topology=stroke_chain_topology(boat.n_seats),
        # A less experienced crew watches less well and wanders more.
        sensory_gain=0.6 + 1.4 * float(args.skill),
        noise=0.05 * (1.0 - float(args.skill)),
        seed=11,
    )
    _omega = 2.0 * np.pi / float(boat.timing.period)

    def advance_crew(step_t, step_state, step_dt):
        crew_timing.step(step_t, step_dt, _omega,
                         hull_roll_rate=float(step_state[9]))
        boat.phase_offsets = crew_timing.phase_offsets()

    loop = FixedStepLoop(simulator, rate=args.physics,
                         on_step=advance_crew)
    # The crew's stroke tables, now that the timing scatter is set and
    # the crew is grouped the way the derivative will see it.  See
    # Boat.warm_crew_tables: built lazily they land on the opening
    # frames, which is the one place a stall is guaranteed to be seen.
    boat.tabulate_crew = True            # the trainer's budget is a frame
    print("   crew tables: %.1f s" % boat.warm_crew_tables())
    loop.start(fresh_state())
    if args.freecam:
        # Started here rather than with the rest of the loop state:
        # fresh_state is defined just above, and reaching for it earlier
        # is an unbound local -- the third time that exact shape of
        # mistake has been made in this file.
        _state0 = fresh_state()
        _eye, _target, _up = seat_camera(_state0, boat)
        _yaw = float(args.cam_yaw)
        if args.cam_relative:
            _yaw += float(_state0[5])
        _heading = np.array([np.cos(_state0[5]), np.sin(_state0[5]), 0.0])
        _where = (_eye - _heading * float(args.cam_back)
                  + np.array([0.0, 0.0, float(args.cam_up)]))
        if args.cam_at:
            _where = np.array([float(v) for v in args.cam_at.split(",")])
        freecam = FreeCamera(_where, yaw=_yaw, pitch=float(args.cam_pitch))

    # The stroke, out loud.  From the bow of a four you cannot see the
    # blades go in, and without the catch there is nothing in the seat
    # view that separates drive from recovery -- which makes calling the
    # boat impossible, and calling is what this is for.
    audio = None
    ambient = None
    # "off" is a mode in the menu and a way of saying no sound at all.
    if args.audio == "off":
        args.no_sound = True
    if not args.no_sound and not args.shot:
        from coxswain.viz.strokeaudio import StrokeAudio
        audio = StrokeAudio(boat, mode=args.audio)
        from coxswain.viz.ambient import AmbientAudio
        ambient = AmbientAudio(args.wind)
        print("   ambient: %s" % ("wind, water and the odd gull"
                                 if ambient.available else
                                 "off (%s)" % ambient.reason))
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
        if sys.platform == "darwin":
            # macOS will not give you a 3.3 core profile unless you also
            # ask for forward-compatible.  Ask for core alone and it
            # hands back a 2.1 legacy context without complaining, and
            # then every `#version 330` shader in this file fails to
            # compile -- so the program dies at the first draw with a
            # message about the shader rather than about the context,
            # which sends you looking in the wrong place entirely.
            #
            # Apple's OpenGL is deprecated but present, and 4.1 is the
            # ceiling.  Nothing here needs past 3.3.
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_FLAGS,
                pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
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

    # How big the drawable actually is, in PIXELS.
    #
    # On a Retina display these are not the window's numbers.  A Mac
    # asked for a 1280x720 window gives you a 1280x720 window measured
    # in POINTS and a 2560x1440 drawable measured in pixels, and the
    # default framebuffer is the drawable.  Size the offscreen buffers
    # from the window and the water samples the scene at half scale,
    # which puts the screen-space refraction and reflection lookups in
    # the wrong place across the whole surface -- subtly wrong, and only
    # on the machines nobody here can test on.
    #
    # So: pixels for anything that is a framebuffer or a viewport;
    # points for the mouse and the HUD layout, which SDL reports in
    # points and which therefore already agree with args.width.
    # Now the context exists: which GPU is actually drawing.
    hardware = probe_hardware(ctx)
    for line in hardware.lines():
        print("   " + line)
    telemetry.section("hardware", hardware.lines()
                      + ["adapters: %s" % "; ".join(hardware.adapters)])
    telemetry.section("settings", [
        "quality: %s" % args.quality, "race: %s" % args.race,
        "boat: %s" % args.boat, "weather: %s" % args.weather,
        "wind: %.1f" % float(args.wind), "samples: %s" % args.samples,
        "physics: %.0f Hz" % float(args.physics),
        "window: %dx%d" % (args.width, args.height),
    ])
    draw_width, draw_height = args.width, args.height
    if not headless:
        try:
            size = ctx.screen.size
            if size and size[0] > 0 and size[1] > 0:
                draw_width, draw_height = int(size[0]), int(size[1])
        except Exception:                                # pragma: no cover
            pass
        if (draw_width, draw_height) != (args.width, args.height):
            print("   drawable %dx%d for a %dx%d window (%.1fx scaling)"
                  % (draw_width, draw_height, args.width, args.height,
                     draw_width / float(args.width)))

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
    resolve_fbo = None
    # An off-screen scene buffer is needed for rich water (it samples
    # the scene for refraction and reflection) OR for drawing at a
    # fraction of the window: the lowest tier renders three-quarter
    # size and lets the blit scale it up, which on an integrated GPU is
    # worth more than every other saving put together, because a
    # fragment shader's cost is fragments.
    _scale = float(getattr(args.tier, "render_scale", 1.0))
    if args.render_scale is not None:
        _scale = float(args.render_scale)
    _offscreen = rich_water or _scale < 0.999
    if _offscreen:
        size = (max(int(draw_width * _scale), 64),
                max(int(draw_height * _scale), 64))
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
        # Same trap as the shadow map: with comparison mode left on, the
        # refraction read a constant instead of the scene's depth, so
        # the water column under every pixel was "infinitely deep".
        scene_depth_tex.compare_func = ""
        scene_depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        scene_depth_tex.repeat_x = scene_depth_tex.repeat_y = False
        resolve_fbo = ctx.framebuffer(color_attachments=[scene_colour],
                                      depth_attachment=scene_depth_tex)
        print("   rich water on a %s buffer"
              % ("%dx multisampled" % samples if samples else "plain"))
    program = ctx.program(vertex_shader=VERTEX_SHADER,
                          fragment_shader=FRAGMENT_SHADER)
    shadow_prog = ctx.program(vertex_shader=SHADOW_VERTEX,
                              fragment_shader=SHADOW_FRAGMENT)
    sky_prog = ctx.program(vertex_shader=SKY_VERTEX,
                           fragment_shader=SKY_FRAGMENT)
    sky_quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3],
                                   dtype="f4").tobytes())
    sky_vao = ctx.vertex_array(sky_prog, [(sky_quad, "2f", "in_pos")])

    def set_sky(prog, weather=None):
        """Every shader that fogs or reflects shares one sky."""
        row = WEATHER.get(weather or args.weather, WEATHER["hazy"])
        zenith, horizon, glow, density, height, scatter, overcast = row
        _optional(prog, sky_zenith=zenith, sky_horizon=horizon,
                  sun_glow=glow, fog_density=density, fog_height=height,
                  fog_scatter=scatter, sky_overcast=overcast,
                  sky_detail=1 if getattr(args.tier, "sky_detail", True) else 0)
    program["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                 / np.linalg.norm([0.42, 0.30, 0.85]))
    # Set only if the shader still wants them.  Both were the flat sky
    # colour and the linear fog range; with the gradient and the
    # exponential fog in, GLSL drops whichever a shader no longer reads,
    # and assigning to a dropped uniform is a KeyError.
    _optional(program, sky=SKY, far=FAR)
    for _prog in (program, sky_prog):
        _optional(_prog, fog_simple=1 if args.tier.fog == "simple" else 0)
    _optional(program, shadow_taps=1 if args.tier.shadow == "single" else 16)

    static, shadow_casters, static_tiles = [], [], []
    for part in mesh.parts:
        # Sorted into square tiles by triangle centroid BEFORE upload, so
        # each tile is one contiguous range of the buffer and the frame
        # can draw only the tiles in view with one call each.  The whole
        # world was submitted every frame; from the seat about half of
        # it is behind the camera, and the world pass measured
        # vertex-bound (unchanged at a quarter of the pixels), so what
        # is not submitted is not paid for.  Tile size is a balance:
        # smaller culls tighter, larger means fewer draw calls, and on
        # an integrated part a Python-issued draw is ~15 us -- 40 tiles
        # a frame is cheap, 400 is not.
        tiles = tile_part(part, TILE_SIZE)
        buffer = ctx.buffer(tiles.packed)
        static.append(ctx.vertex_array(
            program, [(buffer, "3f 4i1 4u1", "in_pos", "in_normal",
                       "in_colour")]))
        static_tiles.append(tiles)
        # The same buffer, read as positions only, for the depth pass.
        shadow_casters.append(ctx.vertex_array(
            shadow_prog, [(buffer, "3f 8x1", "in_pos")]))
        # The GPU has it now.  Nothing reads the CPU copy again -- the
        # shadow pass and every frame draw from the buffer just made --
        # and 1.3 M triangles of float32 vertices, colours and normals
        # is 140 MB sitting in RAM for nothing.  On a laptop with 8 GB
        # and an iGPU sharing it, that is not nothing.
        part.vertices = part.colours = part.normals = None

    print("   world in %d tiles of %.0f m across %d parts"
          % (sum(len(t.first) for t in static_tiles), TILE_SIZE, len(static)))

    # -- the shadow map, baked once -------------------------------------
    #
    # The world does not move and the sun does not cross the sky during
    # a five kilometre race, so this is rendered exactly once and then
    # only sampled.  That is the whole reason it is affordable here: a
    # per-frame shadow pass over a million triangles would not run on an
    # integrated part, and a static one costs nine texture taps.
    sun_dir = np.array([0.42, 0.30, 0.85])
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    shadow_map = shadow_fbo = None
    if not args.no_shadows:
        # Bounded to the COURSE, not to the mesh.  The mesh reaches out
        # to the skyline 6.5 km away, and a map stretched over that was
        # 4.8 m a texel: every shadow a blob.  What a crew sees shadowed
        # is the water they row on and the banks either side of it, so
        # the map covers the course line plus a margin, and its size is
        # chosen for the texel that gives, not the other way round.
        line = np.asarray(course, dtype=float)[:, :2]
        low = np.append(line.min(axis=0) - SHADOW_MARGIN, -5.0)
        high = np.append(line.max(axis=0) + SHADOW_MARGIN, 120.0)
        centre_world = 0.5 * (low + high)
        extent = float(np.max(high[:2] - low[:2])) * 0.5
        # The depth range is set by the world's extent ALONG THE SUN,
        # not by its height.  With the sun 58 degrees up, a box two
        # kilometres across is two kilometres deep from where the sun
        # sits; sized from the height alone the far plane fell inside
        # the world, everything beyond it read as lit, and the map
        # shadowed nothing at all -- which is exactly how it shipped
        # the first time.
        depth_span = 2.0 * extent + float(high[2] - low[2]) + 400.0
        if args.shadow_size <= 0:
            wanted = 2.0 * extent / SHADOW_TEXEL
            power = int(np.ceil(np.log2(max(wanted, 2.0))))
            args.shadow_size = int(min(4096, max(2048, 2 ** power)))        # Look down the sun at the middle of the world.
        eye_sun = centre_world + sun_dir * (0.5 * depth_span + 50.0)
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(float(sun_dir @ up_hint)) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0])
        sun_view = look_at(eye_sun, centre_world, up_hint)
        sun_proj = orthographic(extent, extent, 1.0, depth_span + 100.0)
        sun_vp = (sun_proj @ sun_view).astype("f4")

        size = int(args.shadow_size)
        shadow_map = ctx.depth_texture((size, size))
        # A depth texture comes with comparison mode ON, for
        # sampler2DShadow.  Read through a plain sampler2D, as here, it
        # then returns the comparison's answer instead of the depth --
        # which is 1.0, which is "lit", which is why a map full of
        # buildings shadowed nothing.  Off, it is a texture of depths.
        shadow_map.compare_func = ""
        print("   shadow map compare_func after clearing: %r"
              % (shadow_map.compare_func,))
        shadow_map.filter = (moderngl.NEAREST, moderngl.NEAREST)
        shadow_map.repeat_x = shadow_map.repeat_y = False
        shadow_fbo = ctx.framebuffer(depth_attachment=shadow_map)
        shadow_fbo.use()
        shadow_fbo.clear()
        shadow_prog["sun_vp"].write(sun_vp.T.tobytes(order="C"))
        # No culling for the bake.  Casting from back faces is the usual
        # trick against acne, but it drops every single-sided caster --
        # the tree impostors are quads -- and the bias handles the acne.
        ctx.disable(moderngl.CULL_FACE)
        for caster in shadow_casters:
            caster.render()
        ctx.enable(moderngl.CULL_FACE)
        if args.dump_shadow:
            # The map as an image, so "are there shadows" can be
            # answered by looking rather than by inference.
            from PIL import Image
            raw = np.frombuffer(shadow_map.read(), dtype="f4")
            raw = raw.reshape(size, size)
            span = raw.max() - raw.min()
            picture = ((raw - raw.min()) / max(span, 1e-9) * 255).astype("u1")
            Image.fromarray(picture[::-1]).save(args.dump_shadow)
            # And the raw numbers beside it, so the map can be checked
            # arithmetically -- which point projects to which texel, and
            # what that texel holds -- rather than by looking at a
            # picture and inferring.
            np.savez(args.dump_shadow + ".npz", depth=raw, sun_vp=sun_vp,
                     low=low, high=high, size=size, sun_dir=sun_dir,
                     depth_range=float(depth_span + 100.0 - 1.0))
            print("   shadow map written to %s (depth %.3f..%.3f)"
                  % (args.dump_shadow, raw.min(), raw.max()))
        print("   shadows: %d x %d over %.0f m, %.1f m a texel, baked once"
              % (size, size, 2.0 * extent, 2.0 * extent / size))

        # The bias is given in metres and converted here.  Given in
        # normalised depth it was 0.0016 of a 2.9 km range -- 4.6 m --
        # which is more than the sun-ray length from the roof of a
        # six-metre boathouse to the ground it shadows, so nothing that
        # size ever cast anything, and that is most of a river bank.
        depth_range = float(depth_span + 100.0 - 1.0)
        shadow_settings = dict(shadow_texel=(1.0 / size, 1.0 / size),
                               shadow_bias=float(args.shadow_bias) / depth_range,
                               shadow_strength=0.75, shadow_map=5,
                               shadow_world_texel=2.0 * extent / size)
        print("   shadow bias %.2f m = %.6f of the depth range"
              % (args.shadow_bias, args.shadow_bias / depth_range))
        shadow_matrix = sun_vp
    else:
        shadow_settings = dict(shadow_strength=0.0, shadow_map=5,
                               shadow_texel=(1.0, 1.0), shadow_bias=0.0,
                               shadow_world_texel=1.0)
        shadow_matrix = np.eye(4, dtype="f4")
    program["sun_vp"].write(shadow_matrix.T.tobytes(order="C"))
    _optional(program, **shadow_settings)
    _optional(program, shadow_debug=int(args.shadow_debug))

    # -- the water ------------------------------------------------------
    water_prog = ctx.program(
        vertex_shader=WATER_VERTEX,
        fragment_shader=(WATER_FRAGMENT_RICH if (scene_fbo is not None
                                                 and rich_water)
                         else WATER_FRAGMENT))
    water_prog["sun"].value = tuple(np.array([0.42, 0.30, 0.85])
                                    / np.linalg.norm([0.42, 0.30, 0.85]))
    _optional(water_prog, sky=SKY, far=FAR)
    water_prog["deep"].value = (0.055, 0.115, 0.155)
    water_prog["hull_length"].value = float(boat.length)
    # WATER_REACH, not args.reach: the first is the moving water
    # patch (110 m), the second is how far the WORLD is built
    # either side of the course (900 m).  Fading at 900 never
    # fires inside a 110 m patch, so the seam stayed exactly
    # where it was.
    _optional(water_prog, patch_reach=float(WATER_REACH))
    from coxswain.viz.planscene import boat_outline
    _ring = np.asarray(boat_outline(boat), dtype=float)
    _optional(water_prog,
              hull_beam=float(_ring[:, 1].max() - _ring[:, 1].min()))
    for _prog in (program, sky_prog, water_prog):
        set_sky(_prog)
    # The second normal and the per-pixel surface are the two things
    # that cost real fragment work, so they are what the quality setting
    # actually buys.  Minimal keeps a short exact radius -- the water
    # right under the eye still has to be smooth, because the mesh
    # showing through it is the most obvious fault there is -- and no
    # ripple at all, which is six simplex evaluations a fragment saved.
    _ripple = RIPPLE_SLOPE if args.tier.rich_water else 0.0
    _exact = float(args.tier.exact_within)
    # Scaled by the wind, so calm water is glass.
    _wind_factor = min(1.0, max(float(args.wind), 0.0) / RIPPLE_FULL_WIND)
    _ripple *= _wind_factor ** 0.5
    if args.ripple is not None:
        _ripple = float(args.ripple)
    if args.exact_within is not None:
        _exact = float(args.exact_within)
    if "sun_vp" in water_prog:
        water_prog["sun_vp"].write(shadow_matrix.T.tobytes(order="C"))
    _optional(water_prog, **shadow_settings)
    _optional(water_prog, shadow_sun=tuple(float(v) for v in sun_dir),
              shadow_boat_size=(0.5 * float(boat.length),
                                0.5 * float(_ring[:, 1].max()
                                            - _ring[:, 1].min())))
    # How much of an incident wave the stem sends back.  A thin stem
    # reflects little; this is the fraction of the beam-to-length ratio
    # that thin-ship scattering gives for a wedge, scaled to sit at
    # about 0.3 for an eight.  Zero for a boat too fine to matter.
    _optional(water_prog, bow_reflect=float(np.clip(
        9.0 * (_ring[:, 1].max() - _ring[:, 1].min()) / float(boat.length),
        0.0, 0.6)))
    _optional(water_prog, heave_gain=HEAVE_GAIN, surge_gain=SURGE_GAIN,
              pixel_angle=math.radians(args.fov) / float(args.height))
    # A flat tier gets no exact band at all: that band is the three
    # surface() sums per fragment, nearest the eye where pixels are
    # densest, and flat water has nothing in it worth resolving.
    _flat = 1 if not getattr(args.tier, "rich_water", True) else 0
    if _flat:
        _exact = 0.0
    _optional(water_prog, ripple_slope_amp=_ripple,
              ripple_scale=RIPPLE_SCALE, ripple_fade=RIPPLE_FADE,
              exact_within=_exact, ripple_wake_gain=RIPPLE_WAKE_GAIN,
              water_flat=_flat)
    #: Texture units 0-2 are the HUD, the near field and the wave table.
    #: 5 is the baked shadow map.
    SCENE_UNIT, DEPTH_UNIT, SHADOW_UNIT = 3, 4, 5
    if scene_fbo is not None:
        _optional(water_prog, scene=SCENE_UNIT)
        _optional(water_prog, scene_depth=DEPTH_UNIT)
        # Rich-only uniforms; the plain shader behind a render-scale
        # buffer declares none of them.
        _optional(water_prog, viewport=(float(scene_fbo.size[0]),
                                        float(scene_fbo.size[1])),
                  near_plane=0.25, far_plane=float(FAR))
        # How hard the surface bends what is behind it, before the 1/range
        # falloff.  Set by eye against the near water: enough that a wave
        # visibly displaces the bank behind it, not so much that the
        # shoreline swims.
        _optional(water_prog, refract_scale=5.5)
        _optional(water_prog, reflect_steps=int(args.tier.reflect_steps))
        _optional(water_prog, fog_simple=1 if args.tier.fog == "simple" else 0)
        _optional(water_prog, shadow_taps=(
            1 if args.tier.shadow == "single" else 16))
    _optional(water_prog, water_detail=0 if not args.tier.rich_water else 1,
              sky_detail=1 if getattr(args.tier, "sky_detail", True) else 0)
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
    if args.water_divisions:
        divisions = int(args.water_divisions)
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

    # The bonus run: pickups laid along the course, drawn through the
    # same float-layout path as the boat, collected in the loop below.
    from coxswain.viz.bonus import BonusRun, pickup_solids
    bonus = (BonusRun.along(course) if getattr(args, "bonus", "off") == "on"
             else None)
    pickup_buffer = ctx.buffer(reserve=256 * 1024, dynamic=True)
    pickup_vao = ctx.vertex_array(
        program, [(pickup_buffer, "3f 3f 3f", "in_pos", "in_normal",
                   "in_colour")])
    if bonus is not None:
        print("   bonus run: %d coins, %d boosts along the course"
              % (bonus.total_coins, sum(1 for p in bonus.pickups
                                         if p.kind == "boost")))

    # Only built at High.  Nothing spawns into it otherwise, so the
    # per-frame upload and draw disappear rather than running on an
    # empty pool.
    # Timer queries are sync points: measured, they cost 2.3 ms of an
    # 11.8 ms frame on an integrated part.  The headline number is
    # taken WITHOUT them; the per-pass breakdown is asked for.
    passes = PassTimer(ctx, enabled=bool(args.bench) and args.bench_passes)
    splashes = (SplashSystem()
                if want_particles and not args.no_particles else None)
    splash_prog = ctx.program(vertex_shader=SPLASH_VERTEX,
                              fragment_shader=SPLASH_FRAGMENT)
    # gl_PointSize is a compile-time no-op in core profile unless this is
    # on -- without it every droplet draws at 1 pixel regardless of what
    # the vertex shader sets.
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    splash_buffer = ctx.buffer(
        reserve=SplashSystem.CAPACITY * 4 * 4, dynamic=True)
    splash_vao = ctx.vertex_array(
        splash_prog, [(splash_buffer, "3f 1f", "in_pos", "in_age")])

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
        # The sky is a full-screen quad.  Drawn FIRST it shades every
        # pixel and the world then paints over most of them; drawn LAST
        # at depth 1.0 with the test on, it shades only what nothing
        # covered -- the actual sky, a fifth of the frame from the seat.
        # Only on the plain-water tiers: the rich water refracts the
        # scene texture the world was copied into, and a sky missing
        # from that copy would show as a black band where the far water
        # meets the horizon.
        sky_last = scene_fbo is None
        if not sky_last:
            ctx.disable(moderngl.DEPTH_TEST)
            with passes.span("sky"):
                sky_vao.render()
            ctx.enable(moderngl.DEPTH_TEST)
        if shadow_map is not None:
            shadow_map.use(SHADOW_UNIT)
        program["colour_scale"].value = 1.0 / 255.0     # packed world
        planes = frustum_planes(projection @ view)
        with passes.span("world"):
            drawn = 0
            for vao, tiles in zip(static, static_tiles):
                for k in visible_tiles(tiles, planes):
                    vao.render(first=int(tiles.first[k]),
                               vertices=int(tiles.count[k]))
                    drawn += 1
            draw.tiles_drawn = drawn
        if scene_fbo is not None:
            # Carry the world forward into the buffer the water draws
            # into, so the water reads an untouched copy of what is
            # behind it rather than the surface it is drawing.
            ctx.copy_framebuffer(resolve_fbo, scene_fbo)
            scene_fbo.use()
        # The water goes on after the land, so the shore reads through it
        # at the edges and the patch does not have to be clipped.
        speed = float(np.hypot(state[6], state[7]))
        # Differenced against the previous frame in SIMULATION time --
        # `t`, which this function is handed -- rather than wall clock,
        # so a stutter or a pause does not read as an acceleration.
        _dt = max(float(t) - draw.last_t, 1e-3)
        _raw = (speed - draw.last_speed) / _dt
        draw.surge = 0.7 * draw.surge + 0.3 * float(np.clip(_raw, -6.0, 6.0))
        draw.last_t = float(t)
        water_prog["mvp"].write((projection @ view).T.tobytes(order="C"))
        water_prog["eye"].value = tuple(float(v) for v in eye)
        water_prog["centre"].value = (float(state[0]), float(state[1]))
        water_prog["boat"].value = (float(state[0]), float(state[1]),
                                    float(state[5]))
        water_prog["speed"].value = speed
        # Heave velocity straight off the state vector; surge
        # acceleration differenced between frames, which is the only
        # place the rate of change of speed is available.
        _optional(water_prog, heave_rate=float(state[8]),
                  surge_accel=float(draw.surge),
                  shadow_boat=(float(state[0]), float(state[1]),
                               float(state[5])))
        draw.last_speed = speed
        # The reserve runs on real elapsed time, not on the stroke, so a
        # paused boat does not quietly recover.
        # What the crew are asked for: the coxswain's call, plus a live
        # boost.  A boost is the crew briefly rowing above themselves,
        # paid for out of the same reserve as any other call.
        call_live = call + (bonus.call_bonus() if bonus is not None else 0.0)
        _elapsed = max(float(t) - draw.last_reserve_t, 0.0)
        if _elapsed > 0.0:
            draw.w_prime = reserve.step(draw.w_prime,
                                        nominal_power * call_live, _elapsed)
            draw.last_reserve_t = float(t)
        water_prog["time"].value = float(t)
        for _prog in (program, sky_prog, water_prog):
            _optional(_prog, sky_time=float(t))
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
                # The oar model is a plan view -- there is no vertical
                # entry speed anywhere in it to measure, the blade tip
                # only ever has x and y.  What IS real and available is
                # how fast the blade is travelling horizontally at the
                # instant it catches, finite-differenced against a
                # moment before: a slammed catch has the blade already
                # moving fast when it grips; a placed one is nearly
                # stopped.  Used as the splash's kick, not as a claim
                # about the actual water entry, which this model does
                # not resolve.
                EPS = 0.01
                before, _ = oar_lines(boat, t - EPS)
                rot = hull_to_abs(np.asarray(state[3:6], dtype=float))
                here = np.asarray(state[0:3], dtype=float)
                # What the crew can give, and then what they actually
                # give.  Asked for at the catch because that is when a
                # rower commits to a stroke: a call lands on the next
                # one, not on the one already being pulled.
                asked = nominal_power * call_live
                if draw.w_prime <= 0.0:
                    # Empty.  This is not the crew choosing to ease off;
                    # it is the rate falling whatever the coxswain says.
                    asked = min(asked, reserve.critical_power)
                    draw.faded = True
                elif asked <= reserve.critical_power:
                    draw.faded = False
                scale = asked / max(reference_power, 1.0)
                if variability is not None and variability.power_sigma > 0.0:
                    variability.apply(boat, base=scale * np.asarray(
                        getattr(boat, "seat_ratios", np.ones(boat.n_seats)),
                        dtype=float))
                else:
                    boat.power_scales = (scale * np.asarray(
                        getattr(boat, "seat_ratios", np.ones(boat.n_seats)),
                        dtype=float))
                for oar, was in zip(lines, before):
                    tip = np.append(np.asarray(oar)[-1], 0.0) @ rot.T + here
                    prior = np.asarray(was)[-1]
                    kick = float(np.hypot(*(np.asarray(oar)[-1] - prior))
                                / EPS)
                    trail.drop(float(tip[0]), float(tip[1]), t)
                    if splashes is not None:
                        splashes.spawn(tip, kick * 0.5, t)
            # A blade dragging on the recovery.  Not the catch: an unset
            # boat carries its low side's blades on the water between
            # strokes, and the physics already charges for that -- the
            # skim drag and the righting moment in BladeContact.loads --
            # so the picture should show what the hull is feeling.
            # Gated on the SAME immersion the loads use, at the same
            # roll, so a blade that splashes is a blade that is slowing
            # the boat, and one that is clear throws nothing.
            contact = getattr(simulator, "blade_contact", None)
            if splashes is not None and contact is not None:
                roll = float(state[3])
                if (contact.immersion(roll, +1) > 0.0
                        or contact.immersion(roll, -1) > 0.0):
                    lines, _ = oar_lines(boat, t)
                    rot = hull_to_abs(np.asarray(state[3:6], dtype=float))
                    here = np.asarray(state[0:3], dtype=float)
                    offsets = np.asarray(getattr(boat, "phase_offsets",
                                                 np.zeros(boat.n_seats)),
                                         dtype=float)
                    key = 0
                    for seat_index, seat in enumerate(boat.rig.seats):
                        for lock in seat.oarlocks:
                            oar = lines[key]
                            key += 1
                            # On this seat's own clock, as it is drawn.
                            seat_t = t - float(offsets[seat_index]) * period
                            if boat.timing.is_drive(seat_t):
                                continue          # in the water on purpose
                            depth = contact.immersion(roll, int(lock.side))
                            if depth <= 0.0:
                                continue
                            if (t - draw.last_drag.get(key, -1e9)
                                    < DRAG_SPLASH_INTERVAL):
                                continue
                            draw.last_drag[key] = float(t)
                            tip = (np.append(np.asarray(oar)[-1], 0.0) @ rot.T
                                   + here)
                            # A trickle, kicked by the boat's speed over
                            # the blade and by how deep it is riding.
                            kick = (float(np.clip(speed, 0.0, 5.0))
                                    * (0.12 + 0.8 * min(float(depth), 0.1)))
                            splashes.spawn(tip, kick, t,
                                           count=DRAG_SPLASH_DROPLETS)
            draw.last_phase = phase
        water_prog["puddles"].write(trail.as_uniform(t).tobytes())
        near_tex.use(NEAR_UNIT)          # never trust the binding
        wave_tex.use(WAVE_UNIT)
        if scene_fbo is not None:
            scene_colour.use(SCENE_UNIT)
            scene_depth_tex.use(DEPTH_UNIT)
        with passes.span("water"):
            water_vao.render()
        vertices, normals, colours = boat_geometry(boat, hull, t, state,
                                                   crew)
        if vertices is not None and len(vertices):
            blob = np.hstack([vertices, normals, colours]).astype("f4")
            if blob.nbytes <= oar_buffer.size:
                with passes.span("boat"):
                    program["colour_scale"].value = 1.0   # float buffer
                    oar_buffer.write(blob.tobytes())
                    oar_vao.render(vertices=len(vertices))
        if bonus is not None:
            solids = pickup_solids(
                bonus.visible(float(state[0]), float(state[1])), float(t))
            if solids is not None:
                p_vertices, p_colours = solids
                p_blob = np.hstack([p_vertices, _face_normals(p_vertices),
                                    p_colours]).astype("f4")
                if p_blob.nbytes <= pickup_buffer.size:
                    program["colour_scale"].value = 1.0
                    pickup_buffer.write(p_blob.tobytes())
                    pickup_vao.render(vertices=len(p_vertices))

        if sky_last:
            # Depth 1.0 against a buffer cleared to 1.0: LESS would
            # reject every sky pixel, so LEQUAL for this one draw.
            ctx.depth_func = "<="
            with passes.span("sky"):
                sky_vao.render()
            ctx.depth_func = "<"
        droplets = (splashes.as_uniform(t) if splashes is not None
                    else ())
        if len(droplets):
            splash_prog["mvp"].write((projection @ view).T.tobytes(
                order="C"))
            splash_buffer.write(droplets.tobytes())
            # Additive and no depth write: a droplet in front of the
            # water brightens it rather than punching a flat-shaded
            # hole, and it never occludes anything behind it -- right
            # for something this small and this brief.
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.enable(moderngl.BLEND)
            ctx.depth_mask = False
            splash_vao.render(moderngl.POINTS, vertices=len(droplets))
            ctx.depth_mask = True
            ctx.disable(moderngl.BLEND)
        if scene_fbo is not None:
            # And out to the screen, so the HUD has something to sit on.
            with passes.span("copy"):
                if scene_fbo.size != tuple(target.size) and scene_colour is not None:
                    # A smaller scene buffer has to be STRETCHED to the
                    # window.  copy_framebuffer blits one-to-one, which
                    # left the scaled scene in the bottom-left corner of
                    # a black screen.
                    if resolve_fbo is not None and resolve_fbo is not scene_fbo:
                        ctx.copy_framebuffer(resolve_fbo, scene_fbo)
                    target.use()
                    scene_colour.use(SCENE_UNIT)
                    _opaque_blit(ctx, SCENE_UNIT)
                else:
                    ctx.copy_framebuffer(target, scene_fbo)
                    target.use()
        passes.end_frame()

    draw.last_phase = 0.0
    draw.last_drag = {}
    draw.hud_last = None
    draw.tiles_drawn = 0
    draw.last_speed = 0.0
    draw.w_prime = reserve.capacity
    draw.last_reserve_t = 0.0
    draw.faded = False
    draw.last_t = 0.0
    draw.surge = 0.0

    if headless:
        # draw() every step, not just at the end.  Catches -- and every
        # effect keyed off one, puddles and splashes both -- are found
        # by watching the stroke phase WRAP from one draw() call to the
        # next, so calling draw() once after N steps can never see a
        # wrap: draw.last_phase starts at 0.0 and the single phase it is
        # compared against is never negative.  A `--shot` of a boat mid-
        # stroke was silently unable to show either effect.
        _phys, _drw = [], []
        for _ in range(int(args.frames or 0)):
            _t0 = time.perf_counter()
            loop.advance(1.0 / 60.0)
            _t1 = time.perf_counter()
            draw(loop.pose(), loop.t)
            ctx.finish()                 # the GPU's time, not the queue's
            _t2 = time.perf_counter()
            _phys.append((_t1 - _t0) * 1000.0)
            _drw.append((_t2 - _t1) * 1000.0)
            telemetry.frame((_t2 - _t0) * 1000.0, _phys[-1], _drw[-1])
        if args.frames:
            # Steady state, not the start: the first frames carry the
            # JIT and the shaders' first use, which are start-up costs
            # the loading screen should own, not frame costs.
            warm = min(10, max(len(_phys) - 1, 0))
            ph = sorted(_phys[warm:]) or [0.0]
            dr = sorted(_drw[warm:]) or [0.0]
            tot = sorted(p + d for p, d in zip(_phys[warm:], _drw[warm:])) or [0.0]
            pick = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
            line = ("bench %s: %d frames after %d warm-up  physics p50 %.1f "
                    "p95 %.1f ms  draw+gpu p50 %.1f p95 %.1f ms  total p50 "
                    "%.1f ms (%.0f fps)  worst %.0f ms"
                    % (args.quality, len(tot), warm, pick(ph, 0.5),
                       pick(ph, 0.95), pick(dr, 0.5), pick(dr, 0.95),
                       pick(tot, 0.5), 1000.0 / max(pick(tot, 0.5), 1e-9),
                       tot[-1]))
            print("   " + line)
            telemetry.note(line)
            detail = passes.report(warm)
            if detail:
                print("   passes (ms p50/p95): " + detail)
                telemetry.note("passes " + detail)
        if not args.frames:
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
    # The HUD is laid out in POINTS, because that is what pygame's fonts
    # and mouse coordinates are in, and it is then stretched over the
    # drawable.  On a 1:1 display that stretch is the identity and
    # NEAREST keeps the text pin-sharp; on a Retina display it is a 2x
    # magnification, where NEAREST would give visibly blocky letters.
    hud_texture = ctx.texture((args.width, args.height), 4)
    _hud_smooth = (draw_width, draw_height) != (args.width, args.height)
    hud_texture.filter = ((moderngl.LINEAR, moderngl.LINEAR) if _hud_smooth
                          else (moderngl.NEAREST, moderngl.NEAREST))
    rudder, split, paused, running, frames = 0.0, 0.0, False, True, 0
    #: What the coxswain is asking for, as a multiple of race pace, and
    #: what the crew has left to give.  The call is what you SAY; the
    #: reserve decides whether you get it.
    call = 1.0
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
                            boat, _made = build_boat(
                                args.boat, args.rate,
                                lineup=getattr(args, "lineup", None))
                            hull = hull_solid(boat)
                            crew = crew_poses(boat)
                            simulator = RowingSimulator(boat, coxswain=cox,
                                                        fast=True)
                            loop.simulator = simulator
                            if audio is not None:
                                audio.boat = boat
                        menu, paused = None, False
                    elif action == "restart":
                        loop.start(fresh_state())
                        rudder = split = 0.0
                        menu, paused = None, False
                    elif action in ("options", "weather", "rowers"):
                        if action == "weather":
                            menu = weather_menu(weather=args.weather,
                                                wind=args.wind)
                        elif action == "rowers":
                            menu = rowers_menu(skill=args.skill,
                                               balance=args.balance)
                        else:
                            menu = options_menu(report=getattr(args, "report", "off"), updates=getattr(args, "updates", "on"), minimap=getattr(args, "minimap", "on"), audio=args.audio,
                                                quality=args.quality)
                    elif action == "back":
                        picked = menu.settings()
                        if picked.get("minimap") not in (None, getattr(args, "minimap", "on")):
                            args.minimap = picked["minimap"]
                            _settings.update(minimap=args.minimap)
                        if picked.get("updates") not in (None, getattr(args, "updates", "on")):
                            args.updates = picked["updates"]
                            _settings.update(updates=args.updates)
                        if picked.get("report") not in (None, getattr(args, "report", "off")):
                            args.report = picked["report"]
                            _settings.update(report=args.report)
                        if "audio" in picked and picked["audio"] != args.audio:
                            args.audio = picked["audio"]
                            # Rebuilt rather than retuned: the mode
                            # decides which envelope is loaded.
                            if args.audio == "off":
                                audio = None
                            else:
                                from coxswain.viz.strokeaudio import StrokeAudio
                                audio = StrokeAudio(boat, mode=args.audio)
                        args.quality = picked.get("quality", args.quality)
                        if "skill" in picked and picked["skill"] != args.skill:
                            args.skill = picked["skill"]
                            variability = for_skill(args.skill)
                        if ("balance" in picked
                                and picked["balance"] != args.balance):
                            args.balance = picked["balance"]
                            # Rebuilt, not retuned: the authority comes
                            # off the rig and the learned trim starts
                            # over, which is right -- a different crew
                            # has not learned this boat yet.
                            cox.balance = balance_for_experience(
                                boat, args.balance)
                        if abs(picked.get("wind", args.wind)
                               - args.wind) > 1e-9:
                            args.wind = picked["wind"]
                            if ambient is not None:
                                ambient.set_wind(args.wind)
                            water_prog["waves"].write(
                                sea_for(args.wind, args.fetch,
                                        np.radians(args.wind_from))
                                .as_uniform().tobytes())
                            _optional(water_prog, ripple_slope_amp=(
                                RIPPLE_SLOPE
                                * (min(1.0, args.wind / RIPPLE_FULL_WIND)
                                   ** 0.5)
                                if args.tier.rich_water else 0.0))
                        if picked.get("weather", args.weather) != args.weather:
                            # Weather is only uniforms, so it can change
                            # mid-outing without rebuilding anything.
                            args.weather = picked["weather"]
                            for _p in (program, sky_prog, water_prog):
                                set_sky(_p, args.weather)
                        menu = pause_menu(rate=args.rate, wind=args.wind)
                    elif action == "controls":
                        showing_controls = True
                    elif action == "quit":
                        menu = confirm_quit_menu()
                    elif action in ("setup", "quit_yes"):
                        restart_session = action == "setup"
                        running = False
                    continue
                if event.key == pygame.K_ESCAPE:
                    menu = pause_menu(rate=args.rate, wind=args.wind)
                    paused = True
                elif event.key == pygame.K_q and freecam is None:
                    # Ask.  See confirm_quit_menu.
                    menu = confirm_quit_menu()
                    paused = True
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
                elif event.key in (pygame.K_UP, pygame.K_PAGEUP):
                    call = float(min(call + 0.03, 1.35))
                elif event.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                    call = float(max(call - 0.03, 0.60))
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

        _frame_t0 = time.perf_counter()
        if not paused and menu is None and not showing_controls:
            loop.advance(frame)
            _phys_ms = (time.perf_counter() - _frame_t0) * 1000.0
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
        if bonus is not None:
            _rot = hull_to_abs(np.asarray(pose[3:6], dtype=float))
            _bow = (np.asarray(pose[0:3], dtype=float)
                    + _rot @ np.array([0.5 * float(boat.length), 0.0, 0.0]))
            for _taken in bonus.collect(float(_bow[0]), float(_bow[1])):
                if audio is not None and hasattr(audio, "catch"):
                    try:
                        audio.catch()          # the nearest sound there is
                    except Exception:
                        pass
        draw(pose, loop.t)

        speed = float(np.hypot(pose[6], pose[7]))
        seconds = 500.0 / speed if speed > 0.2 else 0.0
        _left = draw.w_prime / max(reserve.capacity, 1.0)
        lines = ["%d:%04.1f   %.2f m/s   rate %.0f"
                 % (int(seconds // 60), seconds % 60, speed, boat.timing.rate),
                 # The call, and what is left to pay for it.  "FADING"
                 # is the crew unable to hold what was asked, which is
                 # the one thing a coxswain must be able to see.
                 ("call %+.0f%%   crew %3.0f%%%s"
                  % (100.0 * (call - 1.0), 100.0 * _left,
                     "   FADING" if draw.faded else "")),
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
                 ("Up/Down call    F1 free camera    Esc menu    V astern"
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
            draw.hud_last = None          # the menu overwrote the HUD
            hud_texture.use(0)
            _blit(ctx, hud_texture)
            pygame.display.flip()
            frames += 1
            continue

        # Only recompose and upload a HUD that changed.  Rendering the
        # text through pygame and pushing a full-window RGBA texture
        # every frame cost about 3 ms on an integrated part -- more
        # than the boat, the sky and the copy together -- for a picture
        # that changes a few times a second.  Its state is the text and
        # where the rudder knob sits, to the pixel; when those match
        # the last frame's, the texture already holds this picture.
        #
        # The stick.  On the mouse there is nothing else to tell you where
        # the rudder is -- the pointer is hidden and the boat answers a
        # second later -- and on a boat with a standing yaw bias, seeing
        # the trim you are holding is most of what makes it steerable.
        bar_w = int(args.width * MOUSE_SPAN)
        bar_x = (args.width - bar_w) // 2
        bar_y = args.height - 40
        knob = bar_x + int(bar_w * (0.5 + 0.5 * rudder / RUDDER_LIMIT))
        # The minimap moves with the boat, which would put the HUD back
        # to an upload every frame.  Its part of the key is the boat's
        # position to 2 m and heading to 5 degrees: a few uploads a
        # second at race pace, none when stopped.
        if bonus is not None:
            lines.append(bonus.hud_line())
        _map_on = getattr(args, "minimap", "on") == "on"
        _map_key = ((int(state[0] / 2.0), int(state[1] / 2.0),
                     int(math.degrees(state[5]) / 5.0)) if _map_on else None)
        _hud_key = (tuple(lines), knob, _map_key)
        _hud_changed = _hud_key != draw.hud_last
        draw.hud_last = _hud_key
        if _hud_changed:
            overlay.fill((0, 0, 0, 0))
            for row, text in enumerate(lines):
                overlay.blit(font.render(text, True, (233, 240, 245)),
                             (14, 12 + row * 20))
            pygame.draw.rect(overlay, (12, 17, 21, 170),
                             (bar_x - 62, bar_y - 20, bar_w + 124, 46))
            pygame.draw.line(overlay, (70, 82, 92), (bar_x, bar_y),
                             (bar_x + bar_w, bar_y), 3)
            pygame.draw.line(overlay, (110, 124, 136), (args.width // 2,
                                                        bar_y - 9),
                             (args.width // 2, bar_y + 9), 2)
            pygame.draw.circle(overlay, (255, 146, 72), (knob, bar_y), 9)
            pygame.draw.circle(overlay, (18, 24, 29), (knob, bar_y), 5)
            for label, at in (("port", bar_x - 52),
                              ("stbd", bar_x + bar_w + 14)):
                overlay.blit(font.render(label, True, (128, 142, 152)),
                             (at, bar_y - 9))
            # The HUD is a texture blitted over the scene: pygame cannot
            # draw into an OpenGL window directly.  One texture, rewritten
            # -- and now only when the picture on it changed.
            if _map_on:
                draw_minimap(pygame, overlay, course, scene.buoys, state,
                             (args.width, args.height))
            hud_texture.write(_surface_bytes(pygame, overlay))
        ctx.disable(moderngl.DEPTH_TEST)
        hud_texture.use(0)
        _hud_blit(ctx)
        ctx.enable(moderngl.DEPTH_TEST)

        _draw_ms = (time.perf_counter() - _frame_t0) * 1000.0 - locals().get(
            "_phys_ms", 0.0)
        _t_present = time.perf_counter()
        pygame.display.flip()
        _present_ms = (time.perf_counter() - _t_present) * 1000.0
        frames += 1
        telemetry.frame((time.perf_counter() - _frame_t0) * 1000.0,
                        locals().get("_phys_ms", 0.0), _draw_ms,
                        _present_ms,
                        context=lambda: "t=%.1f speed=%.2f quality=%s "
                                        "paused=%s menu=%s"
                        % (loop.t, float(np.hypot(loop.state[6],
                                                  loop.state[7])),
                           args.quality, paused, menu is not None))
        _phys_ms = 0.0
        if args.frames and frames >= args.frames:
            running = False

    pygame.quit()
    print("%d frames, %d physics steps" % (frames, loop.steps))
    # The performance report, last, from what the telemetry already
    # knows.  Off unless switched on; see coxswain/viz/phonehome.py.
    try:
        from coxswain.viz import phonehome
        telemetry.dropped_seconds = float(getattr(loop, "dropped", 0.0))
        report = phonehome.build_report(
            telemetry, probe=hardware,
            settings={"race": args.race, "boat": args.boat,
                      "quality": args.quality, "physics": float(args.physics),
                      "window": "%dx%d" % (args.width, args.height),
                      "weather": args.weather, "wind": float(args.wind)},
            tier=getattr(args.tier, "key", args.quality),
            extra={"build_s": round(float(telemetry.build_seconds), 1),
                   "exceptions": int(telemetry.exceptions)})
        outcome = phonehome.send(report, phonehome.report_url(args.report_url),
                                 enabled=(args.report == "on"),
                                 log=telemetry.note)
        print("   report: " + outcome)
    except Exception as error:                              # never fatal
        print("   report: not sent (%s)" % type(error).__name__)
    if bonus is not None:
        from coxswain.viz import settings as _settings_mod
        best = int(_settings_mod.load().get("bonus_best") or 0)
        score = bonus.score()
        telemetry.note("bonus run: %d coins of %d, %d boosts, score %d"
                       % (bonus.coins, bonus.total_coins, bonus.boosts, score))
        if score > best:
            _settings_mod.update(bonus_best=score)
            print("   bonus run: %d -- a new best" % score)
        else:
            print("   bonus run: %d (best %d)" % (score, best))
    telemetry.close()
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


#: Side of a world tile, metres.  See the upload loop.
TILE_SIZE = 350.0


class TiledPart:
    """A part's triangles sorted by tile, with each tile's range and sphere."""

    __slots__ = ("packed", "first", "count", "centre", "radius", "total")

    def __init__(self, packed, first, count, centre, radius, total):
        self.packed = packed
        self.first = first          # (T,) first vertex of each tile
        self.count = count          # (T,) vertices in each tile
        self.centre = centre        # (T, 3) bounding-sphere centres
        self.radius = radius        # (T,)
        self.total = int(total)


def tile_part(part, size: float) -> TiledPart:
    """Sort a part's triangles into ``size``-metre tiles.

    Everything is done on the triangle level, so a triangle is never
    split and never drawn twice: it lives in the tile its centroid is
    in, and the tile's sphere is grown to hold every vertex of every
    triangle in it -- a building on a tile edge simply makes that
    tile's sphere a little larger.
    """
    vertices = np.asarray(part.vertices, dtype="f4")
    n = len(vertices)
    if n == 0:
        empty = np.zeros(0, dtype=int)
        return TiledPart(part.packed(), empty, empty, np.zeros((0, 3)),
                         np.zeros(0), 0)
    tri = vertices.reshape(-1, 3, 3)
    centroid = tri[:, :, :2].mean(axis=1)
    cell = np.floor(centroid / float(size)).astype(np.int64)
    key = cell[:, 0] * 1_000_003 + cell[:, 1]
    order = np.argsort(key, kind="stable")
    key = key[order]
    tri = tri[order]
    # tile boundaries in the sorted order
    edges = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(key)]])
    first = starts * 3
    count = (stops - starts) * 3
    centre = np.zeros((len(starts), 3), dtype="f4")
    radius = np.zeros(len(starts), dtype="f4")
    for k, (a, b) in enumerate(zip(starts, stops)):
        pts = tri[a:b].reshape(-1, 3)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre[k] = 0.5 * (lo + hi)
        radius[k] = float(np.linalg.norm(hi - centre[k]))
    # rebuild the part in the sorted order so packed() matches the ranges
    from coxswain.viz.worldmesh import MeshPart
    vert_order = (order[:, None] * 3 + np.arange(3)[None, :]).ravel()
    sorted_part = MeshPart(part.name, vertices[vert_order],
                           np.asarray(part.colours, dtype="f4")[vert_order],
                           None if part.normals is None
                           else np.asarray(part.normals, dtype="f4")[vert_order])
    return TiledPart(sorted_part.packed(), first, count, centre, radius, n)


def frustum_planes(vp: np.ndarray) -> np.ndarray:
    """Six planes ``(a, b, c, d)`` from a view-projection matrix, with the
    normals pointing INTO the frustum, so a point is inside when every
    ``a x + b y + c z + d`` is positive."""
    m = np.asarray(vp, dtype="f8")
    rows = [m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1],
            m[3] + m[2], m[3] - m[2]]
    planes = np.array(rows)
    norm = np.linalg.norm(planes[:, :3], axis=1)
    return planes / np.maximum(norm, 1e-12)[:, None]


def visible_tiles(tiles: TiledPart, planes: np.ndarray) -> np.ndarray:
    """Indices of the tiles whose spheres touch the frustum."""
    if tiles.total == 0:
        return np.zeros(0, dtype=int)
    d = tiles.centre @ planes[:, :3].T + planes[:, 3][None, :]   # (T, 6)
    return np.flatnonzero(np.all(d > -tiles.radius[:, None], axis=1))


class PassTimer:
    """GPU time per named pass, from timer queries, for --bench.

    A frame's cost on an integrated GPU turned out not to follow the
    triangle count or the pixel count -- ultra at 300k triangles drew in
    27.8 ms and minimal at 510k in 29 -- so the cost is somewhere
    specific, and a total cannot say where.  Each pass gets its own
    query; ``report`` gives medians over the bench's frames.  Off (a
    no-op context) unless the bench asks, because a query is a sync
    point and the game must not pay for it.
    """

    def __init__(self, ctx, enabled: bool = False):
        self.ctx = ctx
        self.enabled = bool(enabled)
        self.samples = {}
        self._frame = {}

    class _Span:
        def __init__(self, timer, name):
            self.timer, self.name = timer, name

        def __enter__(self):
            if self.timer.enabled:
                self.query = self.timer.ctx.query(time=True)
                self.query.__enter__()
            return self

        def __exit__(self, *exc):
            if self.timer.enabled:
                self.query.__exit__(*exc)
                self.timer._frame[self.name] = (
                    self.timer._frame.get(self.name, 0.0)
                    + self.query.elapsed / 1e6)         # ns -> ms
            return False

    def span(self, name):
        return PassTimer._Span(self, name)

    def end_frame(self):
        if not self.enabled:
            return
        for name, ms in self._frame.items():
            self.samples.setdefault(name, []).append(ms)
        self._frame = {}

    def report(self, warm: int = 10) -> str:
        if not self.samples:
            return ""
        rows = []
        for name, values in self.samples.items():
            v = sorted(values[warm:] or values)
            rows.append((v[len(v) // 2], name, v[min(len(v) - 1,
                                                    int(0.95 * len(v)))]))
        rows.sort(reverse=True)
        return "  ".join("%s %.1f/%.1f" % (name, p50, p95)
                         for p50, name, p95 in rows)


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


def add_bonus_row(menu, value: str = "off") -> None:
    """Put the Bonus run row on a setup menu, once, above the last row."""
    from coxswain.viz.menu import REPORT_CHOICES, Choice

    if any(row.key == "bonus" for row in menu.rows):
        return
    row = Choice("bonus", "Bonus run", list(REPORT_CHOICES),
                 index=1 if value == "on" else 0)
    menu.rows.insert(max(len(menu.rows) - 1, 0), row)


#: Minimap box: size in pixels, and the margin from the corner.
MINIMAP_SIZE = 170
MINIMAP_MARGIN = 14


def draw_minimap(pygame, overlay, course, buoys, state, size) -> None:
    """The course from above, in the top-right corner.

    The whole course fits the box -- a head race is long and thin, so
    the box is what is long and thin about it -- with the buoys as
    dots and the boat as an arrow pointing the way it is heading.  It
    is a map, not a radar: north is up, it does not rotate with the
    boat, because the thing a coxswain wants from it is "which way does
    the river bend next", and that is a question about the map.
    """
    course = np.asarray(course, dtype=float)
    if len(course) < 2:
        return
    width, height = size
    box = MINIMAP_SIZE
    x0 = width - box - MINIMAP_MARGIN
    y0 = MINIMAP_MARGIN
    lo = course[:, :2].min(axis=0)
    hi = course[:, :2].max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    scale = (box - 16) / float(span.max())

    def to_px(east, north):
        return (int(x0 + 8 + (east - lo[0]) * scale
                    + 0.5 * ((box - 16) - span[0] * scale)),
                int(y0 + 8 + (hi[1] - north) * scale
                    + 0.5 * ((box - 16) - span[1] * scale)))

    pygame.draw.rect(overlay, (12, 17, 21, 170), (x0, y0, box, box))
    pygame.draw.rect(overlay, (70, 82, 92), (x0, y0, box, box), 1)
    points = [to_px(e, n) for e, n in course[:, :2]]
    if len(points) >= 2:
        pygame.draw.lines(overlay, (150, 164, 176), False, points, 2)
    if buoys is not None:
        for e, n in np.asarray(buoys, dtype=float)[:, :2]:
            pygame.draw.circle(overlay, (255, 146, 72), to_px(e, n), 2)
    bx, by = to_px(float(state[0]), float(state[1]))
    heading = float(state[5])
    # an arrow: tip forward, two tail corners
    tip = (bx + 7 * math.cos(heading), by - 7 * math.sin(heading))
    left = (bx - 5 * math.cos(heading) + 4 * math.sin(heading),
            by + 5 * math.sin(heading) + 4 * math.cos(heading))
    right = (bx - 5 * math.cos(heading) - 4 * math.sin(heading),
             by + 5 * math.sin(heading) - 4 * math.cos(heading))
    pygame.draw.polygon(overlay, (233, 240, 245), [tip, left, right])


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
