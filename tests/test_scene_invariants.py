r"""Things the scene must keep doing, that only ever got checked by eye.

Each of these was verified once, by looking at a render or reading the
source, and then nothing held it.  A shadow pass that quietly stops
including the buildings, or a sky-noise amplitude nudged from 0.045 to
0.45, would look like a rendering change nobody could bisect.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def source():
    with open(os.path.join(ROOT, "scripts", "fpv.py"), encoding="utf-8") as f:
        return f.read()


def test_everything_in_the_world_casts_a_shadow():
    """Buildings and trees must shadow each other, not only the ground.

    The shadow map is baked from ``shadow_casters``, and that list has to
    be built from EVERY mesh part.  Building it from the terrain alone
    would still produce shadows -- the ground would be striped where the
    sun is blocked by nothing -- and would look like a subtle lighting
    bug rather than a missing pass.

    Checked structurally: the caster is appended in the same loop body,
    from the same buffer, as the drawable.  One append each, so the two
    lists cannot drift apart.
    """
    text = source()
    loop = text[text.index("for part in mesh.parts:"):]
    loop = loop[:loop.index("\n\n")]
    assert loop.count("static.append(") == 1, loop
    assert loop.count("shadow_casters.append(") == 1, loop
    # And the bake draws all of them.
    assert "for caster in shadow_casters:" in text


def test_the_world_receives_shadows_on_its_own_normals():
    """The other half: a surface has to be shadowed AS the surface it is.

    Casting is only half a shadow.  The world shader has to call
    ``sun_visibility`` with the fragment's real normal, because the
    lookup point is pushed out along that normal to clear its own
    texel.  Hand it a fixed up-vector instead and every wall is offset
    as though it were flat ground: the building still casts, still goes
    dark in roughly the right place, and its shadow creeps a texel up
    its own face -- a bug that reads as bad art rather than a mistake.

    The flat-normal calls are the water, which really is horizontal.
    """
    text = source()
    calls = [c for c in re.findall(r"sun_visibility\(([^;]*?)\)", text)
             if "float sun_visibility" not in c]
    assert len(calls) >= 3, calls

    on_normal = [c for c in calls if re.search(r",\s*n\s*,", c)]
    assert on_normal, "no surface passes its own normal: %r" % (calls,)

    # And that normal must be normalised where the offset is applied.
    # An un-normalised one scales the push by the vector's length and
    # brings back the striping the offset exists to cure.
    body = text[text.index("float sun_visibility("):]
    body = body[:body.index("\n}")]
    assert "normalize(normal)" in body, body
    assert "shadow_world_texel" in body, body


def test_the_world_actually_contains_things_to_cast():
    """The loop above is only worth having if the parts exist."""
    from coxswain.viz.worldmesh import build_world

    mesh, _scene = build_world("charles", reach=350.0, step=12.0,
                               water_level=-0.15, guide=False)
    names = {part.name for part in mesh.parts}
    assert any("land" in n for n in names), names
    assert any("building" in n or "wall" in n for n in names), names


def test_the_sky_noise_stays_under_the_gradient():
    """Texture, not weather.

    The zenith-to-horizon gradient is the sky's shape; the simplex is
    meant to sit under it so an overcast reads as sky rather than as a
    wall.  A few percent does that.  Anything approaching the gradient
    itself would be a different sky.
    """
    text = source()
    match = re.search(r"base \*= 1\.0 \+ ([0-9.]+) \* sky_overcast", text)
    assert match, "the sky noise term has moved or gone"
    assert float(match.group(1)) <= 0.08, match.group(1)
    # And it is gated on overcast, so a clear day barely takes any.
    # gated on overcast AND on the tier's sky detail, since the low
    # tiers pay for these octaves in every water pixel too
    assert "if (sky_overcast > 0.0 && sky_detail > 0)" in text


def test_the_bow_foam_is_tied_to_the_hull_and_the_speed():
    """A bow wave that does not know the hull is a white rectangle.

    It was exactly that once: a smoothstep box at the stem times a
    uniform white.  What makes it a bow wave is that it follows the
    waterline and dies when the boat does.
    """
    text = source()
    body = text[text.index("float waterline_foam(vec2 p) {"):]
    body = body[:body.index("\n}")]
    assert "speed" in body, "foam must scale with boat speed"
    assert "hull_frame" in body, "foam must be in the hull's own frame"
    assert "half_beam" in body or "beam" in body, body[:200]


@pytest.mark.parametrize("name", ["SplashSystem"])
def test_the_splash_pool_exists_and_empties(name):
    """Droplets at the catch, and none that live for ever."""
    import numpy as np

    import fpv

    pool = getattr(fpv, name)()
    pool.spawn(np.array([0.0, 0.0, 0.0]), speed=1.5, t=0.0)
    assert len(pool.as_uniform(0.05)) > 0
    assert len(pool.as_uniform(30.0)) == 0


# ---------------------------------------------------------------------------
# the drawn crew against the crew the physics carries
# ---------------------------------------------------------------------------
def test_a_heavier_rower_is_drawn_heavier():
    """Girth follows mass, because the dynamics carry mass.

    Limb *lengths* already came from each rower's own de Leva segments,
    so the crew were the right heights -- but every one of them was
    drawn the same width, which put a 70 kg bow on screen as slight as
    a 54 kg stroke.
    """
    from coxswain.crew.anthropometry import RowerAnthropometry
    from coxswain.viz.worldmesh import BONE_REFERENCE, _build_factor

    light = RowerAnthropometry(mass=54.4, stature=1.664, sex="female")
    heavy = RowerAnthropometry(mass=70.3, stature=1.600, sex="female")
    assert _build_factor(heavy) > _build_factor(light) * 1.10

    # The reference build is the one the girths were drawn for, so it
    # must come back unscaled or every existing crew silently changes.
    reference = RowerAnthropometry(mass=88.0, stature=1.90, sex="male")
    assert _build_factor(reference) == pytest.approx(1.0, abs=1e-9)

    # Square-root scaling, not linear: doubling the mass at fixed
    # stature widens a body by 41%, not by 100%.
    doubled = RowerAnthropometry(mass=108.8, stature=1.664, sex="female")
    assert (_build_factor(doubled) / _build_factor(light)
            == pytest.approx(2.0 ** 0.5, rel=1e-9))
    assert BONE_REFERENCE == pytest.approx(88.0 / 1.90)


def test_the_drawn_crew_are_not_all_the_same_size():
    """End to end, on the boat this was noticed in."""
    import numpy as np

    from coxswain.boats import catalog
    from coxswain.crew.anthropometry import RowerAnthropometry
    from coxswain.viz.worldmesh import _build_factor

    people = [RowerAnthropometry(mass=m, stature=s, sex="female")
              for m, s in ((54.4, 1.664), (54.4, 1.575),
                           (56.7, 1.613), (70.3, 1.600))]
    boat = catalog.coxed_four(
        rate=30.0, rower_mass=float(np.mean([p.mass for p in people])),
        rower_stature=float(np.mean([p.stature for p in people])),
        coxswain_mass=72.6, bow_loaded=True, anthropometry=people,
        rig_pattern="bucket, stbd stroke")

    widths = [_build_factor(m.rower.anthropometry) for m in boat.crew]
    assert len(set("%.4f" % w for w in widths)) == len(widths), widths
    # and the bodies themselves still build
    from coxswain.viz.worldmesh import crew_solids
    mesh = crew_solids(boat, 0.0)
    assert mesh is not None and len(mesh.vertices) > 0


# ---------------------------------------------------------------------------
# a blade dragging on the recovery
# ---------------------------------------------------------------------------
def test_a_dragging_blade_trickles_and_a_catch_bursts():
    """The two are told apart by count, and the drag is a few drops."""
    import numpy as np

    import fpv

    pool = fpv.SplashSystem()
    pool.spawn(np.zeros(3), 1.0, 0.0)
    catch = int((pool.birth > -1e8).sum())
    assert catch == fpv.SplashSystem.PER_BLADE

    pool = fpv.SplashSystem()
    pool.spawn(np.zeros(3), 1.0, 0.0, count=fpv.DRAG_SPLASH_DROPLETS)
    drag = int((pool.birth > -1e8).sum())
    assert drag == fpv.DRAG_SPLASH_DROPLETS
    assert 0 < drag < catch

    # count=0 is a no-op rather than an error, so the trickle can be
    # turned off by setting the constant to zero.
    pool = fpv.SplashSystem()
    pool.spawn(np.zeros(3), 1.0, 0.0, count=0)
    assert int((pool.birth > -1e8).sum()) == 0


def test_the_drag_splash_is_gated_on_the_physics_not_on_the_picture():
    """It must ask BladeContact.immersion at the hull's roll, and skip
    the drive.  A splash keyed off the drawn blade height instead would
    fire on a level boat whenever the picture dipped an oar, and would
    disagree with the drag the boat is actually paying."""
    text = source()
    block = text[text.index("A blade dragging on the recovery"):
                 text.index("draw.last_phase = phase")]
    assert "contact.immersion(roll" in block
    assert "is_drive(seat_t)" in block, "must skip the drive"
    assert "DRAG_SPLASH_INTERVAL" in block, "must be rate-limited"
    assert "phase_offsets" in block, "must use each seat's own clock"
    # And a level boat throws nothing: immersion is zero both sides
    # below the roll at which the low blade first touches.
    from coxswain.crew.blade_contact import BladeContact
    contact = BladeContact()
    touch = contact.roll_to_touch()
    assert contact.immersion(0.0, +1) == 0.0
    assert contact.immersion(0.0, -1) == 0.0
    assert contact.immersion(touch * 0.5, +1) == 0.0
    assert contact.immersion(touch * 1.5, +1) > 0.0
    assert contact.immersion(-touch * 1.5, -1) > 0.0
