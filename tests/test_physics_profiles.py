"""The game's physics is frozen, and this is what holds it still.

Two known defects are being fixed offline -- the blade carries no velocity
term, and the crew kinematics come from a stationary ergometer.  The fixes
live in the ``research`` profile and must not reach the shipped trainer
until they have been justified against the validation scorecard.

Nothing enforces that except these tests.  The first group checks the
registry behaves; the second is a *source scan*, in the same spirit as
``test_no_undefined_names.py``: it reads the code the game runs and fails
if anything in it names a profile other than the frozen one.  A behaviour
test cannot catch a profile passed down a path no test exercises -- which
is exactly how v0.12 shipped a crash through a windowed loop every bench
skipped -- so the scan looks at the text instead.
"""

import ast
import io
import os

import pytest

from coxswain import physics


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Everything the released trainer can execute.  ``scripts/fpv.py`` is the
#: entry point; ``coxswain/viz`` is what it drives.
GAME_SOURCES = [
    os.path.join(ROOT, "scripts", "fpv.py"),
    os.path.join(ROOT, "coxswain", "viz"),
]


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------
def test_shipped_is_tier_zero_and_frozen():
    """What ships is the prescribed model, and it is marked unmovable."""
    shipped = physics.resolve(physics.SHIPPED)
    assert shipped.blade_tier == 0
    assert shipped.rower == "prescribed"
    assert shipped.frozen is True
    assert shipped.is_shipped()


def test_resolve_defaults_to_shipped():
    """Forgetting to pass a profile must not promote research physics.

    This is the safety property the whole registry rests on: the default
    is the frozen configuration, not the newest one.
    """
    assert physics.resolve().name == physics.SHIPPED
    assert physics.resolve(None).name == physics.SHIPPED


def test_resolve_is_idempotent_on_a_profile():
    profile = physics.resolve("research")
    assert physics.resolve(profile) is profile


def test_unknown_profile_names_what_there_is():
    with pytest.raises(KeyError) as caught:
        physics.resolve("tier9000")
    message = str(caught.value)
    assert "tier9000" in message
    for name in physics.names():
        assert name in message


def test_shipped_is_listed_first():
    """So a reader of ``names()`` sees the frozen one before the rest."""
    assert physics.names()[0] == physics.SHIPPED


def test_tier_zero_has_no_blade_model():
    """Tier 0 means the simulator uses the oar's fixed efficiency.

    ``Boat.blade_model is None`` is what the simulator already tests for,
    so tier 0 must produce exactly that and not a blade model that happens
    to be configured to do nothing.
    """
    assert physics.resolve(physics.SHIPPED).blade_model(None) is None


def test_tiers_above_zero_produce_a_blade_model():
    from coxswain.crew.oarlock import BladeModel

    for name in ("research", "learned"):
        model = physics.resolve(name).blade_model(None)
        assert isinstance(model, BladeModel), name


def test_blade_coefficient_follows_the_rig():
    """[CR06] fit C2 separately for sweep (84.5) and sculling (58.7)."""
    from coxswain.boats import catalog

    research = physics.resolve("research")
    sweep = research.blade_model(catalog.build("4+", rate=30.0))
    scull = research.blade_model(catalog.build("1x", rate=30.0))
    assert sweep.c2 == pytest.approx(84.5)
    assert scull.c2 == pytest.approx(58.7)


def test_blade_outboard_comes_from_the_boat_not_the_default():
    """The rig's own geometry, not [CR06]'s 2.28 m.

    Slip is ``l theta_dot + v cos theta`` and is linear in the outboard
    length, so taking the paper's value for a boat rigged differently
    would be a systematic error in every blade force.
    """
    from coxswain.boats import catalog

    boat = catalog.build("4+", rate=30.0)
    rigged = boat.rig.seats[0].oarlocks[0].oar.outboard
    model = physics.resolve("research").blade_model(boat)
    assert model.outboard == pytest.approx(rigged)
    assert model.outboard != pytest.approx(2.28)


def test_bad_tier_and_driver_are_rejected():
    with pytest.raises(ValueError):
        physics.PhysicsProfile("x", "", blade_tier=3, rower="prescribed")
    with pytest.raises(ValueError):
        physics.PhysicsProfile("x", "", blade_tier=0, rower="telepathy")


def test_apply_stamps_the_boat():
    """Every figure must be able to say which physics produced it."""
    from coxswain.boats import catalog

    boat = physics.resolve("research").apply(catalog.build("4+", rate=30.0))
    assert boat.physics_profile == "research"
    assert boat.blade_model is not None


# ---------------------------------------------------------------------------
# the freeze
# ---------------------------------------------------------------------------
def test_the_menu_builds_shipped_physics():
    """The one place the trainer's physics is pinned."""
    from coxswain.viz.menu import build_boat

    boat, _made = build_boat("4+", 30.0)
    assert boat.physics_profile == physics.SHIPPED
    assert boat.blade_model is None


def test_every_boat_the_menu_offers_is_shipped_physics():
    from coxswain.viz.menu import build_boat

    for key in ("1x", "2x", "4+", "8+"):
        boat, made = build_boat(key, 30.0)
        assert boat.physics_profile == physics.SHIPPED, key
        assert boat.blade_model is None, (key, made)


def _python_files(paths):
    for path in paths:
        if os.path.isfile(path):
            yield path
            continue
        for folder, _dirs, files in os.walk(path):
            if "__pycache__" in folder:
                continue
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(folder, name)


def _profile_strings(tree):
    """Every string constant in ``tree`` that names a known profile."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in physics.PROFILES:
                found.append((node.value, getattr(node, "lineno", 0)))
    return found


def test_the_game_never_names_a_research_profile():
    """A source scan, because a behaviour test cannot cover every path.

    If this fails, something the trainer can execute mentions a profile
    other than the frozen one. That is either a mistake or a deliberate
    promotion -- and a deliberate promotion should be a decision recorded
    in the plan, which is what failing here forces.
    """
    offences = []
    for path in _python_files(GAME_SOURCES):
        source = io.open(path, encoding="utf-8").read()
        if "physics" not in source:
            continue
        for name, line in _profile_strings(ast.parse(source, path)):
            if name != physics.SHIPPED:
                offences.append("%s:%d names %r"
                                % (os.path.relpath(path, ROOT), line, name))
    assert not offences, (
        "the released trainer must run the frozen physics profile:\n  "
        + "\n  ".join(offences))


def test_the_scan_would_catch_a_promotion():
    """The guard above is only worth having if it can fail.

    ``test_no_undefined_names`` had a sibling that asserted the *presence*
    of a crashing line and so pinned the bug in place; a guard that cannot
    fail is the same mistake wearing a different hat. So this feeds the
    scan a module that does the forbidden thing and checks it is seen.
    """
    planted = ast.parse('from coxswain import physics\n'
                        'boat = physics.resolve("research")\n')
    found = [name for name, _line in _profile_strings(planted)]
    assert "research" in found
    assert physics.SHIPPED not in found


def test_the_research_profile_advertises_that_it_is_unstable():
    """So nobody resolves it and quotes a boat speed.

    Tier 1 currently resolves to the efficiency-only wiring, which has no
    operating point -- the eight collapses from 3.92 m/s to 0.63. The
    profile is deliberately left at tier 1 rather than dropped to tier 0,
    because the collapse is the finding and a profile that silently
    behaved like ``shipped`` would hide it. What it must not do is look
    safe. See ``tests/test_blade_tier1.py``.
    """
    research = physics.resolve("research")
    assert research.blade_tier == 1
    assert research.frozen is False
    summary = research.summary.lower()
    assert "unstable" in summary
    assert "collapse" in summary


def test_the_learned_profile_advertises_that_it_is_unbuilt():
    """Nothing behind it exists yet.

    It is declared so the shape of the programme is visible in the code
    rather than only in the plan, which is worth doing -- but a profile
    whose name promises a trained policy and whose behaviour is a
    prescribed rower has to say so.
    """
    learned = physics.resolve("learned")
    assert learned.blade_tier == 2
    assert learned.rower == "learned"
    assert "unbuilt" in learned.summary.lower()


def test_no_unfrozen_profile_looks_safe():
    """The general rule the two tests above are instances of.

    Any profile that is not the frozen one is, by construction, a place
    where the physics is being changed. Until one of them is finished it
    must carry a warning in its own summary -- so that resolving it and
    reading the summary is enough, without also having read the plan.
    """
    for name in physics.names():
        profile = physics.resolve(name)
        if profile.frozen:
            continue
        summary = profile.summary.lower()
        assert ("unstable" in summary or "unbuilt" in summary), (
            "profile %r carries no warning in its summary. If it is now "
            "finished and trustworthy, say so here and in "
            "docs/PHYSICS_PROGRAMME.md rather than deleting this check -- "
            "it is the tripwire that stops a half-built profile looking "
            "like a finished one." % name)
