r"""The packaging has to work on a machine nobody here can test on.

Everything in this file is about a Mac, which is exactly the problem:
there is no Mac in this project, so the macOS paths cannot be exercised
by running them.  What can be checked is that the pieces agree with each
other -- that the requirements file covers the imports the build
declares, that the platform branches exist, and that the instructions
match the thing being shipped.

These are cheap and they catch the failure that actually happens: a
build that succeeds on the runner and then does not start for the
person who downloaded it.
"""

from __future__ import annotations

import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def test_the_runner_installs_every_module_the_build_declares_hidden():
    """``HIDDEN`` names packages PyInstaller cannot see; CI must install them.

    This is the drift that bites: someone adds a dynamic import, adds it
    to ``HIDDEN`` so the local build picks it up, and the CI runner --
    which installs from requirements-build.txt and nothing else -- never
    gets the package at all.  The build still succeeds, because
    ``--hidden-import`` of a missing module is only a warning.
    """
    import sys

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from build_exe import EXCLUDED, HIDDEN

    pinned = {line.split("==")[0].strip().lower()
              for line in read("requirements-build.txt").splitlines()
              if "==" in line and not line.startswith("#")}
    # Map the import name to the distribution that provides it.
    provides = {"pil": "pillow", "opengl": "pyopengl"}
    for module in HIDDEN:
        top = module.split(".")[0].lower()
        top = provides.get(top, top)
        assert top in pinned, (
            "%s is a hidden import but no requirements-build.txt line "
            "installs it, so the CI build ships without it" % module)

    # And the heavy things stay out: they were 194 MB of a 756 MB build.
    for heavy in ("pyvista", "vtk", "casadi", "imageio"):
        assert heavy in EXCLUDED or heavy not in pinned, heavy
        assert heavy not in pinned, (
            "%s is excluded from the build, so CI should not spend "
            "minutes installing it" % heavy)


def test_macos_asks_for_a_forward_compatible_context():
    """Without this flag macOS hands back GL 2.1 and every shader fails.

    Requesting a 3.3 core profile alone succeeds on macOS and gives you
    a legacy context, so the failure surfaces later as a shader compile
    error and sends you looking at the shaders.
    """
    source = read("scripts", "fpv.py")
    assert "GL_CONTEXT_FORWARD_COMPATIBLE_FLAG" in source
    flag = source.index("GL_CONTEXT_FORWARD_COMPATIBLE_FLAG")
    guard = source.rindex('sys.platform == "darwin"', 0, flag)
    assert flag - guard < 900, "the flag must sit under the darwin guard"


def test_the_drawable_size_is_used_for_framebuffers_not_the_window_size():
    """Retina makes the drawable twice the window, and they are not
    interchangeable: framebuffers and the viewport uniform are in
    pixels, the mouse and the HUD layout are in points."""
    source = read("scripts", "fpv.py")
    assert "draw_width, draw_height" in source
    # The screen-space water lookup divides gl_FragCoord by this, so in
    # points it would be wrong by the scale factor across the surface.
    viewport = re.search(r'water_prog\["viewport"\]\.value = \(([^)]*)\)',
                         source, re.S)
    assert viewport is not None
    assert "draw_width" in viewport.group(1), viewport.group(1)


def test_the_mac_instructions_cover_gatekeeper():
    """Gatekeeper refuses the first launch and offers only Cancel.

    Someone not told about right-click-then-Open simply cannot run it,
    and will reasonably report the download as broken.  The README also
    has to distinguish the two messages, because they mean different
    things: "cannot be verified" is expected and is got past by opening
    it, while "damaged" after ad-hoc signing means the download really
    did arrive corrupt and should be fetched again.
    """
    text = read("packaging", "README-mac.txt").lower()
    assert "right-click" in text
    assert "quarantine" in text
    assert "cannot be verified" in text
    assert "damaged" in text


def test_every_platform_has_its_own_instructions():
    """The awkward part differs per platform, so one shared README would
    be two thirds irrelevant to every reader."""
    for leaf, must in (("README.txt", "unknown publisher"),
                       ("README-mac.txt", "right-click"),
                       ("README-linux.txt", "glibc")):
        text = read("packaging", leaf).lower()
        assert must in text, (leaf, must)
        # Every one of them has to say what to do first.
        assert "coxswain" in text


def test_the_macos_runner_label_is_one_that_still_exists():
    """A retired runner label does not fail -- it queues for ever.

    This began as macos-13, the last free Intel runner, so that one
    x86_64 download would also cover Apple Silicon through Rosetta.
    GitHub retired the label, and the job sat queued for fifty minutes
    with no runner able to match it while the other two were picked up
    in seconds.  Nothing reported an error; there is simply no runner.

    So the label is checked against the ones GitHub actually publishes.
    If this fails, look at what images exist before changing the number.
    """
    yaml = pytest.importorskip("yaml")
    with io.open(os.path.join(ROOT, ".github", "workflows", "release.yml"),
                 encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    # The VALUES, not the file's text: the comment above the matrix
    # explains the macos-13 history and would trip a substring search.
    runners = [entry["os"] for entry
               in spec["jobs"]["build"]["strategy"]["matrix"]["include"]]
    retired = {"macos-11", "macos-12", "macos-13"}
    assert not (set(runners) & retired), (
        "%s is retired; a job asking for it queues for ever"
        % (set(runners) & retired))
    assert any(r.startswith("macos-") for r in runners), runners

    workflow = read(".github", "workflows", "release.yml")
    assert "ditto" in workflow, (
        "a plain zip breaks the symlinks inside a .app bundle")


def test_the_mac_readme_says_apple_silicon_only():
    """Every free macOS runner is arm64 now, so the build cannot run on
    an Intel Mac at all.  Someone on a 2019 MacBook has to be told that
    on the way in, not after downloading 200 MB."""
    text = read("packaging", "README-mac.txt").lower()
    assert "apple silicon" in text
    assert "intel" in text, "Intel Macs must be named as unsupported"


def test_the_bundle_is_ad_hoc_signed():
    """Signing is what turns "damaged" into "cannot be verified", which
    is the difference between a download people abandon and one they
    open."""
    workflow = read(".github", "workflows", "release.yml")
    assert "codesign --force --deep --sign -" in workflow


def test_linux_is_built_against_an_old_enough_glibc():
    """A Linux binary will not start against a glibc older than the one
    it was built on, and that is most people's machines if you build on
    the newest runner.  tar, not zip, so the executable bit survives."""
    workflow = read(".github", "workflows", "release.yml")
    assert "ubuntu-22.04" in workflow, "ubuntu-latest is too new to ship"
    assert "tar -czf" in workflow
    assert "chmod +x" in workflow
