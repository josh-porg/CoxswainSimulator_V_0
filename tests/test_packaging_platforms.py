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
    """An unsigned bundle is refused outright on first open, with only a
    Cancel button, and the message says "damaged" -- which it is not.
    Someone who is not told about right-click-then-Open simply cannot
    run it, and will report it as broken."""
    text = read("packaging", "README-mac.txt").lower()
    assert "right-click" in text
    assert "quarantine" in text
    assert "damaged" in text, "the exact wording macOS shows should appear"


def test_the_workflow_builds_for_intel_so_it_runs_everywhere():
    """An arm64 build will not start on an Intel Mac; an x86_64 build
    runs on Apple Silicon under Rosetta.  One download beats two."""
    workflow = read(".github", "workflows", "release.yml")
    assert "macos-13" in workflow, "macos-14+ runners are Apple Silicon"
    assert "ditto" in workflow, (
        "a plain zip breaks the symlinks inside a .app bundle")
