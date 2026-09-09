r"""Names that do not exist, and the loop that never gets tested.

v0.12 shipped a crash: the minimap's HUD key reached for ``state`` in
the windowed main loop, where the pose is called ``pose``.  Every
bench and every screenshot passed, because ``--shot`` returns from a
headless path long before that loop -- so the one code path a player
actually runs was the one nothing exercised.

Two guards, because the bug had two halves.  ``pyflakes`` reads every
scope and reports names that are not defined in it, which is this
exact class and costs a second.  And the windowed loop is driven for
real, which catches what no static pass can.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_undefined_names_anywhere():
    """Every module, every scope.

    Only "undefined name" is fatal here -- unused imports and shadowed
    variables are style, and this file is about crashes.  When it
    fires, the message names the file, the line and the name.
    """
    pyflakes = pytest.importorskip("pyflakes")
    targets = [os.path.join(ROOT, "scripts", "fpv.py"),
               os.path.join(ROOT, "coxswain"),
               os.path.join(ROOT, "run.py")]
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes"] + targets,
        capture_output=True, text=True, cwd=ROOT)
    bad = [line for line in result.stdout.splitlines()
           if "undefined name" in line.lower()]
    assert not bad, "undefined names:\n  " + "\n  ".join(bad)


def test_the_shipped_scripts_at_least_compile():
    import py_compile

    for path in ("scripts/fpv.py", "run.py", "tools/build_exe.py"):
        py_compile.compile(os.path.join(ROOT, path), doraise=True)


@pytest.mark.slow
@pytest.mark.parametrize("extra", [
    [],                                    # the plain windowed loop
    ["--bonus", "on"],                     # pickups drawn and collected
    ["--no-minimap"],                      # the map switched off
])
def test_the_windowed_loop_runs(extra):
    """``--frames N`` WITHOUT ``--shot`` is the windowed loop -- the one
    a player runs, and the one every bench skips.  Forty frames is
    enough to draw the HUD, the minimap, the boat and the pickups, and
    to step the physics.

    Needs a GL context, so it is skipped where there is none.
    """
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "fpv.py"),
         "--race", "hotl", "--quality", "minimal", "--weather", "overcast",
         "--boat", "4+", "--frames", "40", "--no-sound", "--no-menu"] + extra,
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    if out.returncode != 0 and ("moderngl" in out.stderr
                                or "OpenGL" in out.stderr
                                or "display" in out.stderr.lower()):
        pytest.skip("no usable GL context here")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "Traceback" not in out.stderr, out.stderr[-2000:]
    assert "40 frames" in out.stdout, out.stdout[-500:]
