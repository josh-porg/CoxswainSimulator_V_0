r"""Package the trainer as a single executable, for people without Python.

    python tools/build_exe.py

The point of this is to be able to hand the thing to a coxswain who has
never opened a terminal.  They get one file, they double-click it, they
get the setup menu.

What has to go in, and why it is not automatic
----------------------------------------------
PyInstaller follows ``import`` statements, and most of what this program
needs at run time is **not imported** -- it is loaded from
``coxswain/data`` by file name: the elevation model, the orthophoto, the
building and tree extracts, the bathymetry, the bridge inventory, the
baked near-field and wave textures, the stroke envelopes.  Miss one and
the executable builds cleanly and then fails on somebody else's machine
with a missing-file error, which is the worst possible way to find out.
So the data directory is listed explicitly and its size is reported,
and :func:`check_payload` refuses to build if something the trainer
needs is absent.

The courses are traced files under ``data/`` at the repository root
rather than inside the package, so those are collected too.

Size
----
Most of the bulk is the scenery: the Seattle elevation model and
orthophoto are tens of megabytes between them.  That is the price of a
build that works away from this machine, and it is worth paying once.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Everything the trainer opens by name at run time.  Anything here that
#: is missing stops the build rather than shipping a broken executable.
NEEDED = (
    "coxswain/data/seattle_dem.npz",
    "coxswain/data/seattle_imagery.jpg",
    "coxswain/data/seattle_structures.npz",
    "coxswain/data/seattle_trees.npz",
    "coxswain/data/seattle_bridges.npz",
    "coxswain/data/charles_dem.tif",
    "coxswain/data/charles_structures.npz",
    "coxswain/data/lake_union_depth.npz",
    "coxswain/data/nearfield.npz",
    "coxswain/data/stroke_envelope.npz",
    "data/seattle_water.json",
    "data/seattle_obstructions.json",
    "data/seattle_bridge_outlines.json",
    "data/totl_course.npy",
    "data/totl_buoys.npy",
    "data/hotl_course.npy",
    "data/hotl_buoys.npy",
)

#: Imported dynamically, so PyInstaller cannot see them from the source.
HIDDEN = ("scipy.spatial", "scipy.ndimage", "scipy.interpolate",
          "scipy.special", "matplotlib.path", "moderngl", "glcontext",
          "pygame", "PIL.Image")


def check_payload():
    """``(present, missing)`` of the data files the trainer needs."""
    present, missing = [], []
    for name in NEEDED:
        path = os.path.join(ROOT, name)
        (present if os.path.exists(path) else missing).append(name)
    return present, missing


def payload_size(names) -> float:
    total = 0
    for name in names:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total / 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="Coxswain")
    parser.add_argument("--onedir", action="store_true",
                        help="a folder instead of one file -- starts much "
                             "faster, since a one-file build unpacks "
                             "itself to a temporary directory every run")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the payload and print the command")
    args = parser.parse_args(argv)

    present, missing = check_payload()
    print("data the trainer needs: %d of %d present, %.0f MB"
          % (len(present), len(NEEDED), payload_size(present)))
    if missing:
        print("\nMISSING -- the build would produce a broken executable:")
        for name in missing:
            print("   %s" % name)
        print("\nRun the fetch tools that produce these first; "
              "tools/bake_nearfield.py and tools/build_stroke_envelope.py "
              "make the two baked ones.")
        return 1

    separator = ";" if os.name == "nt" else ":"
    command = [sys.executable, "-m", "PyInstaller",
               "--noconfirm", "--clean", "--windowed",
               "--name", args.name,
               "--paths", os.path.join(ROOT, "scripts"),
               "--onedir" if args.onedir else "--onefile"]
    for name in NEEDED:
        target = os.path.dirname(name).replace("\\", "/")
        command += ["--add-data",
                    "%s%s%s" % (os.path.join(ROOT, name), separator, target)]
    for module in HIDDEN:
        command += ["--hidden-import", module]
    command.append(os.path.join(ROOT, "scripts", "fpv.py"))

    if args.dry_run:
        print("\nwould run:\n  " + " ".join(command))
        return 0

    if shutil.which("pyinstaller") is None:
        try:
            import PyInstaller                       # noqa: F401
        except ImportError:
            print("\nPyInstaller is not installed:\n"
                  "    pip install pyinstaller")
            return 1

    print("\nbuilding ...")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode
    where = os.path.join(ROOT, "dist",
                         args.name if args.onedir else args.name + ".exe")
    print("\nwrote %s" % where)
    if os.path.isfile(where):
        print("   %.0f MB" % (os.path.getsize(where) / 1e6))
    print("Hand that to anyone; it needs no Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
