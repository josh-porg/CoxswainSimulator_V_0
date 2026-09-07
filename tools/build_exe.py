r"""Package the trainer as a single executable, for people without Python.

    python tools/build_exe.py

The point of this is to be able to hand the thing to a coxswain who has
never opened a terminal.  They get one file, they double-click it, they
get the setup menu.

What has to go in, and why the list is read out of the code
----------------------------------------------------------
PyInstaller follows ``import`` statements, and most of what this program
needs at run time is **not imported** -- it is loaded by file name: the
elevation model, the orthophoto, the building and tree extracts, the
bathymetry, the bridge inventory, the traced courses and their buoys,
the baked water textures, the stroke envelopes.  Miss one and the build
succeeds and the program dies on somebody else's machine, which is the
worst possible way to find out.

That is not hypothetical.  The first version of this listed the files by
hand, got seventeen of the twenty-three, and the six it missed included
``charles_isobaths.csv`` -- so the packaged build printed "building
charles ..." and stopped.  It passed its own test only because that test
ran the Seattle course.  :func:`discover_payload` now reads the list out
of the source, so adding a data file to the code adds it to the build.

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

#: Where data files live, relative to the repository root.
DATA_DIRS = ("coxswain/data", "data")

#: Extensions that are data rather than code or output.
DATA_SUFFIXES = (".npz", ".npy", ".json", ".csv", ".tif", ".tiff", ".jpg",
                 ".mp3")

#: Names that are *written* rather than read, so they are not payload.
NOT_PAYLOAD = ("frame.png",)


def discover_payload():
    """Every data file the trainer opens by name, found by reading the code.

    **This is derived, not hand-written, and that is the whole point.**
    The first version of this list was typed out from memory.  It had
    seventeen entries and the trainer needs twenty-three; the six it
    missed included ``charles_isobaths.csv``, so the packaged build
    started, said "building charles ...", and died -- on somebody else's
    machine, which is exactly the failure this check exists to prevent.
    It passed my own test only because that test happened to run the
    Seattle course.

    So the list is now read out of the source: every string literal that
    looks like a data file name, resolved against the two directories
    they live in.  Adding a new data file to the code adds it here
    automatically.
    """
    import re

    pattern = re.compile(r'["\']([A-Za-z0-9_\-.]+\.(?:%s))["\']'
                         % "|".join(x.lstrip(".") for x in DATA_SUFFIXES))
    sources = []
    for base, dirs, files in os.walk(os.path.join(ROOT, "coxswain")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        sources += [os.path.join(base, f) for f in files if f.endswith(".py")]
    for name in ("fpv.py", "render_totl.py", "render_hotl.py"):
        candidate = os.path.join(ROOT, "scripts", name)
        if os.path.exists(candidate):
            sources.append(candidate)

    wanted = set()
    for path in sources:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            wanted.update(pattern.findall(handle.read()))

    found = []
    for name in sorted(wanted):
        if name in NOT_PAYLOAD:
            continue
        for folder in DATA_DIRS:
            relative = "%s/%s" % (folder, name)
            if os.path.exists(os.path.join(ROOT, relative)):
                found.append(relative)
                break
    return found


#: Filled on first use by :func:`discover_payload`.
NEEDED = tuple(discover_payload()) if os.path.isdir(ROOT) else ()

#: Left out of the build, with what each costs.
#:
#: PyInstaller pulls in whatever is importable, and this project is a
#: research codebase that also does PyVista rendering, movie writing and
#: symbolic optimisation.  None of that runs in the trainer, and it was
#: 194 MB of a 756 MB build: a quarter of the download for code that is
#: never executed.
#:
#: **Numba is kept**, deliberately, and it is the largest single thing in
#: the build at 115 MB of LLVM.  It buys 2.95 ms a step against 3.85 --
#: 23% -- and while both fit inside the 10 ms budget for 100 Hz physics
#: on this machine, the people this is being sent to may not have this
#: machine.  Headroom on an unknown laptop is worth more than a smaller
#: download.
EXCLUDED = (
    "vtk", "vtkmodules", "pyvista", "pyvistaqt",      # ~110 MB, PyVista
    "imageio_ffmpeg", "imageio",                      # ~84 MB, movies
    "casadi",                                         # the MPC work
    "tkinter", "IPython", "pytest", "sphinx",
    "matplotlib.backends._backend_tk",
)

#: Imported dynamically, so PyInstaller cannot see them from the source.
HIDDEN = ("scipy.spatial", "scipy.ndimage", "scipy.interpolate",
          "scipy.special", "matplotlib.path", "moderngl", "glcontext",
          "pygame", "PIL.Image")


def check_payload():
    """``(present, missing)`` of the data files the trainer needs."""
    present, missing = [], []
    for name in NEEDED or discover_payload():
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
    print("data the trainer needs: %d of %d present, %.0f MB "
          "(found by reading the source, not by hand)"
          % (len(present), len(present) + len(missing),
             payload_size(present)))
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
    for name in (NEEDED or discover_payload()):
        target = os.path.dirname(name).replace("\\", "/")
        command += ["--add-data",
                    "%s%s%s" % (os.path.join(ROOT, name), separator, target)]
    for module in HIDDEN:
        command += ["--hidden-import", module]
    for module in EXCLUDED:
        command += ["--exclude-module", module]
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

    # The README ships *inside* the folder, so it has to be copied in
    # after PyInstaller has written it.  It lives in the repository
    # rather than being typed into dist/ by hand: the first copy was,
    # and a `rm -rf dist` erased it silently -- the zip would have gone
    # out with no instructions in it and nothing would have complained.
    readme = os.path.join(ROOT, "packaging", "README.txt")
    if args.onedir and os.path.exists(readme):
        shutil.copy2(readme, os.path.join(where, "README.txt"))
        print("   README.txt copied in")
    elif args.onedir:
        print("   WARNING: %s is missing, so the folder has no "
              "instructions in it" % readme)

    print("\nwrote %s" % where)
    if os.path.isfile(where):
        print("   %.0f MB" % (os.path.getsize(where) / 1e6))
    print("Hand that to anyone; it needs no Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
