r"""What this machine is, and what graphics tier it should start on.

The trainer has to run on an old work laptop with integrated graphics
and on a gaming desktop, and the same settings cannot be right for
both.  This module answers two questions once, at start-up, from the
machine itself rather than from a guess:

1. **Which GPU is actually drawing.**  Not which one the machine has --
   which one the OpenGL context landed on.  A laptop with an Intel
   iGPU and an NVIDIA or AMD chip beside it hands an unknown program to
   the iGPU by default, and that alone can be the difference between
   "smooth" and "unusably sluggish" with no setting in the game able
   to change it.  So the renderer string is read from the live context,
   the machine's adapters are listed from the OS, and if a dedicated
   GPU exists that is NOT the one drawing, the user is told so and
   told how to fix it.

2. **Which tier to start on.**  A recommendation from the renderer
   class and the core count, used only when the user has not chosen.

What it deliberately does not do
--------------------------------
It does not change the GPU assignment by itself.  On Windows that is a
per-application system preference, and this program writes system
settings only when asked to (``--prefer-dedicated-gpu``).  It does not
try to use the GPU for the physics: that is a small, stiff 6-DOF
system on tiny arrays, where a GPU's launch latency loses to a CPU
JIT by orders of magnitude -- the compiled hull kernel
(:mod:`coxswain.hydro._hullkernel`) is the right tool and is already
used.  And there is nothing here for a TPU, because nothing in this
program is a tensor workload.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

#: Substrings that identify an integrated (shared-memory) GPU in a
#: renderer or adapter name.  Deliberately conservative: a miss here
#: means "assume dedicated", which only changes the default tier.
INTEGRATED = ("intel(r) hd", "intel(r) uhd", "intel(r) iris", "intel iris",
              "intel hd", "intel uhd", "intel(r) arc(tm) a3", "radeon(tm) graphics",
              "radeon graphics", "vega 3", "vega 6", "vega 7", "vega 8",
              "amd radeon(tm) vega", "llvmpipe", "swiftshader", "microsoft basic",
              "apple m1", "apple m2", "apple m3", "apple m4", "mesa intel")

#: Substrings that mark a software renderer, which no tier can rescue.
SOFTWARE = ("llvmpipe", "swiftshader", "softpipe", "microsoft basic render",
            "gdi generic", "mesa offscreen")


@dataclass
class Probe:
    """One machine, as seen at start-up."""

    renderer: str = ""
    vendor: str = ""
    gl_version: str = ""
    adapters: List[str] = field(default_factory=list)
    cpu_count: int = 1
    ram_gb: float = 0.0
    system: str = ""
    python: str = ""

    @property
    def software(self) -> bool:
        low = self.renderer.lower()
        return any(tag in low for tag in SOFTWARE)

    @property
    def integrated(self) -> bool:
        """True when the GPU that is DRAWING is an integrated one."""
        low = self.renderer.lower()
        if self.software:
            return True
        return any(tag in low for tag in INTEGRATED)

    @property
    def dedicated_adapters(self) -> List[str]:
        """Adapters on the machine that look like dedicated GPUs."""
        out = []
        for name in self.adapters:
            low = name.lower()
            if any(tag in low for tag in INTEGRATED + SOFTWARE):
                continue
            if any(v in low for v in ("nvidia", "geforce", "quadro", "rtx",
                                      "gtx", "radeon rx", "radeon pro",
                                      "arc a5", "arc a7", "arc b")):
                out.append(name)
        return out

    @property
    def dedicated_idle(self) -> Optional[str]:
        """A dedicated GPU that exists but is not the one drawing."""
        if not self.integrated:
            return None
        for name in self.dedicated_adapters:
            return name
        return None

    def recommended_tier(self) -> str:
        """Where to start when the user has not chosen.

        Conservative on purpose: a first launch that stutters is what
        makes someone close it, and stepping UP a tier from a smooth
        start is a pleasant discovery.  Thresholds are coarse because
        the renderer string is all there is without a benchmark.
        """
        if not self.renderer:
            # Before a GL context exists -- the tier gates the world
            # build, which comes first -- decide from the adapter list.
            # A dedicated card present is probably the one that will
            # draw; the notice after the context is made says if not.
            if self.dedicated_adapters:
                return "standard"
            if self.adapters:
                return "ultra" if self.cpu_count <= 4 else "minimal"
            return "standard"
        if self.software:
            return "ultra"
        if self.integrated:
            return "ultra" if self.cpu_count <= 4 else "minimal"
        low = self.renderer.lower()
        if any(tag in low for tag in ("rtx 40", "rtx 30", "rtx 50", "rx 7",
                                      "rx 6", "rx 9", "arc b")):
            return "high"
        return "standard"

    def lines(self) -> List[str]:
        """The probe as the loading screen and the log print it."""
        out = ["graphics: %s" % (self.renderer or "unknown"),
               "cpu: %d cores, %.1f GB RAM, %s" % (self.cpu_count,
                                                   self.ram_gb, self.system)]
        idle = self.dedicated_idle
        if idle:
            out.append("NOTE: this machine has a %s, but the game is being "
                       "drawn by the integrated graphics." % idle)
            out.append(dedicated_gpu_advice())
        elif self.software:
            out.append("NOTE: no GPU driver is in use -- this is software "
                       "rendering and will be slow at any setting.")
        return out


def _ram_gb() -> float:
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2.0 ** 30
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return pages * size / 2.0 ** 30
    except Exception:                                    # pragma: no cover
        pass
    return 0.0


def _adapters() -> List[str]:
    """Every display adapter the OS knows about, by name."""
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=8)
            return [line.strip() for line in out.stdout.splitlines()
                    if line.strip()]
        if sys.platform == "darwin":
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                                 capture_output=True, text=True, timeout=8)
            return re.findall(r"Chipset Model:\s*(.+)", out.stdout)
        out = subprocess.run(["lspci"], capture_output=True, text=True,
                             timeout=8)
        return [line.split(":", 2)[-1].strip() for line in out.stdout.splitlines()
                if "VGA" in line or "3D controller" in line]
    except Exception:
        return []


def probe(ctx=None) -> Probe:
    """Read the machine.  ``ctx`` is a live moderngl context, if any."""
    found = Probe(cpu_count=os.cpu_count() or 1, ram_gb=_ram_gb(),
                  system="%s %s" % (platform.system(), platform.release()),
                  python=platform.python_version(), adapters=_adapters())
    if ctx is not None:
        try:
            info = ctx.info
            found.renderer = str(info.get("GL_RENDERER", ""))
            found.vendor = str(info.get("GL_VENDOR", ""))
            found.gl_version = str(info.get("GL_VERSION", ""))
        except Exception:                                # pragma: no cover
            pass
    return found


def dedicated_gpu_advice() -> str:
    """How to make the OS use the dedicated GPU, in the user's words."""
    if sys.platform.startswith("win"):
        return ("To use it: Windows Settings > System > Display > Graphics, "
                "add Coxswain.exe, set it to High performance. Or start the "
                "game once with --prefer-dedicated-gpu and it will ask "
                "Windows for you.")
    if sys.platform == "darwin":
        return ("On a Mac with two GPUs, System Settings > Battery > "
                "Options: turn off Automatic graphics switching.")
    return ("On Linux with PRIME: run with DRI_PRIME=1, or "
            "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia.")


def request_dedicated_gpu(executable: str = None) -> str:
    """Ask Windows to draw this program on its high-performance GPU.

    This writes a per-application preference under the current user's
    registry hive -- the same thing the Settings page does -- and only
    when explicitly asked for with ``--prefer-dedicated-gpu``.  It is
    the one system setting this program touches, and it says so.
    """
    if not sys.platform.startswith("win"):
        return "only Windows keeps a per-program GPU preference"
    executable = executable or sys.executable
    if getattr(sys, "frozen", False):
        executable = sys.executable
    try:
        import winreg

        key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\DirectX\UserGpuPreferences")
        winreg.SetValueEx(key, executable, 0, winreg.REG_SZ,
                          "GpuPreference=2;")
        winreg.CloseKey(key)
        return ("asked Windows to use the high-performance GPU for %s; "
                "takes effect on the next start" % executable)
    except Exception as error:                           # pragma: no cover
        return "could not write the preference: %s" % error
