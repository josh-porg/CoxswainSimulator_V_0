r"""A diagnostics file the player can send back.

Why a file, and not a service
-----------------------------
"It was sluggish on my laptop" is not something anyone can act on.
What is needed is the machine, the GPU that actually drew, the tier,
how long the world took to build, what the frame times looked like,
and -- above all -- what was happening in the frames that stalled.
That is what this writes, as plain text, to the operating system's
own log folder, and the player attaches it to a message.

It is a local file on purpose.  Nothing here phones home: the people
testing this are rowers doing a favour, and shipping their machine
details to a server they never agreed to is not a favour returned.  The
path is printed at start-up and shown in the menu, so it is findable.

What goes in it
---------------
* the build (version tag, platform, Python), once;
* the hardware probe (:mod:`coxswain.viz.hardware`), once;
* every setting that affects cost, once and again on change;
* world-build timings, once;
* a **frame-time summary** every ten seconds -- median, 95th
  percentile, worst, and the physics/draw/present split -- rather
  than every frame, so a twenty-minute outing is a few hundred lines;
* every **stall** over :data:`STALL_MS`, with a one-line snapshot of
  what the simulation was doing, because a stall's cause is in its
  context;
* any exception, with its traceback.
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import platform
import sys
import time
import traceback
from typing import Callable, List, Optional

#: A frame longer than this is logged on its own, with context.
STALL_MS = 100.0
#: How often the running summary is written, seconds.
SUMMARY_EVERY = 10.0
#: How many old logs to keep beside the new one.
KEEP_LOGS = 10


def log_directory() -> str:
    """Where the OS expects an application's logs."""
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Coxswain", "logs")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Logs",
                            "Coxswain")
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "coxswain", "logs")


def build_version() -> str:
    """The release tag this build was cut from, or 'source'.

    ``tools/build_exe.py`` writes ``packaging/VERSION`` from the CI tag
    and ships it beside the program; a source checkout has none.
    """
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", ""),
                                       "packaging", "VERSION"))
        candidates.append(os.path.join(os.path.dirname(sys.executable),
                                       "packaging", "VERSION"))
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    candidates.append(os.path.join(here, "packaging", "VERSION"))
    for path in candidates:
        try:
            with io.open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
                if text:
                    return text
        except OSError:
            continue
    return "source"


class Telemetry:
    """The session's diagnostics file.  Cheap to call every frame."""

    def __init__(self, directory: str = None, enabled: bool = True,
                 clock: Callable[[], float] = time.perf_counter):
        self.enabled = bool(enabled)
        self.clock = clock
        self.path: Optional[str] = None
        self._handle = None
        self._frames: List[float] = []
        self._physics: List[float] = []
        self._draw: List[float] = []
        self._present: List[float] = []
        self._stalls = 0
        self._started = self.clock()
        self._last_summary = self._started
        self.total_frames = 0
        if not self.enabled:
            return
        directory = directory or log_directory()
        try:
            os.makedirs(directory, exist_ok=True)
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = os.path.join(directory, "coxswain-%s.log" % stamp)
            self._handle = io.open(self.path, "w", encoding="utf-8")
            self._prune(directory)
        except OSError:
            self.enabled = False
            self._handle = None
            return
        self.section("build", [
            "version: %s" % build_version(),
            "platform: %s %s (%s)" % (platform.system(), platform.release(),
                                      platform.machine()),
            "python: %s" % platform.python_version(),
            "frozen: %s" % bool(getattr(sys, "frozen", False)),
            "started: %s" % _dt.datetime.now().isoformat(timespec="seconds"),
        ])

    # -- writing -----------------------------------------------------------
    def _prune(self, directory: str) -> None:
        try:
            logs = sorted(name for name in os.listdir(directory)
                          if name.startswith("coxswain-")
                          and name.endswith(".log"))
            for name in logs[:-KEEP_LOGS]:
                os.remove(os.path.join(directory, name))
        except OSError:
            pass

    def _write(self, text: str) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(text)
            if not text.endswith("\n"):
                self._handle.write("\n")
            self._handle.flush()
        except OSError:
            self._handle = None
            self.enabled = False

    def _stamp(self) -> str:
        return "%8.2f" % (self.clock() - self._started)

    def section(self, name: str, lines) -> None:
        """A titled block of lines, written once."""
        if not self.enabled:
            return
        self._write("[%s] %s" % (self._stamp(), name))
        for line in lines:
            self._write("    %s" % line)

    def note(self, text: str) -> None:
        if self.enabled:
            self._write("[%s] %s" % (self._stamp(), text))

    def exception(self, error: BaseException) -> None:
        """A traceback, whatever else is going on."""
        if not self.enabled:
            return
        self._write("[%s] EXCEPTION %s: %s" % (self._stamp(),
                                              type(error).__name__, error))
        for line in traceback.format_exception(type(error), error,
                                               error.__traceback__):
            self._write("    " + line.rstrip("\n"))

    # -- frames ------------------------------------------------------------
    def frame(self, total_ms: float, physics_ms: float = 0.0,
              draw_ms: float = 0.0, present_ms: float = 0.0,
              context: Callable[[], str] = None) -> None:
        """One frame's timings.  ``context`` is only called on a stall."""
        self.total_frames += 1
        if not self.enabled:
            return
        self._frames.append(float(total_ms))
        self._physics.append(float(physics_ms))
        self._draw.append(float(draw_ms))
        self._present.append(float(present_ms))
        if total_ms > STALL_MS:
            self._stalls += 1
            what = ""
            if context is not None:
                try:
                    what = str(context())
                except Exception as error:               # pragma: no cover
                    what = "(context failed: %s)" % error
            self._write("[%s] STALL %.0f ms  physics %.0f  draw %.0f  "
                        "present %.0f  %s"
                        % (self._stamp(), total_ms, physics_ms, draw_ms,
                           present_ms, what))
        now = self.clock()
        if now - self._last_summary >= SUMMARY_EVERY:
            self.summary()
            self._last_summary = now

    def summary(self) -> Optional[str]:
        """Write and return the running frame-time summary."""
        if not self._frames:
            return None
        frames = sorted(self._frames)
        n = len(frames)
        p50 = frames[n // 2]
        p95 = frames[min(n - 1, int(0.95 * n))]
        worst = frames[-1]
        mean = sum(frames) / n
        fps = 1000.0 / mean if mean > 0 else 0.0
        line = ("frames %d  fps %.1f  ms p50 %.1f  p95 %.1f  worst %.0f  "
                "physics %.1f  draw %.1f  present %.1f  stalls %d"
                % (n, fps, p50, p95, worst,
                   sum(self._physics) / n, sum(self._draw) / n,
                   sum(self._present) / n, self._stalls))
        self._write("[%s] %s" % (self._stamp(), line))
        self._frames.clear()
        self._physics.clear()
        self._draw.clear()
        self._present.clear()
        self._stalls = 0
        return line

    def close(self) -> None:
        if not self.enabled:
            return
        self.summary()
        self.note("closed after %d frames" % self.total_frames)
        try:
            if self._handle is not None:
                self._handle.close()
        finally:
            self._handle = None


def install_excepthook(telemetry: Telemetry) -> None:
    """Route uncaught exceptions into the log before Python prints them."""
    previous = sys.excepthook

    def hook(kind, value, trace):
        try:
            telemetry.exception(value)
            telemetry.close()
        finally:
            previous(kind, value, trace)

    sys.excepthook = hook
