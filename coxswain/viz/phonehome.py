r"""Performance reports that come back to whoever is making this faster.

The rowers who test the trainer report "too slow" and nothing else,
because nothing else is visible to them.  The diagnostics file already
knows the answer -- GPU, tier, frame percentiles, stalls, how much
physics time was dropped -- but it sits in a folder on their laptop.
This sends a summary of it home, once, at the end of a session.

What is sent, and what is not
-----------------------------
A JSON object of numbers and product strings: version, OS family,
CPU count and RAM, the GPU adapter names and the live renderer string,
the tier, the settings the run used, the world build time, frame-time
percentiles, stall and dropped-time counts, and the count (not the text)
of exceptions.  ``scrub`` refuses to send anything containing a user
name, a home directory, a file path or an e-mail, and the test holds it
to that on a real report.

No secrets ship in the binary
-----------------------------
The destination is a URL the program is *told*: ``--report-url``, the
``COXSWAIN_REPORT_URL`` environment variable, or a ``report_url.txt``
beside the executable.  The intended endpoint is a Google Apps Script
that appends a row to a spreadsheet (``packaging/phonehome/Code.gs``),
which needs no key on this side.  With no URL the report is written
beside the diagnostics log as ``report-<stamp>.json`` and its path
printed, so a tester can paste it instead.

Consent
-------
Off unless switched on: the setup menu's **Send performance reports**
row, remembered in the per-user settings.  Nothing is sent without it,
and the outcome -- sent, refused, no URL, failed -- is written to the
diagnostics log so the choice is auditable.
"""

from __future__ import annotations

import getpass
import io
import json
import os
import platform
import re
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional

from .telemetry import build_version, log_directory

#: Seconds to wait on the network before giving up.  A closing program
#: must not hang on a dead endpoint.
TIMEOUT = 6.0

URL_FILE = "report_url.txt"
ENV_VAR = "COXSWAIN_REPORT_URL"


def report_url(explicit: Optional[str] = None) -> Optional[str]:
    """Where to send, by precedence: flag, environment, file beside the
    program.  ``None`` when none is set."""
    if explicit:
        return explicit.strip() or None
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env
    base = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.getcwd())
    path = os.path.join(base, URL_FILE)
    try:
        with io.open(path, encoding="utf-8") as handle:
            url = handle.read().strip()
        return url or None
    except OSError:
        return None


# -- what goes in the report ---------------------------------------------------
def build_report(telemetry, probe=None, settings: Dict[str, Any] = None,
                 tier: str = "", extra: Dict[str, Any] = None
                 ) -> Dict[str, Any]:
    """The summary, from the objects that already know it."""
    snap = telemetry.snapshot() if hasattr(telemetry, "snapshot") else {}
    report: Dict[str, Any] = {
        "schema": 1,
        "version": build_version(),
        "sent": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "tier": tier,
        "settings": dict(settings or {}),
        "frames": snap,
    }
    if probe is not None:
        report["hardware"] = {
            "renderer": getattr(probe, "renderer", "") or "",
            "adapters": list(getattr(probe, "adapters", []) or []),
            "cpu_count": int(getattr(probe, "cpu_count", 0) or 0),
            "ram_gb": float(getattr(probe, "ram_gb", 0.0) or 0.0),
            "integrated": bool(getattr(probe, "integrated", False)),
            "software": bool(getattr(probe, "software", False)),
        }
    if extra:
        report.update(extra)
    return report


_PATHISH = re.compile(r"([A-Za-z]:\\|/home/|/Users/|\\Users\\|@[\w.-]+\.\w+)")


def scrub(report: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse to send anything that could name the person.

    Raises ``ValueError`` if a user name, a home directory, a file path
    or an e-mail address is found anywhere in the report's strings.  A
    report is numbers and product names; if something else got in, the
    right response is to send nothing, not to redact and hope.
    """
    names = set()
    for candidate in (getpass.getuser if hasattr(getpass, "getuser") else None,
                      socket.gethostname):
        try:
            value = candidate() if candidate else ""
            if value and len(value) >= 3:
                names.add(value.lower())
        except Exception:                                   # pragma: no cover
            pass
    home = os.path.expanduser("~")
    if home and home != "~":
        names.add(home.lower())

    def check(value, where):
        if isinstance(value, dict):
            for key, inner in value.items():
                check(inner, where + "." + str(key))
        elif isinstance(value, (list, tuple)):
            for index, inner in enumerate(value):
                check(inner, "%s[%d]" % (where, index))
        elif isinstance(value, str):
            low = value.lower()
            if _PATHISH.search(value):
                raise ValueError("%s looks like a path or address" % where)
            for name in names:
                if name in low:
                    raise ValueError("%s contains an identifying name" % where)

    check(report, "report")
    return report


# -- sending ---------------------------------------------------------------------
def _context():
    """TLS context: the OS store, or certifi's bundle where the OS gives
    Python nothing (a frozen macOS build)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def post(url: str, report: Dict[str, Any], timeout: float = TIMEOUT) -> str:
    """POST the report as JSON.  Returns a one-line outcome; never raises."""
    import urllib.request
    import urllib.error

    body = json.dumps(report, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "Coxswain/%s" % report.get("version", "?")})
    try:
        kwargs = {"timeout": timeout}
        if url.lower().startswith("https://"):
            kwargs["context"] = _context()
        with urllib.request.urlopen(request, **kwargs) as response:
            status = getattr(response, "status", 200)
            return "sent (%s)" % status
    except urllib.error.HTTPError as error:
        return "refused (%s)" % error.code
    except Exception as error:
        return "failed (%s)" % type(error).__name__


def write_local(report: Dict[str, Any], directory: str = None) -> Optional[str]:
    """The fallback: a file beside the logs a tester can paste."""
    directory = directory or log_directory()
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "report-%s.json"
                            % time.strftime("%Y%m%d-%H%M%S"))
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        return path
    except OSError:
        return None


def send(report: Dict[str, Any], url: Optional[str], enabled: bool,
         log=None, wait: bool = True) -> str:
    """The whole policy in one place.  Returns the outcome line, and
    writes it to ``log`` (a callable taking a string) if given."""
    def out(line: str) -> str:
        if log is not None:
            try:
                log("report: " + line)
            except Exception:                               # pragma: no cover
                pass
        return line

    if not enabled:
        return out("not sent (reports are off)")
    try:
        scrub(report)
    except ValueError as error:
        return out("not sent (%s)" % error)
    if not url:
        path = write_local(report)
        return out("no report URL; written to %s" % path if path
                   else "no report URL and nowhere to write")

    result = {"line": None}

    def worker():
        result["line"] = post(url, report)

    thread = threading.Thread(target=worker, name="coxswain-report",
                              daemon=True)
    thread.start()
    if wait:
        thread.join(TIMEOUT + 1.0)
    return out(result["line"] or "sending in the background")
