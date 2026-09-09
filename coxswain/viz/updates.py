r"""Is there a newer release than the one running?

One GET to GitHub's public releases API, in a daemon thread, at start.
It asks for ``/releases/latest`` and compares the tag with
:func:`~coxswain.viz.telemetry.build_version`.  Nothing is downloaded
and nothing is installed: the answer is a line on the setup menu with
the link, and the person decides.

What it sends
-------------
An HTTPS request to ``api.github.com`` with the program's name and
version as the user agent, which GitHub requires.  That is the whole
disclosure: GitHub learns an IP address ran Coxswain, the same as it
learns from the download.  It is on by default because a rower running
a two-month-old build is the commonest reason the numbers they report
do not match the ones here -- and it is one switch away from off, in
the menu or with ``--no-update-check``, and remembered.

What it must never do
---------------------
Delay the start or fail it.  The thread is a daemon with a short
timeout; a machine with no network gets ``None`` and no message, and
the trainer starts exactly as fast either way.  Every failure -- DNS,
TLS, a rate limit, a malformed reply -- is swallowed into ``None``.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

#: Where the releases live.  The page a person is sent to is the tag's,
#: so the link on the menu is the same one handed out by e-mail.
REPOSITORY = "josh-porg/CoxswainSimulator_V_0"
API = "https://api.github.com/repos/%s/releases/latest" % REPOSITORY
PAGE = "https://github.com/%s/releases/latest" % REPOSITORY
TIMEOUT = 4.0

_TAG = re.compile(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?$")


def parse_version(tag: str):
    """``"v0.10"`` -> ``(0, 10, 0)``; anything else -> ``None``.

    Numeric tuples, so ``v0.10`` beats ``v0.9`` -- a string compare
    would have it the other way round, and this project is exactly
    where that bites.
    """
    if not tag:
        return None
    match = _TAG.match(str(tag).strip())
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


@dataclass(frozen=True)
class Update:
    """A newer release: its tag and where to get it."""

    tag: str
    url: str

    def line(self) -> str:
        return "%s is out -- %s" % (self.tag, self.url)


def newer(running: str, latest: str) -> bool:
    """Is ``latest`` strictly newer than ``running``?

    A source checkout ("source", "local") is never told about updates:
    it is ahead of every release by definition, and the person running
    it is the one cutting them.
    """
    have = parse_version(running)
    want = parse_version(latest)
    if have is None or want is None:
        return False
    return want > have


def fetch_latest_tag(timeout: float = TIMEOUT, running: str = "") -> Optional[str]:
    """The latest release's tag from GitHub, or ``None`` for any failure."""
    from .phonehome import _context

    request = urllib.request.Request(
        API, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "Coxswain/%s" % (running or "source")})
    try:
        kwargs = {}
        context = _context()
        if context is not None:
            kwargs["context"] = context
        with urllib.request.urlopen(request, timeout=timeout, **kwargs) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        tag = body.get("tag_name")
        return str(tag) if tag else None
    except Exception:
        return None


def check(running: str, fetch: Callable[..., Optional[str]] = None
          ) -> Optional[Update]:
    """Synchronous: the update if there is one, else ``None``.

    ``fetch`` is injectable so the tests never touch the network.
    """
    if parse_version(running) is None:
        return None                      # a source checkout, or unstamped
    fetch = fetch or fetch_latest_tag
    try:
        latest = fetch(running=running) if fetch is fetch_latest_tag else fetch()
    except Exception:
        return None
    if latest and newer(running, latest):
        return Update(tag=str(latest), url=PAGE)
    return None


class UpdateCheck:
    """The check, started in the background and read whenever asked.

    ``result`` is ``None`` until the thread has answered, and stays
    ``None`` if the answer was "no" or "could not tell" -- callers draw
    nothing in either case, which is the right picture for both.
    """

    def __init__(self, running: str, enabled: bool = True,
                 fetch: Callable[..., Optional[str]] = None):
        self.running = running
        self.enabled = bool(enabled)
        self._fetch = fetch
        self._result: Optional[Update] = None
        self._done = threading.Event()
        self._thread = None

    def start(self) -> "UpdateCheck":
        if not self.enabled or parse_version(self.running) is None:
            self._done.set()
            return self

        def worker():
            try:
                self._result = check(self.running, self._fetch)
            finally:
                self._done.set()

        self._thread = threading.Thread(target=worker, name="coxswain-update",
                                        daemon=True)
        self._thread.start()
        return self

    @property
    def result(self) -> Optional[Update]:
        return self._result if self._done.is_set() else None

    def wait(self, timeout: float = TIMEOUT + 1.0) -> Optional[Update]:
        self._done.wait(timeout)
        return self.result
