"""Progress reporting for the long jobs.

Optimising a line, simulating a leg and rendering a movie all take minutes,
and a silent terminal for minutes is indistinguishable from a hang -- which
during this project it repeatedly was.  These wrap :mod:`tqdm` where it is
installed and degrade to plain prints where it is not, so nothing here has
a hard dependency on it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

__all__ = ["progress", "stage", "HAVE_TQDM"]

try:
    from tqdm.auto import tqdm as _tqdm
    HAVE_TQDM = True
except ImportError:                      # pragma: no cover - optional
    _tqdm = None
    HAVE_TQDM = False

_BAR = "{desc:<34}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} {elapsed}"


def progress(iterable=None, total=None, desc="", leave=True, unit="it"):
    """A tqdm bar where tqdm exists, otherwise the iterable untouched."""
    if not HAVE_TQDM:
        if iterable is None:
            return _Silent(total=total, desc=desc)
        return iterable
    return _tqdm(iterable, total=total, desc=desc, leave=leave, unit=unit,
                 bar_format=_BAR, file=sys.stdout, dynamic_ncols=True)


@contextmanager
def stage(desc, total=None):
    """A bar for a job whose steps are reported by hand."""
    bar = progress(total=total, desc=desc)
    try:
        yield bar
    finally:
        close = getattr(bar, "close", None)
        if close is not None:
            close()


class _Silent:
    """Stand-in with tqdm's interface, for when tqdm is absent."""

    def __init__(self, total=None, desc=""):
        self.total = total
        self.n = 0
        if desc:
            print("%s ..." % desc)

    def update(self, count=1):
        self.n += count

    def set_description(self, text):
        print("   %s" % text)

    def set_postfix_str(self, text):
        pass

    def close(self):
        pass
