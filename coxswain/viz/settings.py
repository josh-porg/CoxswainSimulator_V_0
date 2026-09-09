r"""Per-user settings that survive a restart.

One JSON file in the OS's per-user application folder -- beside the
diagnostics logs -- holding the few choices that should not have to be
made again every launch: whether performance reports may be sent, and
later the graphics tier once someone has picked one by hand.

Deliberately tiny.  A settings system that stores everything is a
second copy of the command line; this stores what has to persist to be
useful at all, and nothing that is not asked for.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict

from .telemetry import log_directory

FILE = "settings.json"


def settings_path() -> str:
    """``.../Coxswain/settings.json``, one level up from the logs."""
    return os.path.join(os.path.dirname(log_directory()), FILE)


def load(path: str = None) -> Dict[str, Any]:
    """The stored settings, or an empty dict.  Never raises: a corrupt
    or missing file is the same as a fresh install."""
    path = path or settings_path()
    try:
        with io.open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(values: Dict[str, Any], path: str = None) -> bool:
    """Write the settings.  Returns False rather than raising if the
    folder is not writable -- a launch must not fail over a preference."""
    path = path or settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
        return True
    except (OSError, TypeError, ValueError):
        return False


def update(**changes) -> Dict[str, Any]:
    """Merge ``changes`` into the stored settings and save."""
    values = load()
    values.update(changes)
    save(values)
    return values
