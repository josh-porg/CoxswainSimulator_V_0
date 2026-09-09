r"""Boats people saved: rowers and rigging, in ``presets.json``.

Beside ``settings.json``, one file, a list of lineups as plain data.
Names are unique -- saving under an existing name replaces it, which
is what "save" means to the person doing it -- and the built-in
presets are never in here: they come from the code, this holds only
what was typed in.

Read errors are an empty list and write errors are ``False``: a broken
file must not stop the trainer starting, and a disk that refuses the
write must not lose the lineup on screen.
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional

from .settings import settings_path

FILE = "presets.json"


def presets_path() -> str:
    return os.path.join(os.path.dirname(settings_path()), FILE)


def load(path: str = None) -> List[Dict]:
    path = path or presets_path()
    try:
        with io.open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return [d for d in data if isinstance(d, dict) and d.get("name")]
    except (OSError, ValueError):
        return []


def save_all(lineups: List[Dict], path: str = None) -> bool:
    path = path or presets_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as handle:
            json.dump(list(lineups), handle, indent=2, sort_keys=True)
        return True
    except OSError:
        return False


def save(lineup_dict: Dict, path: str = None) -> bool:
    """Add or replace by name."""
    name = str(lineup_dict.get("name", "")).strip()
    if not name:
        return False
    lineup_dict = dict(lineup_dict, name=name)
    kept = [d for d in load(path) if d.get("name") != name]
    kept.append(lineup_dict)
    return save_all(kept, path)


def delete(name: str, path: str = None) -> bool:
    kept = [d for d in load(path) if d.get("name") != name]
    return save_all(kept, path)


def names(path: str = None) -> List[str]:
    return [d["name"] for d in load(path)]


def get(name: str, path: str = None) -> Optional[Dict]:
    for d in load(path):
        if d.get("name") == name:
            return d
    return None
