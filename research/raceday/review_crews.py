"""Which crew a race review is for, loaded from the private crew list.

The crews themselves (coxswains' names, events, bows, places, CoxBox sessions) are private and live in
data/local/raceday/crews.json, which is gitignored; this module only reads it. Set REVIEW_CREW to one of
that file's keys (default "ours") before running race_review_2026.py, race_review_opt.py,
race_review_crew_page.py and race_review_site.py:

    REVIEW_CREW=<key> .venv/Scripts/python.exe research/raceday/race_review_2026.py

Each crew has its CoxBox session, the wall-clock start of the session in US Eastern time (CoxBox clocks are
set to whatever time zone the unit was set up in; the session start matched to the event's start list
decides), the race-card crew (boat model, card, wind table), the day factor used to calibrate the wind, and
the result. COX gives each key's coxswain; ALIASES names the roles older diagnostics refer to (alias()).
"""
import json
import os
from datetime import datetime

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(CODE, "..", "..", "data", "local", "raceday"))
CB = os.path.join(DATA, "..", "coxbox", "Sharing file  CoxBox %s.csv")
BASE_2025 = 1.02

_RAW = json.load(open(os.path.join(DATA, "crews.json"), encoding="utf-8"))


def _crew(d):
    d = dict(d)
    d["coxbox"] = CB % d.pop("session")
    d["start_et"] = datetime.fromisoformat(d["start_et"])
    return d


def _line_review(d):
    d = dict(d)
    d["csv"] = CB % d.pop("session")
    d["start"] = datetime.fromisoformat(d["start"])
    d["median"] = tuple(d["median"])
    return d


CREWS = {k: _crew(v) for k, v in _RAW["crews"].items()}
COX = {k: v["cox"] for k, v in _RAW["crews"].items()}
LINE_REVIEW = {k: _line_review(v) for k, v in _RAW.get("line_review", {}).items()}
ALIASES = _RAW.get("aliases", {})


def alias(role):
    """The crew key that plays a named role (e.g. "eight") in the older diagnostics."""
    return ALIASES.get(role, role)


WHO = os.environ.get("REVIEW_CREW", "ours")
CFG = CREWS[WHO]

#: REVIEW_STARBOARD=shore: the Cambridge limit follows the water (card.CourseModel.build(starboard="shore"),
#: aground depth 0.5 m, research boats only), the crews' lines are clamped to each side's own limit, and every
#: output gets a "_shore" suffix so the published (navigable-corridor) reviews are not overwritten.
STARBOARD = os.environ.get("REVIEW_STARBOARD", "navigable")
if STARBOARD not in ("navigable", "shore"):
    raise SystemExit("REVIEW_STARBOARD must be navigable or shore")


def shore_name(name):
    """An output file name, suffixed "_shore" under the shore corridor."""
    if STARBOARD == "navigable":
        return name
    root, ext = os.path.splitext(name)
    return root + "_shore" + ext


if STARBOARD == "shore":
    CFG = dict(CFG, json=shore_name(CFG["json"]), html=shore_name(CFG["html"]))
