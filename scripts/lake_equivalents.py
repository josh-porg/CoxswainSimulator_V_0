r"""Each race-card goal as a Lake Sammamish training split (still air, same power).

    python scripts/lake_equivalents.py

The Charles is shallow, slightly adverse and open; Lake Sammamish is deep and still. The same crew
at the same power is slower on the Charles, so a Charles target split is not a training number.
``scripts/splits.py`` converts one: the power that rows the Charles time, then that power's speed on
deep still water (SOURCES sec. 87). This runs it for every goal on both cards and writes
``data/local/raceday/lake_equivalents.json``, which ``scripts/race_card.py`` shows under each goal.
The boats are ``splits.py``'s category boats, not our crews rower by rower.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import race_card as RC                       # noqa: E402

OUT = os.path.join(ROOT, "data", "local", "raceday", "lake_equivalents.json")


def split_seconds(text: str) -> float:
    m, s = text.split(":")
    return int(m) * 60 + float(s)


def lake_split(boat: str, target: float) -> dict:
    out = subprocess.run([sys.executable, os.path.join(HERE, "splits.py"), "--boat", boat, "--target", "%.1f" % target],
                         capture_output=True, text=True, cwd=ROOT, timeout=600).stdout
    charles = re.search(r"target on the Charles\s+\S+\s+\((\d+:\d+\.\d) per 500\)", out)
    lake = re.search(r"SPLIT ON SAMMAMISH\s+(\d+:\d+\.\d) per 500", out)
    if not (charles and lake):
        raise RuntimeError("splits.py output not understood for %s %.1f" % (boat, target))
    return dict(charles_split=split_seconds(charles.group(1)), lake_split=split_seconds(lake.group(1)))


def main():
    result = {}
    for crew, spec in RC.CREWS.items():
        result[crew] = {}
        for g in RC.crew_goals(crew):
            r = lake_split(spec["boat"], g["central"])
            result[crew][g["name"]] = dict(r, still_air_s=g["central"])
            print("%-6s %-12s Charles %s -> Sammamish %s" % (crew, g["name"], RC.C.fmt(r["charles_split"]),
                                                             RC.C.fmt(r["lake_split"])))
    json.dump(result, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
