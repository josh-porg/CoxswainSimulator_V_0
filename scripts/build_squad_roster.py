r"""Turn the squad's erg spreadsheet into an anonymous roster.

    python scripts/build_squad_roster.py path/to/export.csv

Why this script exists at all
-----------------------------
The spreadsheet is a training diary kept by real people who did not
agree to be characters in a game.  Their erg scores are the only thing
the simulator needs; their names are not.  So the sheet itself is never
committed, and this is the one place the two ever meet: it reads the
export, throws the identities away, and writes
``data/squad_roster.csv``, which is what the game loads.

What is kept, and what is blurred
---------------------------------
Kept, because the model needs it:

* **rowing age** -- masters racing is age-banded, and a 5k at 34 is not
  the same performance as the same 5k at 62;
* **weight** -- it sets displacement and drag, and it is also what the
  erg's own weight adjustment is computed from;
* **5k time** -- the whole point.

Blurred, because a name is not the only way to identify somebody.  An
exact weight beside an exact time to the tenth of a second is close to
a unique key for anyone holding the original sheet, so the weight is
rounded to five pounds and the time to the whole second.  Neither
rounding is large enough to matter to the physics: five pounds is 0.5%
of a boat's all-up mass, and a second over 5 km is 0.1% of the pace,
which is 0.3% of the power.

The row order is shuffled under a fixed seed as well, so position in
the file does not put the identities back in the order the sheet had
them.

Best and typical
----------------
Both are written.  The **best** is a personal best, set on one good day;
the **median** is what the person actually holds, and it is the median
rather than the mean because a single mis-set drag factor or an
abandoned piece drags a mean around and a median shrugs it off.  The
game races the median and reports the best, which is the right way
round: nobody rows their PB on demand.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Rows the sheet keeps as scaffolding rather than as people.
NOT_PEOPLE = {"example", "test", "name", "women", "men", ""}

#: A stamp the sheet writes as ``mm:ss`` or ``mm:ss.s``.
STAMP = re.compile(r"(\d{1,2}):(\d{2}(?:\.\d+)?)$")

#: The seed.  Fixed so the file regenerates identically -- an anonymous
#: roster that reshuffled on every build would look like new data.
SEED = 20261019


def seconds(stamp: str):
    match = STAMP.match(stamp.strip())
    if not match:
        return None
    return int(match.group(1)) * 60 + float(match.group(2))


def stamp_of(total: float) -> str:
    """Back to ``mm:ss``, whole seconds -- see the module docstring."""
    total = int(round(total))
    return "%d:%02d" % (total // 60, total % 60)


def read_sheet(path: str):
    """Every person in the export, with each 5k they ever posted."""
    rows = list(csv.reader(io.open(path, encoding="utf-8", newline="")))
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    # The header row is the one that names the columns; every season
    # repeats the same block across the sheet, so there is one "5k total
    # time" column per season and they are found by name rather than by
    # counting, which would break the first time a season was inserted.
    header = next(row for row in rows if "5k total time" in row)
    time_columns = [i for i, cell in enumerate(header)
                    if cell.strip() == "5k total time"]

    squad = None
    people = []
    for row in rows:
        label = row[0].strip()
        if label.upper() in ("WOMEN", "MEN"):
            squad = label.lower()
            continue
        if squad is None or label.lower() in NOT_PEOPLE:
            continue
        times = [t for t in (seconds(row[c]) for c in time_columns) if t]
        if not times:
            continue                       # on the sheet, never tested
        age = row[2].strip()
        weight = row[4].strip()
        people.append({
            "squad": squad,
            "age": int(float(age)) if age.replace(".", "").isdigit() else None,
            "pounds": float(weight) if weight.replace(".", "").isdigit()
            else None,
            "times": sorted(times),
        })
    return people


def anonymise(people):
    """Drop the names, blur the keys, and shuffle the order."""
    random.Random(SEED).shuffle(people)
    counts = {}
    out = []
    for person in people:
        squad = person["squad"]
        counts[squad] = counts.get(squad, 0) + 1
        times = person["times"]
        middle = times[len(times) // 2] if len(times) % 2 else (
            0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2]))
        out.append({
            "id": "%s%02d" % ("W" if squad == "women" else "M", counts[squad]),
            "squad": squad,
            "age": person["age"] or "",
            # five pounds, and the time to the second -- see the docstring
            "pounds": ("" if person["pounds"] is None
                       else int(round(person["pounds"] / 5.0) * 5)),
            "erg_5k": stamp_of(middle),
            "erg_5k_best": stamp_of(times[0]),
            "pieces": len(times),
        })
    return out


FIELDS = ("id", "squad", "age", "pounds", "erg_5k", "erg_5k_best", "pieces")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", help="CSV export of the erg spreadsheet")
    parser.add_argument("--out", default=os.path.join("data",
                                                      "squad_roster.csv"))
    args = parser.parse_args(argv)

    people = anonymise(read_sheet(args.sheet))
    with io.open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(people)

    for squad in ("women", "men"):
        group = [p for p in people if p["squad"] == squad]
        if not group:
            continue
        fast = min(group, key=lambda p: seconds(p["erg_5k"]))
        print("%-6s %2d rowers, %3d pieces, fastest median %s"
              % (squad, len(group), sum(p["pieces"] for p in group),
                 fast["erg_5k"]))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
