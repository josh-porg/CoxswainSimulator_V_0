"""Kleshnev's sculling averages as numbers (shared by their author 2026-11-12; SOURCES §239).

A spreadsheet of the BioRow database's sculling means at stroke rates 32, 36 and 40: per 10 % of the drive's time and
per 10 % of the recovery's time, the stroke length (oar angle, fraction of the arc), the handle force (fraction of its
peak), the handle, seat and upper-body velocities (fractions of the peak drive handle velocity; negative through the
recovery) and the vertical oar angle (degrees; the blade's centre at the water at 0, covered at -3). The same
quantities as RBN 2023/12's figures (rbn2023.py), for sculling alone and without reading a figure.

    python research/validation/sculling_averages.py     # the spreadsheet -> sculling_averages.csv, and a check

The files live in data/local/literature/biorow_sculling_averages/ (not in the public repo).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "data", "local", "literature", "biorow_sculling_averages")
CSV = os.path.join(DIR, "sculling_averages.csv")
RATES = (32, 36, 40)
#: the spreadsheet's row labels -> this module's keys
ROWS = (("Stroke Length", "length"), ("Handle Force", "force"), ("Handle Velocity", "handle_v"),
        ("Seat Velocity", "seat_v"), ("Upper body Velocity", "upper_v"), ("Vertical angle", "vertical"))
KEYS = ("time", "length", "force", "handle_v", "seat_v", "upper_v", "vertical")


def convert():
    """Read the spreadsheet (xlrd) and write the CSV: one row per rate, phase and sample."""
    import xlrd
    sh = xlrd.open_workbook(os.path.join(DIR, "Sculling averages.xls")).sheet_by_name("Data")
    table = {}
    r = 0
    while r < sh.nrows:
        label = str(sh.cell_value(r, 0))
        key = next((k for name, k in ROWS if label.startswith(name)), None)
        if key is not None:
            for j, rate in enumerate(RATES):
                row = sh.row_values(r + j)
                if int(row[1]) != rate:
                    raise ValueError("row %d: rate %r, expected %d" % (r + j, row[1], rate))
                table[(key, rate)] = np.array(row[2:23], float)
            r += len(RATES)
        else:
            r += 1
    with open(CSV, "w", encoding="utf-8") as fh:
        fh.write("# Kleshnev (BioRow), sculling averages, shared 2026-11-12 (research/validation/sculling_averages.py;\n"
                 "# SOURCES §239). k: the sample, one per 10 % of the phase's time (drive: catch to finish; recovery:\n"
                 "# finish to catch). length: fraction of the oar's arc from the catch; force: fraction of its peak;\n"
                 "# velocities: fractions of the peak drive handle velocity; vertical: degrees.\n"
                 "rate,phase,k,length,force,handle_v,seat_v,upper_v,vertical\n")
        for rate in RATES:
            for phase, idx in (("drive", range(0, 11)), ("recovery", range(10, 21))):
                for k, i in enumerate(idx):
                    fh.write("%d,%s,%d," % (rate, phase, k)
                             + ",".join("%.4f" % table[(key, rate)][i] for key in KEYS[1:]) + "\n")
    return CSV


def reference(rate=32, phase="drive"):
    """The curves at ``rate`` (32, 36 or 40) through ``phase`` as a dict of arrays in the units of rbn2023.reference():
    ``time`` and ``length`` in %, ``force`` and the velocities in % of their peaks, ``vertical`` in degrees."""
    import csv
    rows = [r for r in csv.DictReader(l for l in open(CSV, encoding="utf-8") if not l.startswith("#"))
            if int(r["rate"]) == int(rate) and r["phase"] == phase]
    if len(rows) != 11:
        raise ValueError("no sculling averages at rate %r (%s): have %s" % (rate, phase, RATES))
    rows.sort(key=lambda r: int(r["k"]))
    out = {"time": 10.0 * np.arange(11)}
    for key in KEYS[1:]:
        scale = 1.0 if key == "vertical" else 100.0
        out[key] = scale * np.array([float(r[key]) for r in rows])
    return out


def main():
    print("wrote", convert())
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rbn2023
    fig1 = rbn2023.reference()
    for rate in RATES:
        d = reference(rate)
        print("rate %d drive: force %% " % rate + " ".join("%4.0f" % v for v in d["force"]))
        print("         handle v %% " + " ".join("%4.0f" % v for v in d["handle_v"]))
        print("         vertical   " + " ".join("%4.1f" % v for v in d["vertical"]))
    print("RBN 2023/12 Fig. 1 (all boats): force %  " + " ".join("%4.0f" % v for v in fig1["force"]))
    # the check on §238's reading of Fig. 2: its sculling curves at rates 20 and 40, interpolated to 40
    cyc = rbn2023.scull_cycle(40.0, n=11)
    print("Fig. 2 as read (§238), rate 40, handle v % " + " ".join("%4.0f" % (100 * v) for v in cyc["drive"]["handle"]))
    print("  the same from the spreadsheet            " + " ".join("%4.0f" % v for v in reference(40)["handle_v"]))


if __name__ == "__main__":
    main()
