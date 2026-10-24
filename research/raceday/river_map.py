"""Geometry for the comparison page's river map: course centreline, water edges and landmarks, local metres.

    .venv/Scripts/python.exe research/raceday/river_map.py

Writes river_map.json: x east, y north (m) from the Charles origin, every 20 m of course.
"""
import json, os, sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from coxswain.raceday import card as C                                       # noqa: E402


def main():
    CM = C.CourseModel.build()
    c = CM.course
    L = float(c.length)
    st = np.append(np.arange(0.0, L, 20.0), L)
    hw = c.water_half_width_at(st)
    ctr = c.offset_position(st, np.zeros(len(st)))
    port = c.offset_position(st, hw)
    stbd = c.offset_position(st, -hw)
    marks = []
    for name, d in CM.landmarks:
        p = c.offset_position(np.array([d]), np.zeros(1))[0]
        marks.append(dict(name=name, m=round(d), x=round(float(p[0]), 1), y=round(float(p[1]), 1)))
    for g, d in CM.gates:
        if not any(abs(m["m"] - d) < 30 for m in marks):
            p = c.offset_position(np.array([d]), np.zeros(1))[0]
            marks.append(dict(name=g.name, m=round(float(d)), x=round(float(p[0]), 1), y=round(float(p[1]), 1)))
    marks.sort(key=lambda m: m["m"])
    r = lambda a: np.round(a, 1).tolist()                                    # noqa: E731
    out = dict(length=round(L, 1), st=st.round(1).tolist(), cx=r(ctr[:, 0]), cy=r(ctr[:, 1]),
               px=r(port[:, 0]), py=r(port[:, 1]), sx=r(stbd[:, 0]), sy=r(stbd[:, 1]), marks=marks)
    json.dump(out, open(os.path.join(HERE, "river_map.json"), "w"))
    print("written river_map.json: %d points, %d landmarks" % (len(st), len(marks)), [m["name"] for m in marks])


if __name__ == "__main__":
    main()
