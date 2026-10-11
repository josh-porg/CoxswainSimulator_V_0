r"""A small map of the course for the race card (R4): banks, the planned line, bridges, landmarks.

The phone draws it as SVG from these coordinates, so it works offline with the rest of the card.
Banks are the water mask's outline (``matplotlib`` contour of the channel raster), simplified to a
few metres; the line is the crew's planned line; bridges are their deck lines with the planned arch
marked. Coordinates are metres east and north from the map's south-west corner, rounded to 1 m.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker, iterative."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = points[a], points[b]
        d = q - p
        n = np.hypot(*d)
        seg = points[a + 1:b]
        if n < 1e-9:
            dist = np.hypot(*(seg - p).T)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / n
        i = int(np.argmax(dist))
        if dist[i] > tolerance:
            k = a + 1 + i
            keep[k] = True
            stack += [(a, k), (k, b)]
    return points[keep]


def course_map(model, route, sections: List[dict], decisions: Optional[List[dict]] = None,
               pad: float = 150.0, tolerance: float = 4.0) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.linspace(0.0, float(model.course.length), 600)
    line = model.course.offset_position(s, route.offset_at(s))
    x0, x1 = line[:, 0].min() - pad, line[:, 0].max() + pad
    y0, y1 = line[:, 1].min() - pad, line[:, 1].max() + pad
    ch = model.raster
    east, north = np.asarray(ch.east), np.asarray(ch.north)
    ix = (east >= x0) & (east <= x1)
    iy = (north >= y0) & (north <= y1)
    water = np.asarray(ch.water, float)[np.ix_(iy, ix)]
    fig = plt.figure()
    cs = plt.contour(east[ix], north[iy], water, levels=[0.5])
    banks = []
    for seg in cs.allsegs[0]:
        if len(seg) < 10:
            continue
        length = float(np.hypot(*np.diff(seg, axis=0).T).sum())
        if length < 250.0:
            continue
        banks.append(_simplify(np.asarray(seg), tolerance))
    plt.close(fig)

    def rel(p):
        return [[int(round(a - x0)), int(round(b - y0))] for a, b in np.atleast_2d(p)]

    planned = {d["bridge"]: d.get("planned") for d in (decisions or [])}
    bridges = []
    for gate, d in model.gates:
        bridges.append(dict(name=gate.name, at_m=float(d), a=rel(gate.start)[0], b=rel(gate.end)[0],
                            arch=planned.get(gate.name)))
    marks = []
    for sec in sections:
        p = model.course.offset_position(np.array([sec["to_m"]]), route.offset_at(np.array([sec["to_m"]])))[0]
        marks.append(dict(name=sec["end"], at_m=sec["to_m"], xy=rel(p)[0]))
    pushes = []
    for sec in sections:
        if sec.get("push"):
            q = np.linspace(sec["from_m"], sec["to_m"], 40)
            pushes.append(rel(model.course.offset_position(q, route.offset_at(q))))
    return dict(width=int(round(x1 - x0)), height=int(round(y1 - y0)),
                banks=[rel(b) for b in banks], line=rel(_simplify(line, 2.0)),
                start=rel(line[0])[0], finish=rel(line[-1])[0], bridges=bridges, marks=marks, pushes=pushes)
