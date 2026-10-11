r"""The race plan's key decisions: which arch at each bridge, and what the alternative costs.

Three bridges on the course have a choice of legal arch: River Street, Western Avenue and Weeks
each have the centre arch and the Cambridge-shore arch (the railway bridge, the BU Bridge, Anderson
and Eliot have one legal arch each; :func:`coxswain.river.bridges.candidate_arches`). For every
combination the line is pinned through those arches (:func:`coxswain.river.lines.pinned_arch_route`),
pulled legal, and timed by the same evaluator and turning cost as the race card, so the cost of an
alternative includes the extra steering it takes.

What a plan needs from this is not the best line (that is the card's line) but the **price of each
alternative**. The alternative is the answer to traffic: a crew being passed must yield the side the
passer declares, and a crew alongside at a bridge takes one arch. The decision rule is then a
comparison of seconds: take the alternative arch when holding the planned one would cost more
(easing, a clash of blades, a wash) than the alternative's price.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from coxswain.river import bridges, lines
from coxswain.raceday.card import CourseModel, _evaluate, _sections


def decisions(model: CourseModel) -> Dict[str, List[str]]:
    """Bridges with more than one legal arch, and their arches (racing arch first)."""
    out = {}
    for gate, _ in model.gates:
        arches = bridges.candidate_arches(gate, model.raster)
        if len(arches) > 1:
            racing = bridges.racing_arch(gate, model.raster)
            labels = [a.label for a in arches]
            if racing is not None and racing.label in labels:
                labels.remove(racing.label)
                labels.insert(0, racing.label)
            out[gate.name] = labels
    return out


def _time(model: CourseModel, boat, route, reference_time: float, wind, rudder_k: float) -> Tuple[float, float]:
    keep = model.route
    try:
        model.route = route
        sec = _sections(model, _evaluate(model, boat, reference_time, wind), wind, rudder_k)
    finally:
        model.route = keep
    return sum(s.time_s for s in sec), sum(s.rudder_s for s in sec)


def arch_costs(model: CourseModel, boat, reference_time: float, rudder_k: float = 0.0,
               wind: Optional[Tuple[float, float]] = None, planned: Optional[Dict[str, str]] = None,
               optimise: bool = False) -> dict:
    """Seconds for arch choices, relative to the planned arches (the racing arches if none given).

    With ``planned``, only the plan and each single-bridge departure from it are timed (what a cox
    weighs at a bridge); without, every combination. ``optimise`` gives each choice its own best line
    (:func:`optimised_time`), which a fair comparison needs and which takes about a minute a line.
    """
    choice = decisions(model)
    names = list(choice)
    plan = {n: (planned or {}).get(n, choice[n][0]) for n in names}
    if planned:
        combos = [dict(plan)] + [dict(plan, **{n: lab}) for n in names for lab in choice[n] if lab != plan[n]]
    else:
        combos = [dict(zip(names, c)) for c in itertools.product(*[choice[n] for n in names])]
    rows = []
    for pins in combos:
        if optimise:
            r = optimised_time(model, boat, reference_time, pins, rudder_k, wind)
            rows.append(dict(pins=pins, time_s=r["time_s"], rudder_s=r["rudder_s"], path_m=r["path_m"]))
            continue
        route = lines.legalise(lines.pinned_arch_route(model.course, model.raster, model.gates, pins, margin=4.0),
                               model.course, model.raster, model.gates, margin=4.0)
        t, rudder = _time(model, boat, route, reference_time, wind, rudder_k)
        rows.append(dict(pins=pins, time_s=t, rudder_s=rudder, path_m=float(route_length(model, route))))
    base = next(r for r in rows if r["pins"] == plan)
    for r in rows:
        r["delta_s"] = r["time_s"] - base["time_s"]
        r["delta_rudder_s"] = r["rudder_s"] - base["rudder_s"]
        r["delta_path_m"] = r["path_m"] - base["path_m"]
    single = {}
    for n in names:
        for label in choice[n]:
            if label == plan[n]:
                continue
            r = next(x for x in rows if x["pins"] == dict(plan, **{n: label}))
            single[n] = dict(planned=plan[n], alternative=label, delta_s=r["delta_s"],
                             delta_rudder_s=r["delta_rudder_s"], delta_path_m=r["delta_path_m"])
    return dict(bridges=choice, plan=plan, combinations=sorted(rows, key=lambda r: r["delta_s"]), single=single)


def route_length(model: CourseModel, route) -> float:
    s = np.linspace(0.0, float(model.course.length), 2000)
    p = model.course.offset_position(s, route.offset_at(s))
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


#: Bridges with one legal arch: the alternative is a penalty, not a choice (HOCR rules: the Boston
#: arch of every bridge, and the Cambridge arch at the railway trestle, Anderson and Eliot, 60 s).
ONE_ARCH = ("Larz Anderson", "Eliot Bridge")
#: Below this, two arch choices are not told apart: the optimised lines' times are good to about a second.
EQUAL_S = 1.5
PENALTY_S = 60.0


def _seconds(lo: float, hi: float) -> str:
    return ("%.0f s" % hi) if round(lo) == round(hi) else ("%.0f-%.0f s" % (lo, hi))


def key_decisions(model: CourseModel, costs: List[dict]) -> List[dict]:
    """The arch decisions for the plan, priced over one or more wind cases (``arch_costs`` results)."""
    at = {g.name: float(d) for g, d in model.gates}
    out = []
    for name in costs[0]["single"]:
        s = [c["single"][name] for c in costs]
        lo, hi = min(x["delta_s"] for x in s), max(x["delta_s"] for x in s)
        path = s[0]["delta_path_m"]
        plan, alt = s[0]["planned"], s[0]["alternative"]
        if max(abs(lo), abs(hi)) < EQUAL_S:
            verdict = "either"
            advice = ("Plan: %s arch. The %s arch is within %.1f s of it (%+.0f m of line), inside what the model "
                      "can tell apart: choose by traffic." % (plan, alt, max(abs(lo), abs(hi)), path))
        elif hi < 0:
            verdict = "quicker"
            advice = ("Plan: %s arch. The %s arch is %s quicker (%.0f m shorter); the plan pays that for its "
                      "reasons (traffic, a straighter run), so take the %s arch if it is clear."
                      % (plan, alt, _seconds(-hi, -lo), -path, alt))
        elif lo < 8.0:
            verdict = "cheap"
            advice = ("Plan: %s arch. The %s arch costs %s (%+.0f m of line): take it rather than ease "
                      "if a crew is holding yours." % (plan, alt, _seconds(max(lo, 0), hi), path))
        else:
            verdict = "last resort"
            advice = ("Plan: %s arch. The %s arch costs %s (%+.0f m), more than easing for a few "
                      "strokes; use it only to avoid a collision." % (plan, alt, _seconds(lo, hi), path))
        out.append(dict(bridge=name, at_m=at.get(name), planned=plan, alternative=alt,
                        price_s=[lo, hi], verdict=verdict, advice=advice))
    for name in ONE_ARCH:
        out.append(dict(bridge=name, at_m=at.get(name), planned="centre", alternative=None,
                        price_s=[PENALTY_S, PENALTY_S], verdict="no choice",
                        advice="Centre only: the other arches are a %.0f s penalty." % PENALTY_S))
    return sorted(out, key=lambda d: d["at_m"] or 0.0)


#: The sharpest turn a line may ask for, deg/s, measured as the GPS measured it (heading over ~30 m,
#: lightly smoothed). The squad's boats peaked at 2.7 deg/s at Tail of the Lake; the race card's
#: turning cost smooths over 40 m (to ignore the chart's centreline wiggle), which alone would let an
#: optimiser hide a short kink.
MAX_TURN = 3.0


def _over_turn(model: CourseModel, route) -> float:
    from coxswain.raceday.card import turn_rate
    s = np.linspace(0.0, float(model.course.length), 1200)
    p = model.course.offset_position(s, route.offset_at(s))
    v = np.full(len(s), 4.0)              # a representative racing speed, m/s
    return float(max(np.max(np.abs(turn_rate(p, v, 30.0, 20.0))) - MAX_TURN, 0.0))


def optimised_time(model: CourseModel, boat, reference_time: float, pins: Dict[str, str],
                   rudder_k: float = 0.0, wind=None, free: Tuple[float, ...] = (1100.0, 1610.0, 2030.0, 2500.0),
                   margin: float = 4.0, maxiter: int = 160) -> dict:
    """The quickest line through the pinned arches: the arch points stay fixed and the line is free
    to move at ``free`` stations between them (Nelder-Mead on the race time with the turning cost).

    A fair comparison of arch choices needs each one's own best line; a line drawn straight from
    arch to arch charges a choice for a poor transition rather than for the arch.
    """
    from scipy.optimize import minimize
    from coxswain.river.route import Route
    pinned = lines.legalise(lines.pinned_arch_route(model.course, model.raster, model.gates, pins, margin=margin),
                            model.course, model.raster, model.gates, margin=margin)
    st0, off0 = np.asarray(pinned.stations, float), np.asarray(pinned.offsets, float)
    free = np.asarray([f for f in free if np.min(np.abs(st0 - f)) > 60.0], float)
    limit = np.array([max(model.course.half_width_at(f) - margin, 0.0) for f in free])
    start = np.clip(pinned.offset_at(free), -limit, limit)

    def route_of(x):
        st = np.concatenate([st0, free])
        off = np.concatenate([off0, np.clip(x, -limit, limit)])
        o = np.argsort(st)
        return Route(st[o], off[o], name="optimised")

    def cost(x):
        r = route_of(x)
        legal = lines.legalise(r, model.course, model.raster, model.gates, margin=margin)
        if len(np.atleast_1d(legal.stations)) != len(np.atleast_1d(r.stations)):
            return 1e6                                   # the free points pulled it out of an arch
        t, _ = _time(model, boat, r, reference_time, wind, rudder_k)
        return t + 50.0 * float(np.sum(np.maximum(np.abs(x) - limit, 0.0))) + 100.0 * _over_turn(model, r)

    t_start = cost(start)
    res = minimize(cost, start, method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=0.5, fatol=0.05, initial_simplex=None))
    best = route_of(res.x)
    t, rudder = _time(model, boat, best, reference_time, wind, rudder_k)
    return dict(pins=pins, time_s=t, rudder_s=rudder, unoptimised_s=t_start, path_m=route_length(model, best),
                free_stations=list(map(float, free)), free_offsets=list(map(float, np.clip(res.x, -limit, limit))),
                evaluations=int(res.nfev), route=best)


def variants(card: dict) -> List[dict]:
    """The start-line choices (plan letters), drafted from the card: A clear water, B the likeliest
    traffic, C a strong headwind. Each says when it applies and what changes; the coxswain edits the
    wording. Built from the card so the details follow the latest traffic and decisions."""
    dec = {d["bridge"]: d for d in card.get("decisions", [])}
    arches = ", ".join("%s %s" % (b.replace(" Footbridge", "").replace(" Avenue", ""), d["planned"].split()[0])
                       for b, d in dec.items() if d.get("alternative"))
    pushes = [s for s in card.get("sections", []) if s.get("push")]
    moves = " and ".join("%s to %s" % (s["start"], s["end"]) for s in pushes) or "as planned"
    turn = next((s for s in card.get("sections", []) if s["start"] == "Big turn"), None)
    out = [dict(letter="A", name="Clear water",
                when="No crew within a few lengths at the start.",
                do=["Arches: %s." % arches if arches else "Planned arches.",
                    "Moves: %s." % moves,
                    "Big turn from just past Mile 2, one even rate%s." % (" to Eliot" if turn else "")])]
    tr = card.get("traffic") or {}
    nb = sorted(tr.get("neighbours", []), key=lambda n: -max(n["p_we_pass"], n["p_they_pass"]))
    if nb and max(nb[0]["p_we_pass"], nb[0]["p_they_pass"]) >= 0.2:
        n = nb[0]
        behind = n["starts_s"] > 0
        where = ("between " + (n.get("near") or "").replace("-", " and ")) if n.get("near") else "early"
        free = [b for b, d in dec.items() if d.get("verdict") in ("either", "quicker", "free", "cheap")]
        if behind:
            out.append(dict(letter="B", name="Passed early",
                            when="Bow %d (%s) closing on us, most likely %s." % (n["bow"], n["club"], where),
                            do=["Yield the side they call, early and cleanly.",
                                "If they take our arch, use the other at %s: it costs little." % " or ".join(free)
                                if free else "Hold our arches unless blocked.",
                                "Keep our moves where planned."]))
        else:
            out.append(dict(letter="B", name="Passing",
                            when="Closing on bow %d (%s), most likely %s." % (n["bow"], n["club"], where),
                            do=["Declare our side at one length of open water.",
                                "Choose the side that leaves us the centre arch at the next bridge.",
                                "[how we make the pass: Joshua / Sophie to write]"]))
    w = card.get("wind") or {}
    out.append(dict(letter="C", name="Hard headwind",
                    when="A westerly or south-westerly of 12 mph or more at the start.",
                    do=["Same line, same arches: re-steering for wind is worth seconds at most.",
                        "Splits slow from Weeks to the finish; judge them against the card, not last week.",
                        "Moves stay in the shallows; hold length and rhythm into the wind."]))
    return out
