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


#: Clearance from a pier face to the boat's centreline through an arch, m, across the boat's path: a
#: sweep blade reaches about 3.7 m beyond the hull's centreline.
PIER_CLEARANCE = 4.0


def arch_window(model: CourseModel, gate, arch, margin: float = 4.0,
                clearance: float = PIER_CLEARANCE) -> Tuple[float, float, float]:
    """``(station, low, high)``: a control point at ``station`` (the arch centre's own station on the
    deck) with an offset in ``[low, high]`` crosses ``gate`` inside ``arch``, for a line that holds
    its offset through the crossing. The arch interval is along the deck; a skewed deck (the trestle,
    41 degrees) gives a boat less room across its path than along the deck, so the pier clearance is
    divided by the arch's squareness. A window narrower than nothing collapses to the centre."""
    pad = clearance / max(float(getattr(arch, "squareness", 1.0)), 0.3)
    lo, hi = arch.interval
    a, b = (lo + pad, hi - pad) if hi - lo > 2.0 * pad else ((lo + hi) / 2.0,) * 2
    station, _ = lines.arch_point(model.course, gate, arch.centre, margin)
    oa = lines.arch_point(model.course, gate, a, margin)[1]
    ob = lines.arch_point(model.course, gate, b, margin)[1]
    low, high = sorted((oa, ob))
    return float(station), float(low), float(high)


def optimise_line(model: CourseModel, boat, reference_time: float, pins: Dict[str, str],
                  rudder_k: float = 0.0, wind=None, starts=(), step: float = 150.0, free=None,
                  margin: float = 4.0, method: str = "Powell", maxfev: int = 5000) -> dict:
    """The quickest line through the pinned arches.

    Variables: where the line crosses every bridge, **anywhere inside its arch** (``arch_window``),
    the start and finish offsets, and a free point every ``step`` metres (or at ``free`` stations)
    away from the bridges. ``pins`` names the arch at some bridges; at the others the arch is the
    legal one nearest where the first warm start crosses. Each of ``starts`` (routes: the crew's own
    line, a published shortest line) and the straight pinned line is tried as a warm start, and the
    search runs from the best.

    Replaces a parametrisation that fixed every bridge point at its arch's centre and both ends on
    the centreline: on HOCR 2026 that, with the trestle rule wrong, left the "optimum" 7.9 s slower
    than a published legal line (TRACKING, 2026-10-22).
    """
    from scipy.optimize import minimize
    from coxswain.river.route import Route
    course = model.course
    L = float(course.length)
    lead = starts[0] if starts else None
    gate_st, gate_lo, gate_hi = [], [], []
    for gate, metres in model.gates:
        legal = bridges.candidate_arches(gate, model.raster)
        if not legal:
            continue
        wanted = pins.get(gate.name)
        match = [a for a in legal if a.label == wanted]
        if match:
            arch = match[0]
        elif lead is not None:
            here = float(lead.offset_at(np.array([metres]))[0])
            along = gate.station_of(course.offset_position(np.array([metres]), np.array([here]))[0])
            arch = min(legal, key=lambda a: abs(a.centre - along))
        else:
            arch = bridges.racing_arch(gate, model.raster) or legal[0]
        st, lo, hi = arch_window(model, gate, arch, margin)
        gate_st.append(st); gate_lo.append(lo); gate_hi.append(hi)
    end_lim = [max(float(course.half_width_at(s)) - margin, 0.0) for s in (0.0, L)]
    fixed = np.array([0.0] + gate_st + [L])
    if free is None:
        free = np.arange(step, L - 0.5 * step, step)
    free = np.array([f for f in free if np.min(np.abs(fixed - f)) > 0.4 * step], float)
    stations = np.concatenate([fixed, free])
    order = np.argsort(stations)
    stations = stations[order]
    for i in range(1, len(stations)):              # two arch points on one station
        if stations[i] <= stations[i - 1] + 0.5:
            stations[i] = stations[i - 1] + 0.5
    lo = np.concatenate([[-end_lim[0]], gate_lo, [-end_lim[1]],
                         [-max(float(course.half_width_at(f)) - margin, 0.0) for f in free]])[order]
    hi = np.concatenate([[end_lim[0]], gate_hi, [end_lim[1]],
                         [max(float(course.half_width_at(f)) - margin, 0.0) for f in free]])[order]

    def route_of(x):
        return Route(stations, np.clip(x, lo, hi), name="optimised")

    def cost(x):
        r = route_of(x)
        t, _ = _time(model, boat, r, reference_time, wind, rudder_k)
        # crossed a bridge outside a legal arch (the true crossing, as the evaluator checks it): a
        # penalty steep enough that no line gain pays for it, graded so the search can see back in
        t += 300.0 * len(lines.illegal_bridges(r, course, model.raster, model.gates))
        return t + 50.0 * float(np.sum(np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0))) \
            + 100.0 * _over_turn(model, r)

    pinned = lines.pinned_arch_route(course, model.raster, model.gates, pins, margin=margin)
    tried = []
    for r in list(starts) + [pinned]:
        x0 = np.clip(r.offset_at(stations), lo, hi)
        tried.append((cost(x0), getattr(r, "name", "start"), x0))
    c0, name0, x0 = min(tried, key=lambda z: z[0])
    if method == "Powell":
        res = minimize(cost, x0, method="Powell", bounds=list(zip(lo, hi)),
                       options=dict(maxfev=maxfev, xtol=0.3, ftol=1e-4))
    else:
        res = minimize(cost, x0, method=method, options=dict(maxfev=maxfev, xatol=0.5, fatol=0.05))
    x = res.x if res.fun <= c0 else x0
    best = route_of(x)
    t, rudder = _time(model, boat, best, reference_time, wind, rudder_k)
    return dict(pins=pins, time_s=t, rudder_s=rudder, unoptimised_s=c0, warm_start=name0,
                warm_starts={n: float(c) for c, n, _ in tried}, path_m=route_length(model, best),
                stations=list(map(float, stations)), offsets=list(map(float, np.clip(x, lo, hi))),
                evaluations=int(res.nfev), route=best)


def optimised_time(model: CourseModel, boat, reference_time: float, pins: Dict[str, str],
                   rudder_k: float = 0.0, wind=None, free: Tuple[float, ...] = (1100.0, 1610.0, 2030.0, 2500.0),
                   margin: float = 4.0, maxiter: int = 160, start_route=None) -> dict:
    """The quickest line through the pinned arches at a few ``free`` stations (Nelder-Mead on the race
    time with the turning cost): :func:`optimise_line` with this function's older defaults, so the
    bridge points move inside their arches and the ends are free here too.

    A fair comparison of arch choices needs each one's own best line; a line drawn straight from
    arch to arch charges a choice for a poor transition rather than for the arch.
    """
    out = optimise_line(model, boat, reference_time, pins, rudder_k, wind,
                        starts=() if start_route is None else (start_route,), free=free, margin=margin,
                        method="Nelder-Mead", maxfev=maxiter * 4)
    fs = set(map(float, free))
    out["free_stations"] = [s for s in out["stations"] if s in fs]
    out["free_offsets"] = [o for s, o in zip(out["stations"], out["offsets"]) if s in fs]
    return out


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
                                "[how we make the pass: the coxswains to write]"]))
    out.append(dict(letter="C", name="Hard headwind",
                    when="12 mph or more at the start, from the west round to the north-west.",
                    do=["Same line, same arches: re-steering for wind is worth seconds at most.",
                        "Splits slow where the wind is ahead (%s); judge them against the card, not last week."
                        % (headwind_stretches(card) or "see the course breakdown"),
                        "Moves stay in the shallows; hold length and rhythm into the wind."]))
    return out


def headwind_stretches(card: dict) -> str:
    """The forecast's headwind stretches as runs of landmarks, e.g. "Start to Weeks Footbridge"."""
    runs, cur = [], None
    for s in card.get("sections", []):
        if s.get("wind_label") == "headwind":
            cur = [s["start"], s["end"]] if cur is None else [cur[0], s["end"]]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return ", ".join("%s to %s" % (a, b) for a, b in runs)
