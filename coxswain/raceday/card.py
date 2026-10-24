r"""Target splits, section by section, for a target time and a wind.

A head-race target is one number (the time that requalifies, or the best the crew can do) but a
coxswain steers by splits, and the right split is not the same on every stretch: the Charles is
shallower in places (a masters crew sits near the transcritical drag rise there), the current is
slight but not zero, and the wind is sheltered on some reaches and open on others. So the target
time is spread over the course in proportion to what the course costs on each stretch.

How
---
The line is the project's legal arch route (:func:`coxswain.river.lines.arch_route`, legalised
through the bridges). :class:`coxswain.river.route.RouteEvaluator` gives the boat's speed over
ground at every point of it for this boat, with the river's depth and current, the steering it
takes, and (if given) the sheltered wind. That speed profile, integrated, gives the time spent on
each section between landmarks; the target time is distributed in the same proportions.

Two totals are reported for every target: the time as stated (a still-air figure, because the
targets come from a field model of average years) and the time **today**, scaled by what the
wind costs this line. Every crew in the event races the same water within the hour, so the target
moves with the conditions (SOURCES sec. 77): a headwind that costs this line 3% moves the
requalifying time by about 3%.

Uncertainty
-----------
A target is a central time with a range (:mod:`coxswain.raceday.goals`: the 50% time, and the 80%
to 20% times), and every split and checkpoint time is given for all three. The wind's cost is
also given at 30% less and 30% more wind than forecast, about the spread of an hourly forecast a
day out; that range is shown beside the target range, not folded into it, because one is a chance
over the field and the other is a what-if on the weather.

Turning
-------
A hard rudder slows the boat, and the river model's steering charge barely registers (its calm splits
are even through the big turn). The cost is measured instead, from the squad's own boats at Tail of
the Lake 2026 (``research/raceday/rudder_totl.py``): per stroke, the speed lost against the turn
rate taken from the GPS track, at the crew's own pace either side, fits ``dv/v = k omega^2`` with
omega in deg/s, k = 0.0071 +- 0.0009 for the four and 0.0099 +- 0.0009 for the eight (a 2 deg/s turn
costs 3-4%). On the Charles the turn rate is the boat's speed times the line's curvature, the
heading taken over 30 m as the GPS heading was. The cost moves time into the bends: the target total
is unchanged (the field's times already include everyone's turns), and the bends' splits slow while
the straights' quicken. Each section reports the seconds its turning costs at the target pace.

Landmarks
---------
The coxswains' own: the bridges, the start of the Powerhouse stretch (the end of the bend past
Magazine Beach, which is the point the coxswains call), the mile markers, the start of the big turn
and the wooden pier before the finish (OpenStreetMap pier 118416673, the fixed pier on the Boston
side about 160 m short of the line). The two bends are found from the line's heading: Powerhouse is
the first point past 600 m from which the heading holds within 8 degrees for 200 m; the big turn is
the first point past about Mile 2 where it swings by more than 20 degrees in 100 m. A coxswain who
has raced it confirmed the big turn and the pier's side (2026-10-08). The two bridges within 200 m of the start (the railway bridge and the BU Bridge) are passed
before the crew is at pace and do not start a section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from coxswain.hydro.canopy import ShelteredWind
from coxswain.river import charles, lines
from coxswain.river.charts import CourseGeometry
from coxswain.river.course import local_tangent_plane
from coxswain.river.route import RouteEvaluator
from coxswain.river.structures import charles_structures
from coxswain.river.trajectory import ReducedModel


MERGE_M = 100.0          # a landmark closer than this to the one before it does not start a section
START_CLEAR_M = 200.0    # landmarks this close to the start line are passed before the crew is at pace
MILE = 1609.344
#: Riverside landmarks off the line, ``(name, (lat, lon), note)``.
RIVERSIDE = (
    ("Wooden pier", (42.3669017, -71.1353417), "Boston side, ~160 m before the line"),
    ("Riverside BC", (42.358056, -71.115959), "short-course start (high wind)"),
)


def label(relative: float) -> str:
    """Head / tail / cross wind from a relative bearing (wind FROM, 0 = from dead ahead)."""
    if not np.isfinite(relative):
        return "calm"
    r = relative % 360.0
    if r <= 45 or r >= 315:
        return "headwind"
    if 135 <= r <= 225:
        return "tailwind"
    return "cross from starboard" if r < 180 else "cross from port"


@dataclass
class Section:
    start: str
    end: str
    from_m: float          # distance from the start line along the course, m
    to_m: float
    path_m: float          # length of the line on this section, m
    time_s: float          # model time on this section, s
    wind_felt: float = float("nan")   # m/s at 1.5 m, mid-section (what the crew feels)
    heading: float = float("nan")     # direction of travel mid-section, degrees from north
    relative: float = float("nan")    # wind FROM, relative to the bow: 0 head, 90 starboard, 180 tail, 270 port
    rudder_s: float = 0.0             # model seconds on this section owed to turning
    peak_turn: float = 0.0            # highest turn rate on the section, deg/s


@dataclass
class CourseModel:
    """The course, its line and landmarks, built once (a few seconds)."""
    raster: object
    course: object
    flow: object
    gates: list
    route: object
    landmarks: List[Tuple[str, float]] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, level_offset: float = 0.0) -> "CourseModel":
        """``level_offset``: the basin's level against the survey's normal pool, metres (USGS 01104715
        at the New Charles River Dam). It moves the depth under the line only; the centreline,
        landmarks and bridges stay those of the normal pool, so stations compare across levels."""
        raster = charles.charles_channel()
        _, _, race_line, _ = charles.hocr_course(raster)
        course = charles.charles_course(centreline=race_line, month=10, level_offset=level_offset)
        flow = charles.ContinuityFlow(course, discharge=charles.monthly_discharge(10))
        geometry = CourseGeometry(channel=raster)
        gates = geometry.gates_on_course()
        route = lines.legalise(lines.arch_route(course, raster, gates, margin=4.0),
                               course, raster, gates, margin=4.0)
        notes = {}
        marks = [(g.name, float(d)) for g, d in gates]
        for name, latlon, note in RIVERSIDE:
            east, north = local_tangent_plane(latlon[0], latlon[1], charles.CHARLES_ORIGIN)
            marks.append((name, float(course.nearest_station(east, north))))
            notes[name] = note
        marks += [("Mile 1", MILE), ("Mile 2", 2.0 * MILE)]
        straight = straight_from(course)
        if straight is not None:
            marks.append(("Powerhouse", straight))
            notes["Powerhouse"] = "end of the Magazine Beach bend; the straight begins"
        turn = big_turn(course)
        if turn is not None:
            marks.append(("Big turn", turn))
            notes["Big turn"] = "where the line starts to swing"
        marks = sorted((d, name) for name, d in marks if START_CLEAR_M <= d < float(course.length) - MERGE_M)
        kept = [("Start", 0.0)]
        for d, name in marks:
            if d - kept[-1][1] >= MERGE_M:
                kept.append((name, d))
        kept.append(("Finish", float(course.length)))
        return cls(raster, course, flow, gates, route, kept, notes)


def _heading(course, at, step: float = 25.0) -> np.ndarray:
    zero = np.zeros_like(at)
    p, q = course.offset_position(at - step, zero), course.offset_position(at + step, zero)
    return np.degrees(np.arctan2(q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]))


def straight_from(course, after: float = 600.0, until: float = 1600.0, span: float = 200.0,
                  tolerance: float = 8.0, step: float = 25.0) -> Optional[float]:
    """Where a bend ends: the first station past ``after`` from which the heading stays within
    ``tolerance`` degrees for ``span`` metres (the start of the Powerhouse stretch)."""
    s = np.arange(after, until + span, step)
    h = np.unwrap(np.radians(_heading(course, s)))
    n = int(round(span / step))
    for i in range(len(s) - n):
        w = np.degrees(h[i:i + n + 1])
        if w.max() - w.min() <= tolerance:
            return float(s[i])
    return None


def big_turn(course, after: float = 2.0 * MILE - 300.0, swing: float = 20.0, over: float = 100.0,
             step: float = 25.0) -> Optional[float]:
    """Where the line begins its big turn: the first station past ``after`` at which the heading
    swings by more than ``swing`` degrees over the next ``over`` metres."""
    s = np.arange(after, float(course.length) - over - step, step)
    zero = np.zeros_like(s)

    def heading(at):
        p, q = course.offset_position(at, zero), course.offset_position(at + step, zero)
        return np.degrees(np.arctan2(q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]))

    turn = np.abs((heading(s + over) - heading(s) + 180.0) % 360.0 - 180.0)
    hit = np.nonzero(turn > swing)[0]
    return float(s[hit[0]]) if len(hit) else None


def wind_area(boat, speed: float, steering: ReducedModel) -> float:
    """The crew's drag area, scaled so the evaluator charges the wind against this boat's resistance.

    The evaluator divides every added drag (wind and steering) by ``critical_power / speed``. The
    steering model is an eight throughout (mass, yaw inertia, turn drag, 3040 W), so its steering
    charges are an eight's ratios, a fair stand-in for any shell. The wind is not: the drag area
    calibrated from a masters four (13% of its own ~120 N resistance at 3.6 m/s) charged against an
    eight's ~840 N cuts the wind's cost several times over. Scaling the area by the ratio of the two
    resistances gives the wind its share of this boat's resistance and leaves the steering alone.
    (Sizing the steering model's power to the boat instead fixes the wind but inflates every turn by
    the same factor, because the turn drag stays an eight's.)
    """
    from coxswain.hydro.wind import AIR_DENSITY, AeroModel
    aero = AeroModel.calibrate(boat, reference_speed=speed)
    water = 0.5 * AIR_DENSITY * aero.total_area * speed ** 2 / aero.still_air_fraction
    return aero.total_area * (steering.critical_power / speed) / water


def depth_boat(boat):
    """The boat the evaluator solves its depth term for, or ``None`` for the default factor path.

    By default the evaluator gets no boat and multiplies the crew's WHOLE resistance by the
    Schlichting wave factor (up to 3x near ``Fr_h = 1``), where [D11] and ``hull_resistance`` scale
    the wave term only. On the HOCR 2026 GPS tracks that charged three crews 62-156 s of depth on
    their own lines and the boats showed none of it (``research/raceday/shallow_check.py``;
    PHYSICS_PROGRAMME). A boat that carries a non-shipped profile (``physics.resolve("research")
    .apply(boat)``) is handed to the evaluator, which then solves the boat's own power balance with
    Sretenskii's finite-depth wave table, so depth moves the wave term and nothing else. Unstamped
    and shipped boats keep the old path: race cards and the report do not change unless asked.
    """
    if getattr(boat, "physics_profile", None) in (None, "shipped"):
        return None
    return boat


def _evaluate(model: CourseModel, boat, reference_time: float, wind: Optional[Tuple[float, float]]):
    ev = RouteEvaluator(model.course, flow=model.flow, reference_speed=model.course.length / reference_time,
                        upstream=True, margin=4.0, minimum_depth=1.2, n_samples=900,
                        boat=depth_boat(boat))
    steering = ReducedModel()
    ev.with_steering(steering, raster=model.raster, gates=model.gates)
    ev.with_exertion()
    if wind is not None and wind[0] > 0.3 and np.isfinite(wind[1]):
        ev.with_wind(ShelteredWind(charles_structures(), model.raster, wind[0], wind[1], height=0.43),
                     drag_area=wind_area(boat, ev.reference_speed, steering))
    return ev.evaluate(model.route)


#: Speed lost to turning, dv/v per (deg/s)^2, from Tail of the Lake 2026 (see the module docstring).
RUDDER_K = {"4+": 0.0071, "8+": 0.0099}
TURN_SPAN_M = 30.0


#: The extracted centreline carries S-wiggles of 150-200 m wavelength (+-2 deg/s, e.g. 725-950 m past
#: the start) that no coxswain steers; the heading is smoothed over this length before it is differentiated.
TURN_SMOOTH_M = 40.0


def turn_rate(points: np.ndarray, speed: np.ndarray, span: float = TURN_SPAN_M,
              smooth: float = TURN_SMOOTH_M) -> np.ndarray:
    """deg/s along a path: speed times curvature, the heading taken over ``span`` metres and then
    smoothed (Gaussian, ``smooth`` metres) so the turn rate is the steered line's, not the chart's."""
    from scipy.ndimage import gaussian_filter1d
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    x = lambda at: np.interp(at, d, points[:, 0])           # noqa: E731
    y = lambda at: np.interp(at, d, points[:, 1])           # noqa: E731
    grid = np.arange(0.0, d[-1] + 5.0, 5.0)
    lo, hi = np.clip(grid - span / 2, 0, d[-1]), np.clip(grid + span / 2, 0, d[-1])
    psi = np.unwrap(np.arctan2(x(hi) - x(lo), y(hi) - y(lo)))
    if smooth > 0:
        psi = gaussian_filter1d(psi, smooth / 5.0, mode="nearest")
    rate = np.interp(d, grid, np.degrees(np.gradient(psi, grid)))
    return rate * speed


def _sections(model: CourseModel, result, wind: Optional[Tuple[float, float]],
              rudder_k: float = 0.0) -> List[Section]:
    station = result.station
    points = model.course.offset_position(station, result.route.offset_at(station))
    seg = np.hypot(*np.diff(points, axis=0).T)
    v = result.speed_ground
    omega = turn_rate(points, v)
    loss = np.clip(rudder_k * omega ** 2, 0.0, 0.25)
    v_turn = v * (1.0 - loss)
    dt0 = seg / np.maximum(0.5 * (v[:-1] + v[1:]), 0.05)
    dt = seg / np.maximum(0.5 * (v_turn[:-1] + v_turn[1:]), 0.05)
    extra = np.concatenate([[0.0], np.cumsum(dt - dt0)])
    cum_t = np.concatenate([[0.0], np.cumsum(dt)])
    cum_d = np.concatenate([[0.0], np.cumsum(seg)])
    felt = None
    if wind is not None and wind[0] > 0.3 and np.isfinite(wind[1]):
        felt = ShelteredWind(charles_structures(), model.raster, wind[0], wind[1], height=1.5)
    out = []
    for (a, sa), (b, sb) in zip(model.landmarks[:-1], model.landmarks[1:]):
        i, j = np.searchsorted(station, [sa, sb])
        j = min(j, len(station) - 1)
        mid = 0.5 * (sa + sb)
        w = float("nan")
        if felt is not None:
            p = model.course.offset_position(np.array([mid]), np.array([0.0]))[0]
            w = float(felt.speed_at(p[0], p[1]))
        q = model.course.offset_position(np.array([mid - 15.0, mid + 15.0]), np.array([0.0, 0.0]))
        heading = float(np.degrees(np.arctan2(q[1, 0] - q[0, 0], q[1, 1] - q[0, 1])) % 360.0)
        rel = float((wind[1] - heading) % 360.0) if (wind is not None and np.isfinite(wind[1])) else float("nan")
        out.append(Section(a, b, sa, sb, float(cum_d[j] - cum_d[i]), float(cum_t[j] - cum_t[i]), w, heading, rel,
                           float(extra[j] - extra[i]), float(np.max(np.abs(omega[i:j + 1])))))
    return out


def split_of(seconds: float, metres: float) -> float:
    """Seconds per 500 m."""
    return seconds / max(metres, 1e-9) * 500.0


def fmt(seconds: float) -> str:
    """m:ss.s"""
    if not np.isfinite(seconds):
        return "-"
    m, s = divmod(round(float(seconds), 1), 60.0)
    return "%d:%04.1f" % (m, s)


#: A stretch where the river (depth, current) costs this much per 500 m in still air, over the course's
#: typical stretch, is marked for a push: the same extra effort moves the boat most there.
PUSH_S500 = 1.0
WIND_SPREAD = 0.3     # the wind's cost is also given at 30% less and 30% more wind than forecast


def _as_goal(name, value) -> dict:
    if isinstance(value, dict):
        return dict(value, name=value.get("name", name))
    return dict(name=name, central=float(value), fast=float(value), slow=float(value))


def race_card(model: CourseModel, boat, targets, wind: Optional[Tuple[float, float]] = None,
              rudder_k: float = 0.0) -> dict:
    """Target splits and checkpoint times by section for each goal, calm and today.

    ``targets``: a list of goal dicts (``name``, ``central``, ``fast``, ``slow``, still-air seconds;
    see :mod:`coxswain.raceday.goals`) or a ``{name: seconds}`` dict.
    ``wind``: (speed m/s at 10 m open exposure, direction FROM in degrees), or None for calm.
    ``rudder_k``: speed lost to turning, dv/v per (deg/s)^2 (:data:`RUDDER_K`); 0 leaves it out.
    """
    goals = [_as_goal(n, v) for n, v in targets.items()] if isinstance(targets, dict) else \
        [_as_goal(g["name"], g) for g in targets]
    reference = float(np.median([g["central"] for g in goals]))
    calm_sections = _sections(model, _evaluate(model, boat, reference, None), None, rudder_k)
    calm_total = sum(s.time_s for s in calm_sections)

    def factor_for(w):
        if w is None:
            return 1.0, calm_sections
        sec = _sections(model, _evaluate(model, boat, reference, w), w, rudder_k)
        return sum(s.time_s for s in sec) / calm_total, sec

    factor, windy_sections = factor_for(wind)
    windy_total = sum(s.time_s for s in windy_sections)
    cost_range = None
    if wind is not None and wind[0] > 0.3:
        lighter, _ = factor_for((wind[0] * (1 - WIND_SPREAD), wind[1]))
        stronger, _ = factor_for((wind[0] * (1 + WIND_SPREAD), wind[1]))
        cost_range = [100.0 * (lighter - 1.0), 100.0 * (stronger - 1.0)]
    path = sum(s.path_m for s in windy_sections)
    # where the river itself is slow: still air, turning cost removed, against the course's median
    river = [split_of(c.time_s - c.rudder_s, c.path_m) for c in calm_sections]
    typical = float(np.median(river))
    slow = [r - typical for r in river]
    card = dict(
        wind=None if wind is None else dict(speed=wind[0], direction=wind[1]),
        wind_factor=factor, wind_cost_pct=100.0 * (factor - 1.0), wind_cost_pct_range=cost_range,
        wind_spread=WIND_SPREAD, rudder_k=rudder_k,
        sections=[dict(start=s.start, end=s.end, from_m=s.from_m, to_m=s.to_m, path_m=s.path_m,
                       wind_felt=s.wind_felt, heading=s.heading, relative=s.relative, wind_label=label(s.relative),
                       share=s.time_s / windy_total, rudder_share=s.rudder_s / windy_total,
                       peak_turn=s.peak_turn, note=model.notes.get(s.end),
                       river_slow_s500=slow[i], push=bool(i > 0 and slow[i] >= PUSH_S500))
                  for i, s in enumerate(windy_sections)],
        targets=[])
    for g in goals:
        out = dict(g)
        for key in ("central", "fast", "slow"):
            today = g[key] * factor
            out["today_" + key] = today
            out["splits_" + key] = [split_of(today * s.time_s / windy_total, s.path_m) for s in windy_sections]
            out["at_landmark_" + key] = list(np.cumsum([today * s.time_s / windy_total for s in windy_sections]))
            out["split_overall_" + key] = split_of(today, path)
            out["rudder_" + key] = [today * s.rudder_s / windy_total for s in windy_sections]
        # the sprint 1 names, kept for the console
        out.update(total_still_air=g["central"], total_today=out["today_central"],
                   split_overall_today=out["split_overall_central"], splits=out["splits_central"],
                   at_landmark=out["at_landmark_central"])
        card["targets"].append(out)
    return card


def rescale(card: dict, new_factor: float, source: str) -> dict:
    """Replace the wind model's day factor with a measured one (``dayfactor``): every target's totals,
    splits, checkpoint times and turning seconds move by ``new_factor / wind_factor``; the split of
    time between stretches (the wind model's) is kept."""
    k = new_factor / card["wind_factor"]
    for t in card["targets"]:
        for key in ("central", "fast", "slow"):
            t["today_" + key] *= k
            t["split_overall_" + key] *= k
            for name in ("splits_", "at_landmark_", "rudder_"):
                t[name + key] = [x * k for x in t[name + key]]
        t.update(total_today=t["today_central"], split_overall_today=t["split_overall_central"],
                 splits=t["splits_central"], at_landmark=t["at_landmark_central"])
    card["model_wind_factor"] = card["wind_factor"]
    card["wind_factor"] = new_factor
    card["factor_source"] = source
    return card


def scale_targets(card: dict, k: float) -> dict:
    """Multiply every target time (totals, splits, checkpoints, turning) by ``k`` (the river's flow
    and level, which every crew pays like the wind)."""
    for t in card["targets"]:
        for key in ("central", "fast", "slow"):
            t["today_" + key] *= k
            t["split_overall_" + key] *= k
            for name in ("splits_", "at_landmark_", "rudder_"):
                t[name + key] = [x * k for x in t[name + key]]
        t.update(total_today=t["today_central"], split_overall_today=t["split_overall_central"],
                 splits=t["splits_central"], at_landmark=t["at_landmark_central"])
    return card



SHORT_START = "Riverside BC"


def short_course(card: dict, start: str = SHORT_START) -> dict:
    """The card for the shortened course: the same river, line and finish, started at ``start``.

    Every section after the start keeps its time and split; the targets are the full course's times
    from the start landmark on (the field's order does not change when everyone rows the same shorter
    course, so the goals scale with the share of time that remains)."""
    import copy
    i0 = next(i for i, x in enumerate(card["sections"]) if x["start"] == start)
    c = copy.deepcopy(card)
    secs = c["sections"][i0:]
    remain = sum(x["share"] for x in secs)
    for x in secs:
        x["share"] = x["share"] / remain
    c["sections"] = secs
    path = sum(x["path_m"] for x in secs)
    for t in c["targets"]:
        for key in ("central", "fast", "slow"):
            before = t["at_landmark_" + key][i0 - 1] if i0 > 0 else 0.0
            t["at_landmark_" + key] = [v - before for v in t["at_landmark_" + key][i0:]]
            t["splits_" + key] = t["splits_" + key][i0:]
            t["rudder_" + key] = t["rudder_" + key][i0:]
            t["today_" + key] = t["at_landmark_" + key][-1]
            t["split_overall_" + key] = t["today_" + key] / path * 500.0
            t[key] = t[key] * remain
        t.update(total_today=t["today_central"], split_overall_today=t["split_overall_central"],
                 splits=t["splits_central"], at_landmark=t["at_landmark_central"])
        t.pop("lake_split", None)
    c["course"] = dict(name="short", start=start, start_m=secs[0]["from_m"], length_m=path, share=remain)
    return c
