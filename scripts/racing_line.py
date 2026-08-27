"""Find and draw the racing line down the Head of the Charles.

    python scripts/racing_line.py
    python scripts/racing_line.py --max-yaw 3.0 --out out/lines
    python scripts/racing_line.py --max-yaw 1.5 2.0 2.5 3.0 --sweep

Optimises a line for minimum time and scores it against several lines a
competent crew might actually row, so the result has something honest to
beat.  Every candidate is judged by the same evaluator, with the same
turn-rate limit and the same 60 second penalty for a forbidden arch.

``--max-yaw`` is the number that matters and it is not well known: the
model makes about 1.5 deg/s at full helm and the coxswain reports about 3.
``--sweep`` runs the whole comparison across a range of it, which is the
more useful output while that disagreement stands.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river import bridges as B          # noqa: E402
from coxswain.river import charles, lines        # noqa: E402
from coxswain.river.charts import CourseGeometry  # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import ReducedModel  # noqa: E402

#: Arch strategies worth costing separately.  The rules leave both the
#: centre and the Cambridge arch open at River Street, Western Avenue and
#: Weeks, and which to take is a real decision: the Cambridge arch is the
#: wider opening at two of the three, but it leaves the boat on the
#: outside for the Weeks turn.  Pinning each and optimising within it
#: prices the choice instead of letting the optimiser bury it.
STRATEGIES = {
    "free choice": {},
    "centre arches": {"River Street": "centre",
                      "Western Avenue": "centre"},
    "Cambridge arches": {"River Street": "Cambridge shore",
                         "Western Avenue": "Cambridge shore"},
    "Cambridge + Weeks": {"River Street": "Cambridge shore",
                          "Western Avenue": "Cambridge shore",
                          "Weeks Footbridge": "Cambridge shore"},
}

INK, MUTED, RULE = "#16211f", "#5c6968", "#dce2e0"
PALETTE = ["#4a5568", "#1f5673", "#2a7f62", "#c8901a", "#a2382a", "#6b3fa0"]


def build(month: int = 10):
    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=month)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(month))
    gates = CourseGeometry(channel=raster).gates_on_course()
    return raster, course, flow, gates


def evaluator(course, flow, raster, gates, model, margin=4.0, pins=None,
              fatigue=True):
    # The Course is built with the *race line* as its centreline, so its
    # station increases in the direction of racing -- which is up the
    # river.  This read False, and the current came out helping the boat
    # by 2.9 s when rowing up the Charles should cost that.
    ev = RouteEvaluator(course, flow=flow, reference_speed=5.2,
                        upstream=True, margin=margin, minimum_depth=1.2,
                        n_samples=1200)
    ev.with_steering(model, raster=raster, gates=gates)
    if fatigue:
        ev.with_exertion()
    if pins:
        ev.required_arches = dict(pins)
    return ev


def strategies(course, raster, flow, gates, model, margin=4.0,
               iterations=70):
    """Optimise a line inside each arch strategy; returns scored routes."""
    out = []
    for name, pins in STRATEGIES.items():
        ev = evaluator(course, flow, raster, gates, model, margin, pins)
        start = lines.pinned_arch_route(course, raster, gates, pins,
                                        margin=margin, name=name)
        best = optimise_route(ev, n_control=13, iterations=iterations,
                              seed=0, initial=start)
        route = Route(best.route.stations, best.route.offsets, name=name)
        out.append((route, ev, ev.evaluate(route)))
    return out


def strategy_table(scored):
    print("  %-20s %9s %8s %8s %8s %9s %9s"
          % ("arch strategy", "RACE time", "dist", "peak yaw", "split",
             "strokes", "W' left"))
    base = None
    for route, _ev, r in scored:
        race = r.elapsed_clean + 60.0 * r.illegal_arches
        if route.name == "centre arches":
            base = race
    for route, _ev, r in scored:
        race = r.elapsed_clean + 60.0 * r.illegal_arches
        delta = "" if base is None else "  %+.1fs" % (race - base)
        print("  %-20s %8.1fs %7.0fm %8.2f %7.0f%% %8.0f %8.0f J%s%s"
              % (route.name, race, r.path_length, r.peak_yaw_rate,
                 100 * r.peak_split, r.split_strokes, r.w_prime_left,
                 "  ILLEGAL" if r.illegal_arches else "", delta))


def loss_table(scored, shortest):
    print()
    print("  where the seconds go, relative to rowing %.0f m in deep still "
          "water" % shortest)
    print("  %-20s %9s %9s %8s %9s %9s %9s"
          % ("line", "ideal", "distance", "depth", "current", "steering",
             "penalty"))
    for route, ev, _r in scored:
        b = ev.loss_breakdown(route, reference_length=shortest)
        print("  %-20s %8.1fs %+8.1fs %+7.1fs %+8.2fs %+8.1fs %+8.1fs"
              % (route.name, b["ideal"], b["distance"], b["depth"],
                 b["current"], b["steering"], b["penalty"]))


def compare(ev, course, raster, gates, margin=4.0, iterations=80):
    candidates = lines.candidate_lines(course, raster, gates, margin=margin)
    best = optimise_route(ev, n_control=13, iterations=iterations, seed=0)
    found = best.route
    candidates.append(Route(found.stations, found.offsets, name="optimised"))
    return [(r, ev.evaluate(r)) for r in candidates]


def table(scored, model):
    reference = (scored[0][1].elapsed_clean
                 + 60.0 * scored[0][1].illegal_arches)
    # `elapsed_clean` excludes penalties, so an illegal line reads fast:
    # "inside the bends" came out 11.7 s quicker than the centreline while
    # taking four 60 s penalties.  Race time is what a crew is scored on,
    # so that is the column to rank by and the one compared against.
    print("  %-22s %9s %8s %8s %9s %9s %8s"
          % ("line", "RACE time", "dist", "peak yaw", "split wanted",
             "strokes", "illegal"))
    for route, result in scored:
        race = result.elapsed_clean + 60.0 * result.illegal_arches
        flag = " " if result.peak_split <= 0.30 else "!"
        print("  %-22s %8.1fs %7.0fm %8.2f %9.0f%%%s %8.0f %6d  %+.1fs"
              % (route.name, race, result.path_length, result.peak_yaw_rate,
                 100.0 * result.peak_split, flag, result.split_strokes,
                 result.illegal_arches, race - reference))


def plot(scored, course, raster, gates, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": RULE,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    water = LinearSegmentedColormap.from_list(
        "w", ["#eaf4f8", "#cbe5f0", "#a9d6e8", "#8ec6dd"])

    station = np.linspace(0.0, course.length, 1200)
    fig = plt.figure(figsize=(15.0, 11.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.0, 1.0], hspace=0.22)

    # -- the river, with every candidate drawn on it ---------------------
    ax = fig.add_subplot(grid[0])
    depth = np.array(raster.depth, dtype=float)
    depth[~raster.water] = np.nan
    ax.imshow(depth, origin="lower",
              extent=[raster.east[0], raster.east[-1],
                      raster.north[0], raster.north[-1]],
              cmap=water, norm=Normalize(0.0, 7.0), interpolation="bilinear",
              zorder=1)
    ax.contour(raster.east, raster.north, raster.navigable.astype(float),
               levels=[0.5], colors=["#c8901a"], linewidths=0.9, zorder=3)

    for gate, metres in gates:
        for arch in B.bridge_arches(gate, raster):
            a, b = arch.interval
            p, q = gate.point_at(a), gate.point_at(b)
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color="#2a7f62" if arch.legal else "#a2382a",
                    lw=3.0, solid_capstyle="butt", zorder=6)
        for pier in gate.piers:
            pt = gate.point_at(pier.centre)
            ax.plot([pt[0]], [pt[1]], marker="s", ms=3.5, color=INK, zorder=7)

    for i, (route, _result) in enumerate(scored):
        pts = course.offset_position(station, route.offset_at(station))
        ax.plot(pts[:, 0], pts[:, 1], color=PALETTE[i % len(PALETTE)],
                lw=2.6 if route.name == "optimised" else 1.5,
                ls="-" if route.name == "optimised" else "--",
                label=route.name, zorder=8, alpha=0.95)

    pad = 140.0
    all_pts = course.offset_position(station, np.zeros_like(station))
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title("Head of the Charles — candidate racing lines\n"
                 "green spans are legal arches, red are 60 s penalties, "
                 "dark squares are piers", loc="left", pad=10)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.94,
              edgecolor=RULE, ncol=2)

    # -- where each line sits across the channel -------------------------
    ax = fig.add_subplot(grid[1])
    half = np.array([course.half_width_at(s) for s in station])
    ax.fill_between(station, -half, half, color="#eaf4f8", zorder=1)
    ax.plot(station, half, color="#c8901a", lw=0.9)
    ax.plot(station, -half, color="#c8901a", lw=0.9)
    for i, (route, _result) in enumerate(scored):
        ax.plot(station, route.offset_at(station),
                color=PALETTE[i % len(PALETTE)],
                lw=2.2 if route.name == "optimised" else 1.2,
                ls="-" if route.name == "optimised" else "--", zorder=4)
    for gate, metres in gates:
        ax.axvline(metres, color="#1f5673", lw=0.8, alpha=0.45)
        ax.annotate(gate.name.replace(" Bridge", "").replace(" RR", ""),
                    (metres, half.max() * 0.92), rotation=90, fontsize=7,
                    color="#1f5673", ha="right", va="top")
    ax.set_ylabel("offset from\ncentreline (m)")
    ax.set_xlim(0, course.length)

    # -- what each line asks of the rudder -------------------------------
    ax = fig.add_subplot(grid[2])
    for i, (route, result) in enumerate(scored):
        pts = course.offset_position(station, route.offset_at(station))
        required = RouteEvaluator._required_yaw(pts, result.speed_ground)
        ax.plot(station, required, color=PALETTE[i % len(PALETTE)],
                lw=1.8 if route.name == "optimised" else 1.0,
                ls="-" if route.name == "optimised" else "--", zorder=4)
    u = 5.2
    rudder_only = np.degrees(model.yaw_control * u * u * model.rudder_limit
                             / (model.yaw_damping * u))
    with_split = np.degrees(
        (model.yaw_control * u * u * model.rudder_limit
         + model.split_control * model.split_limit)
        / (model.yaw_damping * u))
    ax.axhline(rudder_only, color="#c8901a", lw=1.3, ls=":")
    ax.annotate("full rudder alone (%.2f deg/s)" % rudder_only,
                (60, rudder_only * 1.05), fontsize=8, color="#c8901a")
    ax.axhline(with_split, color="#a2382a", lw=1.4, ls=":")
    ax.annotate("rudder + %.0f%% split (%.2f deg/s)"
                % (100 * model.split_limit, with_split),
                (60, with_split * 1.03), fontsize=8, color="#a2382a")
    for gate, metres in gates:
        ax.axvline(metres, color="#1f5673", lw=0.8, alpha=0.45)
    ax.set_ylabel("yaw rate the\nline needs (deg/s)")
    ax.set_xlabel("distance from the start line (m)")
    ax.set_xlim(0, course.length)
    ax.set_ylim(0, None)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def loss_chart(scored, shortest, path):
    """Stacked bars: what each line spends its seconds on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.edgecolor": RULE, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    terms = ("distance", "depth", "current", "steering", "penalty")
    colours = {"distance": "#4a5568", "depth": "#1f5673",
               "current": "#2a7f62", "steering": "#c8901a",
               "penalty": "#a2382a"}

    names, table = [], []
    for route, ev, _r in scored:
        names.append(route.name)
        table.append(ev.loss_breakdown(route, reference_length=shortest))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [2.0, 1.0]})

    y = np.arange(len(names))
    left = np.zeros(len(names))
    for term in terms:
        values = np.array([t[term] for t in table])
        ax.barh(y, values, left=left, color=colours[term], label=term,
                height=0.62)
        left = left + values
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.0, color=INK, lw=0.8)
    ax.set_xlabel("seconds lost against the shortest line in deep still water")
    ax.set_title("Where the race time goes", loc="left", fontsize=11, pad=8)
    ax.legend(fontsize=8, ncol=5, loc="lower right", framealpha=0.95,
              edgecolor=RULE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    race = np.array([t["race"] for t in table])
    best = race.min()
    ax2.barh(y, race - best, color="#6b3fa0", height=0.62)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("seconds behind the best line")
    ax2.set_title("Net", loc="left", fontsize=11, pad=8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    for i, value in enumerate(race - best):
        ax2.annotate("%+.1f s" % value, (value, i), fontsize=8,
                     va="center", ha="left" if value >= 0 else "right",
                     xytext=(4 if value >= 0 else -4, 0),
                     textcoords="offset points")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-limit", type=float, nargs="+",
                        default=[0.30],
                        help="largest port/starboard pressure split to ask "
                             "the crew for (default 0.30). 0 is rudder only")
    parser.add_argument("--sweep", action="store_true",
                        help="compare across every --split-limit given")
    parser.add_argument("--fit", action="store_true",
                        help="fit the steering model to the 6-DOF simulator "
                             "instead of using the documented defaults")
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--margin", type=float, default=4.0,
                        help="metres to keep off each bank")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--out", default="out/lines")
    parser.add_argument("--strategies", action="store_true",
                        help="optimise a line inside each arch strategy "
                             "(centre vs Cambridge through the Powerhouse) "
                             "and break down where the seconds go")
    parser.add_argument("--no-fatigue", action="store_true",
                        help="leave the crew's anaerobic reserve out of it")
    args = parser.parse_args(argv)

    raster, course, flow, gates = build(args.month)
    print("race course %.0f m, %d bridges" % (course.length, len(gates)))
    print()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    from coxswain.river.trajectory import ReducedModel, fit_reduced_model
    if args.fit:
        from coxswain.boats import catalog
        base = fit_reduced_model(catalog.eight(rate=28.0),
                                 reference_speed=5.2)
        print("steering model fitted to the 6-DOF simulator")
    else:
        base = ReducedModel()
    print("  rudder %.0f deg, split_control %.0f, split_drag %.0f"
          % (np.degrees(base.rudder_limit), base.split_control,
             base.split_drag))
    print()

    if args.strategies:
        model = ReducedModel()
        scored = strategies(course, raster, flow, gates, model, args.margin,
                            args.iterations)
        strategy_table(scored)
        shortest = min(r.path_length for _rt, _ev, r in scored)
        loss_table(scored, shortest)
        print()
        print("  wrote", loss_chart(scored, shortest,
                                    os.path.join(args.out,
                                                 "racing_line_losses.png")))
        plain = [(rt, r) for rt, _ev, r in scored]
        print("  wrote", plot(plain, course, raster, gates, model,
                              os.path.join(args.out,
                                           "racing_line_strategies.png")))
        return 0

    limits = args.split_limit if args.sweep else args.split_limit[:1]
    for limit in limits:
        model = dataclasses.replace(base, split_limit=float(limit))
        ev = evaluator(course, flow, raster, gates, model, args.margin)
        scored = compare(ev, course, raster, gates, args.margin,
                         args.iterations)
        print("  pressure split up to %.0f%%:" % (100 * limit))
        table(scored, model)
        name = ("racing_lines_split%02d.png" % round(100 * limit)
                if len(limits) > 1 else "racing_lines.png")
        print("  wrote", plot(scored, course, raster, gates, model,
                              os.path.join(args.out, name)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
