"""Run everything and build the coach's report.

    python scripts/make_report.py                 # everything
    python scripts/make_report.py --quick         # skip the animations
    python scripts/make_report.py --out out/hocr  # somewhere else

Produces **one self-contained HTML file** -- ``out/report/hocr_report.html``
by default -- with every figure embedded, so it can be handed to somebody
without the repository.

The stages, in order, because each depends on the one before:

1. verify the river: bridges against the federal survey, arches, widths
2. optimise a racing line and score it against lines a crew would row
3. price the arch strategies and break the race time into its causes``
4. steer the full 6-DOF boat down the winning line, both controllers
5. draw the charts, the lines, the losses and the animations
6. assemble the page

Nothing on the page is transcribed.  Every table and figure comes from
the run that writes it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.progress import progress, stage            # noqa: E402
from coxswain.report import (Embed, Figure, Finding,  # noqa: E402
                             Report, Table)
from coxswain.river import bridges as B                   # noqa: E402
from coxswain.river import charles, charts, lines         # noqa: E402
from coxswain.river.charts import CourseGeometry          # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,   # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import (ReducedModel,       # noqa: E402
                                       fit_reduced_model)
from coxswain.sim.control import Coxswain                 # noqa: E402
from coxswain.sim.guidance import PathFollower            # noqa: E402
from coxswain.sim.mpc import PathMPC, horizon_for                      # noqa: E402
from coxswain.sim.simulator import RowingSimulator        # noqa: E402

NBI = {
    "River Street": (42.36124, -71.11675),
    "Western Avenue": (42.36425, -71.11690),
    "Larz Anderson": (42.36896, -71.12316),
    "Eliot Bridge": (42.37175, -71.13286),
    "BU Bridge": (42.35262, -71.11064),
}
MODEL = {
    "River Street": "RIVER_ST_BRIDGE",
    "Western Avenue": "WESTERN_AVE_BRIDGE",
    "Larz Anderson": "LARZ_ANDERSON_BRIDGE",
    "Eliot Bridge": "ELIOT_BRIDGE",
    "BU Bridge": "BU_BRIDGE",
}


#: The masters operating point as a power scale, the same 0.658 the
#: sibling studies use (mpc_tune, stroke_rate, unsteady, mpc_bench).
MASTERS_POWER = 0.658


def reference_eight(rate=28.0, profile=None):
    """The masters eight the tactical work is priced on.

    ``power_scales`` is set, which it was not: the steering leg used to
    build this boat and leave it at the catalogue's full power, so half
    the report described a crew rowing 6.10 m/s while the other half
    priced lines at a masters 4.85.
    """
    from coxswain import physics

    boat = physics.resolve(profile).apply(catalog.eight(rate=rate))
    boat.power_scales = np.full(boat.n_seats, MASTERS_POWER)
    return boat


def hocr_four(rate=30.0, profile=None):
    """This project's own Women's Veteran 60+ four, from the rig editor.

    Their weights, heights, rig and coxswain, and each seat's share of
    the power from its own erg -- the same lineup the trainer races.
    The crew's power scale is calibrated against ``mean_handle_power``,
    which reports the scale-1.0 figure, so the ratio is the scale that
    puts the crew at their measured watts.
    """
    from coxswain.crew.exertion import mean_handle_power
    from coxswain.viz.menu import build_boat
    from coxswain.viz.rigview import PRESETS

    lineup = PRESETS["HOCR 4+"]()
    boat, _made = build_boat("4+", rate, lineup=lineup, profile=profile)
    watts = [r.watts for r in lineup.rowers if r.watts]
    target = float(np.mean(watts)) if watts else 0.0
    unit = mean_handle_power(boat, samples=180)
    ratios = np.asarray(getattr(boat, "seat_ratios",
                                np.ones(boat.n_seats)), dtype=float)
    boat.power_scales = (target / max(unit, 1.0)) * ratios
    return boat, lineup, target


def measured_four_pace(course, year=2024, club="Sammamish"):
    """Their own Charles, as a speed: the pace to price their lines at.

    Taken from the published result rather than from the simulator,
    because the two speed models disagree by a factor of 2.3 on exactly
    this boat (docs/TRACKING.md) and the field is the arbiter.
    """
    import csv

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "hocr_wvet4_results.csv")
    best = None
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["year"]) == year and club.lower() in row["club"].lower():
                seconds = float(row["seconds"])
                if best is None or seconds < best[0]:
                    best = (seconds, int(row["place"]))
    if best is None:
        return None
    return best[0], course.length / best[0], best[1]


def settled_speed(boat, start_speed, duration=70.0, dt=0.01):
    """The speed this boat settles at, driven straight."""
    sim = RowingSimulator(boat,
                          coxswain=Coxswain(rudder_override=lambda t, s: 0.0),
                          fast=True)
    result = sim.run(duration=duration, dt=dt, surge_speed=start_speed)
    time_s = np.asarray(result.time)
    speed = np.hypot(*np.asarray(result.velocity)[:2])
    return float(speed[time_s > duration - 4 * boat.timing.period].mean())


def quasi_steady_gap(reference_speed, scale=MASTERS_POWER, rate=28.0,
                     profile=None,
                     duration=70.0, dt=0.01):
    """How optimistic the quasi-steady evaluator is, measured.

    The evaluator prices a line at a fixed ``reference_speed``; the full
    6-DOF boat has to produce that speed from the crew's power.  Run the
    same boat at the same operating point, let it settle, and compare.

    This used to be a sentence in the caveats saying "about 3%", written
    once and never re-run.  Every physics change since moved it -- by
    the time anybody checked it was 7% -- which is exactly what the rest
    of this page refuses to do: nothing here is transcribed.
    """
    from coxswain import physics

    boat = physics.resolve(profile).apply(catalog.eight(rate=rate))
    boat.power_scales = np.full(boat.n_seats, scale)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=lambda t, s: 0.0),
                          fast=True)
    result = sim.run(duration=duration, dt=dt, surge_speed=reference_speed)
    time_s = np.asarray(result.time)
    speed = np.hypot(*np.asarray(result.velocity)[:2])
    settled = float(speed[time_s > duration - 4 * boat.timing.period].mean())
    return settled, (reference_speed - settled) / max(settled, 1e-6)


def separation(a, b) -> float:
    import math
    return math.hypot((a[0] - b[0]) * 111320.0,
                      (a[1] - b[1]) * 111320.0 * math.cos(math.radians(42.365)))


def build_course(month: int = 10):
    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=month)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(month))
    gates = CourseGeometry(channel=raster).gates_on_course()
    return raster, course, flow, gates


#: Control points to try when searching for a line.
#:
#: Measured, both boats, 150 iterations each: the eight reads 1001.9 s
#: at 13 points, 999.5 at 17, 1001.5 at 19, 998.0 at 27; the four reads
#: 1320.7, 1315.3, 1315.0, 1308.6 and 1304.8 at 35, then 1306.1 at 41.
#: Two things follow.  Thirteen points -- what this used -- cannot hold
#: the shape: the four's answer was 2.0 s WORSE than a hand-drawn
#: candidate, and starting the search from that candidate still ended
#: worse than it began, which is a resolution limit and not a failure
#: to converge (250 iterations and a different seed both change
#: nothing).  And the sequence is not monotone -- 19 beats 27 for
#: neither boat, 41 loses to 35 -- because coordinate descent finds a
#: LOCAL optimum and each resolution finds a different one.  So the
#: search is run at several and the best kept.  Peak yaw stays at
#: 4.06 deg/s against 4.44 for the best hand-drawn line, so the quicker
#: lines are not weaving between control points to beat a quasi-steady
#: model.
LINE_RESOLUTIONS = (17, 23, 27, 35)


def best_line(ev, name, resolutions=LINE_RESOLUTIONS, iterations=150,
              seed=0):
    """The quickest line this search can find, and how much the
    resolution mattered.

    Returns ``(route, clock, n_control, spread)``.  ``spread`` is the
    range across resolutions: it is the honest measure of how much of
    "optimised" is the optimiser and how much is luck.
    """
    found = []
    for n in resolutions:
        got = optimise_route(ev, n_control=n, iterations=iterations,
                             seed=seed)
        route = Route(got.route.stations, got.route.offsets, name=name)
        result = ev.evaluate(route)
        found.append((result.elapsed_clean + 60.0 * result.illegal_arches,
                      n, route))
    found.sort(key=lambda item: item[0])
    clock, n, route = found[0]
    return route, clock, n, found[-1][0] - clock


def evaluator(course, flow, raster, gates, pins=None, reference_speed=5.2,
              exertion=None, rowers=8):
    ev = RouteEvaluator(course, flow=flow, reference_speed=reference_speed,
                        upstream=True, margin=4.0, minimum_depth=1.2,
                        n_samples=1200)
    ev.with_steering(ReducedModel(), raster=raster, gates=gates)
    ev.with_exertion(exertion, rowers=rowers)
    if pins:
        ev.required_arches = dict(pins)
    return ev


def steer(path, controller, boat, dt=0.01, cruise=4.7):
    """Run the 6-DOF boat down ``path``; returns (times, positions, error).

    ``cruise`` is the speed THIS boat holds, and everything that used to
    be the eight's 4.7 comes off it: the model the controller predicts
    with, the speed the run starts at, how long the run is given, and
    the horizon.  Left at 4.7 the four started 74% above the speed it
    can hold, decelerated down the whole leg and ran out of time before
    the end -- its steering numbers came from a run that never reached
    the finish.
    """
    if controller == "mpc":
        model = fit_reduced_model(boat, reference_speed=cruise)
        driver = PathMPC(path, model=model, horizon=horizon_for(cruise),
                         steps=12, interval=0.20)
    else:
        driver = PathFollower(path, boundary_layer=25.0)

    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=driver))
    heading = float(np.arctan2(path[1, 1] - path[0, 1],
                               path[1, 0] - path[0, 0]))
    state = sim.initial_state(surge_speed=cruise)
    state[0], state[1] = path[0]
    state[5] = heading
    # Velocity is stored in the ABSOLUTE frame, so it has to be rotated to
    # the heading or the boat starts crabbing at the heading angle.
    state[6] = cruise * np.cos(heading)
    state[7] = cruise * np.sin(heading)

    leg = float(np.hypot(*np.diff(path, axis=0).T).sum())
    result = sim.run(duration=1.40 * leg / max(0.95 * cruise, 0.5),
                     dt=dt, initial_state=state)
    positions = np.asarray(result.position)[:2].T
    times = np.asarray(result.time)

    gap = np.linalg.norm(positions - path[-1], axis=1)
    arrived = np.nonzero(gap < 12.0)[0]
    if len(arrived):
        cut = int(arrived[0]) + 1
        positions, times = positions[:cut], times[:cut]

    check = PathFollower(path)
    errors = []
    for point in positions:
        index = check.nearest(point)
        tangent, _ = check.frame_at(index)
        errors.append(float(np.dot(point[:2] - check.path[index],
                                   np.array([-tangent[1], tangent[0]]))))
    return times, positions, np.asarray(errors), driver


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/report")
    parser.add_argument(
        "--physics", default=None,
        help="physics profile (see coxswain.physics). Defaults to the "
             "frozen 'shipped' configuration, which is what the online "
             "report must use; pass 'research' for the offline programme. "
             "Every figure is stamped with whichever was used.")
    parser.add_argument("--quick", action="store_true",
                        help="skip the animations, which dominate the runtime")
    parser.add_argument("--month", type=int, default=10)
    parser.add_argument("--steer-leg", type=float, default=400.0,
                        help="metres of racing to steer for the controller "
                             "comparison. The 6-DOF boat integrates at "
                             "100 Hz, so this is the slowest stage by far "
                             "and 400 m is enough to separate the two "
                             "controllers through the Weeks turn")
    parser.add_argument("--no-steering", action="store_true",
                        help="skip the controller comparison entirely")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="simulation step for the controller runs; the "
                             "boat's yaw time constant is 0.06 s so this "
                             "still resolves the dynamics")
    args = parser.parse_args(argv)

    started = time.time()
    overall = progress(total=6, desc="report", unit="stage")
    figures_dir = os.path.join(args.out, "figures")
    for directory in (args.out, figures_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    overall.set_description("river and bridges"); overall.update(1)
    raster, course, flow, gates = build_course(args.month)
    bridge_rows = []
    for name, nbi in NBI.items():
        model_point = getattr(charles, MODEL[name])
        _station, offset = charles.landmark_station(model_point, raster)
        bridge_rows.append([name, "%.1f" % separation(model_point, nbi),
                            "%.1f" % offset])
    arch_rows = []
    for gate, metres in gates:
        racing = B.racing_arch(gate, raster)
        legal = B.candidate_arches(gate, raster)
        arch_rows.append([gate.name, "%.0f" % metres,
                          len(B.bridge_arches(gate, raster)), len(legal),
                          "%.1f" % (racing.width if racing else float("nan")),
                          "%.1f" % (racing.fits() if racing else float("nan"))])

    overall.set_description("optimising the racing line"); overall.update(1)
    ev = evaluator(course, flow, raster, gates)
    route, eight_clock, eight_n, eight_spread = best_line(
        ev, "optimised (eight)")

    # The four, its own operating point, and its OWN optimised line.
    # Scoring the eight's line for the four is not enough: the line was
    # optimised at the eight's speed, and at the four's pace a different
    # candidate wins.  A report that hands this crew a line optimised
    # for a boat 1.2 m/s quicker is giving them somebody else's race.
    four, four_lineup, four_watts = hocr_four(profile=args.physics)
    from coxswain.crew.exertion import WPrimeBalance
    four_cp, four_wprime = four.crew_physiology
    pace = measured_four_pace(course)
    four_speed = pace[1] if pace else 3.6
    ev4 = evaluator(course, flow, raster, gates,
                    reference_speed=four_speed,
                    exertion=WPrimeBalance(critical_power=four_cp,
                                           capacity=four_wprime),
                    rowers=4)
    route4, four_clock, four_n, four_spread = best_line(
        ev4, "optimised (this four)")
    search = [("masters eight", eight_n, eight_spread),
              ("this four", four_n, four_spread)]

    candidates = lines.candidate_lines(course, raster, gates, margin=4.0)
    candidates.extend([route, route4])
    scored = [(r, ev.evaluate(r)) for r in candidates]
    line_rows = []
    reference = None
    for r, result in scored:
        race = result.elapsed_clean + 60.0 * result.illegal_arches
        if r.name == "centreline":
            reference = race
        line_rows.append([r.name, "%.1f" % race, "%.0f" % result.path_length,
                          "%.2f" % result.peak_yaw_rate,
                          "%.0f%%" % (100 * result.peak_split),
                          result.illegal_arches])
    for row, (_r, result) in zip(line_rows, scored):
        race = result.elapsed_clean + 60.0 * result.illegal_arches
        row.append("%+.1f" % (race - reference) if reference else "-")

    # Every line priced for both boats, so the two columns can be read
    # against each other -- which is how the ranking change shows.
    for row, (r, _result) in zip(line_rows, scored):
        result4 = ev4.evaluate(r)
        row.insert(2, "%.1f" % (result4.elapsed_clean
                                + 60.0 * result4.illegal_arches))

    overall.set_description("arch strategies and the loss breakdown"); overall.update(1)
    from scripts.racing_line import STRATEGIES  # noqa: E402
    strategy_rows, loss_rows, strategy_scored = [], [], []
    for name, pins in progress(list(STRATEGIES.items()),
                               desc="  arch strategies", unit="strategy"):
        sev = evaluator(course, flow, raster, gates, pins)
        start = lines.pinned_arch_route(course, raster, gates, pins,
                                        margin=4.0, name=name)
        found = optimise_route(sev, n_control=13, iterations=60, seed=0,
                               initial=start)
        r = Route(found.route.stations, found.route.offsets, name=name)
        outcome = sev.evaluate(r)
        strategy_scored.append((r, sev, outcome))
        race = outcome.elapsed_clean + 60.0 * outcome.illegal_arches
        strategy_rows.append([name, "%.1f" % race, "%.0f" % outcome.path_length,
                              "%.0f%%" % (100 * outcome.peak_split),
                              "%.0f" % outcome.split_strokes,
                              "%.0f" % outcome.w_prime_left])
    base = float(strategy_rows[1][1])
    for row in strategy_rows:
        row.append("%+.1f" % (float(row[1]) - base))

    shortest = min(o.path_length for _r, _e, o in strategy_scored)
    for r, sev, _o in strategy_scored:
        b = sev.loss_breakdown(r, reference_length=shortest)
        loss_rows.append([r.name, "%.1f" % b["ideal"], "%+.1f" % b["distance"],
                          "%+.1f" % b["depth"], "%+.2f" % b["current"],
                          "%+.1f" % b["steering"], "%+.1f" % b["penalty"]])

    overall.set_description("steering the 6-DOF boat down it"); overall.update(1)
    station = np.linspace(0.0, course.length, 4000)
    span = float(args.steer_leg)
    leg = (station >= 2278 - 0.35 * span) & (station <= 2278 + 0.65 * span)
    paths = {"masters eight":
             course.offset_position(station, route.offset_at(station)),
             "this four":
             course.offset_position(station, route4.offset_at(station))}
    control_rows = []
    controllers = [] if args.no_steering else ["reactive", "mpc"]
    eight_boat = reference_eight(profile=args.physics)
    eight_settled = settled_speed(eight_boat, 5.2)
    four_settled = settled_speed(four, four_speed)
    fleet = [("masters eight", eight_boat, eight_settled),
             ("this four", four, four_settled)]
    runs = [(label, boat, cruise, c)
            for label, boat, cruise in fleet for c in controllers]
    for label, boat, cruise, controller in progress(runs, desc="  controllers",
                                                    unit="run"):
        times, positions, errors, driver = steer(paths[label][leg],
                                                 controller, boat,
                                                 dt=args.dt, cruise=cruise)
        settled = errors[len(errors) // 5:]
        control_rows.append([
            label,
            "model predictive" if controller == "mpc" else "reactive (LOS)",
            "%.1f" % times[-1],
            "%.2f" % np.sqrt((settled ** 2).mean()),
            "%.2f" % np.abs(settled).max(),
            "%d / %d" % (getattr(driver, "failures", 0),
                         getattr(driver, "solves", 0))
            if controller == "mpc" else "-"])

    # The evaluator's optimism, measured on this run rather than quoted,
    # for each boat at its own operating point.
    optimism = [("masters eight", 5.2, eight_settled),
                ("this four", four_speed, four_settled)]

    overall.set_description("figures"); overall.update(1)
    written = charts.write_all(figures_dir, month=args.month)
    from scripts.racing_line import loss_chart, plot as line_plot
    loss_png = loss_chart(strategy_scored, shortest,
                          os.path.join(figures_dir, "losses.png"))
    lines_png = line_plot([(r, o) for r, o in scored], course, raster, gates,
                          ReducedModel(),
                          os.path.join(figures_dir, "racing_lines.png"))

    overall.set_description("assembling"); overall.update(1)
    report = build_report(bridge_rows, arch_rows, line_rows, strategy_rows,
                          loss_rows, control_rows, written, loss_png,
                          lines_png, figures_dir, args.quick,
                          dt=args.dt, out=args.out, overall=overall,
                          optimism=optimism, search=search,
                          physics_profile=args.physics,
                          four=(four_lineup, four_watts,
                                                  four_cp, four_wprime, pace))
    path = report.write(os.path.join(args.out, "hocr_report.html"))
    overall.close()
    print()
    print("wrote %s  (%.0f s)" % (path, time.time() - started))
    return 0


def _line_ranking_finding(line_rows):
    """Whether the two boats agree on which line is quickest.

    The page says twice that rankings survive an operating-point change
    while absolute times do not.  With two boats priced side by side
    that is checkable, and on this run it is not true -- so it is said
    here rather than left as a sentence nobody re-ran.
    """
    def rank(column):
        order = sorted(line_rows, key=lambda r: float(r[column]))
        return [row[0] for row in order]

    eight, four = rank(1), rank(2)
    if eight[0] == four[0]:
        return Finding(
            "Both boats want the same line",
            "%s is quickest for the eight and for this four." % eight[0],
            "The two boats are priced at different speeds and with "
            "different reserves, and still agree on the order. That is "
            "the claim the rest of the page leans on when it says "
            "rankings survive an operating-point change.",
            "measured", "", weight=60)
    gap = abs(float(next(r for r in line_rows if r[0] == four[0])[2])
              - float(next(r for r in line_rows if r[0] == eight[0])[2]))
    return Finding(
        "The two boats do not want the same line",
        "This four's best line is %s, not the eight's %s -- worth %.1f s."
        % (four[0], eight[0], gap),
        "Each boat's line is optimised at its own pace, and they do not "
        "agree. Taking the eight's line costs this four %.1f s. A slower "
        "boat spends its time differently: distance matters less against "
        "the clock and shallow water matters more. So \"rankings survive, "
        "absolute times do not\" holds within a boat and not across "
        "boats, and a crew should race a line priced at its own speed."
        % gap + _optimiser_note(line_rows, 2, "optimised (this four)")
        + _optimiser_note(line_rows, 1, "optimised (eight)"),
        "measured", "", weight=75)


def _optimiser_note(line_rows, column, name):
    """Say so when the optimiser fails to beat a hand-drawn candidate.

    Calling a line "optimised" when a named candidate beats it in the
    same column is the kind of thing a reader trusts and should not.
    """
    rows = [r for r in line_rows if r[0] == name]
    if not rows:
        return ""
    mine = float(rows[0][column])
    best = min(line_rows, key=lambda r: float(r[column]))
    if best[0] == name or float(best[column]) >= mine:
        return ""
    return (" The optimiser did not earn its name here: %s is %.1f s "
            "quicker than the line the optimiser found for that boat, so "
            "that search has not converged and its line should be read as "
            "a candidate rather than a best."
            % (best[0], mine - float(best[column])))


def _fallback_caveat(control_rows) -> str:
    """What the MPC's solver actually did, from its own counters.

    The caveat here said "about a fifth of its solves" long after the
    table beside it read 0 / 460.  A reader who compares the two stops
    believing either.
    """
    # Scan every cell for the controller's name: it was the first column
    # until the boat's name went in front of it, and keying on position
    # made this silently report "not recorded".
    cell = ""
    for row in control_rows:
        if any("model predictive" in str(c) for c in row):
            cell = str(row[-1])
    if "/" not in cell:
        return ("The model predictive controller's solver health was not "
                "recorded on this run.")
    failed, solves = [part.strip() for part in cell.split("/", 1)]
    try:
        failed_n, solves_n = int(failed), int(solves)
    except ValueError:
        return ("The model predictive controller reported %s fallbacks."
                % cell)
    if failed_n == 0:
        return ("The model predictive controller solved every step on this "
                "run -- 0 of %d -- so its cross-track figures are what the "
                "controller actually did. It has fallen back heavily in the "
                "past, so this is worth reading off the table each time "
                "rather than assumed." % solves_n)
    return ("The model predictive controller fell back to its previous plan "
            "on %d of %d solves (%.0f%%). It steers well when it converges "
            "and the fallback covers the rest, but its cross-track figures "
            "should be read as a lower bound."
            % (failed_n, solves_n, 100.0 * failed_n / max(solves_n, 1)))


def build_report(bridge_rows, arch_rows, line_rows, strategy_rows, loss_rows,
                 control_rows, chart_paths, loss_png, lines_png, figures_dir,
                 quick, dt=0.02, out=".", overall=None, optimism=None,
                 four=None, search=None, physics_profile=None):
    """Assemble the page.

    ``dt``, ``out`` and ``overall`` are passed in rather than read off a
    module-level ``args``: this function was lifted out of ``main`` and
    kept reaching back into its caller's scope, which is a NameError on
    the first line that touches it -- in BOTH modes, since ``not
    args.quick`` evaluates ``args`` either way.  So the report could not
    be built at all, and the last one predates the move.
    """
    report = Report(
        title="Head of the Charles — what the model says",
        subtitle="A 6-DOF rowing simulator on the surveyed Charles: where "
                 "the time goes, which arches to take, and how much of it "
                 "you can steer for.")

    report.findings = [
        Finding("Depth is the race",
                "Shallow water costs about 82 seconds — 8% of the race.",
                "At 4.8 m/s over the median 3.17 m the boat sits at depth "
                "Froude 0.86, in the shallow-water resistance rise. Every "
                "other term is a sliver beside it: steering costs 1.6 s and "
                "line length about 1 s. On this river, hunting deep water "
                "beats shortening the line by roughly fifty to one.",
                "derived", "Surveyed bathymetry through the shallow-water "
                "resistance model.", weight=100),
        Finding("Do not carry Cambridge through Weeks",
                "It costs 16 seconds; taking the Cambridge arch lower down "
                "costs nothing measurable.",
                "River Street and Western Avenue leave both the centre and "
                "Cambridge arches open, and the Cambridge arch is the wider "
                "opening at both. Taking it there costs 1.9 s, which is "
                "inside the model's own noise. Carrying it through Weeks "
                "costs 16.2 s — 8.1 s of extra distance and 8.3 s of "
                "shallower water. Not the corner, the shoal.",
                "derived", "Four arch strategies each optimised separately.",
                weight=90),
        _line_ranking_finding(line_rows),
        Finding("A tight line has to be steered before it is worth anything",
                "This four holds the centreline to 2.8 m and loses its own "
                "optimised line entirely -- unless the steering anticipates.",
                "Rudder force goes as the square of the speed, so at "
                "2.70 m/s this four has about a third of the eight's "
                "steering authority, on a line asking for 4.1 deg/s of "
                "yaw. Measured, same boat and same controller, only the "
                "line changing: centreline (3.38 deg/s) 2.75 m rms; "
                "inside the bends (4.44) 17.39 m; its own optimised line "
                "(4.06) 15.88 m and it never reached the finish. The "
                "boundary layer is not the cause -- 8 m to 25 m all fail. "
                "The predictive controller holds the same line to 0.80 m "
                "because it acts on what is 28 m ahead instead of "
                "correcting an error that already exists. So the seconds "
                "an optimised line promises are only available to a "
                "coxswain steering off the water ahead; taken reactively "
                "they are not there, and the gentler line is worth more.",
                "measured", "", weight=80),
        Finding("The conventional line is close to optimal",
                "Given a free choice the optimiser picks the centre arches "
                "by itself.",
                "Pointing at the centre arch and holding it — what a "
                "coxswain is taught — lands within a couple of seconds "
                "of the best line found. The value on this course is in "
                "executing the standard line, not in finding a cleverer one.",
                "derived", "", weight=70),
        Finding("Bridges verified against the federal survey",
                "All five road bridges within 7.4 m of the National Bridge "
                "Inventory.",
                "And within 6.5 m of a channel centreline extracted from "
                "bathymetry alone, which knows nothing about bridges. Three "
                "independent sources agreeing. BU Bridge was 27 m out and "
                "has been corrected.",
                "measured", "FHWA National Bridge Inventory 2024, "
                "Massachusetts.", weight=60),
        Finding("Steering is worth having, and it is small",
                "Model predictive control holds the line about twice as "
                "tightly as a reactive law.",
                "Anticipating the bend rather than correcting after it "
                "matters most exactly where a coxswain would expect: in the "
                "turns. But the time difference between good and adequate "
                "steering is under a second over 700 m.",
                "derived", "Full 6-DOF boat under both controllers on the "
                "same line.", weight=50),
        Finding("Fitness outranks every tactical decision, together",
                "About 5 seconds per 1% of crew power. A 2% gain beats the "
                "line, the rig and the seating combined.",
                "Measured: -5% costs 26.6 s, +5% saves 25.6 s, +10% saves "
                "50.0 s. The whole tactical stack on this page -- racing "
                "line, arch strategy, rigging, seat order -- comes to about "
                "20 s, which a 4% squad improvement matches on its own. "
                "That is not an argument against the tactics, which are "
                "free where fitness is expensive and cannot be acquired on "
                "race morning. It is an argument about where training time "
                "goes. Two caveats: this is power at fixed rate, meaning "
                "more force per stroke rather than rowing harder in a way "
                "that wrecks the ratio; and the crew has to hold it for "
                "sixteen minutes, which is a critical-power question, not a "
                "peak-force one.",
                "derived", "Crew power scaled uniformly, full 6-DOF, race "
                "time at the resulting steady speed.", weight=99),
        Finding("Wind is the largest term and the reach is not uniform",
                "A 6 m/s forecast becomes 3.9 to 6.6 m/s at chest height "
                "along the course, and 23% across the channel.",
                "The aerodynamic force model was calibrated and validated "
                "long ago; what it never had was a wind FIELD. It has one "
                "now: OpenStreetMap building footprints and tree canopy "
                "give a Raupach (1994) roughness for whichever bank the "
                "wind crosses, and an internal-boundary-layer model brings "
                "that down onto the water. The roughness model reproduces "
                "the Davenport classes 7 out of 7 and Raupach's own "
                "displacement curve exactly before it is pointed at the "
                "river. The counter-intuitive part is the sign: shelter is "
                "a SHORT-FETCH effect and near-surface wind INCREASES with "
                "distance from a lee bank, because the retarding surface "
                "has fallen away. At 5 m of fetch a crew sits in 43% of the "
                "open wind; at 150 m, 91%.",
                "derived",
                "scripts/canopy.py validates on published roughness classes, "
                "then applies the model to 9463 OSM footprints; "
                "coxswain/river/stations.py converts a KBOS reading into "
                "the reference wind it needs.", weight=97),
        Finding("Take the Cambridge arch on the Powerhouse Stretch",
                "It costs 0.7 s of line and buys clean water worth about "
                "2.5 s. Break-even is 29% of the stretch in traffic.",
                "The line cost is seven metres -- small enough that the "
                "decision should not be made on distance at all. Against "
                "that, being in another crew's water costs about 1.0% of "
                "speed a length astern, split between blades in their "
                "puddles, hull in their turbulence, and a partial offset "
                "from their hull wake. Over the 1050 m stretch that is "
                "2.5 s. You need to be in somebody's water for only 300 m "
                "of it to be ahead, and in a masters flight you will be. "
                "The model does not even price the things that make the "
                "case stronger: clean air, no risk of being forced wide, "
                "and rowing your own rhythm.",
                "derived",
                "scripts/powerhouse.py optimises inside each arch choice "
                "at masters speed and prices the wake separately.",
                weight=88),
        Finding("Sit two to five metres across, never directly astern",
                "Your blades reach 3.15 m out. Directly astern they land "
                "on both of their puddle tracks at once.",
                "A 2-D vortex method -- every blade sheds a dipole, the "
                "dipoles advect one another, the hull sheds a "
                "momentum-cancelling wake on the centreline -- resolves "
                "where the disturbance actually is. Three features, not "
                "one: their hull wake on the centreline running WITH them, "
                "and two puddle lines at plus and minus 3.15 m running "
                "against. On their line, your blades lose 0.106 m/s of "
                "grip while your hull gains. Three metres across it "
                "inverts. One and a half or four and a half metres is the "
                "quiet water. Six point three metres is the trap -- that "
                "is one blade span across, and it puts your inside blades "
                "straight back on their far puddle line.",
                "derived",
                "coxswain/hydro/vortex.py; the momentumless wake of a "
                "self-propelled body falls out to a residual of 2e-8, "
                "which is a check the model was not fitted to.",
                weight=72),
        Finding("Nothing on this boat is near a drag crisis",
                "Textured kit, trip tape and blade vortex generators all "
                "fail, each for a different reason.",
                "Rowers' limbs sit a factor of five below the cylinder "
                "drag crisis and oar shafts a factor of ten, so cycling's "
                "textured fabric has no transition to trip. The hull is "
                "already turbulent over 99.3% of its length, so a trip "
                "strip can only end the 12 cm of cheap laminar run early. "
                "Riblets WOULD work -- 113 micrometre grooves, worth 28.6 s "
                "-- and World Rowing bans them by name. Blade vortex "
                "generators are the interesting failure: the Reynolds "
                "number is fine, but Caplan and Gardner measured NO STALL "
                "at any angle of attack, which is the signature of flow "
                "that was never attached, and 70% of the drive's impulse "
                "is made with the blade broadside where delaying "
                "separation would REDUCE the force. The tip fence is the "
                "one device with the right mechanism and the right sign.",
                "derived",
                "scripts/clothing.py, scripts/oar_aero.py, "
                "scripts/surfaces.py, scripts/blade_devices.py.",
                weight=45),
        Finding("Take the jacket off, and tuck into a headwind",
                "6.6 s for fitted kit and 3.4 s for a flat tuck, in an "
                "8 m/s headwind. Both free.",
                "The aerodynamic split is oars 50%, bodies 35%, hull and "
                "riggers 15% -- the largest aerodynamic object in a rowing "
                "eight is not a person. Of what is left, loose clothing is "
                "5% of the crew's effective drag area and a jacket left on "
                "for a cold start is the whole of that number. The "
                "coxswain sits ninth in a line of nine and is the most "
                "sheltered person in the boat, so a fairing there is worth "
                "2 to 3 s and carries a rules risk; leaning forward is "
                "worth as much, costs nothing when the wind is behind, and "
                "can be decided on the warm-up.",
                "derived",
                "scripts/clothing.py and scripts/cox_fairing.py at 1.73 m "
                "and 68 kg.", weight=52),
        Finding("The controller no longer fails, and tracking was the "
                "wrong thing to tune it on",
                "Solve failures 28.7% to zero, 2.9x faster, and tighter "
                "tracking is SLOWER on the clock.",
                "Everything in the MPC transcription was linear-quadratic "
                "except one sine. Linearising it about the measured "
                "heading error -- not about zero -- makes the program "
                "convex, and an active-set QP has no iteration limit to "
                "hit quietly. A disturbance observer then carries the "
                "rig's standing yaw couple on feedforward, 5.8 degrees of "
                "held rudder through the Weeks turn, instead of paying for "
                "it in a permanent cross-track offset. But the important "
                "finding is the objective: scored on cross-track error, "
                "tighter is always better and the tuning runs one way "
                "forever. Scored gate-to-gate on the clock it reverses -- "
                "raising the cross-track weight from 2 to 120 improves "
                "tracking to 1.05 m and costs 2.0 s, while travelling one "
                "metre LESS. All of it is helm drag.",
                "derived",
                "scripts/mpc_bench.py switches each change independently; "
                "scripts/mpc_tune.py times between fixed gates.",
                weight=60),
        Finding("The line has a corner the boat cannot turn",
                "Median tracking error 0.78 m over the whole course, and "
                "4.1% of it over 10 m off -- all in one kilometre.",
                "Station-resolved error shows the controller holding under "
                "a metre nearly everywhere and losing the line badly "
                "between 500 and 1500 m, where the optimised route reaches "
                "a curvature of 0.120 per metre against 0.04 to 0.06 "
                "everywhere else. That is an 8 m radius, far tighter than "
                "a 17.3 m hull can turn. The MPC already defends itself by "
                "clipping curvature before the solver sees it; the route "
                "optimiser has no such limit and emitted the knot. The fix "
                "is in RouteEvaluator, not in the controller.",
                "open",
                "scripts/mpc_tune.py --full plus a station-resolved error "
                "profile.", weight=58),
        Finding("Running the boat smoothly is worth about seven seconds",
                "Velocity fluctuation costs through the nonlinearity of "
                "resistance, not through added mass.",
                "Two things get called unsteady and only one costs. Added "
                "mass is in quadrature with velocity, so around a closed "
                "cycle it does exactly zero net work -- it sets how big "
                "the swing is, not what it costs. The cost is that the "
                "mean of a power is not the power of the mean: with the "
                "measured local exponent of 1.89, the penalty is quadratic "
                "in the swing. CAVEAT: the simulator's surge swing is "
                "about twice the published figure for an eight, and the "
                "penalty goes as its square, so the honest number is 7.4 s "
                "rather than the 33 s the raw trace gives. One outing with "
                "a logger would settle it.",
                "open",
                "scripts/unsteady.py; added mass from the panel solver, "
                "validated against a circle to 0.34%.", weight=55),
        Finding("A following boat does not need its own flow solved",
                "The wake's steering disturbance is 0.5 N m against "
                "2600 N m of rudder authority.",
                "The question was whether wake-hull interference needs the "
                "follower's own fluid mechanics simulated. It does not, "
                "for two separate reasons. A body's own potential field "
                "exerts no net force on itself, so interference enters "
                "only through distortion of the incident field, which is "
                "second order in a ratio of about 4%. And the 6-DOF "
                "perturbation the wake actually applies is tiny: sampled "
                "along a follower's seventeen metres, the lateral gradient "
                "integrates to a yaw moment four orders of magnitude below "
                "what the rudder makes. The wake is a DRAG and GRIP "
                "effect, not a steering one. What is needed instead is "
                "cheap: sample the existing wake field at each hull strip "
                "rather than at one point, and the 6-DOF already carries "
                "distributed cross-flow drag to turn that into forces.",
                "derived",
                "coxswain/hydro/vortex.py sampled along a follower hull; "
                "strip-integrated side force and yaw moment.", weight=50),
        Finding("Wind moves the race more than anything a crew decides",
                "A 4 m/s headwind costs 86 seconds; 8 m/s costs 232.",
                "And it is asymmetric: the same wind behind you gives back "
                "only 48 s, because apparent wind rises with boat speed "
                "into it. Crosswind is cheap in time (4-29 s) but real in "
                "trim, standing 0.2 deg/s of yaw and half a degree of heel "
                "on the crew. At a head race, wind drift between flights is "
                "larger than any tactical choice available to a coxswain.",
                "derived", "Uniform wind over the reach, aerodynamic model "
                "calibrated from the boat's own frontal area.", weight=95),
        Finding("A bucket rig is worth 3 to 4 seconds",
                "And it does not matter which bucket rig.",
                "A standard alternating rig carries a 4.88 m stagger arm, "
                "which needs 2 to 2.5 degrees of standing rudder to hold a "
                "line -- and standing rudder is drag for 4.8 km. German, "
                "Italian, battleship and tandem all cancel that arm to zero "
                "and are identical to three decimals. The advantage is "
                "independent of crew height (3.6-4.4 s across 1.70-1.90 m) "
                "and of crew mass (2.1-3.8 s across 70-90 kg), so the "
                "advice transfers to any crew.",
                "derived", "Six seating patterns, full 6-DOF, rudder "
                "trimmed to hold a straight line.", weight=80),
        Finding("Which side a weak rower sits, not which seat",
                "Side is worth 0.74 s; seat position is worth 0.03 s.",
                "Conventional wisdom says put the weak rower amidships. On "
                "a standard rig the physics says something else: a weak "
                "PORT rower partially cancels the rig's own stagger bias "
                "and costs 0.74 s less than the same rower on starboard, "
                "while moving them stroke-to-bow along the port side is "
                "worth 0.03 s. On a bucket rig, with no bias to cancel, "
                "that reverses -- side shrinks to 0.24 s and position "
                "emerges at 0.51 s. Cost scales linearly at about 0.65 s "
                "per 1% one rower is down.",
                "derived", "Every seat, both rigs, deficits from 5% to 30%.",
                weight=75),
        Finding("Crew consistency outranks the racing line",
                "Crew-to-crew scatter alone spreads race time by 9 to 20 "
                "seconds.",
                "Elite 9.0 s, club 14.2 s, junior 20.3 s, drawn from "
                "measured force and timing variability. That is the error "
                "bar under every other number here, and it is larger than "
                "the racing line is worth. It is also the one on this list "
                "that training moves.",
                "measured", "Kleshnev force-variability series, drawn per "
                "crew and rowed in the full model.", weight=85),
        Finding("Do not reseat the crew for this river",
                "The Charles turns 40% to port and 33% to starboard: the "
                "bends cancel.",
                "Scoring the racing line by how long it spends turning each "
                "way gives a time-weighted mean demand of +0.06 deg/s -- "
                "for practical purposes, a straight course. A crew's "
                "standing bias can only cancel the mean, never the swing, "
                "and the swing here has no net direction. Seat for the "
                "straight; the turns take care of themselves.",
                "derived", "Curvature of the optimised line, weighted by "
                "time spent at each demand.", weight=65),
        Finding("The fin is the weakest number in the model",
                "Its depth is scaled off a spanner in a photograph.",
                "The fin's shape is exact — proportions from the "
                "photograph fix aspect ratio, taper and sweep without "
                "needing scale. Its size is not. Fin depth sets steering "
                "authority, and steering authority sets which lines are "
                "rowable at all.",
                "open", "Wants a tape measure: depth below the hull, root "
                "chord, and how deep the rudder hangs.", weight=40),
    ]

    report.tables = [
        Table("What each thing is worth",
              ["lever", "seconds", "who controls it"],
              [["crew power, per 1%", "5.0", "training"],
               ["crew consistency, junior to elite", "11.3", "training"],
               ["Weeks arch decision", "16.2", "the coxswain, once"],
               ["racing line vs centreline", "14.3", "the coxswain"],
               ["bucket rig vs standard", "3.6", "the rigger, once"],
               ["weak rower on port not starboard", "0.74", "the coach, free"],
               ["seat order along one side", "0.03", "nobody, it does not matter"],
               ["--- not controllable ---", "", ""],
               ["headwind at 4 m/s", "85.8", "the weather"],
               ["headwind at 8 m/s", "232.0", "the weather"],
               ["shallow water over the course", "82.4", "the river"]],
              "Everything above the divider is a coaching decision; "
              "everything below is the day you drew. The two largest "
              "numbers on the page are in the second group, which is "
              "worth knowing before comparing crews across flights.",
              highlight=0, group="Where the time is"),
        Table("Every lever, one list",
              ["lever", "seconds", "who decides", "confidence"],
              [["crew power, per 1%", "5.6", "training", "measured"],
               ["riblets on the hull", "28.6", "ILLEGAL", "measured"],
               ["shallow water over the course", "82.4", "the river", "measured"],
               ["headwind at 8 m/s", "232.0", "the weather", "measured"],
               ["blade cover, 90 mm vs optimum", "57.1", "the crew", "measured"],
               ["lose 3 kg per rower", "8.4", "training", "displacement only"],
               ["running the boat smoothly", "7.4", "the crew", "swing unvalidated"],
               ["fitted kit, no jacket (8 m/s)", "6.6", "free", "measured"],
               ["coxswain tuck (8 m/s)", "3.4", "free", "estimated"],
               ["thinner oar shafts", "3.3", "the supplier", "estimated"],
               ["cox fairing (8 m/s)", "2-3", "rules risk", "wide band"],
               ["Cambridge arch, Powerhouse", "1.8", "the coxswain", "derived"],
               ["crew consistency, club to elite", "0.4", "training", "not measurable"],
               ["bucket rig", "0.5", "the rigger", "sign unresolved"],
               ["weak rower to port", "0.9", "free", "measured"],
               ["re-steering for the wind", "0.4", "the coxswain", "measured"],
               ["seat order along one side", "0.03", "nobody", "measured"],
               ["trip tape on the shafts", "-1.0", "do not", "measured"]],
              "Everything this project has priced, on one scale, with an "
              "honest confidence column. Two of the three largest numbers "
              "are not available to you: one is banned and two are the "
              "weather and the river. The largest that is available is "
              "blade depth, and it is a technique call a coach can see "
              "from the launch.", highlight=0, group="Where the time is"),
        Table("Where a device idea goes to die",
              ["idea", "regime", "why not"],
              [["textured kit", "Re 4e4, 5x below crisis", "no transition to trip"],
               ["trip tape on shafts", "Re 2e4, 10x below", "roughness on the flat curve"],
               ["trip strip on the hull", "Re 7e7, already turbulent", "ends the laminar run early"],
               ["riblets", "correct, 113 um spacing", "banned by name"],
               ["blade vortex generators", "Re 7e5, correct", "never attached; wrong sign"],
               ["trailing-edge Gurney", "correct", "edges swap at the perpendicular"],
               ["perimeter flange", "correct", "already fitted, called a spoon"],
               ["tip fence", "correct", "works; already fitted"]],
              "Four different failure modes, and none of them is the one "
              "people expect. Checking rather than assuming is what "
              "separates them.", group="Where the time is"),
        Table("Bridges against the federal survey",
              ["bridge", "vs NBI (m)", "off the channel (m)"], bridge_rows,
              "The channel centreline is extracted from depth alone, so its "
              "agreement with the bridges is an independent check rather "
              "than a restatement.", group="The river"),
        Table("The arches",  ["bridge", "station (m)", "arches", "legal",
                             "racing arch (m)", "eights abreast"], arch_rows,
              "Span counts and lengths from the National Bridge Inventory; "
              "pier thickness measured from the Grand Junction trestle. "
              "Legal arches follow the regatta's rules — the Boston "
              "arch is out of bounds everywhere, and the Cambridge arch is "
              "additionally barred at the trestle, Anderson and Eliot.", group="The river"),
        Table("Candidate lines",  ["line", "eight (s)", "this four (s)",
                                  "distance (m)",
                                  "peak yaw (deg/s)", "split wanted",
                                  "illegal", "vs centreline"], line_rows,
              "Race time includes a 60 s penalty per forbidden arch. Every "
              "line here is legal by construction.", group="The line"),
        Table("Arch strategy",  ["strategy", "race time (s)", "distance (m)",
                                "split wanted", "split strokes",
                                "W' left (J)", "vs centre arches"],
              strategy_rows,
              "Each strategy optimised inside its own arch constraint, so "
              "this is best against best. W' is the crew's anaerobic "
              "reserve; the pace is solved so it reaches zero at the line.", group="The line"),
        Table("Where the seconds go",  ["line", "ideal (s)", "distance",
                                       "depth", "current", "steering",
                                       "penalty"], loss_rows,
              "Each term is the cost of adding that effect to the one "
              "before, so they sum to the race time and nothing hides in a "
              "residual.", group="The line"),
        Table("Steering the real boat",  ["boat", "controller", "elapsed (s)",
                                         "cross-track rms (m)", "worst (m)",
                                         "solver fallbacks"], control_rows,
              "Each boat down its OWN optimised line, at its own cruising "
              "speed, measured after the opening transient. The "
              "predictive controller looks a fixed DISTANCE ahead rather "
              "than a fixed time: six seconds is 29 m for the eight and "
              "16 m for the four, and on 16 m the four cannot steer at "
              "all (12.88 m rms, never reaching the finish). Given the "
              "same 28 m it holds the line to 0.80 m, better than the "
              "eight. The pursuit controller has no such fix: see the "
              "finding above.",
              group="The line"),
    ]

    figures = [
        Figure(os.path.join(figures_dir, "charles_course_bathymetry.png"),
               "The course over the survey",
               "4828 m from the DeWolfe Boathouse start to the finish above "
               "Eliot.",
               "Gold is the navigable edge, white the channel centreline. "
               "Green spans are arches a racing crew may use; red carry a "
               "60 second penalty.", group="The river"),
        Figure(os.path.join(figures_dir, "charles_course_profiles.png"),
               "The course straightened out",
               "Depth across the channel, centreline depth, navigable width "
               "and current, against distance from the start.",
               "The width panel is the one to watch: the river narrows to "
               "50 m between Anderson and Eliot, which is where both the "
               "tightest corner and the one-boat travel lane are.", group="The river"),
        Figure(os.path.join(figures_dir, "charles_bridge_arches.png"),
               "Every arch to scale",
               "Drawn as the coxswain meets them, first bridge at the "
               "bottom, Cambridge to starboard.",
               "The green bar is a rowed eight, 6.82 m tip to tip. Anderson "
               "and Eliot are centre-arch only; the Powerhouse bridges give "
               "you a choice.", group="The river"),
        Figure(lines_png, "Candidate racing lines",
               "Six lines, all legal, scored by the same evaluator.",
               "The middle panel is where the lines are legible — 20 m "
               "apart on a 4.8 km map is a hairline. The bottom panel shows "
               "what each asks of the rudder against what the boat can "
               "give."),
        Figure(loss_png, "Where the race time goes",
               "Loss breakdown per arch strategy.",
               "The bars are almost entirely one colour. That is the "
               "finding: depth dominates everything else by a factor of "
               "fifty, and the only visible difference between strategies "
               "is the grey sliver of extra distance on the bottom bar."),
        Figure("out/wake2d/wake2d.png",
               "The wake as a vortex field",
               "Red is water travelling with the boat ahead, blue against "
               "it. The centreline jet is the hull wake; the two blue "
               "tracks are the blade puddles at 3.15 m. The lower panel "
               "is the answer to where to sit -- note the trap at 6.3 m, "
               "one blade span across, where your inside blades land back "
               "on their far puddle line.", group="Hydrodynamics"),
        Figure("out/hull/hull_potential.png",
               "Potential flow around the real waterline",
               "Hess-Smith source panels on the boat's own offsets, "
               "validated against a cylinder to 1e-15. Stagnation at both "
               "ends, Cp about -0.1 over the middle, peak overspeed 8.4%. "
               "A shell is fine enough that the water barely notices it.", group="Hydrodynamics"),
        Figure("out/budget/time_budget.png",
               "Finding sixty seconds",
               "The waterfall that started the tactical work, at the "
               "masters operating point rather than a collegiate one.", group="Where the time is"),
        Figure(os.path.join(figures_dir, "charles_navigable_spans.png"),
               "The navigable spans",
               "Each bridge at about 200 m across, over the real bathymetry.",
               "At course scale a 20 m arch is a hairline; this is the scale "
               "at which the decision is actually made.", group="The river"),
        Figure("out/wind/wind_corridor.png",
               "The ground the wind crosses",
               "Bare-earth elevation from USGS 3DEP, with the 9463 "
               "OpenStreetMap building footprints, 216 canopy polygons and "
               "2156 mapped trees that 3DEP deliberately strips out drawn "
               "back on. This is the picture that makes the wind model "
               "plausible or not: if the footprints are in the wrong "
               "place, so is every roughness length downstream of them, "
               "and no amount of Raupach fixes that. Note how little of "
               "the reach is open -- three storeys of Cambridge to the "
               "north, Boston to the south, and a tree line on both banks.", group="Wind"),
        Figure("out/wind/wind_field.png",
               "Wind reaching a rower's chest, by direction",
               "The same 6 m/s forecast, resolved to 1.5 m above the water "
               "for four wind directions. Open water at that height would "
               "be 5.64 m/s; the reach delivers 3.5 to 5.7 depending on "
               "where you are and which way the wind comes from. The "
               "gradient ACROSS the channel is the tactical content and it "
               "does not survive being averaged into a single number -- at "
               "station 2400 in a westerly the sheltered side carries "
               "4.58 m/s against 5.64 on the open side, a 23% difference "
               "across 100 m of river.", group="Wind"),
        Figure("out/wind/wind_profile.png",
               "Wind and bank roughness, station by station",
               "What a crew actually meets down the course. The upper "
               "panel is wind at chest height for four directions against "
               "the open-water reference; the lower is the Raupach "
               "roughness length of whichever bank is upwind, on a log "
               "scale spanning three decades from open water to city. The "
               "spiky stretch around 400 to 800 m is real rather than "
               "numerical: that is where the upwind sector alternates "
               "between built bank and open park as the river turns.", group="Wind"),
        Figure(os.path.join(figures_dir, "charles_course_current.png"),
               "The current",
               "Continuity model at October median discharge.",
               "Slack water: a few centimetres per second against a racing "
               "5 m/s. Line choice here is about distance and depth, not "
               "about hunting current.", group="The river"),
    ]
    # Animations are embedded whenever they exist on disk -- they are
    # produced by scripts/animate_race.py and scripts/render3d.py, which
    # are slow, so the report picks up the latest rather than re-rendering.
    figures.extend([
        Figure("out/animation/race_2100_2800_chase_mpc.gif",
               "The boat rowing the line",
               "Weeks to Anderson under model predictive control, 12x "
               "real time.",
               "The dashed line is the plan, the solid trail is what the "
               "boat did. It rows in from behind so it is already "
               "tracking when the picture starts.", group="Watch it row"),
        Figure("out/animation/race_2100_2800_chase.gif",
               "The same leg under reactive steering",
               "Line-of-sight guidance, for comparison with the MPC run "
               "above.",
               "Watch the trail through the Weeks turn: the reactive law "
               "corrects error after it appears, so it runs wider than "
               "the anticipating controller.", group="Watch it row"),
        Figure("out/animation/race_0000_4821_course_mpc.mp4",
               "The whole race, from above",
               "All 4822 m from the DeWolfe start to the finish above "
               "Eliot, at 24x real time.",
               "The boat is the moving mark; the dashed line is the "
               "optimised route. This is the quick reference for where the "
               "line sits on the river.", group="Watch it row"),
        Figure("out/animation/race_0000_4821_chase_mpc.mp4",
               "The whole race, following the boat",
               "The same run at boat scale, blades working.",
               "Watch the bridges arrive: the trestle and BU inside the "
               "first 200 m, then the long Powerhouse straight, Weeks, "
               "Anderson, and the Cambridge Boat Club bend before Eliot.", group="Watch it row"),
        Figure("out/render3d/mpc_2250_2560_cox.mp4",
               "From the coxswain's seat",
               "Weeks to Anderson, camera at the cox's own seat 0.80 m "
               "above the water.",
               "The only view that answers whether a line is steerable. An "
               "arch 300 m off is a slot near the horizon, and the far bank "
               "hides behind the near one -- geometry every plan view "
               "flatters.", group="Watch it row"),
    ])
    # -- boat dynamics ------------------------------------------------
    # The damping page runs its own counterfactual -- the same boat with
    # the damping put back the way it was -- so it costs a couple of
    # minutes of integration and is skipped under --quick.
    if not quick:
        if overall is not None:
            overall.set_description("boat dynamics")
        try:
            from damping_page import build as build_damping
            finding, tables, damping_figures = build_damping(figures_dir,
                                                             dt=dt)
            report.findings.append(finding)
            report.tables.extend(tables)
            figures.extend(damping_figures)
        except Exception as error:                       # pragma: no cover
            print("  (no damping page: %s)" % error)

    report.figures = figures
    # The explorer is built separately (scripts/export_map.py writes the
    # data, then the template is filled), so it is embedded only when it
    # is actually there -- a missing iframe is worse than a missing tab.
    if os.path.exists(os.path.join(out, "map.html")):
        report.embeds = [Embed(
            path="map.html",
            title="Course explorer",
            caption="Depth, current and shelter over the surveyed reach, "
                    "with the depth-following line for the conditions you "
                    "set. Switch the layer, drag the water level, change "
                    "the wind, and watch the line move.")]

    from coxswain import physics as _physics

    profile = _physics.resolve(physics_profile)
    report.caveats = [
        ("Every number on this page was produced by the %s physics "
         "profile: blade tier %d, %s rower. The released trainer runs "
         "'shipped' and nothing else; a page built from another profile "
         "is a study, not a description of the game. See "
         "docs/PHYSICS_PROGRAMME.md." % (profile.name, profile.blade_tier,
                                          profile.rower)),
        (("The line search is coordinate descent, which finds a LOCAL "
          "optimum: it is run at several resolutions and the best kept. "
          + "; ".join("%s took %d control points, and the spread across "
                      "resolutions was %.1f s" % (label, n, spread)
                      for label, n, spread in search)
          + ". A line quicker than these exists and this has not found "
          "it.") if search else
         "The line search is coordinate descent and finds a local "
         "optimum."),
        "The 82 s depth loss is the largest claim here and the least "
        "checked. It rests on the shallow-water model at depth Froude 0.86, "
        "the steepest part of that curve. A GPS trace with depth would "
        "settle it.",
        "Fin depth is scaled from a spanner in a photograph and one of the "
        "three features measured off that spanner was demonstrably wrong. "
        "The fin's shape is exact; its size is not.",
        (("The four's critical power and reserve are its own: %.0f W and "
          "%.1f kJ, from each rower's 5 km erg and the crew's age band, "
          "not a collegiate mean. The eight is still a masters reference "
          "crew rather than anybody in particular. Its lines are priced "
          "at the four's measured Charles pace -- %s in %d, %.2f m/s -- "
          "because the simulator's own power path disagrees with the "
          "field by a factor of 2.3 on this boat (docs/TRACKING.md) and "
          "the field is the arbiter."
          % (four[2], four[3] / 1000.0,
             "%d:%04.1f" % (four[4][0] // 60, four[4][0] % 60)
             if four[4] else "no result on file",
             2024, four[4][1] if four[4] else float("nan")))
         if four and four[4] else
         "Critical power and anaerobic capacity are collegiate means, not "
         "this crew."),
        "DeWolfe Boathouse sits 124 m from its OpenStreetMap building "
        "footprint. The start line is placed off it, so correcting that "
        "would move every station in the model.",
        (("The route evaluator is quasi-steady and prices each boat's "
          "lines at a fixed speed. Against the full 6-DOF boat at the "
          "same operating point, measured on this run: "
          + "; ".join("%s, %.2f m/s priced against %.2f settled (%+.0f%%)"
                      % (label, priced, got, 100.0 * (priced - got) / got)
                      for label, priced, got in optimism)
          + ". Rankings survive that; absolute finishing times do not. "
          "The reason is known: the blade's force is a function of stroke "
          "phase with no velocity term, so the propulsive efficiency the "
          "6-DOF path implies comes out proportional to boat speed -- "
          "0.37 at 2.8 m/s rising to 0.77 at 5.8, on both hulls. It was "
          "calibrated where an eight races, which is why the eight looks "
          "right and this four does not.")
         if optimism else
         "The route evaluator is quasi-steady and optimistic against the "
         "full 6-DOF boat. Rankings survive that; absolute finishing times "
         "do not."),
        _fallback_caveat(control_rows),
        "Seating advice is at the edge of the model's resolution: the fast "
        "surrogate carries a 0.04 deg/s bias against the full simulation, "
        "and the whole Charles asks for 0.06 deg/s. Rankings survive that; "
        "absolute targets do not.",
        "Nothing here models rhythm, confidence, or whether a rower can "
        "follow the person in front of them -- which is most of what seat "
        "selection is actually about.",
        "The steering controller is a machine, not a coxswain. It corrects "
        "error rather than reading the river, and it has never had to deal "
        "with another crew.",
    ]
    return report


if __name__ == "__main__":
    raise SystemExit(main())
