"""The fastest racing line as an optimal-control problem, solved by Radau IIA collocation.

Why
---
:func:`coxswain.raceday.plan.optimise_line` searches over offsets at control points every 150 m (and at
the bridges) joined by a shape-preserving cubic. Where neighbouring points jump - two bridges 25 m apart
by Magazine Beach, a point held at the river's edge - the line kinks, and no boat steers a kink. Here
the line is the solution of a continuous problem: the boat's curvature is the control, the heading and
the sideways position follow from it, and the race time is minimised. The line is as smooth as the
physics allows, at the mesh's resolution (15 m elements, three Radau stages each: a point every 5 m).

The frame
---------
The course centreline is extracted from the banks and carries S-wiggles of 150-200 m wavelength that no
coxswain steers (``card.TURN_SMOOTH_M``). :class:`SmoothFrame` is a reference curve without them: the
centreline's heading smoothed with a Gaussian (``sigma`` m) and integrated again, the end point pinned.
Smoothing the heading, not the coordinates, keeps every bend's total turn, so the curve does not shrink
toward the inside of a bend (smoothing x and y moves a bend of radius R inward by about sigma^2 / 2R).
Offsets in this frame have no artificial kinks, so it is also the display frame for lines.

The problem (station s along the frame, Frenet coordinates)
-----------------------------------------------------------
states n (sideways position, + port as the course), chi (heading against the frame), t (time);
control kappa (the boat's path curvature), piecewise constant per element:

    h = 1 - n kappa_c(s)
    dn/ds   = h tan(chi)
    dchi/ds = kappa h / cos(chi) - kappa_c(s)
    dt/ds   = h / (cos(chi) V)
    V = V_field(s, n) * (1 - k omega^2),   omega = (180/pi) V_field kappa   [deg/s]

``V_field`` is the full course model's ground speed at the crew's pace (depth, current, sheltered wind,
the research boat) on lines parallel to the frame, interpolated (B-spline) in (s, n); ``k`` is the
measured turning loss for the boat (``card.RUDDER_K``: the square rule, dv/v = k omega^2). Minimise
t(end) + W * sum((kappa_{i+1} - kappa_i) / ds)^2 * ds, a light penalty on how fast the curvature (the
rudder) changes. Bounds: |kappa| below the turn-rate limit (``plan.MAX_TURN`` at 4 m/s); n inside the
water less ``edge`` m (a blade's reach plus clearance); at every bridge n inside the legal arch's window
(``plan.arch_window``), with mesh boundaries placed exactly at the bridges. IPOPT, warm-started from the
crew's own line. The answer is then timed again with the full model (``plan._time``), so its time is on
the same footing as every other line's.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from coxswain.river.collocation import RadauIIA

DEG = 180.0 / np.pi


class SmoothFrame:
    """A smooth reference curve for the course: heading-smoothed centreline, re-integrated."""

    def __init__(self, course, sigma: float = 60.0, ds: float = 2.0, drift_sigma: float = 150.0):
        from scipy.ndimage import gaussian_filter1d
        self.course = course
        L = float(course.length)
        s = np.arange(0.0, L + ds, ds)
        p = course.offset_position(s, np.zeros_like(s))
        seg = np.hypot(*np.diff(p, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        g = np.arange(0.0, arc[-1], ds)
        x, y = np.interp(g, arc, p[:, 0]), np.interp(g, arc, p[:, 1])
        th = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        ths = gaussian_filter1d(th, sigma / ds, mode="nearest")
        xs = x[0] + np.concatenate([[0.0], np.cumsum(np.cos(ths[:-1]) * ds)])
        ys = y[0] + np.concatenate([[0.0], np.cumsum(np.sin(ths[:-1]) * ds)])
        # integrating a smoothed heading drifts (tens of metres over 4.8 km): pull the curve back onto
        # the centreline at long scales only (a wide low-pass of the difference), twice
        for _ in range(2):
            xs = xs + gaussian_filter1d(x - xs, drift_sigma / ds, mode="nearest")
            ys = ys + gaussian_filter1d(y - ys, drift_sigma / ds, mode="nearest")
        th2 = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
        self.s, self.x, self.y, self.heading = g, xs, ys, th2
        self.curvature = np.gradient(th2, g)
        self.length = float(g[-1])
        from scipy.spatial import cKDTree
        self._tree = cKDTree(np.column_stack([xs, ys]))
        # sign: + is the course's + side (port)
        q = course.offset_position(np.array([L / 2]), np.array([10.0]))[0]
        self._sign = 1.0
        self._sign = 1.0 if self.from_plane(q[0], q[1])[1] > 0 else -1.0

    def to_plane(self, s, n):
        s = np.asarray(s, float); n = np.asarray(n, float)
        x = np.interp(s, self.s, self.x); y = np.interp(s, self.s, self.y); th = np.interp(s, self.s, self.heading)
        return np.column_stack([x - self._sign * n * np.sin(th), y + self._sign * n * np.cos(th)])

    def from_plane(self, x, y):
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        _, i = self._tree.query(np.column_stack([x, y]))
        th = self.heading[i]
        dx, dy = x - self.x[i], y - self.y[i]
        along = dx * np.cos(th) + dy * np.sin(th)
        across = -dx * np.sin(th) + dy * np.cos(th)
        return self.s[i] + along, self._sign * across

    def from_course(self, stations, offsets):
        p = self.course.offset_position(np.asarray(stations, float), np.asarray(offsets, float))
        return self.from_plane(p[:, 0], p[:, 1])

    def to_course_route(self, s, n, name="line"):
        """A course-frame Route through the frame points (s, n)."""
        from coxswain.river.route import Route
        p = self.to_plane(s, n)
        st = np.array([self.course.nearest_station(a, b) for a, b in p])
        of = np.array([self.course.offset_of(a, b, q) for (a, b), q in zip(p, st)])
        st = np.maximum.accumulate(st)
        keep = np.concatenate([[True], np.diff(st) > 0.5])
        return Route(st[keep], of[keep], name=name)


def speed_field(model, boat, reference_time, wind, frame: SmoothFrame, n_grid=None, s_step: float = 10.0):
    """Ground speed V(s, n) from the full model on lines parallel to the frame (at the crew's pace)."""
    from coxswain.raceday.card import _evaluate
    n_grid = np.arange(-84.0, 84.1, 6.0) if n_grid is None else np.asarray(n_grid, float)
    s_grid = np.arange(0.0, frame.length + 2 * s_step, s_step)          # past both ends of the race
    V = np.zeros((len(s_grid), len(n_grid)))
    keep = model.route
    try:
        for j, n in enumerate(n_grid):
            r = frame.to_course_route(s_grid, np.full(len(s_grid), n), name="parallel")
            model.route = r
            res = _evaluate(model, boat, reference_time, wind)
            # back onto the frame stations: the evaluator's stations are course stations along r
            fs, _ = frame.from_course(res.station, r.offset_at(res.station))
            order = np.argsort(fs)
            V[:, j] = np.interp(s_grid, fs[order], np.asarray(res.speed_ground)[order])
    finally:
        model.route = keep
    return s_grid, n_grid, V


def water_bounds(model, frame: SmoothFrame, s_nodes, edge: float = 6.0, reach: float = 95.0):
    """At each frame station, the sideways interval inside the water less ``edge`` metres."""
    course = model.course
    lo, hi = np.zeros(len(s_nodes)), np.zeros(len(s_nodes))
    probe = np.arange(-reach, reach + 0.1, 1.0)
    for i, s in enumerate(s_nodes):
        p = frame.to_plane(np.full(len(probe), s), probe)
        st = np.array([course.nearest_station(a, b) for a, b in p])
        of = np.array([course.offset_of(a, b, q) for (a, b), q in zip(p, st)])
        hw = course.water_half_width_at(st)
        ok = np.abs(of) <= hw - edge
        if not ok.any():
            lo[i] = hi[i] = 0.0
            continue
        c = int(np.argmin(np.abs(probe)))
        # the contiguous admissible run nearest the frame line
        idx = np.flatnonzero(ok)
        j = idx[np.argmin(np.abs(idx - c))]
        a = j
        while a - 1 >= 0 and ok[a - 1]:
            a -= 1
        b = j
        while b + 1 < len(ok) and ok[b + 1]:
            b += 1
        lo[i], hi[i] = probe[a], probe[b]
    # the water's edge in the data is jagged at the metre scale; a line riding it would copy the jags. A
    # smooth, conservative envelope: the tightest bound within +-30 m, then a light Gaussian
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
    ds_ = max(float(np.median(np.diff(s_nodes))), 1.0)
    w = max(int(round(60.0 / ds_)), 1)
    lo = gaussian_filter1d(maximum_filter1d(lo, w, mode="nearest"), 20.0 / ds_, mode="nearest")
    hi = gaussian_filter1d(minimum_filter1d(hi, w, mode="nearest"), 20.0 / ds_, mode="nearest")
    return lo, np.maximum(hi, lo)


def gate_windows(model, frame: SmoothFrame, pins: Dict[str, str], lead=None, margin: float = 4.0):
    """Each bridge's legal arch window as (frame station, n_low, n_high)."""
    from coxswain.raceday.plan import arch_window
    from coxswain.river import bridges
    course = model.course
    out = []
    for gate, metres in model.gates:
        legal = bridges.candidate_arches(gate, model.raster)
        if not legal:
            continue
        match = [a for a in legal if a.label == pins.get(gate.name)]
        if match:
            arch = match[0]
        elif lead is not None:
            here = float(lead.offset_at(np.array([metres]))[0])
            along = gate.station_of(course.offset_position(np.array([metres]), np.array([here]))[0])
            arch = min(legal, key=lambda a: abs(a.centre - along))
        else:
            arch = bridges.racing_arch(gate, model.raster) or legal[0]
        st, lo, hi = arch_window(model, gate, arch, margin)
        s_lo, n_lo = frame.from_course([st], [lo])
        s_hi, n_hi = frame.from_course([st], [hi])
        out.append(dict(name=gate.name, arch=arch.label, s=float(0.5 * (s_lo[0] + s_hi[0])),
                        lo=float(min(n_lo[0], n_hi[0])), hi=float(max(n_lo[0], n_hi[0]))))
    return sorted(out, key=lambda g: g["s"])


def solve(model, boat, reference_time: float, wind, pins: Dict[str, str], rudder_k: float,
          start_route, frame: Optional[SmoothFrame] = None, element: float = 15.0, stages: int = 3,
          edge: float = 6.0, smooth_weight: float = 2.0e5, max_turn: Optional[float] = None,
          field=None, verbose: bool = False) -> dict:
    """Radau-collocation optimum through the pinned arches; see the module docstring."""
    import casadi as ca
    from coxswain.raceday import plan as P
    frame = frame or SmoothFrame(model.course)
    course = model.course
    # the race runs from the start line (course station 0) to the finish (course length)
    s_a = float(frame.from_course([0.0], [0.0])[0][0]); s_b = float(frame.from_course([course.length], [0.0])[0][0])
    gates = [g for g in gate_windows(model, frame, pins, lead=start_route) if s_a + 5 < g["s"] < s_b - 5]
    # mesh: boundaries at every bridge, elements of about `element` metres between
    marks = [s_a] + [g["s"] for g in gates] + [s_b]
    bounds_s = [s_a]
    for a, b in zip(marks[:-1], marks[1:]):
        m = max(int(np.ceil((b - a) / element)), 1)
        bounds_s += list(np.linspace(a, b, m + 1)[1:])
    bounds_s = np.array(bounds_s)
    N = len(bounds_s) - 1
    ds = np.diff(bounds_s)
    scheme = RadauIIA(stages)
    # the speed field and the frame curvature as CasADi functions
    if field is None:
        field = speed_field(model, boat, reference_time, wind, frame)
    sg, ng, V = field
    Vf = ca.interpolant("V", "bspline", [sg, ng], V.ravel(order="F"))
    kc_f = ca.interpolant("kc", "linear", [frame.s], frame.curvature)
    k = float(rudder_k)

    def dyn(x, u, s):
        n, chi = x[0], x[1]
        h = 1 - n * kc_f(s)
        vf = Vf(ca.vertcat(ca.fmin(ca.fmax(s, sg[0]), sg[-1]), ca.fmin(ca.fmax(n, ng[0]), ng[-1])))
        om = DEG * vf * u[0]
        v = ca.fmax(vf * (1 - k * om ** 2), 0.3)
        return ca.vertcat(h * ca.tan(chi), u[0] * h / ca.cos(chi) - kc_f(s), h / (ca.cos(chi) * v))

    opti = ca.Opti()
    X = opti.variable(3, N + 1)
    Z = [opti.variable(3, stages) for _ in range(N)]
    U = opti.variable(1, N)
    opti.subject_to(scheme.defects(dyn, X, Z, U, bounds_s[:-1], ds) == 0)
    opti.subject_to(X[2, 0] == 0)
    # bounds
    vmean = float(np.mean(V))
    kmax = (max_turn if max_turn is not None else P.MAX_TURN) / (DEG * 4.0)
    opti.subject_to(opti.bounded(-kmax, U, kmax))
    opti.subject_to(opti.bounded(-0.6, X[1, :], 0.6))
    blo, bhi = water_bounds(model, frame, bounds_s, edge=edge)
    opti.subject_to(opti.bounded(blo, X[0, :].T, bhi))
    for i in range(N):                                     # the stages too (they lie between nodes)
        sl = np.interp(bounds_s[i] + scheme.points[1:] * ds[i], bounds_s, blo)
        sh = np.interp(bounds_s[i] + scheme.points[1:] * ds[i], bounds_s, bhi)
        opti.subject_to(opti.bounded(sl, Z[i][0, :].T, sh))
    for g in gates:
        i = int(np.argmin(np.abs(bounds_s - g["s"])))
        opti.subject_to(opti.bounded(g["lo"], X[0, i], g["hi"]))
    smooth = ca.sumsqr(ca.diff(U, 1, 1) / ca.DM(np.sqrt(0.5 * (ds[:-1] + ds[1:]))).T)
    opti.minimize(X[2, N] + smooth_weight * smooth)
    # warm start: the crew's own line, in the frame
    st0 = np.linspace(0.0, float(course.length), 2000)
    fs, fn = frame.from_course(st0, start_route.offset_at(st0))
    o = np.argsort(fs)
    n0 = np.interp(bounds_s, fs[o], fn[o])
    n0 = np.clip(n0, blo + 0.5, bhi - 0.5)
    for g in gates:
        i = int(np.argmin(np.abs(bounds_s - g["s"])))
        n0[i] = np.clip(n0[i], g["lo"] + 0.3, g["hi"] - 0.3)
    from scipy.ndimage import gaussian_filter1d
    n0 = gaussian_filter1d(n0, 1.0, mode="nearest")
    kcn = np.interp(bounds_s, frame.s, frame.curvature)
    chi0 = np.arctan(np.gradient(n0, bounds_s) / (1 - n0 * kcn))
    kap0 = np.clip((np.gradient(chi0, bounds_s) + kcn) * np.cos(chi0), -0.9 * kmax, 0.9 * kmax)
    t0 = np.concatenate([[0.0], np.cumsum(ds / vmean)])
    opti.set_initial(X, np.vstack([n0, chi0, t0]))
    for i in range(N):
        sj = bounds_s[i] + scheme.points[1:] * ds[i]
        opti.set_initial(Z[i], np.vstack([np.interp(sj, bounds_s, n0), np.interp(sj, bounds_s, chi0), np.interp(sj, bounds_s, t0)]))
    opti.set_initial(U, 0.5 * (kap0[:-1] + kap0[1:]))
    opti.solver("ipopt", {"print_time": False}, {"max_iter": 3000, "tol": 1e-7, "print_level": 5 if verbose else 0,
                                                  "mu_strategy": "adaptive"})
    try:
        sol = opti.solve()
        status = "solved"
    except RuntimeError:
        sol = opti.debug
        status = "not converged: " + str(opti.stats().get("return_status"))
    Xs = np.asarray(sol.value(X)); Us = np.atleast_1d(np.asarray(sol.value(U))).ravel()
    dense_s = np.arange(s_a, s_b, 5.0)
    dense_n = np.interp(dense_s, bounds_s, Xs[0])
    # the stage values too, for the 5 m line
    sv, nv = [], []
    for i in range(N):
        Zi = np.asarray(sol.value(Z[i]))
        sv += list(bounds_s[i] + scheme.points[1:] * ds[i]); nv += list(Zi[0])
    sv = np.concatenate([[bounds_s[0]], sv]); nv = np.concatenate([[Xs[0, 0]], nv])
    dense_n = np.interp(dense_s, sv, nv)
    route = frame.to_course_route(np.concatenate([[s_a - 20], dense_s, [s_b + 20]]),
                                  np.concatenate([[dense_n[0]], dense_n, [dense_n[-1]]]), name="radau optimum")
    t_full, rudder = P._time(model, boat, route, reference_time, wind, rudder_k)
    return dict(status=status, iterations=int(opti.stats().get("iter_count", -1)), ocp_time_s=float(Xs[2, -1]),
                time_s=float(t_full), rudder_s=float(rudder), route=route, frame_s=sv.tolist(), frame_n=nv.tolist(),
                curvature=Us.tolist(), mesh=bounds_s.tolist(), gates=gates, edge_m=edge,
                path_m=float(P.route_length(model, route)))
