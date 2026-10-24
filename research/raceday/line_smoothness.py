"""HOCR 2026: each coxswain's intended line and their steering of it, separated by length scale.

    .venv/Scripts/python.exe research/raceday/line_smoothness.py

Why in the boat's own frame. A first version decomposed the offset from the course centreline. Its fast
part was 0.68-0.97 correlated between crews: it was the centreline's own kinks, which every crew's offset
inherits, not steering. So the decomposition here works on each track's own geometry.

For every crew in review_crews with a review JSON:
1. The race part of the per-stroke CoxBox track in the plane. GPS glitches removed: a stroke whose
   distance from the previous fix disagrees with the CoxBox's distance per stroke by more than 4 m, or
   whose heading jumps more than 4 MAD from its 7-stroke neighbourhood and back. Resampled to equal steps
   of DS m along the track's own arc length; heading theta(s) from the resampled points.
2. theta(s) is split into a slow part (the intended line's heading) and a fast part (steering), two ways:
   a. Wavelets: the isotropic undecimated ("a trous", starlet) transform, B3-spline kernel; shift-
      invariant, levels add back exactly. The level-J smooth is a low-pass with half-power wavelength
      about 3.08 * 2^J * DS (kernel variance 4^j samples^2 per level): with DS = 6.5 m, J = 2, 3, 4 cut
      at about 80, 160 and 320 m.
   b. Diffusion maps: a Gaussian kernel on the track points with the alpha = 1 density normalisation, so
      the generator converges to the Laplace-Beltrami operator of the curve (Coifman & Lafon 2006). Its
      eigenvectors are the curve's own Fourier modes; theta is projected on them with a Gaussian taper that
      halves at the same cut-off (each mode's wavelength from its zero crossings).
3. Steering: the fast heading theta - theta_slow, without the finest level (under about 20 m, two to three
   strokes, which is mostly the GPS unit's own filtering and differs between units); its sideways effect
   d(s) = integral of it along s, high-passed at the same cut-off; heading-detail energy per scale band.
   The intended line = the track moved sideways by -d along its normal (no curve shrinkage, unlike
   smoothing x and y directly).
4. Times, in every crew's setting (research boat, as line_compare.py): the raw track and the intended
   line; the steering cost = raw - intended.
Check: the fast heading of different crews on a common station grid should now be uncorrelated.
Writes line_smoothness.json and line_smoothness.png.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.course import local_tangent_plane                        # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402
import race_review_2026 as RR                                                # noqa: E402  (strokes)
from review_crews import CREWS                                               # noqa: E402

DS = 6.5
LEVELS = {80: 2, 160: 3, 320: 4}
MAIN = 160
FT = 0.3048
from review_crews import COX                                 # noqa: E402  names: data/local/raceday/crews.json (private)


def starlet(sig, J):
    h = np.array([1, 4, 6, 4, 1], float) / 16.0
    c = np.asarray(sig, float)
    details = []
    for j in range(J):
        step = 2 ** j
        pad = 2 * step
        ext = np.concatenate([c[pad:0:-1], c, c[-2:-pad - 2:-1]])
        cn = np.zeros_like(c)
        for k, hk in zip(range(-2, 3), h):
            cn += hk * ext[pad + k * step: pad + k * step + len(c)]
        details.append(c - cn)
        c = cn
    return c, details


def lowpass_w(sig, cutoff):
    return starlet(sig, LEVELS[cutoff])[0]


def diffusion_basis(x, y, eps_m=3.0 * DS):
    P = np.column_stack([x, y])
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / eps_m ** 2)
    q = W.sum(1)
    Wa = W / np.outer(q, q)
    dd = Wa.sum(1)
    vals, vecs = np.linalg.eigh(Wa / np.sqrt(np.outer(dd, dd)))
    order = np.argsort(vals)[::-1][:min(len(x) - 1, 260)]
    psi = vecs[:, order] / np.sqrt(dd)[:, None]
    L = (len(x) - 1) * DS
    zc = np.array([np.sum(np.diff(np.sign(psi[:, k])) != 0) for k in range(psi.shape[1])])
    wavelength = np.where(zc > 0, 2.0 * L / np.maximum(zc, 1), np.inf)
    return psi, dd, wavelength


def lowpass_d(sig, basis, cutoff):
    psi, dd, wl = basis
    taper = np.exp(-np.log(2.0) * (cutoff / wl) ** 2)
    coef = (psi * dd[:, None]).T @ sig / np.sum(psi ** 2 * dd[:, None], axis=0)
    return psi @ (taper * coef)


def track(key, course):
    RR.COXBOX = CREWS[key]["coxbox"]
    S = RR.strokes()                                  # t, distance, speed, rate, lat, lon
    x, y = local_tangent_plane(S[:, 4], S[:, 5], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(a, b) for a, b in zip(x, y)])
    race = (st > 2.0) & (st < course.length - 2.0)
    x, y, st, dist = x[race], y[race], st[race], S[race, 1]
    # glitches: fix-to-fix distance against the CoxBox's own distance per stroke, and heading spikes
    seg = np.hypot(np.diff(x), np.diff(y)); dps = np.diff(dist)
    bad = np.zeros(len(x), bool)
    bad[1:] |= np.abs(seg - dps) > 4.0
    hd = np.unwrap(np.arctan2(np.diff(y), np.diff(x)))
    dh = np.degrees(np.diff(hd))
    for i in range(len(dh)):
        w = dh[max(i - 3, 0):i + 4]
        med = np.median(w); mad = 1.4826 * np.median(np.abs(w - med)) + 0.5
        if abs(dh[i] - med) > 4 * mad and i + 1 < len(dh) and np.sign(dh[i]) != np.sign(dh[i + 1]):
            bad[i + 1] = True
    removed = [dict(station=round(float(s), 0)) for s in st[bad]]
    x, y = x[~bad], y[~bad]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    g = np.arange(0.0, arc[-1], DS)
    xg, yg = np.interp(g, arc, x), np.interp(g, arc, y)
    th = np.unwrap(np.arctan2(np.gradient(yg), np.gradient(xg)))
    return dict(arc=g, x=xg, y=yg, theta=th, removed=removed, n_strokes=int(len(S)))


def decompose(T, cutoff, method, basis=None):
    th = T["theta"]
    slow = lowpass_w(th, cutoff) if method == "wavelet" else lowpass_d(th, basis, cutoff)
    fine = starlet(th, 1)[1][0]                                # the finest level (< ~20 m): GPS filtering
    fast = th - slow - fine
    d = np.cumsum(fast) * DS
    d = d - (lowpass_w(d, cutoff) if method == "wavelet" else lowpass_d(d, basis, cutoff))
    nx, ny = -np.sin(slow), np.cos(slow)
    return slow, fast, d, T["x"] - d * nx, T["y"] - d * ny


def on_course(course, x, y):
    st = np.array([course.nearest_station(a, b) for a, b in zip(x, y)])
    of = np.array([course.offset_of(a, b, s) for a, b, s in zip(x, y, st)])
    st = np.maximum.accumulate(st)
    u, idx = np.unique(np.round(st, 1), return_index=True)
    return st[idx], of[idx]


def route_of(course, st, of):
    L = float(course.length)
    s = np.concatenate([[0.0], st, [L]]); o = np.concatenate([[of[0]], of, [of[-1]]])
    hw = course.half_width_at(s)
    return Route(s, np.clip(o, -(hw - 1.0), hw - 1.0), name="line")


def timer(cfg):
    model = C.CourseModel.build(level_offset=cfg["level_ft"] * FT)
    spec = RC.CREWS[cfg["boat"]]
    boat = RC.build_boat(cfg["boat"], spec["rate"], profile="research")
    k = C.RUDDER_K[spec["boat"]]

    def t(route):
        model.flow = charles.ContinuityFlow(model.course, discharge=cfg["q"])
        model.route = route
        sec = C._sections(model, C._evaluate(model, boat, cfg["ref"], cfg["wind"]), cfg["wind"], k)
        return sum(z.time_s for z in sec), sum(z.path_m for z in sec)
    return t


def main():
    course = C.CourseModel.build().course
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    out = dict(ds=DS, cutoffs=list(LEVELS), main_cutoff=MAIN, crews={}, timed={}, check={})
    fasts, planes = {}, {}
    for k in keys:
        T = track(k, course)
        basis = diffusion_basis(T["x"], T["y"])
        th = T["theta"]
        _, det = starlet(th, 6)
        c = dict(cox=COX.get(k, k), n_strokes=T["n_strokes"], removed=T["removed"], wobble_deg={}, lateral_rms_m={},
                 bands=[dict(level=j + 1, to_m=round(3.08 * 2 ** (j + 1) * DS), from_m=round(3.08 * 2 ** j * DS) if j else 0,
                             heading_rms_deg=round(float(np.degrees(np.sqrt(np.mean(d ** 2)))), 3)) for j, d in enumerate(det)])
        lines = {}
        for cut in LEVELS:
            for method in ("wavelet", "diffusion"):
                slow, fast, d, xi, yi = decompose(T, cut, method, basis)
                c["wobble_deg"]["%s_%d" % (method, cut)] = round(float(np.degrees(np.sqrt(np.mean(fast ** 2)))), 3)
                c["lateral_rms_m"]["%s_%d" % (method, cut)] = round(float(np.sqrt(np.mean(d ** 2))), 3)
                if method == "wavelet" and cut == MAIN:
                    fasts[k] = (T, fast)
                    planes[k] = dict(raw=(T["x"], T["y"]), intended=(xi, yi), fast=fast, slow=slow)
                lines["%s_%d" % (method, cut)] = (xi, yi, slow)
        xw, yw, sw = lines["wavelet_%d" % MAIN]; xd, yd, sd = lines["diffusion_%d" % MAIN]
        c["methods_agree"] = dict(position_rms_m=round(float(np.sqrt(np.mean((xw - xd) ** 2 + (yw - yd) ** 2))), 3),
                                  heading_rms_deg=round(float(np.degrees(np.sqrt(np.mean((sw - sd) ** 2)))), 3))
        st_r, of_r = on_course(course, T["x"], T["y"])
        c["raw"] = dict(st=np.round(st_r, 1).tolist(), off=np.round(of_r, 1).tolist())
        c["intended"] = {}
        for name, (xi, yi, _) in lines.items():
            st_i, of_i = on_course(course, xi, yi)
            c["intended"][name] = dict(st=np.round(st_i, 1).tolist(), off=np.round(of_i, 1).tolist())
        out["crews"][k] = c
        print("%-7s %4d strokes, %d glitches; heading wobble (wavelet/diffusion, 160 m) %.2f/%.2f deg; lateral %.2f/%.2f m; methods agree %.2f m"
              % (k, T["n_strokes"], len(T["removed"]), c["wobble_deg"]["wavelet_160"], c["wobble_deg"]["diffusion_160"],
                 c["lateral_rms_m"]["wavelet_160"], c["lateral_rms_m"]["diffusion_160"], c["methods_agree"]["position_rms_m"]))
    # check: fast headings of different crews on a common course grid should be uncorrelated now
    grid = np.arange(150.0, 4550.0, DS)
    F = []
    for k in keys:
        T, fast = fasts[k]
        st_r = np.maximum.accumulate(np.array([course.nearest_station(a, b) for a, b in zip(T["x"], T["y"])]))
        F.append(np.interp(grid, st_r, fast))
    M = np.corrcoef(F)
    out["check"]["inter_crew_fast_heading_correlation"] = round(float(M[np.triu_indices(len(keys), 1)].mean()), 3)
    print("check: mean inter-crew correlation of the fast heading %.3f (the centreline version was 0.68-0.97)" % out["check"]["inter_crew_fast_heading_correlation"])
    # times in every setting
    for setk in keys:
        R = json.load(open(os.path.join(HERE, CREWS[setk]["json"])))
        cal = R["calibrated"]
        cfg = dict(boat=CREWS[setk]["crew"], level_ft=R["level_ft"], q=R["flow_cfs"] * 0.0283168466,
                   wind=(cal["wind"]["speed"], cal["wind"]["direction"]), ref=cal["ref_s"])
        t = timer(cfg)
        res = dict(best=t(Route(np.asarray(cal["best"]["stations"], float), np.asarray(cal["best"]["offsets"], float), name="best"))[0])
        for k in keys:
            c = out["crews"][k]
            raw = t(route_of(course, np.asarray(c["raw"]["st"]), np.asarray(c["raw"]["off"])))
            ints = {n: t(route_of(course, np.asarray(v["st"]), np.asarray(v["off"]))) for n, v in c["intended"].items()}
            res[k] = dict(raw=raw[0], raw_path=raw[1], intended={n: v[0] for n, v in ints.items()}, intended_path={n: v[1] for n, v in ints.items()})
        out["timed"][setk] = res
        print("setting %-7s best line %.1f s" % (setk, res["best"]))
        for k in keys:
            r = res[k]; w = "wavelet_%d" % MAIN
            print("   %-7s raw %.1f  intended %.1f (diffusion %.1f)  steering cost %+.1f s  path %+.1f m"
                  % (k, r["raw"], r["intended"][w], r["intended"]["diffusion_%d" % MAIN], r["raw"] - r["intended"][w], r["raw_path"] - r["intended_path"][w]))
    own_metrics(out, keys, planes)
    setting_metrics(out, keys, planes)
    json.dump(out, open(os.path.join(HERE, "line_smoothness.json"), "w"))
    figure(out, keys)


BOAT_L = {"four": 13.4, "eight": 17.7}       # hull length, m (a coxed four, an eight)


def sparc(sig, ds, fc_max=1.0 / 20.0, amp_th=0.05, pad=4):
    """Spectral arc length (Balasubramanian et al. 2015) of a profile sampled every ``ds`` m: the arc length
    of its normalised magnitude spectrum up to an adaptive cut-off (where the spectrum last exceeds
    ``amp_th``, at most ``fc_max`` cycles per metre). More negative = less smooth."""
    n = int(2 ** np.ceil(np.log2(len(sig)) + pad))
    f = np.fft.rfftfreq(n, d=ds)
    M = np.abs(np.fft.rfft(sig, n))
    M = M / max(M.max(), 1e-12)
    sel = f <= fc_max
    f, M = f[sel], M[sel]
    above = np.flatnonzero(M >= amp_th)
    fc = f[above[-1]] if len(above) else f[-1]
    sel = f <= fc
    f, M = f[sel], M[sel]
    fn = f / max(fc, 1e-12)
    return -float(np.sum(np.sqrt(np.diff(fn) ** 2 + np.diff(M) ** 2)))


def own_metrics(out, keys, planes):
    """Each crew judged in its own boat at its own speed: the plan (intended line) and the execution (steering)."""
    from coxswain.raceday.card import turn_rate
    for k in keys:
        cfg = CREWS[k]
        R = json.load(open(os.path.join(HERE, cfg["json"])))
        boat = cfg["crew"]; kk = C.RUDDER_K[RC.CREWS[boat]["boat"]]; Lb = BOAT_L[boat]
        P_ = planes[k]
        xr, yr = P_["raw"]; xi, yi = P_["intended"]
        pr, pi = np.column_stack([xr, yr]), np.column_stack([xi, yi])
        seg_r = np.hypot(*np.diff(pr, axis=0).T); seg_i = np.hypot(*np.diff(pi, axis=0).T)
        v = float(R["calibrated"]["ours"]["path_m"]) / float(R["race_s"])          # the crew's own mean speed
        # turn rate as the turning loss was measured (heading over 30 m, no extra smoothing)
        om_r = turn_rate(pr, np.full(len(pr), v), 30.0, 0.0)
        om_i = turn_rate(pi, np.full(len(pi), v), 30.0, 0.0)
        cost = lambda om, seg: float(np.sum(seg / v * kk * 0.5 * (om[:-1] ** 2 + om[1:] ** 2)))   # noqa: E731
        plan_turn_s = cost(om_i, seg_i)
        steer_turn_s = cost(om_r, seg_r) - plan_turn_s
        steer_path_s = (seg_r.sum() - seg_i.sum()) / v
        kap_i = np.radians(om_i) / v; kap_f = P_["fast"]                           # fast heading (rad) per DS
        kap_fast = np.gradient(kap_f, DS)
        # corrections: sign changes of the fast heading between excursions of more than 0.5 degrees
        f_deg = np.degrees(kap_f)
        peaks, last = 0, 0
        for val in f_deg:
            sgn = 1 if val > 0.5 else (-1 if val < -0.5 else 0)
            if sgn and sgn != last:
                peaks += 1; last = sgn
        km = seg_r.sum() / 1000.0
        om_fast = np.degrees(kap_fast) * v
        seg_len = int(500.0 / DS)
        sp = [sparc(om_r[i:i + seg_len], DS) for i in range(0, len(om_r) - seg_len, seg_len)]
        lat = np.asarray(out["crews"][k]["lateral_rms_m"]["wavelet_%d" % MAIN])
        out["crews"][k]["own"] = dict(
            boat=boat, speed=round(v, 3), rudder_k=kk, boat_length=Lb,
            plan=dict(turning_s=round(plan_turn_s, 2), peak_turn_deg_s=round(float(np.max(np.abs(om_i))), 2),
                      min_radius_m=round(float(1.0 / max(np.max(np.abs(kap_i)), 1e-6)), 1),
                      peak_rudder_index=round(float(Lb * np.max(np.abs(kap_i))), 3)),
            steer=dict(seconds=round(steer_turn_s + steer_path_s, 2), turning_s=round(steer_turn_s, 2), path_s=round(steer_path_s, 2),
                       weave_m=float(lat), weave_per_length=round(float(lat) / Lb, 4),
                       wobble_deg=out["crews"][k]["wobble_deg"]["wavelet_%d" % MAIN],
                       turn_rate_wobble_deg_s=round(float(np.sqrt(np.mean(om_fast ** 2))), 3),
                       rudder_index_rms=round(float(Lb * np.sqrt(np.mean(kap_fast ** 2))), 4),
                       corrections_per_km=round(peaks / km, 1), sparc=round(float(np.mean(sp)), 3)))
        o = out["crews"][k]["own"]
        print("%-7s own boat %s at %.2f m/s | plan: turning %.1f s, peak %.1f deg/s, R_min %.0f m, L/R %.3f | steering: %.1f s (turn %.1f + path %.1f), weave %.2f m (%.3f L), %.1f corrections/km, SPARC %.2f"
              % (k, boat, v, o["plan"]["turning_s"], o["plan"]["peak_turn_deg_s"], o["plan"]["min_radius_m"], o["plan"]["peak_rudder_index"],
                 o["steer"]["seconds"], o["steer"]["turning_s"], o["steer"]["path_s"], o["steer"]["weave_m"], o["steer"]["weave_per_length"],
                 o["steer"]["corrections_per_km"], o["steer"]["sparc"]))


def setting_metrics(out, keys, planes):
    """Every line in every crew's setting, in that setting's boat at that setting's speed (the eight's
    setting rows every line as an eight; a four's setting rows the eight's line as a four): the turning
    the intended line builds in, and the steering cost (extra turning + extra distance) of the track."""
    from coxswain.raceday.card import turn_rate
    out["by_setting"] = {}
    for setk in keys:
        cfg = CREWS[setk]
        R = json.load(open(os.path.join(HERE, cfg["json"])))
        boat = cfg["crew"]; kk = C.RUDDER_K[RC.CREWS[boat]["boat"]]; Lb = BOAT_L[boat]
        v = float(R["calibrated"]["ours"]["path_m"]) / float(R["race_s"])
        res = {}
        for k in keys:
            P_ = planes[k]
            pr, pi = np.column_stack(P_["raw"]), np.column_stack(P_["intended"])
            seg_r = np.hypot(*np.diff(pr, axis=0).T); seg_i = np.hypot(*np.diff(pi, axis=0).T)
            om_r = turn_rate(pr, np.full(len(pr), v), 30.0, 0.0); om_i = turn_rate(pi, np.full(len(pi), v), 30.0, 0.0)
            cost = lambda om, seg: float(np.sum(seg / v * kk * 0.5 * (om[:-1] ** 2 + om[1:] ** 2)))   # noqa: E731
            plan = cost(om_i, seg_i)
            steer_turn = cost(om_r, seg_r) - plan
            steer_path = (seg_r.sum() - seg_i.sum()) / v
            res[k] = dict(plan_turn_s=round(plan, 2), steer_s=round(steer_turn + steer_path, 2), steer_turn_s=round(steer_turn, 2),
                          steer_path_s=round(steer_path, 2), peak_LR=round(float(Lb * np.max(np.abs(np.radians(om_i) / v))), 3))
        out["by_setting"][setk] = dict(boat=boat, speed=round(v, 3), rudder_k=kk, boat_length=Lb, lines=res)
        print("setting %-7s (%s, %.2f m/s): " % (setk, boat, v) + "  ".join("%s steer %.1f / plan %.1f" % (k, r["steer_s"], r["plan_turn_s"]) for k, r in res.items()))


def figure(out, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(keys), 1, figsize=(11, 2.0 * len(keys)), sharex=True, constrained_layout=True)
    for a, k in zip(np.atleast_1d(ax), keys):
        c = out["crews"][k]
        a.plot(c["raw"]["st"], c["raw"]["off"], color="#999", lw=0.8, label="GPS")
        w = c["intended"]["wavelet_%d" % MAIN]; d = c["intended"]["diffusion_%d" % MAIN]
        a.plot(w["st"], w["off"], color="#c0392b", lw=1.5, label="intended, wavelet")
        a.plot(d["st"], d["off"], color="#1f4e79", lw=1.0, ls="--", label="intended, diffusion map")
        a.set_ylabel(c["cox"] + "\noffset (m)")
    np.atleast_1d(ax)[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    np.atleast_1d(ax)[-1].set_xlabel("distance along the course (m)")
    fig.savefig(os.path.join(HERE, "line_smoothness.png"), dpi=130)
    print("written line_smoothness.json and line_smoothness.png")


if __name__ == "__main__":
    main()
