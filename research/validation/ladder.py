"""The validation ladder (docs/VALIDATION_LADDER.md): every rung's criteria against its band, from the runs' outputs.

    python research/validation/ladder.py --rung2 <rung2 json> --rung2-half <rung2 json at half the step>
        --bench <bench json> --bench-case "<case name>" --c1 <handle-force json> [--scorecard <log>] [--tests]

Reads; runs nothing heavy (A1's beam FE is milliseconds; ``--tests`` runs A2/A4's unit tests, ~5 min). Prints the
table and writes data/local/validation/ladder.json. The bands are the document's, copied here once (BANDS); a change to
either is a change to both, recorded in the document's history.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "crew")]

#: docs/VALIDATION_LADDER.md, set 2026-11-09 before the first run
BANDS = dict(
    A1_ratio=(0.95, 1.05),
    A3_speed=0.005, A3_ivv=0.005,
    B1_bend_rms_deg=1.5, B1_peak_at=(75.0, 95.0), B1_mz_impulse=(0.85, 1.15),
    C1_rms_frac=0.15, C1_peak_deg=10.0,
    D1_speed=0.03, D3_power=0.10,
    # D2 / D2b since 2026-11-10 (user: "source the band"; SOURCES §233): 0.75 x the between-rower relative SD of the
    # measure in Leao et al. 2025 (23 single scullers, 2000 m races; sex-matched) x the athlete's measured value.
    # 0.75 keeps the original band's stringency (3 points over Loschner & Smith's ~4-point SD). Was +-3 points.
    D2_rel_sd={"br24": 1.05 / 5.78, "cr06": 1.11 / 5.50}, D2b_rel_sd={"br24": 2.02 / 14.13, "cr06": 1.93 / 11.64},
    D2_fraction=0.75,
    D4_centre=(0.44, 0.54), D4_peak=(424.0 * 0.85, 424.0 * 1.15), D4_mid_min=0.0,
    D5_progress=0.05,
)
ATHLETES = ("cr06", "br24")
#: [BR24]'s rig (BioRow file header) and BioRow's sculling force point from the handle's top (the data's author)
BIOROW_INBOARD, BIOROW_FORCE_POINT = 0.875, 0.06


def row(rung, what, value, band, ok, note=""):
    return dict(rung=rung, what=what, value=value, band=band, ok=bool(ok), note=note)


def rung_a1():
    import oar_modes
    chk = np.array(oar_modes.bl2010_check())
    r = float(np.mean(chk[:, 1] / chk[:, 0]))
    lo, hi = BANDS["A1_ratio"]
    return [row("A1", "shaft first frequency, predicted / measured (mean of 8)", "%.3f" % r, "%.2f-%.2f" % (lo, hi),
                lo <= r <= hi)]


def rung_a3(main, half):
    out = []
    for a in ATHLETES:
        s0, s1 = main[a]["rung2"], half[a]["rung2"]
        ds = abs(s1["speed"] / s0["speed"] - 1.0)
        di = abs(s1["ivv"] - s0["ivv"])
        out.append(row("A3", "%s half step: speed / IVV change" % a, "%.2f %% / %.2f pt" % (100 * ds, 100 * di),
                       "< %.1f %% / < %.1f pt" % (100 * BANDS["A3_speed"], 100 * BANDS["A3_ivv"]),
                       ds < BANDS["A3_speed"] and di < BANDS["A3_ivv"]))
    return out


def rung_tests():
    tests = ["tests/unit/test_bend_dynamics.py::test_a_stiff_dynamic_shaft_converges_to_the_rigid_oar_with_the_added_mass",
             "tests/unit/test_bend_dynamics.py::test_release_after_leaves_a_run_without_an_early_release_unchanged",
             "tests/unit/test_bend_dynamics.py::test_the_blade_track_is_the_gate_plus_the_bend"]
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    out = []
    for rung, t in zip(("A2", "A4", "A4"), tests):
        rc = subprocess.run([py, "-m", "pytest", "-q", t], cwd=ROOT, capture_output=True, text=True).returncode
        out.append(row(rung, t.split("::")[1].replace("_", " "), "pass" if rc == 0 else "fail", "test passes", rc == 0))
    return out


def rung_b1(bench, case):
    r = bench["results"][case]
    rms, at, imp = r["bend_rms_deg"], r["bend_peak_at"], r["score"]["Mz"]["impulse_ratio"]
    lo, hi = BANDS["B1_peak_at"]
    ilo, ihi = BANDS["B1_mz_impulse"]
    return [row("B1", "tank trial bend rms (%s)" % case, "%.2f deg" % rms, "<= %.1f deg" % BANDS["B1_bend_rms_deg"],
                rms <= BANDS["B1_bend_rms_deg"]),
            row("B1", "tank trial bend peak at", "%.0f deg" % at, "%.0f-%.0f deg" % (lo, hi), lo <= at <= hi),
            row("B1", "tank trial M_z impulse ratio", "%.2f" % imp, "%.2f-%.2f" % (ilo, ihi), ilo <= imp <= ihi)]


def rung_c1(c1):
    r = c1["bending"]
    d = abs(r["peak_at_deg"] - r["measured_peak_at_deg"])
    return [row("C1", "[CR06] rung 1 handle force rms / her peak", "%.1f %%" % (100 * r["rms_frac"]),
                "<= %.0f %%" % (100 * BANDS["C1_rms_frac"]), r["rms_frac"] <= BANDS["C1_rms_frac"]),
            row("C1", "[CR06] rung 1 force peak angle vs hers", "%.1f deg off" % d, "<= %.0f deg" % BANDS["C1_peak_deg"],
                d <= BANDS["C1_peak_deg"])]


def measured_speed(athlete):
    """The athlete's measured cycle of boat speed (m/s), as validate_blade reads it."""
    if athlete == "cr06":
        import validate_blade as V
        return V._cr06_series("boat_velocity_mps")(np.linspace(0.0, V.CR06_T, 2000, endpoint=False))
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    return np.asarray(D["Vs"], float)


def rung_d(main):
    out = []
    for a in ATHLETES:
        s, m = main[a]["rung2"], main[a]["measured"]
        ds = s["speed"] / m["speed"] - 1.0
        di = s["ivv"] - m["ivv"]
        dp = s["power"] / m["power"] - 1.0
        b2 = BANDS["D2_fraction"] * BANDS["D2_rel_sd"][a] * m["ivv"]
        out += [row("D1", "%s speed %.3f vs %.3f m/s" % (a, s["speed"], m["speed"]), "%+.1f %%" % (100 * ds),
                    "±%.0f %%" % (100 * BANDS["D1_speed"]), abs(ds) <= BANDS["D1_speed"]),
                row("D2", "%s IVV %.1f vs %.1f %%" % (a, 100 * s["ivv"], 100 * m["ivv"]), "%+.1f pt" % (100 * di),
                    "±%.1f pt" % (100 * b2), abs(di) <= b2, "band sourced 2026-11-10; at the first ±3 it fails")]
        st = s.get("speed_trace")
        if st is not None:                                        # D2b: Kleshnev's own measure, SD / mean (§233)
            v = np.asarray(st["v"], float)
            vm = measured_speed(a)
            cm, cv = float(vm.std() / vm.mean()), float(v.std() / v.mean())
            b2b = BANDS["D2_fraction"] * BANDS["D2b_rel_sd"][a] * cm
            out.append(row("D2b", "%s CVV (SD/mean) %.2f vs %.2f %%" % (a, 100 * cv, 100 * cm),
                           "%+.2f pt" % (100 * (cv - cm)), "±%.2f pt" % (100 * b2b), abs(cv - cm) <= b2b))
        out += [
                row("D3", "%s power %.0f vs %.0f W" % (a, s["power"], m["power"]), "%+.1f %%" % (100 * dp),
                    "±%.0f %%" % (100 * BANDS["D3_power"]), abs(dp) <= BANDS["D3_power"]),
                row("D5", "%s oar progress rms" % a, "%.3f" % s["progress_rms"], "<= %.2f" % BANDS["D5_progress"],
                    s["progress_rms"] <= BANDS["D5_progress"])]
    fs = main["br24"]["rung2"].get("force_shape")
    if fs is None:
        out.append(row("D4", "br24 force shape", "missing", "-", False, "run predates force_shape (§228)"))
        return out
    clo, chi = BANDS["D4_centre"]
    plo, phi = BANDS["D4_peak"]
    # BioRow's handle force is the inboard bending torque expressed at 0.06 m from the handle's top (sculls; the
    # data's author, 2026-10-28); the model's is torque over the whole inboard. Compared on BioRow's lever (S12, §233).
    lever = BIOROW_INBOARD / (BIOROW_INBOARD - BIOROW_FORCE_POINT)
    out += [row("D4", "br24 force centre (measured 0.49)", "%.2f" % fs["centre"], "%.2f-%.2f" % (clo, chi),
                clo <= fs["centre"] <= chi),
            row("D4", "br24 peak force on BioRow's lever (measured 424 N)", "%.0f N" % (lever * fs["peak"]),
                "%.0f-%.0f N" % (plo, phi), plo <= lever * fs["peak"] <= phi),
            row("D4", "br24 least force 10-80 % of the drive", "%.0f N" % (lever * fs["mid_min"]), "> 0",
                fs["mid_min"] > 0.0)]
    return out


def rung_b2(flex, rigid):
    """The population's own stroke against its own force curve (RBN 2023/12; SOURCES §237). Added 2026-11-13 after the
    first run was seen, so the criterion is comparative, not a tuned band: the flexible shaft must beat the rigid oar."""
    return [row("B2", "population force curve, rms vs RBN 2023/12 (rigid %.1f)" % rigid["force_rms"],
                "%.1f pt" % flex["force_rms"], "< rigid", flex["force_rms"] < rigid["force_rms"],
                "comparative (added after the first run)"),
            row("B2", "population force peak, % of drive time", "%.0f %%" % flex["force_peak_time"], "40-50 %",
                40.0 <= flex["force_peak_time"] <= 50.0, "the reference curve's own peak span")]


def rung_c2(open_loop):
    """The blade and shaft alone on each athlete's measured motion (open_loop_shaft.py; §237): comparative, as B2."""
    out = []
    for a in ATHLETES:
        r = open_loop[a]
        out.append(row("C2", "%s open-loop handle force rms / measured peak (rigid %.1f %%)"
                       % (a, 100 * r["rigid"]["rms_frac"]), "%.1f %%" % (100 * r["hinge"]["rms_frac"]), "< rigid",
                       r["hinge"]["rms_frac"] < r["rigid"]["rms_frac"], "comparative (added after the first run)"))
    return out


def rung_e1(log):
    lines = [l for l in open(log, encoding="utf-8", errors="replace") if l[:3] in ("ok ", "FAI")]
    fails = [l.split()[1] + " " + l.split()[2] for l in lines if l.startswith("FAIL")]
    return [row("E1", "research scorecard targets (%d scored)" % len(lines), "%d fail" % len(fails), "0 fail",
                not fails, "; ".join(fails))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung2", required=True)
    ap.add_argument("--rung2-half", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--bench-case", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--scorecard", default=None)
    ap.add_argument("--population", nargs=2, default=None, metavar=("FLEXIBLE", "RIGID"),
                    help="population_kinematics JSONs, flexible then rigid (rung B2)")
    ap.add_argument("--open-loop", default=None, help="open_loop_shaft.json (rung C2)")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    main_r, half_r = json.load(open(a.rung2)), json.load(open(a.rung2_half))
    rows = rung_a1()
    if a.tests:
        rows += rung_tests()
    rows += rung_a3(main_r, half_r)
    rows += rung_b1(json.load(open(a.bench)), a.bench_case)
    if a.population:
        rows += rung_b2(*[json.load(open(p)) for p in a.population])
    rows += rung_c1(json.load(open(a.c1)))
    if a.open_loop:
        rows += rung_c2(json.load(open(a.open_loop)))
    rows += rung_d(main_r)
    if a.scorecard:
        rows += rung_e1(a.scorecard)
    print("validation ladder %s (docs/VALIDATION_LADDER.md)\n" % a.label)
    w = max(len(r["what"]) for r in rows)
    for r in rows:
        print("%-4s %-4s %-*s  %-16s %-16s %s" % ("ok" if r["ok"] else "FAIL", r["rung"], w, r["what"], r["value"],
                                                r["band"], r["note"]))
    n_fail = sum(not r["ok"] for r in rows)
    print("\n%d criteria, %d fail" % (len(rows), n_fail))
    json.dump(dict(label=a.label, inputs=vars(a), bands=BANDS, rows=rows, passed=n_fail == 0),
              open(os.path.join(ROOT, "data", "local", "validation", "ladder.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
