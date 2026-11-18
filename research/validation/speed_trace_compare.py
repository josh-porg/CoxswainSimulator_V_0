"""Boat speed through the cycle, rung 2 against each athlete's measured cycle, both from the catch (§228).

    python research/validation/speed_trace_compare.py <rung2 json> [<rung2 json> ...]

The model's catch is the last stroke's first wet sample; [CR06]'s her oar's greatest angle (Fig. 3), [BR24]'s his
least (BioRow's M1x file). Speeds at fractions of the cycle, and each trace's least and greatest speed and when.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]


def measured():
    import validate_blade as V
    g = np.linspace(0.0, V.CR06_T, 2000, endpoint=False)
    v = V._cr06_series("boat_velocity_mps")(g)
    k = int(np.argmax(V._cr06_series("oar_angle_deg")(g)))
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    T = 60.0 / 32.4093
    kb = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    return {"cr06": (g / V.CR06_T, np.roll(v, -k)),
            "br24": (np.arange(len(D)) / len(D), np.roll(D["Vs"], -kb))}


def model(run, athlete):
    st = run[athlete]["rung2"]["speed_trace"]
    t, v, air = np.array(st["t"]), np.array(st["v"]), np.array(st["air"], dtype=bool)
    i0 = int(np.flatnonzero(~air)[0])
    period = float(t[-1])
    ph = np.mod(t - t[i0], period) / period
    o = np.argsort(ph)
    return ph[o], v[o]


def main():
    runs = [(os.path.basename(p), json.load(open(p))) for p in sys.argv[1:]]
    meas = measured()
    for a in ("cr06", "br24"):
        traces = [("measured", *meas[a])] + [(name[:28], *model(r, a)) for name, r in runs]
        print("\n== %s: boat speed from the catch, m/s" % a)
        for name, ph, v in traces:
            print("  %-28s least %.3f at %.2f; greatest %.3f at %.2f; IVV %.1f %%"
                  % (name, v.min(), ph[np.argmin(v)], v.max(), ph[np.argmax(v)], 100 * np.ptp(v) / v.mean()))
        print("  cycle | " + " ".join("%7s" % n[0][:7] for n in traces))
        for q in np.arange(0.0, 1.0, 0.05):
            print("  %.2f  | " % q + " ".join("%7.2f" % np.interp(q, ph, v, period=1.0) for _n, ph, v in traces))
        # the two catches are defined differently (model: first wet sample; measured: the oar's turning point), so
        # also: the model shifted by the lag that best overlays it on the measured trace, and the residual by part of
        # the cycle (§241). The swing's excess is read from that residual, not from a phase offset.
        q = np.linspace(0.0, 1.0, 400, endpoint=False)
        vm = np.interp(q, *traces[0][1:], period=1.0)
        for name, ph, v in traces[1:]:
            lags = np.linspace(-0.1, 0.1, 201)
            err = [np.sqrt(np.mean((np.interp(q + d, ph, v, period=1.0) - vm) ** 2)) for d in lags]
            d = float(lags[int(np.argmin(err))])
            vs = np.interp(q + d, ph, v, period=1.0)
            res = vs - vm
            print("  %s: best overlay with the model lagged %+.3f of the cycle; rms %.3f m/s (unshifted %.3f)"
                  % (name, d, min(err), err[100]))
            for lo, hi, lab in ((0.0, 0.15, "catch to speed minimum"), (0.15, 0.45, "drive"), (0.45, 0.55, "finish"),
                                (0.55, 0.85, "recovery to speed maximum"), (0.85, 1.0, "approach to the catch")):
                m = (q >= lo) & (q < hi)
                print("     %-27s mean residual %+.3f m/s, rms %.3f" % (lab, res[m].mean(), np.sqrt(np.mean(res[m] ** 2))))
            print("     aligned extremes: least %+.3f, greatest %+.3f m/s against measured" % (vs.min() - vm.min(),
                                                                                         vs.max() - vm.max()))


if __name__ == "__main__":
    main()
