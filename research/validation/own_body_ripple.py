"""Is the own-body field's 6 Hz push in the athlete's measured channels or made by our mapping? (§240)

rung 2's "own body" times the model's legs and trunk by the athlete's measured seat (Ls) and trunk (Lt) channels:
each channel is smoothed (GCV), normalised, and *warped* onto the model's own channel (hip - ankle, shoulder - hip) by
inverting the model's curve (research/biorow/measured_body.warp, channel_field). Here, finely through the cycle:

* the measured seat's acceleration (the smoothed channel, differentiated) against the model's lower-trunk (seat)
  acceleration from the warped leg field -- they should be the same motion;
* the same for the trunk channel (shoulder relative to the seat);
* the warp's own time-law derivatives (tau', tau''), where an inversion through a flat stretch of the model's curve
  would show up.

Prints the content above 3 Hz of each (rms), and the largest harmonics.

    python research/validation/own_body_ripple.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "crew"), os.path.join(ROOT, "research", "biorow"),
                os.path.join(ROOT, "research", "blade")]
import rung2_population as RP                                    # noqa: E402


def above(x, dt, f0=3.0):
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(x.size, d=dt)
    amp = np.abs(spec) / x.size * 2
    top = np.argsort(amp)[::-1][:5]
    return float(np.sqrt(np.sum(amp[f > f0] ** 2) / 2)), ", ".join("%.1f Hz %.2f" % (f[i], amp[i]) for i in top)


def main():
    RP.BODY, RP.BLADE = "own", "crossflow_normal"
    RP.C2.TAKEOVER_DEG = RP.C2.BIOROW_CATCH_TO_BURIED_DEG
    m = RP.V.ATHLETES["br24"][0]()
    boat = RP.athlete_boat("br24", m)
    import measured_body as MB
    MB.CHANNEL_SMOOTHING = sys.argv[1] if len(sys.argv) > 1 else "gcv"
    print("channel smoothing:", MB.CHANNEL_SMOOTHING)
    P = float(boat.timing.period)
    k = 800
    t = np.arange(k) * P / k
    dt = P / k
    for name, fn, label in (("Ls", lambda j: j["hip"][0] - j["ankle"][0], "seat (legs)"),
                            ("Lt", lambda j: j["shoulder"][0] - j["hip"][0], "trunk on the seat")):
        m_c, s_c, c_c, travel = MB.measured_channel(name)
        mm, tau, cm, model_travel = MB.model_channel(boat, fn)
        wp = MB.warp(s_c, c_c, tau, cm)
        field = MB.channel_field(boat, wp, m_c, mm)
        # measured: the channel as the model's clock sees it (phase since its minimum), metres; differentiated
        x_meas = np.interp(np.mod(t / P - m_c, 1.0), s_c, c_c, period=1.0) * travel
        a_meas = np.gradient(np.gradient(x_meas, dt), dt)
        # model: the same channel of the warped field, from the field's own joint accelerations
        rows = []
        for tt in t:
            _mass, pos, vel, acc = field(float(tt))
            rows.append((pos, acc))
        rower = boat.crew[0].rower
        # the warped field's channel position, via the model's joints at the warped time (positions only), differentiated
        tm = np.array([P * (np.floor(tt / P - m_c) + float(wp(np.mod(tt / P - m_c, 1.0))) + mm) for tt in t])
        x_model = np.array([fn(rower.joint_positions(float(u))) for u in tm])
        a_model = np.gradient(np.gradient(x_model, dt), dt)
        tp = np.array([float(wp.derivative(1)(np.mod(tt / P - m_c, 1.0))) for tt in t])
        tpp = np.array([float(wp.derivative(2)(np.mod(tt / P - m_c, 1.0))) for tt in t])
        # the model channel itself on its own clock (no warp), differentiated
        x_own = np.array([fn(rower.joint_positions(float(u))) for u in t])
        a_own = np.gradient(np.gradient(x_own, dt), dt)
        print("\n== %s: travel measured %.3f m, model %.3f m" % (label, travel, model_travel))
        for lab, x in (("measured channel accel, m/s2", a_meas), ("warped model channel accel", a_model),
                       ("model channel, own clock", a_own), ("warp tau'", tp), ("warp tau''", tpp)):
            r, top = above(x, dt)
            print("  %-30s rms %7.3f  peak %7.2f | above 3 Hz rms %7.3f | %s" % (lab, np.sqrt(np.mean((x - x.mean()) ** 2)),
                                                                           np.abs(x).max(), r, top))
        print("  every 0.02 of the cycle (catch clock): measured | warped model accel, m/s2")
        for j in range(0, k, k // 50):
            print("    %.2f  %7.2f  %7.2f   tau' %.3f" % (t[j] / P, a_meas[j], a_model[j], tp[j]))


if __name__ == "__main__":
    main()
