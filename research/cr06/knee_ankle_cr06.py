"""Step 4.3: knee and ankle moments on [CR06]'s woman, now that a women's foot-force direction is sourced.

    .venv/Scripts/python.exe research/cr06/knee_ankle_cr06.py

Until now the knee and ankle were indeterminate in the model's plane: the stretcher's vertical force and the
seat's share one equation. [BU15] (data/literature/buckeridge2015_foot_force.csv) gives twenty elite women's
foot-force direction at maximum handle force: horizontal / resultant RF = 0.81 +- 0.07 (an ergometer with
instrumented footplates). This script:

1. The stretcher's horizontal force from her own measured stroke ([CR06] Fig. 3, data/literature/
   cr06_fig3_measured.csv), the body's momentum balance along the boat with a frictionless seat, as step 4.3's
   first rung: F_h = m (a_boat + a_rel) + handle pull along the boat, with [CR06] eq. 5's centre of mass
   (all 75 kg on the seat plus 0.4 x the back channel). The handle convention (per handle or both) is
   decided by reproducing that rung's recorded numbers (peak stretcher 675-726 N; 408-448 N at the catch).
2. The vertical part from the sourced direction: F_v = F_h tan(acos RF), upward on the feet (they press down
   and forward into the footplate), RF held at its MHF value through the drive (an assumption, flagged), with
   the +-1 SD band.
3. Her leg posture: the package rower on the women's table at her 1.758 m (as female_segments_cr06.py), its
   seat height and segment lengths, placed at her measured leg displacement (two-link chain, ankle on the
   stretcher).
4. Net joint moments of both legs together (bottom-up: feet, shanks, then thighs to the hip), with segment
   weights and inertia. The hip from below is checked against rung 2's hip from above (trunk against the
   handle, no foot data): they agree at maximum handle force (3.46 against 3.5 N.m/kg), not at the catch.
   The point on the footplate where the force acts (heel to ball of the foot) is NOT measured by [BU15], and
   the ankle moment depends on it almost one for one; it is carried as a range. The knee-minus-ankle moment
   does not depend on it at all, and is the result the sourced direction settles.
Writes data/local/cr06_study/knee_ankle_cr06.json and .png.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import her_stroke as HS                                        # noqa: E402  the shared definitions of her stroke and body
from her_stroke import T, M, STATURE, G, DE_LEVA_FEMALE, series, periodic_spline   # noqa: E402,F401

HERE = HS.OUT                                                  # outputs (gitignored)
RF, RF_SD = 0.81, 0.07      # [BU15] Table 4, at maximum handle force


def her_rower():
    return HS.her_rower()[0]


def main(lam_v=1e-5, lam_x=1e-5, write=True):
    tv, v = series("boat_velocity_mps"); tF, F = series("handle_force_N"); ta, ang = series("oar_angle_deg")
    tl, leg = series("leg_disp_m"); tb, back = series("back_disp_m")
    t = np.linspace(0.0, T, 777, endpoint=False)
    a_boat = periodic_spline(tv, v, lam_v).derivative(1)(t)
    legS, backS = periodic_spline(tl, leg, lam_x), periodic_spline(tb, back, lam_x)
    a_rel = legS.derivative(2)(t) + 0.4 * backS.derivative(2)(t)
    Fh_handle = np.interp(t, tF, F, period=T) * np.cos(np.radians(np.interp(t, ta, ang, period=T)))
    k_land = int(np.argmin(a_boat))                         # [LE26]'s catch landmark: minimum boat acceleration
    # the handle convention: per handle (x2 for the pair of sculls) or both handles together
    conv = {}
    for name, n in (("per handle, x2", 2.0), ("both handles", 1.0)):
        Fs = M * (a_boat + a_rel) + n * Fh_handle
        conv[name] = dict(peak=float(Fs.max()), landmark=float(Fs[k_land]), gate_peak=float((n * np.interp(t, tF, F, period=T)).max()))
    target_peak = (675.0, 726.0)
    pick = min(conv, key=lambda c: min(abs(conv[c]["peak"] - target_peak[0]), abs(conv[c]["peak"] - target_peak[1])))
    nH = 2.0 if pick.startswith("per") else 1.0
    Fh = M * (a_boat + a_rel) + nH * Fh_handle
    print("handle convention:", {k: {kk: round(vv, 1) for kk, vv in d.items()} for k, d in conv.items()}, "->", pick)
    # posture: her leg displacement placed on the women's-table rower's chain
    r = her_rower()
    seat_h, Lt, Ls = float(r.station.seat_height), float(r.thigh_length), float(r.shank_length)
    tm = np.linspace(0, r.timing.period if hasattr(r, "timing") else T, 400, endpoint=False)
    hx_model = np.array([r.joint_positions(float(x))["hip"][0] - r.joint_positions(float(x))["ankle"][0] for x in tm])
    hx = hx_model.min() + (legS(t) - legS(t).min())            # hip ahead of the ankle, catch at the model's closest
    hz = seat_h
    d = np.hypot(hx, hz)
    # knee above the ankle-hip line (two-link chain)
    cosk = np.clip((Ls ** 2 + d ** 2 - Lt ** 2) / (2 * Ls * d), -1, 1)
    base_ang = np.arctan2(hz, hx); shank_ang = base_ang + np.arccos(cosk)
    kx, kz = Ls * np.cos(shank_ang), Ls * np.sin(shank_ang)   # knee, ankle at the origin
    # Frame: x toward the bow (feet to seat), z up, ankle at the origin; the 2-D moment of force F at r is
    # r x F = rx Fz - rz Fx (counter-clockwise positive). The footplate pushes the feet toward the bow and up:
    # F = (Fh, Fv). At the knee (knee above, hip toward +x) a counter-clockwise external moment folds the leg,
    # so the knee EXTENSORS supply it. At the ankle a force ahead of the ankle along the plate (toward the toes)
    # turns the foot clockwise (dorsiflexing), so the PLANTARFLEXORS supply minus the external moment.
    # Hence  knee extensor + ankle plantarflexor = (ankle - knee) x F + shank/foot weight and inertia terms:
    # where on the plate the force acts cancels (Winter's support moment, in this plane).
    spec = {s: DE_LEVA_FEMALE[s] for s in ("shank", "foot")}
    m_sh, m_ft = 2 * M * spec["shank"].mass_fraction, 2 * M * spec["foot"].mass_fraction
    c_sh = spec["shank"].com_fraction                          # shank centre of mass, from the knee
    sx, sz = kx * (1 - c_sh), kz * (1 - c_sh)
    dt = t[1] - t[0]
    acc = lambda y: (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / dt ** 2          # noqa: E731  periodic
    ax_sh, az_sh = acc(sx) + a_boat, acc(sz)                   # absolute: the hull's acceleration added
    I_sh = m_sh * (0.271 * Ls) ** 2                            # sagittal radius of gyration ~27% of the length
    alpha_sh = acc(np.unwrap(shank_ang))
    cross = lambda rx, rz, fx, fz: rx * fz - rz * fx           # noqa: E731
    # thighs (both): hip at (hx, hz) from the ankle, centre of mass 36.12% from the hip (de Leva female)
    hx_a = np.asarray(hx) * np.ones_like(t); hz_a = hz * np.ones_like(t)
    hx, hz = hx_a, hz_a
    m_th = 2 * M * DE_LEVA_FEMALE["thigh"].mass_fraction
    c_th = DE_LEVA_FEMALE["thigh"].com_fraction
    tcx, tcz = hx + c_th * (kx - hx), hz + c_th * (kz - hz)
    ax_th, az_th = acc(tcx) + a_boat, acc(tcz)
    I_th = m_th * (0.369 * Lt) ** 2                            # sagittal radius of gyration ~37% (de Leva female thigh)
    alpha_th = acc(np.unwrap(np.arctan2(kz - hz, kx - hx)))
    PHI = np.radians(42.0)                                     # footplate incline ([BU15]); taken from the horizontal
    ux, uz = -np.cos(PHI), np.sin(PHI)                         # along the plate, heel to toes (toward the stern, up)
    nx, nz = np.sin(PHI), np.cos(PHI)                          # plate normal, toward the rower
    SOLE = 0.07                                                # ankle joint centre above the sole, m
    # where the force acts, along the plate from below the ankle, m. Bounds: heel, mid-foot, ball. Measured: [VI17]
    # (Vieira et al. 2017, data/literature/vieira2017_cop.csv) puts it 196 +- 15 mm from the heel at peak force and
    # 206-212 mm from the catch to peak force on one rower: about 0.73 +- 0.05 of foot length. On her foot (de Leva
    # women's foot length at her stature) with the ankle 0.05 m ahead of the heel:
    FOOT = DE_LEVA_FEMALE["foot"].length_fraction * STATURE
    COP = {"heel": -0.05, "mid-foot": 0.04, "ball": 0.13,
           "measured-1SD": 0.68 * FOOT - 0.05, "measured": 0.73 * FOOT - 0.05, "measured+1SD": 0.78 * FOOT - 0.05}
    drive = Fh > 30.0
    i_mhf = int(np.argmax(np.where(drive, np.interp(t, tF, F, period=T), -1)))
    out = dict(source="[BU15] RF at MHF %.2f +- %.2f, held through the drive (assumption)" % (RF, RF_SD), handle_convention=pick,
               geometry=dict(seat_height=seat_h, thigh=Lt, shank=Ls, stature=STATURE, footplate_deg=42.0, sole_m=SOLE, cop_m=COP),
               sign="knee: + = extensor moment; ankle: + = plantarflexor moment; both legs together; N.m", cases={})
    for label, rf in (("RF-1SD", RF - RF_SD), ("RF", RF), ("RF+1SD", min(RF + RF_SD, 0.999))):
        th = np.arccos(rf)
        Fv = Fh * np.tan(th)
        # shank and foot weights and the shank's inertia, about the knee (the foot's mass held at the ankle)
        ext_knee_body = (cross(sx - kx, sz - kz, 0.0, -m_sh * G) + cross(-kx, -kz, 0.0, -m_ft * G)
                         - cross(sx - kx, sz - kz, m_sh * ax_sh, m_sh * az_sh) - cross(-kx, -kz, m_ft * a_boat, 0.0) - I_sh * alpha_sh)
        support = cross(-kx, -kz, Fh, Fv) + ext_knee_body      # knee extensor + ankle plantarflexor (CoP-free)
        # the thighs, for the hip from below: the force the thighs put on the shanks (Newton on shanks + feet) and
        # the thighs' own weight and inertia; hip extensor = the moment the pelvis puts on the thighs (+ = opening the hip)
        Fkx = m_sh * ax_sh + m_ft * a_boat - Fh
        Fkz = m_sh * az_sh + (m_sh + m_ft) * G - Fv
        res = {}
        for name, s in COP.items():
            px, pz = s * ux - SOLE * nx, s * uz - SOLE * nz
            ext_ankle = cross(px, pz, Fh, Fv)
            plantar = -ext_ankle
            knee_ext = cross(px - kx, pz - kz, Fh, Fv) + ext_knee_body
            M_k = -knee_ext                                    # moment the thighs put on the shanks
            hip = (M_k + cross(kx - hx, kz - hz, Fkx, Fkz) - cross(tcx - hx, tcz - hz, 0.0, -m_th * G)
                   + I_th * alpha_th + cross(tcx - hx, tcz - hz, m_th * ax_th, m_th * az_th))
            res[name] = dict(ankle=plantar, knee=knee_ext, hip=hip)
        assert np.allclose(res["ball"]["knee"] + res["ball"]["ankle"], support) and np.allclose(res["heel"]["knee"] + res["heel"]["ankle"], support)
        pick_i = lambda i: dict(t=float(t[i]), Fh=float(Fh[i]), Fv=float(Fv[i]), support=float(support[i]),            # noqa: E731
                                ankle={c: float(res[c]["ankle"][i]) for c in res}, knee={c: float(res[c]["knee"][i]) for c in res},
                                hip={c: float(res[c]["hip"][i]) for c in res})
        out["cases"][label] = dict(theta_deg=float(np.degrees(th)), at_mhf=pick_i(i_mhf), at_landmark=pick_i(k_land),
                                   drive_peak_support=float(support[drive].max()), t_peak_support=float(t[drive][np.argmax(support[drive])]),
                                   series=dict(t=t[::7].round(4).tolist(), support=support[::7].round(1).tolist(),
                                               ankle_ball=res["ball"]["ankle"][::7].round(1).tolist(), ankle_heel=res["heel"]["ankle"][::7].round(1).tolist(),
                                               knee_ball=res["ball"]["knee"][::7].round(1).tolist(), knee_heel=res["heel"]["knee"][::7].round(1).tolist()))
    knee_deg = np.degrees(np.pi - np.arccos(np.clip((Lt ** 2 + Ls ** 2 - d ** 2) / (2 * Lt * Ls), -1, 1)))
    out["checks"] = dict(stretcher_peak=float(Fh.max()), stretcher_at_landmark=float(Fh[k_land]),
                         knee_flexion_deg_at_landmark=float(knee_deg[k_land]), knee_flexion_deg_at_mhf=float(knee_deg[i_mhf]),
                         shank_incline_deg_at_mhf=float(np.degrees(shank_ang[i_mhf])), shank_incline_deg_at_landmark=float(np.degrees(shank_ang[k_land])),
                         handle_force_at_mhf=float(np.interp(t[i_mhf], tF, F, period=T)), smoothing=dict(velocity=lam_v, body=lam_x))
    if not write:
        return out
    # the catch sits on a second derivative: carry the smoothing as a band (rung 1 used 3e-6 to 3e-5)
    band = {}
    for lv in (3e-6, 1e-5, 3e-5):
        for lx in (3e-6, 1e-5, 3e-5):
            o = main(lv, lx, write=False)
            for lab, cc in o["cases"].items():
                for where in ("at_landmark", "at_mhf"):
                    band.setdefault((lab, where), []).append((cc[where]["Fh"], cc[where]["support"]))
    out["smoothing_band"] = {"%s %s" % k: dict(Fh=[round(min(a for a, _ in v), 0), round(max(a for a, _ in v), 0)],
                                                support=[round(min(b for _, b in v), 0), round(max(b for _, b in v), 0)]) for k, v in band.items()}
    print("smoothing band:", out["smoothing_band"])
    print("shank incline above horizontal: %.0f deg at the catch landmark, %.0f deg at MHF; force direction %.0f deg (RF %.2f)"
          % (out["checks"]["shank_incline_deg_at_landmark"], out["checks"]["shank_incline_deg_at_mhf"], np.degrees(np.arccos(RF)), RF))
    json.dump(out, open(os.path.join(HERE, "knee_ankle_cr06.json"), "w"), indent=1)
    print("stretcher: peak %.0f N, at the catch landmark %.0f N (t %.3f s); seat height %.3f m, thigh %.3f, shank %.3f; knee flexion %.0f deg at the landmark, %.0f at MHF"
          % (Fh.max(), Fh[k_land], t[k_land], seat_h, Lt, Ls, knee_deg[k_land], knee_deg[i_mhf]))
    for lab, cc in out["cases"].items():
        for where in ("at_landmark", "at_mhf"):
            m = cc[where]
            print("%-7s theta %4.1f | %-11s t %.2f: Fh %4.0f Fv %4.0f N | support %5.0f N.m | ankle plantarflexor heel/mid/ball %s | knee extensor %s"
                  % (lab, cc["theta_deg"], where, m["t"], m["Fh"], m["Fv"], m["support"],
                     "/".join("%.0f" % m["ankle"][k] for k in ("heel", "mid-foot", "ball")), "/".join("%.0f" % m["knee"][k] for k in ("heel", "mid-foot", "ball"))))
            print("        measured CoP (0.73 +- 0.05 of the foot): ankle plantarflexor %.0f [%.0f, %.0f], knee extensor %.0f [%.0f, %.0f], hip extensor from below %.0f N.m (%.2f N.m/kg; rung 2 top-down: 2.7 at the catch, 3.5 at MHF)"
                  % (m["ankle"]["measured"], m["ankle"]["measured-1SD"], m["ankle"]["measured+1SD"], m["knee"]["measured"], m["knee"]["measured+1SD"], m["knee"]["measured-1SD"],
                     m["hip"]["measured"], m["hip"]["measured"] / M))
        print("        peak support in the drive %.0f N.m at t %.2f s" % (cc["drive_peak_support"], cc["t_peak_support"]))


if __name__ == "__main__":
    main()
