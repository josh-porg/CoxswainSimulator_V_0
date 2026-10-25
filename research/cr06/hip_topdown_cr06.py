"""Step 4.3, rung 2 (rebuilt 2026-10-24): the hip moment of [CR06]'s woman, top-down from the handle.

    .venv/Scripts/python.exe research/cr06/hip_topdown_cr06.py

The original (cr06/hip_moment_cr06.py, hip_moment_terms_cr06.py, 2026-09-14) lived in a session scratchpad
and was lost; its results are recorded in docs/TRACKING.md ("Step 4.3, second rung"). This rebuild follows
that record and prints its numbers beside the recorded ones.

Free body: everything above the hip joint (head, three trunk segments, both arms), no foot data needed.
- Hip: on the hull frame, moved by her leg channel (her_stroke.Stroke.leg) from the package rower's catch
  position (women's table at 1.758 m); height fixed at the rower's seat height above the ankle.
- Trunk: one rigid link hip-to-shoulder, L = 0.570 m (the package rower's). Its angle from her back channel:
  shoulder minus hip, horizontally, x(t) = x_catch + back(t), a(t) = arccos(x / L), where x_catch is the
  rower's own value at its catch (-0.354 m, 128.4 deg). That is the record's rigid remap with K = her back
  travel / the model's = 0.771, written directly.
- Masses (de Leva women, 75 kg): head and the three trunk segments at the package rower's own fixed
  distances along the link; both arms as one point mass at the shoulder.
- Handle: both handles' measured force, horizontal, toward the stern, acting at the oarlock height
  (0.32 m in the hull frame; the record's lever was 0.240 m at the catch and 0.228 m at maximum force).
- Hull: her measured boat velocity, differentiated (the record used the model's hull, which follows hers to
  0.030 m/s rms).
Extension moment of both hips (+ = opening the hip, trunk rotating toward the bow), and its terms: handle,
gravity, inertia of the point masses, and the segments' spin.

Writes data/local/cr06_study/hip_topdown_cr06.json.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import her_stroke as H                                         # noqa: E402

L_TRUNK = 0.570
Z_HAND = 0.32                                                  # oarlock height, hull frame, m
RECORDED = {"catch": dict(phase=0.002, total=206, per_kg=2.74, handle=6, gravity=108, inertia=89 + 2.7, trunk_deg=127),
            "max handle force": dict(phase=0.262, total=258, per_kg=3.45, handle=125, gravity=70, inertia=61 + 2.5, trunk_deg=110),
            "drive peak": dict(phase=0.243, total=267, per_kg=3.56)}


def upper_body():
    """(name, mass, distance from the hip along the link, spin inertia about its own centre): the package rower's
    own placement (women's table, 75 kg, 1.758 m), read off its segment field; both arms as one point mass at
    the shoulder, as the record has it."""
    from coxswain.crew.kinematics import SEGMENT_ORDER
    r, _ = H.her_rower()
    P, _, _ = r.segment_state(0.0)
    j = r.joint_positions(0.0)
    u = (j["shoulder"] - j["hip"])[[0, 2]]; u = u / np.linalg.norm(u)
    masses = dict(zip(SEGMENT_ORDER, r.segment_masses))
    out = []
    for name in ("lower_trunk", "mid_trunk", "upper_trunk", "head"):
        d = float((P[SEGMENT_ORDER.index(name)] - j["hip"])[[0, 2]] @ u)
        L_seg = H.segment(name)[1]
        out.append((name, float(masses[name]), d, float(masses[name]) * (0.3 * L_seg) ** 2))
    m_arms = float(sum(v for k, v in masses.items() if "arm" in k))
    out.append(("arms", m_arms, L_TRUNK, 0.0))
    return out


def trunk_angle(S, x_catch):
    x = x_catch + (S.back - S.back[np.argmin(S.leg)])          # zero at her catch (leg channel minimum)
    return np.arccos(np.clip(x / L_TRUNK, -1, 1))


def moment(S, a, ad=None, add=None, hip=None):
    """Extension moment of both hips and its terms for trunk angle a (rad, from the bow direction, so 90 deg is
    upright and the catch is ~128 deg) and its rates (from a when not given).

    Euler about the moving hip for the free body above it, 2-D moments counter-clockwise (bow toward up):
        M_ccw (thighs on the trunk) + sum r x m g + r_hand x F_hand = sum r x m a_abs + sum I alpha
    Hip extension turns the trunk toward the bow (clockwise), so the extensor moment is M_ext = -M_ccw:
        M_ext = r_hand x F_hand  +  sum r x m g  -  sum r x m a_abs  -  sum I alpha
              =    handle      +   gravity    +     inertia       +    spin
    with F_hand = (-F, 0) (the handles pull the rower toward the stern) and g = (0, -9.81)."""
    hx, hz, hxdd = hip
    ad = S.vel(a) if ad is None else ad
    add = S.acc(a) if add is None else add
    cross = lambda rx, rz, fx, fz: rx * fz - rz * fx           # noqa: E731
    handle = cross(0.0, Z_HAND - hz, -S.F_handle * np.cos(np.radians(S.oar_deg)), 0.0)
    gravity = np.zeros_like(a); inertia = np.zeros_like(a); spin = np.zeros_like(a)
    for _, m, d, I in upper_body():
        rx, rz = d * np.cos(a), d * np.sin(a)
        ax = S.a_boat + hxdd + d * (-np.sin(a) * add - np.cos(a) * ad ** 2)     # hull + hip + rotation
        az = d * (np.cos(a) * add - np.sin(a) * ad ** 2)
        gravity += cross(rx, rz, 0.0, -m * H.G)
        inertia -= cross(rx, rz, m * ax, m * az)
        spin -= I * add
    ext = dict(handle=handle, gravity=gravity, inertia=inertia, spin=spin)
    ext["total"] = handle + gravity + inertia + spin
    return ext


def setup(S):
    r, b = H.her_rower()
    tm = np.linspace(0, b.timing.period, 600, endpoint=False)
    J = [r.joint_positions(float(x)) for x in tm]
    hipx = np.array([j["hip"][0] for j in J]); k = int(np.argmin(hipx - np.array([j["ankle"][0] for j in J])))
    x_catch = float(J[k]["shoulder"][0] - J[k]["hip"][0])
    hz = float(J[k]["hip"][2])
    hx = hipx[k] + (S.leg - S.leg.min())
    return x_catch, (hx, hz, S.acc(hx))



def main(lam_v=H.LAM_DEFAULT, lam_x=H.LAM_DEFAULT, verbose=True):
    S = H.Stroke(lam_v, lam_x)
    x_catch, hip = setup(S)
    a = trunk_angle(S, x_catch)
    ext = moment(S, a, hip=hip)
    out = dict(smoothing=[lam_v, lam_x], x_catch=x_catch, hip_height=hip[1], upper_body=[(n, round(m, 2), round(d, 3)) for n, m, d, _ in upper_body()],
               instants={})
    for label, rec in RECORDED.items():
        i = int(round(rec["phase"] * len(S.t))) % len(S.t)
        if label == "catch":
            i = S.k_landmark
        elif label == "max handle force":
            i = S.i_mhf
        elif label == "drive peak":
            drive = S.F_handle > 30
            i = int(np.argmax(np.where(drive, ext["total"], -1e9)))
        row = dict(t=float(S.t[i]), phase=float(S.t[i] / H.T), trunk_deg=float(np.degrees(a[i])),
                   **{k: float(v[i]) for k, v in ext.items()}, per_kg=float(ext["total"][i] / H.M), recorded=rec)
        out["instants"][label] = row
        if verbose:
            print("%-17s phase %.3f (recorded %.3f) trunk %5.1f deg (rec %s) | total %5.0f N.m = %.2f N.m/kg (rec %s, %s) | handle %4.0f (rec %s) gravity %4.0f (rec %s) inertia %4.0f + spin %.1f (rec %s)"
                  % (label, row["phase"], rec["phase"], row["trunk_deg"], rec.get("trunk_deg", "-"), row["total"], row["per_kg"], rec["total"], rec["per_kg"],
                     row["handle"], rec.get("handle", "-"), row["gravity"], rec.get("gravity", "-"), row["inertia"], row["spin"], rec.get("inertia", "-")))
    rec_min = float(ext["total"].min() / H.M)
    out["flexion_minimum_per_kg"] = rec_min
    out["series"] = dict(t=S.t[::7].round(4).tolist(), total=ext["total"][::7].round(1).tolist(), trunk_deg=np.degrees(a[::7]).round(2).tolist())
    if verbose:
        print("flexion minimum %.2f N.m/kg at phase %.3f (recorded -0.80 at 0.573)" % (rec_min, S.t[np.argmin(ext["total"])] / H.T))
    return out, S, a, ext, hip


if __name__ == "__main__":
    out, *_ = main()
    band = {}
    for lv in H.LAM_BAND:
        o, *_ = main(lv, lv, verbose=False)
        for k, v in o["instants"].items():
            band.setdefault(k, []).append(v["per_kg"])
    out["smoothing_band_per_kg"] = {k: [round(min(v), 2), round(max(v), 2)] for k, v in band.items()}
    print("smoothing band (N.m/kg):", out["smoothing_band_per_kg"])
    json.dump(out, open(os.path.join(H.OUT, "hip_topdown_cr06.json"), "w"), indent=1)
    print("written", os.path.join(H.OUT, "hip_topdown_cr06.json"))
