"""The crew's hold on its pace that the joints' own torque-velocity gives (SOURCES §221).

The hold of §220 scales the crew's force by 1 + beta (1 - v / v_ref). A muscle at a fixed activation supplies its own:
for a joint torque T = a C1 g(theta) f(omega), a fractional speed error d(omega)/omega changes the torque by
dT/T = (omega f'(omega) / f(omega)) d(omega)/omega, so

    beta_joint(omega) = -omega f'(omega) / f(omega),

independent of the activation. This evaluates it with Anderson et al. (2007)'s concentric hyperbola for hip and knee
extension (the drive's main movers; coxswain.crew.joint_muscles), at the joint speeds of the research body's drive on
[BR24]'s and [CR06]'s boats, and weights the two joints by their share of the joints' power capacity, |omega| T_max.

    python research/crew/hold_from_strength.py

Not counted: reflexes (the trunk's measured 40 ms loop, §208, would add to this), co-contraction, the arms (elbow
from Frey-Law 2012 is tabulated, shoulder unsourced). So the result is a floor on the hold a rower's muscles supply.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]

from coxswain.crew.joint_muscles import anderson                 # noqa: E402


def _angle(a, b, c):
    u = np.stack([a[0] - b[0], a[1] - b[1]])
    v = np.stack([c[0] - b[0], c[1] - b[1]])
    cos = (u * v).sum(0) / np.maximum(np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0), 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def beta(joint, omega, step=1e-3):
    """-omega f'(omega) / f(omega) for the concentric velocity factor; 0 at rest."""
    w = np.maximum(np.asarray(omega, float), 0.0)
    f = joint.velocity_factor(w)
    fp = (joint.velocity_factor(w + step) - joint.velocity_factor(np.maximum(w - step, 0.0))) / (
        np.where(w > step, 2 * step, step + w))
    return np.where(f > 1e-6, -w * fp / np.maximum(f, 1e-6), np.nan)


def athlete(name, sex):
    import rung2_clock as C2
    m = C2.R.V.ATHLETES[name][0]()
    boat = C2.R.boat_for(name, m, "on-water", drive_law=True, catch_posture="as-built")
    rower = boat.crew[0].rower
    mass, stature = float(rower.total_mass), float(rower.anthropometry.stature)   # as muscle_capacity.py reads them
    tm = boat.timing
    D = float(tm.drive_fraction) * float(tm.period)
    t = np.linspace(0.0, D, 400)
    ch = rower._chain(t)
    p = {k: (np.asarray(v[0].value, float), np.asarray(v[1].value, float)) for k, v in ch.items()
         if k in ("ankle", "knee", "hip", "shoulder")}
    knee_flex = np.pi - _angle(p["ankle"], p["knee"], p["hip"])          # 0 straight, + flexion
    trunk = np.arctan2(p["shoulder"][0] - p["hip"][0], p["shoulder"][1] - p["hip"][1])
    thigh = np.arctan2(p["knee"][0] - p["hip"][0], p["knee"][1] - p["hip"][1])
    hip_flex = np.pi - np.abs(trunk - thigh)                             # 0 straight, + flexion (sagittal)
    w_knee = -np.gradient(knee_flex, t)                                  # extension speed, + while extending
    w_hip = -np.gradient(hip_flex, t)
    group = "M18-25" if sex == "M" else "F18-25"
    KE, HE = anderson("KE", group, mass, stature), anderson("HE", group, mass, stature, passive=False)
    b_k, b_h = beta(KE, w_knee), beta(HE, w_hip)
    cap_k = np.maximum(w_knee, 0) * KE.max_active(knee_flex, np.maximum(w_knee, 0))
    cap_h = np.maximum(w_hip, 0) * HE.max_active(hip_flex, np.maximum(w_hip, 0))
    weight = cap_k + cap_h
    ok = weight > 1e-6
    combined = (np.nan_to_num(b_k) * cap_k + np.nan_to_num(b_h) * cap_h)[ok].sum() / weight[ok].sum()
    print("%s (%s, %.0f kg, %.2f m): knee extension speed peak %.1f rad/s, hip %.1f rad/s" % (
        name, group, mass, stature, np.max(w_knee), np.max(w_hip)))
    for frac in (0.2, 0.4, 0.6, 0.8):
        i = int(frac * (t.size - 1))
        print("  %.0f %% of the drive: knee %.1f rad/s beta %.2f; hip %.1f rad/s beta %.2f" % (
            100 * frac, w_knee[i], float(b_k[i]), w_hip[i], float(b_h[i])))
    print("  power-capacity-weighted beta over the drive: %.2f" % combined)
    print("  Anderson %s KE: C4 %.2f, C5 %.2f rad/s (75 / 50 %% of isometric); HE: C4 %.2f, C5 %.2f" % (
        group, KE.C4, KE.C5, HE.C4, HE.C5))
    return combined


def main():
    print("The hold the joints' own torque-velocity supplies (Anderson et al. 2007), research body's drive\n")
    athlete("br24", "M")
    print()
    athlete("cr06", "F")
    print("\nFor comparison: the hold that stops the bending shaft ringing in rung 2 (SOURCES §220): beta 1 ([CR06]),"
          " 2 ([BR24]).")


if __name__ == "__main__":
    main()
