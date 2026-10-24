"""The coxswain's felt motion, resolved into boat coordinates.

The camera is head-mounted, so its IMU is a proxy for the vestibular input: what the
coxswain's inner ear actually receives. That is the wrong signal for measuring the boat
(head motion contaminates it) but the right signal for asking what the coxswain is
responding to, because it is literally what they feel.

The quaternion puts acceleration in world axes; GPS heading per stroke then rotates
world into boat axes, giving surge, sway and heave. Angular rates come from the
quaternion derivative and are the semicircular-canal input.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

d = np.load("imu.npz"); it, acc, q = d["t"], d["acc"], d["quat"]
R = np.load("race.npz")
gt, lat, lon = R["t"], R["lat"], R["lon"]
fs = 1 / np.median(np.diff(it))


def rot(q, v):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    qv = np.stack([x, y, z], 1); tt = 2 * np.cross(qv, v)
    return v + w[:, None] * tt + np.cross(qv, tt)


aw = rot(q, acc)                       # world frame, gravity on -Z
aw = aw - aw.mean(0)                   # remove gravity and any bias

# boat heading from the GPS track, per stroke, then interpolated to IMU time
mlat = np.radians(lat.mean())
ex = np.radians(lon) * np.cos(mlat) * 6371000.0
ey = np.radians(lat) * 6371000.0
hx = np.gradient(ex); hy = np.gradient(ey)
hd = np.unwrap(np.arctan2(hy, hx))
hd_i = np.interp(it, gt, hd)
ch, sh = np.cos(hd_i), np.sin(hd_i)
surge = aw[:, 0] * ch + aw[:, 1] * sh          # along the boat
sway = -aw[:, 0] * sh + aw[:, 1] * ch          # across it
heave = aw[:, 2]

# angular rate in head frame: the semicircular-canal signal
def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], 1)
qc = q.copy(); qc[:, 1:] *= -1
om = 2 * qmul(qc[:-1], q[1:])[:, 1:] / np.diff(it)[:, None]
omt = it[:-1]

CH = {"surge": (it, surge), "sway": (it, sway), "heave": (it, heave),
      "wroll": (omt, om[:, 0]), "wpitch": (omt, om[:, 1]), "wyaw": (omt, om[:, 2])}


def per_stroke(tt, x, gt):
    """sd, peak-to-peak and mean|.| of a channel within each stroke interval."""
    out = np.zeros((len(gt), 3))
    edges = np.concatenate([[gt[0] - 2.0], (gt[:-1] + gt[1:]) / 2, [gt[-1] + 2.0]])
    for i in range(len(gt)):
        m = (tt >= edges[i]) & (tt < edges[i + 1])
        if m.sum() < 4: out[i] = np.nan; continue
        s = x[m]
        out[i] = (s.std(), s.max() - s.min(), np.abs(s).mean())
    return out


if __name__ == "__main__":
    print("heading range %.0f deg; IMU %.1f Hz over %.0f s" % (np.degrees(np.ptp(hd)), fs, it[-1]))
    for k, (tt, x) in CH.items():
        print("  %-6s sd %.4f  (per-stroke sd of sd: %.4f)"
              % (k, x.std(), np.nanstd(per_stroke(tt, x, gt)[:, 0])))

F, FN = [], []
for k, (tt, x) in CH.items():
    ps = per_stroke(tt, x, gt)
    for j, lbl in enumerate(("sd", "ptp", "absmean")):
        F.append(ps[:, j]); FN.append("%s %s" % (k, lbl))
VEST = np.column_stack(F)
med = np.nanmedian(VEST, axis=0)
ix = np.where(np.isnan(VEST))
VEST[ix] = np.take(med, ix[1])
