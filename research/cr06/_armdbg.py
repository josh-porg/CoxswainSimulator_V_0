import os, warnings
import numpy as np
warnings.simplefilter("ignore")
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "cr06_study"))   # its data and outputs (gitignored)
MARKER = "\n" + "grid = np.linspace(0.0, 1.0, 400"
exec(open(os.path.join(CODE, "female_segments_cr06.py")).read().split(MARKER)[0])

boat = research.apply(her_boat(stature))
r = boat.crew[0].rower


def vec(p):
    return np.array([float(getattr(c, "value", c)) for c in np.atleast_1d(p)], dtype=float)


side = sorted(r._hand_tracks)[0]
print("sides", sorted(r._hand_tracks), " using", side)
lock = boat.rig.seats[0].oarlocks[0]
print("lock position", np.round(np.asarray(lock.position, dtype=float), 3),
      "inboard", round(float(lock.oar.inboard), 3), "side", lock.side)
print("arm: upper %.3f  forearm+hand %.3f  total %.3f"
      % (r.upper_arm_length, r.forearm_length, r.upper_arm_length + r.forearm_length))
for t in (0.0, 0.35, 0.897):
    h = vec(r._hand_tracks[side](t))
    s = vec(r._shoulder_tracks[side](t))
    J = r.joint_positions(t)
    e = vec(r._elbow_tracks[side](t))
    print()
    print("t=%.3f" % t)
    print("  hand track      ", np.round(h, 3))
    print("  elbow track     ", np.round(e, 3))
    print("  shoulder track  ", np.round(s, 3))
    print("  jp shoulder", np.round(J["shoulder"], 3), " jp hand", np.round(J["hand"], 3))
    print("  jp hip     ", np.round(J["hip"], 3), " jp ankle", np.round(J["ankle"], 3))
    print("  |hand-shoulder| = %.3f ;  |sh-elbow| %.3f + |elbow-hand| %.3f = %.3f"
          % (np.linalg.norm(h - s), np.linalg.norm(e - s), np.linalg.norm(h - e),
             np.linalg.norm(e - s) + np.linalg.norm(h - e)))

print()
print("=== how far short is the catch reach? ===")
h = vec(r._hand_tracks[side](0.0)); s = vec(r._shoulder_tracks[side](0.0))
L = r.upper_arm_length + r.forearm_length
drop = abs(h[2] - s[2])
need = float(np.sqrt(max(L ** 2 - drop ** 2, 0.0)))
have = float(np.hypot(h[0] - s[0], h[1] - s[1]))
print("  handle height %.3f, shoulder height %.3f -> vertical drop %.3f m" % (h[2], s[2], drop))
print("  horizontal reach needed for a straight arm %.3f m; actual %.3f m; short by %.3f m"
      % (need, have, need - have))
print("  the oar cannot make it up: handle is at r_h sin(theta) = %.3f m aft of the pin;"
      % (float(lock.oar.inboard) * np.sin(float(boat.oar_sweep.catch_angle))))
print("  another %.3f m would need sin(theta) = %.2f" % (need - have,
      (float(lock.oar.inboard) * np.sin(float(boat.oar_sweep.catch_angle)) + (need - have)) / float(lock.oar.inboard)))
print()
print("  geometry that sets it: pin x %.3f, ankle x %.3f -> pin is %.3f m bow-ward of the feet"
      % (float(np.asarray(lock.position, dtype=float)[0]), float(r.joint_positions(0.0)["ankle"][0]),
         float(np.asarray(lock.position, dtype=float)[0]) - float(r.joint_positions(0.0)["ankle"][0])))
print("  at the catch the handle sits %.3f m STERN-ward of the feet."
      % (float(r.joint_positions(0.0)["ankle"][0]) - h[0]))
