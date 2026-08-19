import numpy as np
# Caplan & Gardner (2010) Table II, stretcher position 1 (standard), 9 male
# university rowers @ 30 spm.  Their shank angle is measured from the ground
# on the STERN side, so my bow-referenced angle is 180 - theirs.
caplan = {              # catch, mid-drive, finish, mid-recovery
    "shank_theirs": [91.6, 140.4, 168.5, 141.2],
    "knee":         [41.0, 125.9, 171.2, 127.2],
    "hip":          [18.7,  65.7, 109.1,  51.1],
    "trunk":        [-38.1, -9.4,  16.6, -24.8],
}
a_s = 180.0 - np.array(caplan["shank_theirs"])   # shank direction from +x (bow)
knee = np.array(caplan["knee"])
a_t = knee - 180.0 + a_s                          # thigh direction from +x
print("derived link direction angles (deg, from bow axis, +ve = up):")
print(f"  shank {np.round(a_s,1)}")
print(f"  thigh {np.round(a_t,1)}")

print("\ncross-check: hip angle = angle between thigh and trunk long axes")
trunk_from_bow = 90.0 - np.array(caplan["trunk"])  # trunk from vertical -> from bow
print(f"  trunk from bow axis {np.round(trunk_from_bow,1)}")
hip_pred = trunk_from_bow - a_t
print(f"  predicted hip angle {np.round(hip_pred,1)}")
print(f"  measured  hip angle {np.round(np.array(caplan['hip']),1)}")
print(f"  residual            {np.round(hip_pred-np.array(caplan['hip']),1)}  <- tests whether the 4 angle sets are one consistent linkage")

print("\nhip height above ankle implied at each keyframe, for a range of limb ratios:")
for Ls,Lt,label in [(0.469,0.456,'my de Leva 1.88 m rower'),(0.44,0.43,'shorter'),(0.49,0.47,'taller')]:
    h = Ls*np.sin(np.radians(a_s)) + Lt*np.sin(np.radians(a_t))
    print(f"  {label:24s} Ls={Ls} Lt={Lt}: {np.round(h,3)}  spread {h.max()-h.min():.3f} m")

# least squares: find Lt/Ls and h minimising the spread
from scipy.optimize import minimize_scalar
def spread(ratio):
    Ls=0.469; Lt=ratio*Ls
    h = Ls*np.sin(np.radians(a_s)) + Lt*np.sin(np.radians(a_t))
    return h.std()
r = minimize_scalar(spread, bounds=(0.7,1.4), method='bounded')
Ls=0.469; Lt=r.x*Ls
h = Ls*np.sin(np.radians(a_s)) + Lt*np.sin(np.radians(a_t))
print(f"\nbest-fit thigh/shank ratio {r.x:.3f} (de Leva gives {0.456/0.469:.3f})")
print(f"  -> hip heights {np.round(h,3)}, mean {h.mean():.3f}, spread {h.max()-h.min():.3f} m")
print("\nseat travel implied (hip x range):")
x = Ls*np.cos(np.radians(a_s)) + Lt*np.cos(np.radians(a_t))
print(f"  hip x at keyframes {np.round(x,3)} -> travel {x.max()-x.min():.3f} m")
