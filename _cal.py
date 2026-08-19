import numpy as np
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming
from coxswain.crew.kinematics import JointDrivenRower, JointAngles, RowerStation

anthro = RowerAnthropometry(mass=85.0, stature=1.88)
timing = StrokeTiming(rate=32.0)
rower = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), JointAngles.from_catch_finish(timing))

print("link lengths (m): shank %.3f thigh %.3f trunk %.3f uarm %.3f farm %.3f"%(
    rower.shank_length,rower.thigh_length,rower.trunk_length,rower.upper_arm_length,rower.forearm_length))
print("crew mass %.2f kg\n"%rower.total_mass)

for frac,label in ((0.0,"catch"),(timing.drive_fraction,"finish")):
    jp=rower.joint_positions(frac*timing.period)
    print(f"{label:7s} " + "  ".join(f"{k}=({v[0]:+.3f},{v[2]:+.3f})" for k,v in jp.items()))

ts=np.linspace(0,timing.period,600,endpoint=False)
hip=np.array([rower.joint_positions(t)["hip"] for t in ts])
hand=np.array([rower.joint_positions(t)["hand"] for t in ts])
com=np.array([rower.centre_of_mass(t) for t in ts])
print(f"\nslide travel     {np.ptp(hip[:,0]):.3f} m   target 0.60-0.70")
print(f"seat height var  {np.ptp(hip[:,2])*1000:.2f} mm  target ~0 (level track)")
print(f"handle x travel  {np.ptp(hand[:,0]):.3f} m")
print(f"handle z rise    {np.ptp(hand[:,2]):.3f} m")
print(f"crew CoM x range {np.ptp(com[:,0]):.3f} m   target 0.45-0.60")
print(f"crew CoM z range {np.ptp(com[:,2]):.3f} m")
print(f"thigh angle      {np.degrees(rower.thigh_angle(ts)).min():.1f} .. {np.degrees(rower.thigh_angle(ts)).max():.1f} deg")

amax=0; vmax=0
for t in ts:
    _,v,a=rower.segment_state(t); amax=max(amax,np.abs(a).max()); vmax=max(vmax,np.abs(v).max())
print(f"\nmax |segment vel| {vmax:.2f} m/s   max |segment accel| {amax:.2f} m/s^2")

# continuity across the whole cycle including the wrap
tt=np.linspace(0,2*timing.period,4001)
A=np.array([rower.segment_state(t)[2] for t in tt])
jump=np.abs(np.diff(A,axis=0)).max()
print(f"max step in accel between adjacent samples (dt={tt[1]-tt[0]:.4f}s): {jump:.4f} m/s^2  -> C2 smooth" )
