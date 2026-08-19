import numpy as np
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming
from coxswain.crew.kinematics import JointDrivenRower, RowerStation, SEGMENT_ORDER, DEFAULT_ARM_POSTURE
anthro = RowerAnthropometry(mass=85.0, stature=1.88); timing = StrokeTiming(rate=32.0)
r = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
L = r.upper_arm_length + r.forearm_length
print("arm IK at keyframes (deg):")
for name in ("catch","mid_drive","finish","mid_recovery"):
    ext, elev = DEFAULT_ARM_POSTURE[name]
    b=np.radians(elev); dx,dz = ext*L*np.cos(b), ext*L*np.sin(b)
    ua, fa = r._solve_arm_angles(dx,dz,name)
    print(f"  {name:13s} upper_arm={ua:8.2f}  forearm={fa:8.2f}")
ts=np.linspace(0,timing.period,400,endpoint=False)
ua=np.degrees(r.joint_angles.upper_arm(ts).value); fa=np.degrees(r.joint_angles.forearm(ts).value)
print(f"\nfitted profiles over stroke: upper_arm {ua.min():.1f}..{ua.max():.1f}   forearm {fa.min():.1f}..{fa.max():.1f}")
amax=np.zeros(len(SEGMENT_ORDER)); vmax=np.zeros(len(SEGMENT_ORDER))
for t in ts:
    _,v,a=r.segment_state(t)
    amax=np.maximum(amax,np.abs(a).max(axis=1)); vmax=np.maximum(vmax,np.abs(v).max(axis=1))
print("\nper-segment max |vel| and |accel|:")
for n,vm,am in zip(SEGMENT_ORDER,vmax,amax): print(f"  {n:24s} {vm:7.2f} m/s   {am:8.2f} m/s^2")
