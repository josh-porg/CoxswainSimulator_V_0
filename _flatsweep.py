import numpy as np
from coxswain.crew.stroke import StrokeTiming, FourierProfile
from coxswain.crew.kinematics import JointDrivenRower, JointAngles, RowerStation, DEFAULT_JOINT_ANGLES, DEFAULT_JOINT_PHASES
from coxswain.crew.anthropometry import RowerAnthropometry
timing = StrokeTiming(32.0); anthro = RowerAnthropometry(mass=88.0, stature=1.90)
print("flatness sweep (eight crew, rate 32):")
for f in (0.0, 0.5, 0.65, 0.75, 0.85, 0.95):
    built={}
    for n,(c,fi) in DEFAULT_JOINT_ANGLES.items():
        from coxswain.crew.kinematics import _shift_phase
        pr = FourierProfile.from_catch_finish(np.radians(c), np.radians(fi), timing, flatness=f)
        built[n]=_shift_phase(pr, DEFAULT_JOINT_PHASES[n])
    r = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), JointAngles(**built))
    ts=np.linspace(0,timing.period,600,endpoint=False)
    com=np.array([r.centre_of_mass(t) for t in ts])
    vel=np.array([(r.segment_masses[:,None]*r.segment_state(t)[1]).sum(0)/r.total_mass for t in ts])
    amax=max(np.abs(r.segment_state(t)[2]).max() for t in ts)
    print(f"  flatness={f:.2f}  CoM travel {np.ptp(com[:,0]):.3f} m  CoM vel range {np.ptp(vel[:,0]):.3f} m/s"
          f"  -> hull fluct ~{0.823*np.ptp(vel[:,0]):.2f} m/s   max seg accel {amax:5.1f} m/s^2")
print("\n  target hull fluctuation for an eight: 1.2-1.5 m/s")
