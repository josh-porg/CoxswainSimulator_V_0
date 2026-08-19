import numpy as np
from coxswain.boats import catalog
from coxswain.crew.oarlock import OarForceProfile
from coxswain.sim import RowingSimulator
from coxswain.crew.kinematics import JointDrivenRower, JointAngles, RowerStation
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming

timing = StrokeTiming(32.0)
anthro = RowerAnthropometry(mass=88.0, stature=1.90)
print("crew CoM velocity range vs joint sequencing:")
best=None
for lab,ph in (("none",{"shank":0,"trunk":0,"upper_arm":0,"forearm":0}),
               ("mild",{"shank":0,"trunk":0.10,"upper_arm":0.18,"forearm":0.20}),
               ("strong",{"shank":0,"trunk":0.16,"upper_arm":0.26,"forearm":0.30}),
               ("very strong",{"shank":0,"trunk":0.22,"upper_arm":0.34,"forearm":0.38})):
    r = JointDrivenRower(anthro, RowerStation(x_ankle=0.0),
                         JointAngles.from_catch_finish(timing, joint_phases=ph))
    ts = np.linspace(0,timing.period,600,endpoint=False)
    com = np.array([r.centre_of_mass(t) for t in ts])
    # CoM velocity by mass-weighting the segment velocities
    vel = np.array([ (r.segment_masses[:,None]*r.segment_state(t)[1]).sum(0)/r.total_mass for t in ts])
    print(f"  {lab:12s} CoM travel {np.ptp(com[:,0]):.3f} m   CoM vel range {np.ptp(vel[:,0]):.3f} m/s"
          f"   -> predicted hull fluct {0.823*np.ptp(vel[:,0]):.2f} m/s")
print("\n  measured for an eight: hull speed fluctuates ~1.2-1.5 m/s peak-to-peak")
