import numpy as np
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming
from coxswain.crew.kinematics import JointDrivenRower, RowerStation
anthro=RowerAnthropometry(mass=85.0,stature=1.88); timing=StrokeTiming(rate=30.0)
r=JointDrivenRower(anthro,RowerStation(x_ankle=0.0),timing)
n=600; ts=np.linspace(0,timing.period,n,endpoint=False)
ch=r._chain(ts)
def rel_speed(a,b):
    dx=ch[a][0].first-ch[b][0].first; dz=ch[a][1].first-ch[b][1].first
    return np.hypot(dx,dz)
drive = ts < timing.drive_duration
print("mean segment speed during the drive vs PMC12289236 'Olympic' @30 spm:")
print(f"  leg   (hip rel ankle)     {rel_speed('hip','ankle')[drive].mean():.2f} m/s   measured 1.15 +- 0.15")
print(f"  trunk (shoulder rel hip)  {rel_speed('shoulder','hip')[drive].mean():.2f} m/s   measured 1.34 +- 0.14")
print(f"  arm   (hand rel shoulder) {rel_speed('hand','shoulder')[drive].mean():.2f} m/s   measured 2.29 +- 0.14")
print(f"\nboat-relative crew CoM travel and the surge it implies for an eight:")
com=np.array([r.centre_of_mass(t)[0] for t in ts]); travel=com.max()-com.min()
crew=8*85.0; total=96.0+crew+55.0
hull_swing=crew*travel/total
print(f"  crew CoM travel {travel:.3f} m -> hull surge {hull_swing:.3f} m about the system CoM")
print(f"  implied hull speed fluctuation ~ +-{hull_swing/2*2*np.pi/timing.period:.2f} m/s")
print(f"  Formaggia Fig.14 (4x) shows Vx 4.5-6.5 m/s about ~5.5 -> +-1.0 m/s")
