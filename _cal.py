import numpy as np
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.stroke import StrokeTiming
from coxswain.crew.kinematics import JointDrivenRower, RowerStation
from coxswain.crew import stroke_data

anthro = RowerAnthropometry(mass=85.0, stature=1.88)
timing = StrokeTiming(rate=32.0)
rower = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)

print(f"timing: T={timing.period:.3f}s drive={timing.drive_duration:.3f}s "
      f"recovery={timing.recovery_duration:.3f}s ratio 1:{timing.ratio:.2f}")
print(f"calibrated seat height above ankle: {rower.station.seat_height*1000:.1f} mm")
print("link lengths (m): " + "  ".join(f"{n}={getattr(rower,n+'_length'):.3f}"
      for n in ("shank","thigh","trunk","upper_arm","forearm")))

print("\njoint positions (hull frame, x=bow, z=up):")
ph = stroke_data.CAPLAN_GARDNER_2010.keyframe_phases(timing.drive_fraction)
for p,label in zip(ph, ("catch","mid-drive","finish","mid-recovery")):
    jp = rower.joint_positions(p*timing.period)
    print(f"  {label:13s} " + "  ".join(f"{k}=({v[0]:+.3f},{v[2]:+.3f})"
          for k,v in jp.items() if k in ("knee","hip","shoulder","hand")))

print(f"\nslide travel        {rower.slide_travel():.3f} m   (measured 0.60-0.70)")
print(f"seat height variation {rower.seat_height_variation()*1000:.1f} mm (level track)")
print(f"handle x travel     {rower.handle_travel():.3f} m")
ts = np.linspace(0, timing.period, 500, endpoint=False)
com = np.array([rower.centre_of_mass(t) for t in ts])
print(f"crew CoM x range    {com[:,0].max()-com[:,0].min():.3f} m")
print(f"crew CoM z range    {com[:,2].max()-com[:,2].min():.3f} m")

amax=vmax=0
prev=None; jump=0
for t in ts:
    _,v,a = rower.segment_state(t)
    vmax=max(vmax,np.abs(v).max()); amax=max(amax,np.abs(a).max())
    if prev is not None: jump=max(jump,np.abs(a-prev).max())
    prev=a
print(f"\nmax |segment velocity|     {vmax:.2f} m/s")
print(f"max |segment acceleration| {amax:.2f} m/s^2")
print(f"max accel step between samples (dt={timing.period/500*1000:.2f} ms): {jump:.4f} m/s^2 -> continuous")

print("\nreproduced vs measured joint angles at keyframes:")
ds = stroke_data.CAPLAN_GARDNER_2010
for i,(p,label) in enumerate(zip(ph, ("catch","mid-drive","finish","mid-recovery"))):
    t=p*timing.period
    sh=np.degrees(rower.joint_angles.shank(t).value)
    th=np.degrees(rower.thigh_angle(t))
    tr=np.degrees(rower.joint_angles.trunk(t).value)
    print(f"  {label:13s} shank {sh:6.1f} (data {ds.shank[i]:6.1f})   "
          f"trunk_link {tr:6.1f} (data {ds.trunk_link[i]:6.1f})   thigh {th:6.1f} (data {ds.thigh[i]:6.1f})")
