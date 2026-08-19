import numpy as np
from coxswain.boats import catalog
from coxswain.sim import RowingSimulator
from coxswain.core.state import State

b = catalog.eight(rate=32.0)
sim = RowingSimulator(b)
h,p = b.trim_attitude(0.0)

print("=== static roll stability about G_h ===")
mass,pos,_,_ = b.crew_field(0.0)
crew_z = (mass*pos[:,2]).sum()/mass.sum()
print(f"crew CoM height above G_h: {crew_z:+.3f} m, crew+cox mass {mass.sum():.0f} kg")
upset = mass.sum()*9.81*crew_z
print(f"crew weight upsetting moment slope: {upset:+8.0f} N m / rad")

for roll_deg in (1.0, 3.0):
    att = np.array([np.radians(roll_deg), p, 0.0])
    pr = b.mesh.submerged(np.array([0,0,h]), att, rho=b.water.density)
    print(f"roll {roll_deg}deg: hydrostatic moment_x {pr.buoyancy_moment[0]:+8.1f} N m"
          f"  -> stiffness {pr.buoyancy_moment[0]/np.radians(roll_deg):+8.0f} N m/rad")
hyd = None
att = np.array([np.radians(1.0), p, 0.0])
hyd = b.mesh.submerged(np.array([0,0,h]), att, rho=b.water.density).buoyancy_moment[0]/np.radians(1.0)
print(f"\nNET roll stiffness = {hyd:+.0f} (hydrostatic) + {upset:+.0f} (crew weight) = {hyd+upset:+.0f} N m/rad")
print("  positive => STATICALLY UNSTABLE in roll (capsizes)" if hyd+upset>0 else "  stable")

print("\n=== net yaw moment from the sweep rig ===")
tot=0
for member, seat in zip(b.crew, b.rig.seats):
    lock = seat.oarlocks[0]
    tot += lock.side*lock.position[0]
print(f"sum(side * x_oarlock) = {tot:+.2f} m  -> a persistent yaw couple each drive")

print("\n=== early growth with everything on ===")
y = sim.initial_state(surge_speed=5.0)
res = sim.run(duration=3.0, initial_state=y, dt=0.01)
for i in range(0, len(res.time), 60):
    print(f"  t={res.time[i]:4.2f}  roll={np.degrees(res.roll[i]):+8.3f}  pitch={np.degrees(res.pitch[i]):+7.3f}"
          f"  yaw={np.degrees(res.yaw[i]):+8.3f}  heave={res.heave[i]:+.4f}  u={res.surge_speed[i]:+.3f}")
