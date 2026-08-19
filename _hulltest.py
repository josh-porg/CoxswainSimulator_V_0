import numpy as np
from coxswain.hydro.hull import parametric_offsets, HullMesh

# an eight: 17.3 m, 0.57 m waterline beam, 0.16 m draft
off = parametric_offsets(length=17.3, max_beam=0.57, max_depth=0.16)
mesh = HullMesh(off)
print(f"panels {mesh.n_panels}, total area {mesh.total_area:.2f} m^2")
print(f"design volume {off.design_volume():.4f} m^3 -> displaces {off.design_displacement():.1f} kg")

mass = 96.0 + 8*85.0 + 55.0    # hull + 8 rowers + cox
print(f"\ntarget all-up mass {mass:.1f} kg")
hz = mesh.equilibrium_heave(mass)
print(f"equilibrium heave G_z = {hz:+.4f} m")
p = mesh.submerged(np.array([0,0,hz]), np.zeros(3))
print(f"  wetted area      {p.wetted_area:8.3f} m^2   (published ~12-14 for an eight)")
print(f"  transverse area  {p.transverse_area:8.4f} m^2  (~beam*draft/2 ~ 0.05)")
print(f"  plan area        {p.plan_area:8.3f} m^2   (~waterplane, ~ 0.7*L*B = 6.9)")
print(f"  volume           {p.volume:8.4f} m^3   (need {mass/1025:.4f})")
print(f"  buoyancy force   {p.buoyancy_force} N  (weight {mass*9.81:.0f})")
print(f"  submerged frac   {p.submerged_fraction:.3f}")

print("\n--- restoring behaviour ---")
for dz in (-0.02, 0.0, 0.02):
    q = mesh.submerged(np.array([0,0,hz+dz]), np.zeros(3))
    print(f"  heave {dz:+.3f} m -> Fz {q.buoyancy_force[2]-mass*9.81:+9.1f} N net")
for ang,name in ((np.radians(2.0),'pitch'),(np.radians(5.0),'roll')):
    att = np.array([ang,0,0]) if name=='roll' else np.array([0,ang,0])
    q = mesh.submerged(np.array([0,0,hz]), att)
    idx = 1 if name=='roll' else 1
    print(f"  {name} {np.degrees(ang):+.1f} deg -> moment {q.buoyancy_moment} N m")

print("\n--- convergence with panel count ---")
for ng in (12, 24, 48, 96):
    m = HullMesh(off, n_girth=ng)
    h = m.equilibrium_heave(mass)
    pp = m.submerged(np.array([0,0,h]), np.zeros(3))
    print(f"  n_girth={ng:3d}  heave={h:+.5f}  wetted={pp.wetted_area:.3f}  vol={pp.volume:.4f}")
