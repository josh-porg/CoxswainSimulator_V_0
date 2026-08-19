import numpy as np
from coxswain.hydro.hull import parametric_offsets, HullMesh
from coxswain.hydro.resistance import hull_resistance, FRESH_WATER, friction_coefficient

for fullness,beam,depth,label in ((2.2,0.57,0.16,"fullness 2.2"),(3.5,0.57,0.16,"fullness 3.5"),(5.0,0.55,0.155,"fullness 5.0")):
    off = parametric_offsets(17.3, beam, depth, fullness=fullness)
    mesh = HullMesh(off)
    mass = 96.0+8*85.0+55.0
    try: hz = mesh.equilibrium_heave(mass, rho=FRESH_WATER.density)
    except ValueError as e: print(f"{label}: {e}"); continue
    p = mesh.submerged(np.array([0,0,hz]), np.zeros(3), rho=FRESH_WATER.density)
    print(f"\n{label}: beam {beam} depth {depth}")
    print(f"  wetted {p.wetted_area:.2f} m^2  transverse {p.transverse_area:.4f}  plan {p.plan_area:.2f}  lateral {p.lateral_area:.2f}")
    for V in (4.5, 5.5, 6.0):
        f,b = hull_resistance(np.array([V,0,0]), p, 17.3, FRESH_WATER)
        print(f"  V={V} m/s: shape {b['shape']:6.1f}  visc {b['viscous']:6.1f}  wave {b['wave']:6.1f}  TOTAL {b['total_longitudinal']:6.1f} N  power {b['total_longitudinal']*V/1000:.2f} kW  Cf={b['friction_coefficient']:.5f}")

print("\n--- published reference points ---")
print("  eight at 5.5-5.9 m/s: hull drag ~ 400-500 N, hull power ~ 2.3-2.8 kW")
print("  (crew ~8 x 450 W = 3.6 kW, of which ~70% reaches the hull)")
print("\n--- ITTC vs the legacy natural-log form ---")
Re = 5.5*17.3/1.139e-6
print(f"  Re={Re:.3e}  C_f(ITTC log10)={friction_coefficient(Re):.5f}  C_f(legacy ln)={0.075/(np.log(Re)-2)**2:.5f}  ratio {friction_coefficient(Re)/(0.075/(np.log(Re)-2)**2):.2f}x")
