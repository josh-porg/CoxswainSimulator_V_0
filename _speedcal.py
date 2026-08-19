import numpy as np
from coxswain.boats import catalog
from coxswain.crew.oarlock import OarForceProfile
from coxswain.sim import RowingSimulator

print("steady speed vs peak oarlock force (eight, rate 32):")
for fmax in (800.0, 950.0, 1100.0):
    b = catalog.eight(rate=32.0, force_profile=OarForceProfile(max_x=fmax))
    sim = RowingSimulator(b)
    res = sim.run(duration=25.0, surge_speed=5.0, dt=0.005)
    s = res.summary()
    print(f"  F_max_x={fmax:6.0f} N -> mean u {s['mean_speed']:.3f} m/s, "
          f"fluct {s['speed_fluctuation']:.3f} ({100*s['speed_fluctuation_ratio']:.1f}%), "
          f"pitch {s['pitch_amplitude_deg']:.3f}deg roll {s['roll_amplitude_deg']:.3f}deg "
          f"heave {s['heave_amplitude']:.4f}m")
print("\ntarget: eight at rate 32 cruises ~5.2-5.5 m/s; speed fluctuation 8-15%")
