import numpy as np, time
from coxswain.boats import catalog
from coxswain.sim import RowingSimulator

print(f"{'boat':18s} {'rate':>5s} {'mean u':>8s} {'fluct':>7s} {'ratio':>7s} {'pitch':>7s} {'roll':>7s} {'yaw':>7s} {'heave':>8s}")
print("-"*82)
t0=time.time()
for name, rates in (("8+",(24.0,32.0,38.0)), ("4+",(32.0,)), ("1x",(20.0,30.0))):
    for r in rates:
        b = catalog.build(name, rate=r)
        sim = RowingSimulator(b)
        res = sim.run(duration=24.0, surge_speed=4.5, dt=0.005)
        s = res.summary()
        print(f"{b.name:18s} {r:5.0f} {s['mean_speed']:8.3f} {s['speed_fluctuation']:7.3f} "
              f"{100*s['speed_fluctuation_ratio']:6.1f}% {s['pitch_amplitude_deg']:7.3f} "
              f"{s['roll_amplitude_deg']:7.3f} {s['yaw_amplitude_deg']:7.3f} {s['heave_amplitude']:8.4f}")
print(f"\n({time.time()-t0:.0f} s wall)")
print("\nPublished reference values:")
print("  eight  @32: 5.2-5.5 m/s   @38 (race): 5.8-6.1 m/s")
print("  4+     @32: 4.7-5.0 m/s")
print("  1x     @30: 4.2-4.6 m/s")
print("  Formaggia Fig.13 (1x @39.5): pitch |phi|<0.02 rad = 1.15 deg, heave |Xz|<0.08 m")
