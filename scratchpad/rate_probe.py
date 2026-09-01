"""Hold delivered power fixed, sweep rate, watch speed and surge swing."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coxswain.boats import catalog
from coxswain.hydro.resistance import hull_resistance
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator

def build(rate, scale):
    b = catalog.eight(rate=rate, rower_mass=74.8, rower_stature=1.86,
                      coxswain_mass=56.7)
    b.power_scales = np.full(b.n_seats, float(scale))
    return b

def steady(boat, guess, duration=26.0, dt=0.008):
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    r = sim.run(duration=duration, dt=dt, surge_speed=guess)
    t = np.asarray(r.time); v = np.hypot(*np.asarray(r.velocity)[:2])
    per = boat.timing.period
    cycles = int((0.5 * t[-1]) // per)
    keep = t >= t[-1] - cycles * per
    v = v[keep]
    return float(v.mean()), float(v.max() - v.min())

def delivered(boat, speed):
    sub = boat.mesh.submerged(np.array([0.,0.,boat.equilibrium_heave()]),
                              np.zeros(3), rho=boat.water.density,
                              gravity=9.80665, water_level=0.0)
    f,_ = hull_resistance(np.array([speed,0.,0.]), sub,
                          mean_wetted_length=boat.length, water=boat.water,
                          coefficients=boat.resistance)
    return abs(float(f[0])) * speed

TARGET = 8 * 0.80 * 313.0   # delivered watts, masters eight

def match(rate, guess):
    lo, hi = 0.10, 3.0
    for _ in range(11):
        s = 0.5*(lo+hi)
        b = build(rate, s)
        v, sw = steady(b, guess)
        p = delivered(b, v)
        if abs(p - TARGET) < 4.0: break
        if p < TARGET: lo = s
        else: hi = s
    return s, v, sw, p

print("delivered power held at %.0f W" % TARGET)
print("  %6s %8s %9s %9s %9s" % ("rate","scale","m/s","swing %","W"))
rows=[]
for rate in (26.,30.,34.,38.):
    s,v,sw,p = match(rate, 4.23)
    rows.append((rate,v))
    print("  %6.0f %8.3f %9.4f %8.1f%% %9.0f" % (rate,s,v,100*sw/v,p))
r = np.array([x[0] for x in rows]); v = np.array([x[1] for x in rows])
slope = np.polyfit(r, 100*np.log(v), 1)[0]
print()
print("  model: %+.3f %% per spm   |  Holt: +0.6 to +1.1 %% per spm" % slope)
