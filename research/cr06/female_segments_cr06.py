"""The [CR06] athlete is a woman; the one-athlete test built her from de Leva's MALE table (the default).
Female table, stature re-anchored so the model's leg travel equals her measured 0.581 m, back-travel ratio
recomputed, then the smooth-body and whole-measured-body builds rerun. Compare with the recorded male-table rows:
  smooth body       4.087 m/s, v min 2.59, rms 0.203
  leg clock + back law & travel (K 0.776)  4.109 m/s, v min 2.93 @0.143, v max 5.09, rms 0.035
"""
import os, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "self_consistent_body_cr06.py")).read().split("grid = np.linspace")[0])

def her_boat(stature, sex="female"):
    base = catalog.single_scull(rate=RATE, rower_mass=75.0, rower_stature=stature)
    rig = build_sculling_rig(n_seats=1, spacing=1.22, stern_station=-0.35, span=0.80, oarlock_height=0.32, oar=CR06_OAR)
    return Boat(name="CR06 women's single", offsets=base.offsets, rig=rig, hull_mass=19.7, hull_inertia=base.hull_inertia,
                timing=base.timing, appendages=base.appendages, water=base.water,
                default_anthropometry=catalog.RowerAnthropometry(mass=75.0, stature=stature, sex=sex),
                force_profile=base.force_profile, oar_sweep=ARC)

def travels(boat):
    r = boat.crew[0].rower
    t = np.linspace(0, boat.timing.period, 800, endpoint=False)
    J = [r.joint_positions(float(x)) for x in t]
    return (np.ptp([j["hip"][0] - j["ankle"][0] for j in J]), np.ptp([j["shoulder"][0] - j["hip"][0] for j in J]))

lo, hi = 1.55, 2.00
for _ in range(14):
    mid = 0.5 * (lo + hi)
    leg, _b = travels(her_boat(mid))
    lo, hi = (mid, hi) if leg < 0.581 else (lo, mid)
stature = 0.5 * (lo + hi)
leg, back = travels(her_boat(stature))
K = 0.398 / back
m_leg, m_back = travels(her_boat(1.787, "male"))
print("female table: stature %.3f m gives leg travel %.3f (her 0.581), back travel %.3f (her 0.398), K %.3f" % (stature, leg, back, K))
print("male table at 1.787: leg %.3f back %.3f K %.3f" % (m_leg, m_back, 0.398 / m_back), flush=True)
b = her_boat(stature)
print("segment masses female:", dict(zip(("head", "up_tr", "mid_tr", "low_tr"), np.round(b.crew[0].rower.segment_masses[:4], 2))),
      "thigh", round(float(b.crew[0].rower.segment_masses[8]), 2), "shank", round(float(b.crew[0].rower.segment_masses[10]), 2), flush=True)

grid = np.linspace(0.0, 1.0, 400, endpoint=False)
v_meas_g = np.interp(grid * T_CR06, t_v, v_boat, period=T_CR06)
early = grid < 0.3
for label, warped, back_on, k_back in (("female, smooth body", False, False, 1.0),
                                       ("female, leg clock + back law & travel", True, True, K)):
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = research.apply(her_boat(stature))
        boat.power_scales = np.ones(boat.n_seats)
        r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
        sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch, coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        sim._torque = lambda slot, ang, state, t: 0.5 * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
        if warped:
            sim.crew_field = body_field(boat, back_on, k_back)
        run = sim.run_strokes(20, surge_speed=4.191, dt=integrators.estimate_step(T_CR06) / 4.0)
    finally:
        oardynamics.InertiaProfile.of = REAL_OF
    t, y = run.last_time, run.last_states
    frac = np.mod(t - t[0], T_CR06) / T_CR06
    o = np.argsort(frac)
    v_m = np.interp(grid, frac[o], np.hypot(y[6], y[7])[o], period=1.0)
    rms = float(np.sqrt(np.mean(((v_m - v_m.mean()) - (v_meas_g - v_meas_g.mean())) ** 2)))
    print("%-40s speed %.3f (%+.1f%%)  v min %.2f @%.3f  v max %.2f  rms %.3f  power %.1f W  last %s" % (
        label, run.settled_speed(), 100 * (run.settled_speed() / 4.191 - 1), v_m[early].min(), grid[early][np.argmin(v_m[early])],
        v_m.max(), rms, run.settled_power(), np.round([s.mean_speed for s in run.strokes[-3:]], 4)), flush=True)
