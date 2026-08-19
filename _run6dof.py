import numpy as np, time
from coxswain.boats import catalog
from coxswain.sim import RowingSimulator

b = catalog.eight(rate=32.0)
sim = RowingSimulator(b)
y0 = sim.initial_state(surge_speed=5.0)
print("initial state:", np.array2string(y0, precision=4))

# check the mass matrix is well conditioned before integrating
from coxswain.core.state import State
M = sim.mass_matrix(0.0, State.from_vector(y0))
w = np.linalg.eigvalsh(M)
print(f"\nmass matrix eigenvalues: {np.array2string(w, precision=1)}")
print(f"  positive definite: {w.min() > 0}   condition number {w.max()/w.min():.1f}")

fb = sim.breakdown(0.0, State.from_vector(y0))
print("\nforce breakdown at t=0 (absolute frame, N):")
for n in ("crew","oar","buoyancy","gravity","resistance","appendage"):
    print(f"  {n:12s} F={np.array2string(getattr(fb,n+'_force'),precision=1,suppress_small=True):>34s}  M={np.array2string(getattr(fb,n+'_moment'),precision=1,suppress_small=True)}")
print(f"  {'TOTAL':12s} F={np.array2string(fb.total_force(),precision=1,suppress_small=True):>34s}  M={np.array2string(fb.total_moment(),precision=1,suppress_small=True)}")

t0=time.time()
res = sim.run(duration=20.0, surge_speed=5.0)
print(f"\nintegrated 20 s in {time.time()-t0:.1f} s wall, {len(res.time)} steps, finite={res.is_finite}")
for k,v in res.summary().items(): print(f"  {k:26s} {v:+.4f}")
