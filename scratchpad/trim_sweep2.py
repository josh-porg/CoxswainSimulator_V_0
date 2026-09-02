"""Same sweep, but WITH the phase authority the test uses."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coxswain.boats import catalog
from coxswain.crew.balance import PhaseAuthority
from coxswain.crew.trim import StrokeTrim
from coxswain.sim.control import BalanceController, Coxswain
from coxswain.sim.simulator import RowingSimulator

def run(eight, trim, n_strokes=14, dt=0.009):
    auth = PhaseAuthority.from_boat(eight)
    ctl = BalanceController(authority=auth, timing=eight.timing, trim=trim)
    sim = RowingSimulator(eight, coxswain=Coxswain(balance=ctl))
    state = sim.initial_state(surge_speed=4.6)
    period = eight.timing.period
    swings = []
    for k in range(n_strokes):
        r = sim.run(duration=period, dt=dt, initial_state=state)
        state = np.asarray(r.states)[:, -1]
        roll = np.degrees(np.asarray(r.roll))
        swings.append(roll.max() - roll.min())
        if trim is not None:
            trim.update(np.asarray(r.time)+k*period, np.asarray(r.roll), eight.timing)
    return np.array(swings)

eight = catalog.eight(rate=32.0)
auth = PhaseAuthority.from_boat(eight)
period = eight.timing.period
w = [auth.window(t, eight.timing) for t in np.linspace(0, period, 24, endpoint=False)]
print("authority window over the stroke: min %.0f max %.0f N m" % (min(w), max(w)))
base = run(eight, None)
print("no trim      : final %.3f deg" % base[-3:].mean())
for gain in (250., 500., 1000., 2000., 4000.):
    t = StrokeTrim(gain=gain)
    s = run(eight, t)
    print("gain %6.0f  : final %.3f  first %.3f  learned RMS %.0f N m  %s"
          % (gain, s[-3:].mean(), s[0], t.effort,
             "DIVERGING" if s[-1] > s[0] else "converging"))
