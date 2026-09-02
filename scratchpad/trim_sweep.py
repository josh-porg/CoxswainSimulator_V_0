"""Find where the ILC loop gain actually converges."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coxswain.boats import catalog
from coxswain.crew.trim import StrokeTrim
from coxswain.sim.control import BalanceController, Coxswain
from coxswain.sim.simulator import RowingSimulator

def run(eight, trim, n_strokes=14, dt=0.004):
    cox = Coxswain(rudder_override=lambda t, s: 0.0)
    cox.balance = BalanceController(timing=eight.timing, trim=trim)
    sim = RowingSimulator(eight, coxswain=cox)
    state = sim.initial_state(surge_speed=4.6)
    period = eight.timing.period
    swings = []
    for k in range(n_strokes):
        r = sim.run(duration=period, dt=dt, initial_state=state)
        state = np.asarray(r.states)[:, -1]
        roll = np.degrees(np.asarray(r.roll))
        swings.append(roll.max() - roll.min())
        if trim is not None:
            trim.update(np.asarray(r.time) + k*period,
                        np.asarray(r.roll), eight.timing)
    return np.array(swings)

eight = catalog.eight(rate=32.0)
base = run(eight, None)
print("no trim            : final %.3f deg" % base[-3:].mean())
for gain in (500., 1000., 1500., 2000., 3000., 4000.):
    s = run(eight, StrokeTrim(gain=gain))
    print("gain %6.0f        : final %.3f deg  (first %.3f)  %s"
          % (gain, s[-3:].mean(), s[0], "DIVERGING" if s[-1] > s[0] else "converging"))
