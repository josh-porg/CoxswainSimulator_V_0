"""The standard low-pass for rowing kinematics, on a periodic (one stroke cycle) signal (SOURCES §240).

A 4th-order zero-lag Butterworth at 6 Hz: a 2nd-order filter run forward and back (Winter's "fourth-order zero-lag"),
the filter and cut-off "previously verified for kinematics studies in rowing" (Urbanczyk 2021 thesis, citing
Buckeridge, Bull & McGregor 2015; Pollock et al. 2009). Chosen from the literature before any run it affects -- not
from the result. The athletes' averaged channels carry a coherent 6-7 Hz ripple that generalised cross-validation keeps
(it removes only white noise) and that the hull's own accelerometer rules out as body motion
(research/validation/own_body_ripple.py).
"""
import numpy as np

CUTOFF_HZ = 6.0
ORDER = 2                                                         # per pass; forward and back -> 4th order, zero lag


def lowpass_periodic(values, period, cutoff=CUTOFF_HZ):
    """``values`` sampled uniformly over one ``period`` (s), endpoint excluded; returns them low-passed. The cycle is
    tiled three times so the filter sees a periodic signal, and the middle copy is returned."""
    from scipy.signal import butter, filtfilt
    y = np.asarray(values, float)
    fs = y.size / float(period)
    if not cutoff < 0.5 * fs:
        raise ValueError("cut-off %.1f Hz is above the Nyquist frequency %.1f Hz of %d samples per cycle"
                         % (cutoff, 0.5 * fs, y.size))
    b, a = butter(ORDER, cutoff / (0.5 * fs))
    tiled = np.concatenate([y, y, y])
    return filtfilt(b, a, tiled)[y.size:2 * y.size]
