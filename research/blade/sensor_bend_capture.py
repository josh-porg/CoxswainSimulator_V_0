"""How much of the shaft's bend does an oar-angle sensor capture, by where it reads the shaft? (§246)

SOURCES §245's hypothesis: if the horizontal oar angle is sensed on the shaft a distance d outboard of the pin, the
measured angle already contains the shaft's slope there, and a model that adds the full bend to it counts part of the
compliance twice; on a measured angle that is the same, quasi-statically, as a shaft 1 / (1 - f) times stiffer, where
f is the fraction of the blade's bend displacement the sensor's slope accounts for:

    f(d) = l * slope(d) / delta(l),   slope(d) = int_0^d M / EI dy,   delta(l) = int_0^l M (l - y) / EI dy,

with M(y) = F (l - y) for a blade load F at the blade centre l (the pin as the root of the outboard cantilever). The
athletes' and the population's preference for x1.5-2 means f ~ 0.33-0.5. EI profiles: uniform, and [L14]'s finding
that the shafts are stiffest at the blade end (up to +64 % for the medium shaft) and most compliant near the sleeve,
taken as linear from the sleeve to the blade end. Prints f(d) and the d that gives f = 0.33 and 0.5.

    python research/blade/sensor_bend_capture.py
"""
import numpy as np

L_CENTRE = 1.805          # blade centre from the pin, m ([CR06]'s scull; [BR24]'s 1.795)
BLADE_END = 1.59          # the shaft's blade end (blade root) from the pin, m


def f_of_d(d, ei):
    y = np.linspace(0.0, L_CENTRE, 4001)
    m = L_CENTRE - y                                             # M / F
    inv = 1.0 / ei(y)
    delta = np.trapezoid(m * (L_CENTRE - y) * inv, y)
    sel = y <= d
    slope = np.trapezoid((m * inv)[sel], y[sel]) if sel.sum() > 1 else 0.0
    return L_CENTRE * slope / delta


def main():
    profiles = {
        "uniform": lambda y: np.ones_like(y),
        "[L14]-like: x1.644 at the blade end": lambda y: 1.0 + 0.644 * np.clip(y / BLADE_END, 0.0, 1.0),
        "compliant sleeve: x2 at the blade end": lambda y: 1.0 + 1.0 * np.clip(y / BLADE_END, 0.0, 1.0),
    }
    ds = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    print("fraction f of the blade's bend displacement captured by a sensor reading the shaft's slope at d from the pin")
    print("%-40s" % "EI profile" + " ".join("d=%.2f" % d for d in ds) + " | d for f=0.33 | d for f=0.5")
    for name, ei in profiles.items():
        fs = [f_of_d(d, ei) for d in ds]
        grid = np.linspace(0.0, 1.0, 1001)
        fg = np.array([f_of_d(d, ei) for d in grid])
        d33, d50 = np.interp(0.33, fg, grid), np.interp(0.5, fg, grid)
        print("%-40s" % name + " ".join("%6.3f" % f for f in fs) + " |    %.2f m    |   %.2f m" % (d33, d50))
    print("\nequivalent stiffness on a measured angle: 1 / (1 - f): f 0.05 -> x1.05, 0.10 -> x1.11, 0.33 -> x1.5, "
          "0.5 -> x2")


if __name__ == "__main__":
    main()
