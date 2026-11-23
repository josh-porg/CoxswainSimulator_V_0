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
    inboard_sensor()


def inboard_sensor(clamp=0.40, l_handle=0.82):
    """The sensor as it is (Kleshnev, email 2026-11-22; SOURCES §258): BioRow's 2D oar-angle sensor on the pin, its arm
    clamped to the shaft ~0.40 m INBOARD of the pin. The model's inboard is rigid, so it reads the measured angle as the
    shaft's tangent at the pin. A real inboard bends too (three-point bending, the moment F_h y' falling to zero at the
    handle): the handle and the clamp lead the pin's tangent, so the sensor's chord angle leads it by

        Delta = F_h (l_h c / 2 - c^2 / 6) / EI_in        (cantilever from the pin, load F_h at l_h, chord to c)

    and the real blade lags the *measured* angle by phi + Delta, not the model's phi = F_h l_h / K_ang: on a measured
    angle the real oar is 1 / (1 + g) times as stiff as the model's, g = Delta / phi -- the opposite sign to §246's
    outboard hypothesis. l_h: the handle force's point, 0.06 m from the top of the 0.88 m inboard (Kleshnev 2026-10-28).
    EI_in is not measured for the inboard; bracketed at 1-3x [L14]'s medium shaft (the handle end is the thicker)."""
    from coxswain.crew.oar_bending import L14_MEDIUM_EI, OarBending
    k_ang = OarBending.laschowski2014_scull(L_CENTRE, 0.43).k_ang
    print("\nthe sensor as it is: clamped %.2f m inboard of the pin (Kleshnev 2026-11-22); model K_ang %.0f N m/rad"
          % (clamp, k_ang))
    for m in (1.0, 2.0, 3.0):
        ei = m * L14_MEDIUM_EI
        g = (l_handle * clamp / 2.0 - clamp ** 2 / 6.0) * k_ang / (ei * l_handle)
        print("  inboard EI %.1f x [L14] (%.0f N m^2): g = %.3f -> on the measured angle the oar is x%.2f as stiff as "
              "the model's (the fits wanted x1.5-2)" % (m, ei, g, 1.0 / (1.0 + g)))


if __name__ == "__main__":
    main()
