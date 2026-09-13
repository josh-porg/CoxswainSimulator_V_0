"""Michell's integral on the Wigley hull, against Lazauskas (2009).

The thesis prints the inviscid wave resistance of the Wigley parabolic hull
(L/B 10, L/T 16) as 1000 C_W against Froude number, C_W on wetted area
S = 0.1487 L^2 (Fig. 7.1, Table 7.1), and quotes 0.89 at Fr 0.2 as the
"true" value from the 1979 Workshop discussion (section 6.5).

The shipped sum puts dx dz on every grid point, ends included, and reads
high: the waterline row, where the depth decay is 1, and the bow and stern
count at full weight rather than half.  Trapezoid weights remove that.  The
shipped default is left exactly as it was; these tests pin both.
"""

import numpy as np
import pytest

from coxswain.hydro.michell import GRAVITY, MichellWave, wigley_offsets

WETTED_AREA = 0.1487          # S / L^2, Lazauskas Table 7.1
#: Read by eye from Lazauskas Fig. 7.1; the reading error is about 0.05.
FIGURE_7_1 = {0.20: 0.89, 0.30: 2.14, 0.345: 1.24, 0.50: 4.52, 1.00: 1.82}
READING = 0.05


def thousand_cw(wave, froude):
    speed = np.asarray(froude, dtype=float) * np.sqrt(GRAVITY * 1.0)
    return 1000.0 * wave.resistance(speed) / (
        0.5 * wave.density * speed ** 2 * WETTED_AREA)


def test_default_is_the_shipped_uniform_sum_exactly():
    """The shipped trainer builds every hull's wave table through this sum."""
    wave = MichellWave.wigley()
    assert wave.quadrature == "uniform"

    speeds = np.array([1.4, 3.1])
    draft = float(np.max(np.abs(wave.level)))
    expected = []
    for speed in speeds:
        k0 = GRAVITY / speed ** 2
        lam_max = max(float(np.sqrt(wave.decay_cutoff / (k0 * draft))), 2.0)
        u_max = float(np.arccosh(lam_max))
        u = (np.arange(wave.angles) + 0.5) * (u_max / wave.angles)
        lam = np.cosh(u)
        weight = np.cosh(u) ** 2 * (u_max / wave.angles)
        decay = np.exp(np.clip(k0 * lam[:, None] ** 2 * wave.level[None, :],
                               -700.0, 0.0))
        phase = k0 * lam[:, None] * wave.station[None, :]
        weighted = np.einsum("lz,xz->lx", decay, wave.slope)
        cosine = np.einsum("lx,lx->l", weighted, np.cos(phase))
        sine = np.einsum("lx,lx->l", weighted, np.sin(phase))
        integral = (cosine ** 2 + sine ** 2) * (wave._dx * wave._dz) ** 2
        expected.append(4.0 * wave.density * GRAVITY ** 2 / (np.pi * speed ** 2)
                        * float(np.sum(integral * weight)))
    assert np.array_equal(wave.resistance(speeds), np.array(expected))


def test_unknown_quadrature_is_refused():
    with pytest.raises(ValueError, match="quadrature"):
        MichellWave.wigley(quadrature="simpson")


@pytest.mark.parametrize("froude", sorted(FIGURE_7_1))
def test_trapezoid_matches_lazauskas_figure_on_the_default_grid(froude):
    """Hump at 0.30, hollow at 0.345, peak at 0.50 -- all within reading."""
    wave = MichellWave.wigley(quadrature="trapezoid")
    assert thousand_cw(wave, [froude])[0] == pytest.approx(
        FIGURE_7_1[froude], abs=READING)


def test_uniform_sum_reads_high_at_the_peak():
    """The finding itself: 4.88 against 4.52 at Fr 0.5 on the default grid."""
    wave = MichellWave.wigley()
    assert thousand_cw(wave, [0.50])[0] > FIGURE_7_1[0.50] + READING


def test_trapezoid_is_below_uniform_everywhere():
    """Halving the waterline and end weights can only remove resistance
    here: every weight it changes carries a positive-definite contribution
    on this hull at these speeds."""
    froude = sorted(FIGURE_7_1)
    uniform = thousand_cw(MichellWave.wigley(), froude)
    trapezoid = thousand_cw(MichellWave.wigley(quadrature="trapezoid"), froude)
    assert np.all(trapezoid < uniform)
