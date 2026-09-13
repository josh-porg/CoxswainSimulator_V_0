"""What the model is measured against, and where each number comes from.

Every target carries its source.  A target whose data has not been obtained
is still listed, marked ``pending`` with the reason -- see the module
docstring of :mod:`coxswain.validation` for why omitting it would be worse.

Nothing here runs anything; :mod:`coxswain.validation.scorecard` does that.
Keeping the *claims* separate from the *measurements* means a target can be
read, argued with and cited without running a simulation, and it means
adding a source is an edit to a data structure rather than to control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["Target", "TARGETS", "ready", "pending", "by_key"]


@dataclass(frozen=True)
class Target:
    """One published (or self-measured) expectation the model must meet.

    ``band`` is the acceptable range of the measured quantity.  It is
    deliberately a *band* and not a point: several of these are known only
    to within a scatter, and quoting a point target for a quantity whose
    source reports a standard deviation invites tuning to noise.
    """

    key: str
    what: str
    #: Where the expectation comes from.  ``"self"`` means it is a
    #: diagnostic of this model rather than an external measurement --
    #: still falsifiable, but it tests internal consistency, not reality.
    source: str
    units: str
    #: ``(low, high)``, or ``None`` for a pending target.
    band: Optional[Tuple[float, float]] = None
    #: ``False`` when the data or the machinery is not there yet.
    implemented: bool = True
    #: Why it is pending, or what the band means. Shown in the table.
    note: str = ""
    #: Minimum blade tier for the target to mean anything. A slip-based
    #: expectation is vacuous against a model with no slip in it.
    min_blade_tier: int = 0


#: The battery.  Ordered roughly by how much each would change a result.
TARGETS = (
    # -- the defect this whole programme exists to fix -------------------
    Target(
        key="blade_efficiency_zero_crossing",
        what="Where fitted blade efficiency reaches zero, "
             "as a multiple of mean speed",
        source="self",
        units="multiples of mean boat speed",
        band=(0.15, float("inf")),
        note="With no velocity term in the blade force, the steady balance "
             "R(v) v = eta P forces eta proportional to v -- a straight "
             "line THROUGH THE ORIGIN. A blade with real slip physics has "
             "no reason to pass through the origin, so a line that reaches "
             "zero efficiency at (or near) zero speed is the signature of "
             "the defect. Measured on the baseline: 0.0204 on the eight and "
             "0.0007 on the four -- the line reaches zero at 0.09 and "
             "0.003 m/s, where the boats race at 4-6. Expressed "
             "as a SPEED rather than as a y-intercept because a speed can "
             "be compared against the boat's own speed with no external "
             "scale: a raw intercept of -0.008 is near zero beside an "
             "efficiency of 0.4 and enormous beside one of 0.001.",
    ),
    Target(
        key="blade_efficiency_linearity",
        what="Spread of eta/v across the racing speed range",
        source="self",
        units="fraction of the mean",
        band=(0.08, 10.0),
        note="The companion to the intercept and the more robust of the "
             "two: if eta/v is constant to within a couple of percent "
             "across a 2x speed range, eta is proportional to v whatever "
             "a fitted intercept says. A model with real slip physics "
             "should show this vary substantially.",
    ),

    # -- what the optimiser is actually denominated in --------------------
    Target(
        key="speed_per_watt",
        what="Boat speed per rower watt at the crew's operating point",
        source="self; guarded by Sports Biomechanics (2026)",
        units="m/s per 100 W of total crew power",
        band=(0.0, 100.0),
        note="Carried because the 2026 blade study found the blade with "
             "the LOWEST slippage and HIGHEST propulsive efficiency was "
             "not the one that produced the most boat speed per watt. "
             "Minimising slip is not the objective. Recorded on every run "
             "so a change cannot improve eta while losing boat speed "
             "without it being visible. The band is deliberately open: "
             "this is a tracked quantity, not yet a pass/fail.",
    ),

    # -- crew motion ------------------------------------------------------
    Target(
        key="surge_swing",
        what="Peak-to-peak surge as a fraction of mean speed",
        source="Day et al. (2011) from Kleshnev (2002), men's pair at 35; "
               "Holt, four boat classes",
        units="percent",
        band=(30.0, 60.0),
        note="QUOTED AT THE FASTEST OPERATING POINT, because the swing is "
             "strongly speed-dependent and a figure at an unstated speed is "
             "not a target: measured on the eight it runs 77.4% at 2.81 m/s "
             "down to 37.6% at 5.93. Any comparison with a published number "
             "has to match the speed and the rate, and it is not clear the "
             "recorded 1.7x comparison did. "
             "Holt puts the model 10-31% high across four boat classes and "
             "[D11] reports ~50% for a pair, which the model brackets. "
             "[IVV25]'s 37.5% for a single is NOT used as the target: its "
             "reported extremes are skewed opposite to a real boat-speed "
             "curve (max-mean 1.66 m/s against mean-min 1.18, where the "
             "catch dip is sharp and the recovery plateau broad), which is "
             "what instrument smoothing does. See docs/SOURCES.md sec. 4.",
    ),

    # -- the blade, once there is a blade model ---------------------------
    Target(
        key="blade_efficiency_level",
        what="Force-weighted blade efficiency over the drive at race pace",
        source="Kleshnev, Propulsive efficiency of rowing",
        units="fraction",
        band=(0.754, 0.816),
        min_blade_tier=1,
        note="78.5% +/- 3.1% for a single. Vacuous below blade tier 1: "
             "with no slip model the efficiency IS the lumped constant, so "
             "the target would be testing the constant against itself.",
    ),

    # -- pending: data located but not obtained ---------------------------
    Target(
        key="grift_immersion_curve",
        what="Blade drag coefficient against immersion depth",
        source="Grift, Vijayaragavan, Tummers & Westerweel (2019), "
               "J. Fluid Mech. 866",
        units="C_D",
        implemented=False,
        min_blade_tier=1,
        note="PENDING DATA. Aspect ratio 2, chosen to resemble an oar "
             "blade, Re 4-8e4. The open-access paper has been read in full "
             "([G19] in docs/SOURCES.md): steady-phase C_D is 1.10 with the "
             "plate's top edge at the surface, peaks at 1.60 at 20 mm "
             "cover (1/5 of the 100 mm plate height, a 45% rise), and is "
             "1.30 fully submerged. That is enough to show our "
             "immersion_factor is QUALITATIVELY wrong (ours is monotone and "
             "saturating; the measurement has an optimum) but not enough to "
             "fit against: the full curve is the paper's figure 4 and still "
             "needs digitising, and its depths are for a 100 mm plate, so "
             "carrying it to a blade needs cover as a fraction of blade "
             "width. Three points are not a basis for physics.",
    ),
    Target(
        key="grift_force_decomposition",
        what="Normal and tangential blade force over the drive",
        source="Grift, Tummers & Westerweel (2021), J. Fluid Mech. 918",
        units="N",
        implemented=False,
        min_blade_tier=2,
        note="PENDING DATA. Time-resolved PIV and six-axis force on a 1:2 "
             "blade on a realistic path, Re 0.8-1.2e5. The primary "
             "validation target for tier 2, and the only source found that "
             "gives the tangential component -- which is identically zero "
             "in the model today. Traces need digitising.",
    ),
    Target(
        key="caplan_gardner_coefficients",
        what="Lift and drag coefficients against sweep angle",
        source="Caplan & Gardner (2007)",
        units="C_L, C_D",
        implemented=False,
        min_blade_tier=2,
        note="PENDING DATA. Quasi-static flume, quarter scale. The "
             "coefficient set tier 2 would be built from.",
    ),
    Target(
        key="coordination_rate_invariance",
        what="Crew coordination variability against stroke rate",
        source="Cuijpers, Zaal & de Poel (2015), PLOS ONE 10(7): e0133527",
        units="relative phase, degrees",
        implemented=False,
        note="PENDING MODEL. Coupled-oscillator theory predicts coordination "
             "stability DEGRADES as rate rises; measured, it did not, and "
             "consistency slightly improved. Reproducing that without "
             "fitting to it is the acceptance test for the learned rower, "
             "and the replication this project is positioned to publish. "
             "Needs the forward-dynamic crew before it can be run.",
    ),
    Target(
        key="gross_efficiency",
        what="Metabolic gross efficiency, and its independence of rate",
        source="Hofmijster & van Soest",
        units="fraction",
        implemented=False,
        note="NOT MODELLED. 0.20, and unaffected by stroke rate. This "
             "model has no metabolic layer -- power enters at the handle -- "
             "so the target is carried to record that the constant is "
             "known and rate-independent, which is why a rate-scaled "
             "internal-loss term would be the wrong shape if one is ever "
             "added.",
    ),
)


def by_key(key: str) -> Target:
    for target in TARGETS:
        if target.key == key:
            return target
    raise KeyError("no validation target %r; have: %s"
                   % (key, ", ".join(t.key for t in TARGETS)))


def ready():
    """Targets that can be measured today."""
    return tuple(t for t in TARGETS if t.implemented)


def pending():
    """Targets carried but not yet runnable, with their reasons."""
    return tuple(t for t in TARGETS if not t.implemented)
