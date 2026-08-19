"""Coxswain -- a validated 6-DOF dynamics model of a rowing shell.

The dynamics follow Formaggia, Miglio, Mola and Montano, "A model for the
dynamics of rowing boats", *Int. J. Numer. Meth. Fluids* **61** (2009)
119-143, generalised from the paper's symmetry-plane (surge/heave/pitch)
formulation to all six degrees of freedom.

See ``docs/validation.md`` for the correspondence between the paper's
equations and this implementation, and for the numbers the regression
suite checks against.
"""

__version__ = "0.1.0"

__all__ = ["boats", "core", "crew", "hydro", "river", "sim"]
