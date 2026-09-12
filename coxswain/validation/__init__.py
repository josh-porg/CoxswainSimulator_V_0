"""One battery of published targets, and every physics profile scored on it.

Why this exists
---------------
Two known defects are being fixed offline (see :mod:`coxswain.physics`), and
the question "is the new model better?" has to have an answer that is not a
matter of taste.  Without a fixed battery it is one: a change that improves
the number somebody is looking at while quietly wrecking one they are not
looks like progress.  That is not hypothetical here -- switching the blade
model on the first time cost 14% of boat speed through a double-counted
immersion term, and it read as a physical finding until the bookkeeping was
checked (``docs/SOURCES.md`` sec. 7).

So: a set of :class:`Target` s, each with its provenance and the expectation
it encodes, and a harness that scores **any** profile against **all** of
them in one table.  The baseline is established against the current model
*before* any of the new work lands, which is the point -- the number to beat
gets set before anybody has a stake in beating it.

Honest bookkeeping
------------------
A target that cannot yet be measured is carried as ``pending`` with the
reason, rather than left out.  A battery that silently omits what it cannot
do reports a clean sheet for the wrong reason, and the shape of what is
missing is exactly what a reader needs to judge the rest.  :func:`run`
reports pending targets in the table with everything else.

Reading the result
------------------
    >>> from coxswain import validation
    >>> rows = validation.run("shipped", boats=("4+",))   # doctest: +SKIP
    >>> print(validation.table(rows))                     # doctest: +SKIP
"""

from .targets import Target, TARGETS, ready, pending
from .scorecard import Score, run, table, measure

__all__ = ["Target", "TARGETS", "ready", "pending",
           "Score", "run", "table", "measure"]
