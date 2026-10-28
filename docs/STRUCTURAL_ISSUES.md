# Structural issues: classes of defect in the code, and the change that removes the class

*Created 2026-10-27 (user: "when you hit issues like that we need to make structural changes to the code... keep a list
of those required and maybe just fixing it because it represents a class of issues"). A bug that can recur in a new
form is a structural issue: it is listed here with every instance found, the structural fix, and its status. Fixing an
instance without the class is not a fix. Indexed in docs/PROJECT_MANAGEMENT.md; memory `structural-issues`.*

| # | class | instances found | structural fix | status |
|---|---|---|---|---|
| S1 | **A load path that bypasses the general blade-load function**, so an option added later is silently ignored on that path | (1) `blade_strips.liftdrag_strips` resolved tier 2's lift and drag itself, ignoring a blade's own `coefficients` (2026-10-27, SOURCES §191); (2) `DynamicOarSimulator._oar_loads` had a fast path for the plain slip law that skipped `_blade_loads_arm`, so the oar's bending had no effect (2026-10-27, §193); (3) the blade-efficiency tally recomputed the slip load directly and read no bending (§193) | every load goes through `_blade_loads_arm` (the fast path removed: identical results, ~33% slower on the slip law); `liftdrag_strips` calls `coefficients` for any blade that overrides it; the tally reads the stored bending history per sample | **fixed** 2026-10-27; tests `test_blade_review.py::test_strips_honour_a_blade_that_overrides_its_coefficients`, `test_oar_bending.py` |
| S2 | **One name, two lengths**: `Oar.outboard` is pin-to-**tip**; the dynamic oar's blade objects' `outboard` is pin-to-**blade centre** (the lever) | `research/blade/crossflow_decomposition.py` took the tip as the lever (2026-10-27): every row 10–30% off until caught by a baseline that did not reproduce §192 | `Oar.tip_outboard` names the tip; `Oar.outboard`'s docstring says it is not the lever; new code names the length it means (`tip_outboard` / `blade_centre_outboard`). Renaming `outboard` itself waits: the frozen shipped game reads it | **partly fixed** (named, documented, tested); migration of existing callers to the explicit names: open |
| S3 | **A study law patched into a private attribute after construction**, outside every check and cache key | the decomposition study set `sim._liftdrag` by hand (2026-10-27) | `DynamicOarSimulator(liftdrag_blade=...)`: a blade instance through the constructor, validated, and carried in `_match_key` (`repr`) and `torque_for_power` | **fixed** 2026-10-27; test `test_oar_bending.py::test_liftdrag_blade_parameter_and_its_cache_key` |
| S4 | **A cache key that misses a field that changes the result** | historic: hull shape (09-13), pull shape (09-13), oar mass (09-14), fins/Munk/roll friction (09-30) (memory `new-physics-switch-check-cache-keys`); 2026-10-27 the face-on value (carried in its name) and bending (added) | **open**: build the key from the simulator's own options (every constructor argument and the profile's fields) so a new option cannot be left out; until then, each new option adds itself and a test | open (class recurring) |
| S5 | **A fit filed as a source** | the research profile's scull C2 (140.88, fitted to one athlete 2026-09-18) carried "sourced" in the ledger; the ledger's "fitted" style had never been used | `docs/CALIBRATION_REGISTER.md`: every fit, the model state it was fitted under, its re-check triggers; the ledger tags fits "fitted" | **fixed** as process 2026-10-27 (memory `calibration-register`) |
| S6 | **Duplicated physics formulas** | the Polhamus/Helmbold slope re-derived for the blade beside the fins' copy (2026-10-27, §190) | one function, `hydro.appendages.polhamus_helmbold_slope`, used by the fins and the blade; a test pins the study copy to it | **fixed** 2026-10-27 |

| S7 | **A unit in a name that the value does not have** | `CoupledCrew.spread_seconds` returned radians (2026-10-28, found writing SOURCES §194); `scripts/recovery.py` traced it as a time | `spread_radians` names the radians; `spread_seconds` converts with the stroke frequency of the last step and refuses before any step; a test pins the conversion | **fixed** 2026-10-28; `test_coupled_crew.py::test_spread_seconds_is_in_seconds` |

## How to use this list

- When a bug is found, ask whether it is an instance of a listed class or a new one. If either, it goes here.
- The fix removes the class (one path, one name, one key builder), and a test pins the class, not only the instance.
- When a fix would touch the frozen shipped game, the class is documented and new code is protected, and the
  shipped side waits for a release decision (as S2).
