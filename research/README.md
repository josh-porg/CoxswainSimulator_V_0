# research/

Study scripts, one folder per study. Committed; the repository is public.

| folder | study | its data (gitignored) |
|---|---|---|
| `raceday/` | HOCR 2026 race reviews, the club lines comparison, steering, wind, skill, adaptation, operator models | `data/local/raceday/` |
| `lakes/` (`sammamish/`, `whatcom/`) | lake conditions pages: build, history, calibration, the lake fog model | `data/local/lakes/`, `data/local/<lake>/` |
| `coxing/` (`analysis/`, `experiment/`, `review/`) | coxing calls and boat response, the coxing paper | `data/local/coxing/` |
| `callmodel/` | the coupled call/boat model (see its README) | `data/local/coxing/` |
| `cr06/` | the [CR06] women's single: one-athlete tests, step 4.3 | `data/literature/`, `data/local/cr06_study/` |
| `biorow/` | [BR24] like-for-like results | `data/local/biorow/` |
| `crew/` (`feeders/`) | the crew model studies; race-entry feeders | `data/local/crew/` |
| `bestline/` | row3r's best line against the model | `data/local/bestline/` |
| `overlay/` | the race-video renders | `data/local/overlay/` |
| `steering/` | constant versus modulated rudder | — |
| `blade/`, `coppel/`, `k05/` | blade validation and digitising | `data/literature/` |

## Conventions

- Run from the repository root: `.venv/Scripts/python.exe research/<study>/<script>.py`.
- A script's `CODE` is its own folder (for importing its siblings) and its `HERE` is its data folder under
  `data/local/`, where it reads inputs and writes outputs. Scripts that work in their working folder start
  with a `chdir` to their data folder.
- No names, personal paths or private results in code. Those live in `data/local/` (for the club crews:
  `data/local/raceday/crews.json`, read through `raceday/review_crews.py`; machine paths:
  `data/local/coxing/local_paths.json`).
- Never in a scratchpad. New work is written here from the start.
