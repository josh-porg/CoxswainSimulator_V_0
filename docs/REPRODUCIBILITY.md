# Reproducibility: how every result in this project can be checked

*Written 2026-10-24. Update when a study, a data store or a convention changes.*

The standard is that an examiner with the two repositories can regenerate every number the project
reports, from its inputs, with one command, and see where each input came from.

## 1. The rules

1. **Every result has a script in the repository.** Study code lives in `research/<study>/` (committed),
   products in `scripts/`, data tools in `tools/`, the simulator in `coxswain/`. Nothing is run from a
   session scratchpad. *Why:* two sets of results were lost that way (the [CR06] traces on 2026-09-27,
   step 4.3's rungs 1–5 after 2026-09-14). They are rebuilt and the rebuild is documented
   (`research/cr06/README.md`).
2. **Every reported number names its script and its record.** A result is written up in docs/SOURCES.md
   (numbered sections) or a study's own document (docs/coxswain_performance.md, docs/RACEDAY.md), with the
   script path; docs/TRACKING.md's Done list names the script and the files it writes.
3. **Scripts print what they claim.** A study script prints the numbers the write-up quotes, and, where a
   result is a rebuild or a validation, the recorded or target value beside its own.
4. **Inputs are files with provenance.** Published numbers are digitised into `data/literature/`
   (committed), each file's header naming the paper, figure or table, and the extraction script. Private
   inputs (recordings, CoxBox files, crews, commercial data) live in `data/local/`.
5. **Assumptions are written where they are made**: in the script's docstring and in the write-up. A
   choice that is not sourced is called a choice.
6. **Disagreements are reported, not tuned away.** "Sourced physics builds; athletes validate": a
   mismatch with a measurement is a finding about missing physics, never a coefficient to fit.
7. **Code and data are committed and pushed at the end of each piece of work**: the code to the public
   repository after a privacy check (no names, personal paths or private results), the private data to the
   private repository.

## 2. Where things are

| what | where | public? |
|---|---|---|
| the simulator, products, tools, studies, documents, digitised literature | https://github.com/josh-porg/CoxswainSimulator_V_0 | yes |
| private data and every study's outputs (`data/local/`): CoxBox files, race and practice recordings, transcripts and call codes, crew lists, lake histories, downloaded papers, fitted models | https://github.com/josh-porg/private_cox (audio through Git LFS; rendered videos excluded, they regenerate from the source videos) | **no** |
| the race-day and lake pages | https://github.com/josh-porg/charles-race-card (pages encrypted where private) | yes |
| layout in detail | docs/PROJECT_MANAGEMENT.md, "Code and data"; `research/README.md` | |

To reproduce a private result, clone the private repository into `data/local/` of a checkout of the
public one (`git lfs install` first, for the audio).

## 3. Environments

| environment | used for | snapshot |
|---|---|---|
| `.venv` (Python 3.13) | the simulator and nearly every study | `research/environment/venv_research.txt` |
| `.venv-tools` (Python 3.13) | encryption of the private pages, spreadsheets, speech-to-text | `research/environment/venv_tools.txt` |
| `research/callmodel/.venv` | the call model's transformer (torch, kept out of the game's environment) | `research/environment/venv_callmodel.txt` |

Regenerate a snapshot with `<venv>/Scripts/python.exe -m pip freeze > research/environment/<name>.txt`.
Random draws are seeded in each script (`np.random.default_rng(<seed>)`), so Monte Carlo and bootstrap
results repeat exactly.

## 4. Register of results (recent work)

Each row: the claim, the command that regenerates it, what it writes, and where it is written up. Older
results are in docs/SOURCES.md and docs/TRACKING.md with the same information.

| result | command (from the repository root) | writes (data/local/...) | written up |
|---|---|---|---|
| Women's foot-force direction 36° at maximum handle force ([BU15]) | — (read from the paper) | — | SOURCES §177; `data/literature/buckeridge2015_foot_force.csv` |
| Footplate centre of pressure 0.73 of foot length ([VI17]) | `.venv/Scripts/python.exe research/cr06/extract_vieira2017.py` | `data/literature/vieira2017_cop.csv` | SOURCES §177 |
| Knee, ankle and support moments; hip from below | `.venv/Scripts/python.exe research/cr06/knee_ankle_cr06.py` | `cr06_study/knee_ankle_cr06.json` | SOURCES §177 |
| Step 4.3 rungs 1–4 and rung 5's linear check, rebuilt | `research/cr06/stretcher_cr06.py`, `hip_topdown_cr06.py`, `trunk_forward_cr06.py` | `cr06_study/*.json` | `research/cr06/README.md`; SOURCES §178 |
| OKID/ERA call-response model | `.venv/Scripts/python.exe research/callmodel/okid.py` (and `--charles-only`) | `coxing/okid*.json`, `.png` | docs/coxswain_performance.md, "OKID/ERA" |
| Lake fog model, Sammamish | `research/lakes/fog_history.py`, `lake_fog.py data/local/sammamish/lake.json` | `sammamish/fog/lake_fog_model.json` | TRACKING Done, "Lake fog model" |
| HOCR 2026 club line comparison and its analyses | `research/raceday/line_compare.py`, `line_smoothness.py`, `wind_perturbation.py`, `skill_separation.py`, `window_robustness.py`, `adaptation.py`, `coxswain_operator.py`, then `compare_page.py` | `raceday/*.json` | TRACKING Done (2026-10-22/23 rows) |

## 5. Checklist for new work

- [ ] The script is in `research/<study>/`, its docstring states the question, inputs, method, assumptions
      and outputs.
- [ ] It prints the numbers the write-up quotes, and the target or recorded value where there is one.
- [ ] Inputs from the literature are digitised into `data/literature/` with a provenance header.
- [ ] The result is written up (SOURCES section or the study's document) with the script path, and a Done
      row in TRACKING names the script and its outputs.
- [ ] A privacy check of the staged changes; commit and push the code; commit and push `data/local/`.
