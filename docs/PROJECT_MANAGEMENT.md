# Project management — the master index

*Read this first. It says which document holds what, who keeps it current, and what
has to be updated when work happens. Last reviewed 2026-10-25.*

The rule behind all of it: **every finished piece of work is recorded before the next
one starts**, so nothing is done twice and nothing is claimed that the record does not
show. Check TRACKING.md's **Done** section before starting anything.

---

## The project-management set

These are kept current on every change. If one of them is out of date, fixing it is
the next task.

| document | where | purpose | update it when |
|---|---|---|---|
| **This index** | `docs/PROJECT_MANAGEMENT.md` | which document holds what | a document is added, retired or changes purpose |
| **Status** | [STATUS.md](STATUS.md) | one page: the state of each workstream, open problems in priority order, what is waiting on others | any workstream moves, a priority changes, or a reply arrives |
| **Issues — tracked and resolved** | [TRACKING.md](TRACKING.md) | every known defect (**Open**), everything finished and where it lives (**Done**), and fixed bugs with how they were found (**Fixed**) | a defect is found, a result lands on an open item, or work finishes |
| **The plan** | *Blade and Body* — [artifact](https://claude.ai/artifact/5ZM6yag3fYTViD3fqwhgv6) | the physics review and staged development plan: the two defects, the tiers, the phases and their gates, decisions, bibliography. Numbered revisions | a phase moves, a gate is passed or failed, a decision is taken |
| **The ledger** | *Rowing Physics Ledger* — [artifact](https://claude.ai/artifact/XBcnRHMy2Par4PAbw8kr6n) | every force the research model applies, as the code computes it, each term marked sourced / derived / fitted / provisional / chosen / failing, and the table of what has been checked | a term, value or status changes, or a validation result arrives |
| **Physics programme** | [PHYSICS_PROGRAMME.md](PHYSICS_PROGRAMME.md) | the repository's copy of the plan's machinery: phase table and gates, **Blocked, and on what**, decisions log, open questions | as the plan, and whenever something blocks or unblocks |
| **Sprint board** | [SPRINT.md](SPRINT.md) | the current sprint: goal, prioritised backlog with the groundwork approach for blocked items, board, findings that may unblock others | an item starts or finishes; a finding changes a priority |
| **Outreach** | [DATA_REQUESTS.md](DATA_REQUESTS.md) | letters to authors and labs, their status, and replies received with drafted follow-ups | a letter is sent or a reply arrives |
| **Race-day product** | [RACEDAY.md](RACEDAY.md) | the Charles race-day insight product: users, outputs, backlog, sprints, open questions (agile, user = Joshua) | a sprint starts or ends, a story is done, or an answer changes the plan |

**Artifacts are published pages.** To change one, republish to its existing link (never
a new page), after reading the full current version. The saved source is the thing to
edit, not a reconstruction. The last-published sources of the plan and ledger are kept in
`data/local/artifacts/` (`blade_and_body.html`, `rowing_physics_ledger.html`); edit those and
republish, after checking no newer version was published elsewhere.

---

## Evidence and reference

Updated as results arrive; they are the record the management set points into.

| document | purpose |
|---|---|
| [SOURCES.md](SOURCES.md) | the evidence: every source, derivation and result, in numbered sections (§1–§179 as of this review; §120a and §169a are second sections that once shared a number, and 144–151 were never assigned). New work appends a section; nothing is rewritten silently |
| [BLADE_HYDRODYNAMICS_REVIEW.md](BLADE_HYDRODYNAMICS_REVIEW.md) | the literature review of the rowing blade's load (started 2026-10-27): sources, what measured loads say about quasi-steady and unsteady blade laws, our tests on them, and the models to build. A living review; numbers in SOURCES §183– |
| [PROVENANCE.md](PROVENANCE.md) | where each data file came from |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | how every result is regenerated: rules, the public and private repositories, environments, a register of results with their commands, the checklist for new work |
| [coxswain_performance.md](coxswain_performance.md) | the coxing research log (calls, boat response, the paper) |
| `research/callmodel/README.md` | the coupled call/boat model and its session format |
| `research/biorow/` | scripts behind the [BR24] like-for-like results (derived numbers only; the data stay in `data/local/`) |

## Code and data: where things live

**Code is committed; private data is not; nothing lives in a scratchpad.** A session's temporary
scratchpad is cleaned without warning: the [CR06] digitised traces were lost that way on
2026-09-27, and step 4.3's hip-moment scripts (`cr06/hip_moment_cr06.py`) after 2026-09-14.
From 2026-10-24 every script is written straight into the repository, and one-off edits are
made with the editor, not with throwaway patch scripts.

| folder | committed? | holds |
|---|---|---|
| `coxswain/` | yes | the package: the simulator and the shipped game |
| `scripts/` | yes | runnable products: race cards, reports, the race-day site, validation |
| `tools/` | yes | fetch, extract and build tools for the data in `data/` |
| `research/<study>/` | yes | study scripts, one folder per study (`raceday`, `lakes`, `coxing`, `cr06`, `biorow`, `crew`, `steering`, ...). They read and write their study's data folder under `data/local/` (a script's `HERE` is that folder; `CODE` is its own folder) |
| `data/` (top level) | yes | course, river, results and roster data the code reads |
| `data/literature/` | yes | numbers digitised from published papers, provenance in each file's header ([README](../data/literature/README.md)) |
| `data/local/<study>/` | **no** (gitignored) | each study's inputs and outputs: race recordings and captions, CoxBox files, crew lists, pages and their encrypted copies, fitted models |
| `data/local/raceday/crews.json` | **no** | the club crews (coxswains' names, events, results); code reads it through `research/raceday/review_crews.py`, so no name is written in committed code |
| `data/local/coxing/local_paths.json` | **no** | machine-specific paths (the video and CoxBox archive) |
| `data/local/biorow/` | **no** | [BR24] — commercial, shared for validation |
| `data/local/literature/` | **no** | downloaded papers and theses (copyright) |
| `data/local/artifacts/` | **no** | last-published sources of the plan and ledger artifacts |
| `data/local/archive/` | **no** | old scratch folders and the 2026-10-24 move (`moved_2026-10-24/MOVES.json` maps every moved script) |
| `data/raw/` | **no** | re-fetchable downloads |

**One exception:** `data/local/raceday/keepalive_relay.py` stays where it is, because the Windows
task "Whatcom relay keepalive" runs it by that path.

**The repository is public.** Before committing anything under `research/`, check it for names,
personal paths and private results; those belong in `data/local/`.

## Subsystem documents

Reference for one part of the code. Updated when that part changes.

| document | purpose |
|---|---|
| [DAMPING.md](DAMPING.md) | hull damping terms |
| [REALTIME.md](REALTIME.md) | the real-time trainer's loop and performance |
| [SCENERY.md](SCENERY.md) | courses and rendering of the river scenery |
| [TRAINER.md](TRAINER.md) | how the trainer is used |
| [PLAN_SYNCHRONISATION_AND_BLADES.md](PLAN_SYNCHRONISATION_AND_BLADES.md) | the August plan for crew synchronisation and blades on the water; superseded as the master plan by *Blade and Body*, kept for its design |
| [RELEASING.md](RELEASING.md), [SIGNING.md](SIGNING.md) | building, signing and publishing releases |
| `README.md` | the repository's front page |

## Other published pages

| page | what it is | kept current? |
|---|---|---|
| *What the Boat Answers* — [artifact](https://claude.ai/artifact/228aScVzR5tgsQKNDrzizq) | the coxing paper's web version (2026-09-21, before the freeze and the coupled-process revision) | no — the paper's source is the reference |
| *Mid-Morning Eight Forecast* — [artifact](https://claude.ai/artifact/C6SPydDt2jsSAgL89uRuqG) | race forecast for the crew | as needed for the crew |
| *Four Weeks to the Charles* — [artifact](https://claude.ai/artifact/4Kxujd2tu1QsKSEFrtNEku) | the crew's training plan to the Head of the Charles — **not** the project plan | as needed for the crew |
| *Model Compatibility Map* — [artifact](https://claude.ai/artifact/K9uBbGqnAHUW8ah1YveSjE) | every source, the code that implements it, whether a printed result is reproduced, and which catch, finish and blade options can run together (SOURCES §163–164) | **yes** — update when an option, a guard or a validation changes, and **add a section whenever something new is developed** (user, 2026-10-27); last republished 2026-10-27 (v14); source in `data/local/artifacts/model_compatibility_map.html` |
| *Shell Model Validation* — [artifact](https://claude.ai/artifact/8pD9EBLo7VfVCvsSRnQv1m) | August validation snapshot | no — superseded by the ledger's checks table |
| *Stroke-Resolved Steering* — [artifact](https://claude.ai/artifact/PKnatD2JmNp6t93YVScmqo) | August steering study | no — snapshot |

---

## What to update, by kind of change

| when this happens | update |
|---|---|
| a piece of work finishes | TRACKING **Done** (or **Fixed** for a bug); the SPRINT board; STATUS at-a-glance row; the plan and ledger if it touches physics |
| a measurement or model result lands | a new SOURCES section; the TRACKING item it answers; the ledger's checks table; STATUS §3 if priorities move |
| something blocks or unblocks | PHYSICS_PROGRAMME **Blocked**; STATUS §4 |
| a decision is taken | PHYSICS_PROGRAMME **Decisions**; the plan's §07 |
| a letter goes out or a reply arrives | DATA_REQUESTS; SOURCES if data arrived; STATUS §4 |
| a document is added or retired | this index |
