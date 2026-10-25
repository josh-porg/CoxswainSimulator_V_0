# Charles Race-Day Insight — product plan

*A separate planning document for the race-day product. Started 2026-10-08. Owner: Joshua
(user and product owner); Claude (developer). Agile: short sprints, a demo and a check-in at the
end of each, the backlog re-prioritised from what the user says at the demo. Update this file
whenever a sprint starts or ends, a story is done, or an answer changes the plan.*

> **Depth numbers below are overstated (flagged 2026-10-21).** Every race card, plan, field map and review
> here priced shallow water through `RouteEvaluator`'s default path, which multiplies the boat's **whole**
> resistance by the shallow-water wave factor (up to 3x) instead of the wave term (9-17% of it). HOCR 2026
> GPS from three crews rejects that term (slope -0.06 +- 0.03 against 1). On a research boat (Sretenskii,
> wave term only) the faster crews' optimum gaps fall from 32-34 s to 6-8 s. Treat as overstated: "shallows +16 /
> +34 s", "push in the shallows" / "push where the boat is slowest", "steering wide for deeper water", the eight's
> lake targets "near the drag rise", and every "out of the shallows: depth +N" line gain. The reviews are being
> re-run on the research boat (`race_card.build_boat(..., profile="research")`); the published race cards are
> not changed. PHYSICS_PROGRAMME, "Shallow water on the Charles, tested against HOCR 2026 GPS"; TRACKING,
> "Analyses priced on the factor-on-total depth path".

> **Status 2026-10-25: post-race.** HOCR 2026 was raced on 17 October. Sections 11b and 11d are the
> pre-race questions, kept as asked. The log (§12) runs to the race review and the post-race analyses; their
> details are in TRACKING's Done list (2026-10-19 to 10-23). Open: whether the race cards move to the research
> depth term, and the Cambridge-side corridor (TRACKING, Open).

---

## 1. Vision

Give me and my fellow coxswains **as much race-day insight as the data can honestly support**,
delivered in the form each moment can take: long and detailed the night before, a few trusted
numbers on race morning, a split and a plan letter at the start. Weather, water, the
competition and the earlier races of the day are pulled in automatically; the physics and line
work already in this project turn them into recommendations.

**The governing principle (the coach's):** whatever conclusion we reach, we must be confident in
it, because a crew that believes it has an edge is worth more than a couple of seconds of
modelled advantage. So the product makes **clear calls**, keeps uncertainty in the coxswain's
and coach's view, and never shows the rowers probabilities.

## 2. Users and what each needs (discovery, 2026-10-08)

| user | needs | when | must not see |
|---|---|---|---|
| **Joshua** (cox, women's masters 4+) | target splits and reference speeds to requalify; the line through the bridges; wind tactics; traffic around his bow number; where to make moves; updates from earlier races | night before (detail), morning (last detail), 1-2 h before launch (a couple of numbers + which plan) | — |
| **The eight's coxswain** (the 8) | the same for the eight; works closely with Joshua | same | — |
| **Other SRA coxswains** (other events; more experienced) | "here is what I have found, use what is useful": course insights that hold for any crew — bridges, wind exposure, lines, traffic patterns | night before, as a presentation | crew-specific odds of other crews |
| **The coach** (physics degree) | the reasoning and the evidence, and a clear, confident conclusion | night before, as a presentation | — (can see everything) |
| **The rowers** | a short, positive pre-race briefing: the course, the plan, the moves, what the cox will call | the day before, at the team meeting | probabilities of medalling, requalifying or placing; anything that undermines belief |

**Moments of use on race day**
- *Night before:* long-form briefings (three decks: coxswains, coach, rowers) and the full race card.
- *Morning / at the trailers:* the last time detail can be taken in — conditions for the day, any change of plan.
- *1-2 h before launch, on land:* a couple of numbers and a few global variables: target split, plan letter, wind, anything changed by the earlier races.
- *At the start line:* only if it can be read at a glance and trusted — one split and a category (plan A/B/C). Not expected to be used.

## 3. What the product delivers

| # | output | for | form |
|---|---|---|---|
| O1 | **Race card** — plan letter, target splits by course section, reference speed to requalify, wind summary, the crews around your bow number | Joshua, the eight's coxswain | private web page, phone-first, refreshes itself |
| O2 | **Plans A/B/C**, decided the night before for the likely conditions (e.g. calm / headwind on the basin / crosswind gusts at the bends), each with its line, moves and splits | Joshua, the eight's coxswain | inside the race card; the morning view says which one applies |
| O3 | **Day update** — how fast crews in earlier events are going compared with their prediction, folded into the target splits | Joshua, the eight's coxswain | race card, automatic |
| O4 | **Coxswains' briefing** — course findings for any crew | other SRA coxswains | slides, presented from a laptop |
| O5 | **Coach briefing** — the evidence and the conclusion | coach | slides |
| O6 | **Rowers' briefing** — the course and the plan, no probabilities | crews | slides, short |

## 4. Product principles

1. **Few numbers on race day.** The morning and pre-launch views show at most a handful of
   numbers and one decision. Everything else lives in the night-before material.
2. **Clear calls.** Every recommendation is stated as a decision, with its reason in one line. The
   uncertainty is in the coach's and coxswains' view, not in the call.
3. **No probabilities for rowers.** The rower deck and anything a rower might see carries no
   odds, placings or requalification chances.
4. **Trust over cleverness.** A number that changes is shown with what changed it. If live data
   are missing, the last good value and its time are shown, never a blank.
5. **Private.** Shared by private link with the team; crews and rowers named generically outside it.
6. **Built on what the project already validates.** Where the physics is not settled (the
   speed fluctuation work), the product uses the parts that are: course, bridges, wind, the
   cross-crew time model.

## 5. What already exists in this repository

| asset | where | ready for the product? |
|---|---|---|
| Charles bathymetry (CRAB / MIT Sea Grant survey) and calibrated flow model | `coxswain/river/charles.py` | yes |
| Six bridges with surveyed deck lines and piers | `coxswain/river/bridges.py`, `data/charles_obstructions.json` | yes |
| Line evaluation and optimisation (distance, depth, current) | `coxswain/river/route.py`, `trajectory.py` | yes for line comparison; the full receding-horizon solve is stalled (STATUS 3.6) |
| Wind: station reading → wind over the river, sheltering by banks and trees | `coxswain/river/stations.py`, `coxswain/hydro/canopy.py`, `wind.py` | yes |
| HOCR passing rules; the wake of the boat ahead | `coxswain/river/passing.py`, `traffic.py` | yes |
| Stochastic line optimisation across scenarios | `coxswain/river/stochastic.py` | research-grade; slow |
| Discharge history at Waltham (USGS 01104500) | `data/charles_discharge_waltham.csv` | history only; live feed to add |
| HOCR 2026 entries (veteran fours, grand master eights) and the requalification forecast (cross-crew model from Tail of the Lake, Pink Ribbon Row) | `data/local/crew/feeders/`, the forecast pages | yes; to be refreshed |

### What the project already knows (findings that shape the product)

These come from earlier work (SOURCES sections in brackets) and change what the product should do:

- **The line barely depends on the weather.** Re-optimising the line for the forecast wind is
  worth 0.2-0.9 s at 4 m/s and up to 5.1 s at 8 m/s (from 340°, the line moving 3.7 m on average);
  the wind itself is worth -26 to +155 s for a masters eight (`scripts/raceday.py`,
  `wind_scenarios.py`; SOURCES §176). These were 0.1-0.5 s and "up to 133 s" before the wind was
  charged against the boat's own resistance. So the product shows **one line**, and puts the
  weather into the **target splits and expectations**, not into a different line; on a hard
  northerly the re-steered line is worth a few seconds and could be shown as an option.
- **Position decisions pay seconds; pacing redistribution barely does.** A 1% power spread
  captures the whole pacing gain; taking a clean line past a slower crew, and which side to send
  them, pay in seconds (§84-86, §88). So the plans' real content is **traffic and arch decisions**.
- **Push where the boat is slowest** — the shallow stretches, where a masters crew sits near the
  transcritical drag rise (`course_pacing.py`, `line_and_pace.py`). That is where power moves go.
  *Flagged 2026-10-21: rests on the overstated depth term (banner above); the shallow stretches showed
  no measurable slowdown in HOCR 2026 GPS. Do not use until re-derived on the research boat.*
- **A day factor across all events.** Every category races the same water on the same day, so
  earlier events' times against their norms give today's factor (§77). That is the machinery for
  R9: target splits updated from earlier races.
- **Targets with error bars**: win, medal and requalify for the 60-69 four from 2019-2025 full
  fields (§89, §95, `scripts/targets.py`); the crew modelled rower by rower (`my_crew.py`,
  `hocr_crew.py`); the Charles split converted to its Lake Sammamish equivalent (§87, `splits.py`).
- **Passing rules and the cost of yielding**, coupled to the river (§82-86, `passing_race.py`).
- **Viewers exist**: a course explorer with layers and a moving line (§81), and a report built
  from containers (§85) — starting points for the phone page and the decks.

## 6. Data the product pulls

| data | source | refresh | notes |
|---|---|---|---|
| Wind and weather forecast, hourly | NWS gridpoint forecast for the course (api.weather.gov) | hourly | free, no key |
| Observed wind | NWS station KBOS (and nearer stations if usable) | 10-60 min | the stations module already corrects for exposure |
| River flow | USGS 01104500 Charles River at Waltham, instantaneous values | 15 min | the basin is impounded; current is small, to be quantified |
| Entries, bow numbers, start order and times | HOCR / RegattaCentral | daily until race day | to confirm where published |
| Earlier races' results on the day | the regatta's live results | every few minutes on race day | source to confirm (O3) |
| Crew histories | Tail of the Lake, Pink Ribbon Row, previous HOCR results | before the race | already used by the forecast |

## 7. Architecture (proposal, to be confirmed in sprint 1)

- **Compute** in Python with the project's own modules: a scheduled job pulls the data, runs the
  day index and the plan selection, and writes the race card and the decks.
- **Publish** a private web page (phone-first, readable offline once loaded) and slide decks that
  run in a browser on a laptop.
- **Precompute the heavy parts** (lines, plans A/B/C, splits) the night before; race day only
  re-selects and re-scales them, so the morning view is fast and does not depend on a solver.

## 8. Backlog

Priority: **M** must for this year's Charles, **S** should, **C** could, **L** later.

| id | story | acceptance | pri |
|---|---|---|---|
| R1 | As a cox, I see **target splits by course section** and a reference speed to requalify for my crew | splits for each section from the start to the finish; the requalify reference stated as a split; numbers reconcile to the forecast | M |
| R2 | As a cox, the race card shows **today's wind** along the course and what it means (headwind, tailwind, crosswind by section) | live NWS data; sections coloured; one-line advice per section | M |
| R3 | As a cox, I have a **race plan** prepared the night before — pacing, moves, turns, focuses, landmarks, stroke rates and splits — with its **key decisions** (which arch, and the conditions for the alternative arch) and the morning view tells me which variant applies | one plan per crew with its decision points; each decision has a trigger (traffic, wind, how the start went) and an alternative; the morning view names the variant | M |
| R4 | As a cox, I see the **line through each bridge** and bend | course map with the recommended arch and line; checked against the piers | M |
| R5 | As a cox, I see **who starts around me** and where passing is likely | bow numbers ±5 with their predicted speeds; passing locations along the course | M |
| R6 | As a presenter, I have a **coxswains' deck** of course findings | 10-15 slides; no other crew's odds; runs in a browser | M |
| R7 | As a presenter, I have a **coach deck** with the evidence and a clear conclusion | the methods and numbers behind R1-R5; the conclusion stated with its confidence | M |
| R8 | As a presenter, I have a **rowers' deck** without probabilities | short; course, plan, moves, calls; a test checks no probabilities appear | M |
| R9 | As a cox, target splits **update from earlier races** of the day | a day-speed index from earlier crews against their predictions; the update and its cause shown | S |
| R10 | As a cox, the page **works with poor signal** | last data kept and timestamped; loads offline after the first visit | S |
| R11 | As a cox, I see **where to make moves** (sheltered stretches, after bridges, passing chances) | moves marked on the course with a reason each | S |
| R12 | As a cox, live **river flow** is shown and its effect stated | USGS feed; the effect on time computed, or shown as negligible | C |
| R13 | As a team, other regattas can use it | course and data adapters beyond the Charles | L |

## 9. Release plan

| sprint | dates | goal | demo / check-in |
|---|---|---|---|
| 0 Discovery | 8 Oct | users, moments, outputs; this document; round 2 questions | answers to round 2 |
| 1 Walking skeleton | 9-11 Oct | R1 + R2 for the 4+ on a private phone page, fed by live weather; a first deck skeleton | Sat 11 Oct: you use it on your phone and say what is wrong |
| 2 Race version | 12-15 Oct | R3-R8 for the 4+ and the eight; R9 if the results source is confirmed | Wed 15 Oct: dress rehearsal of the three briefings |
| Race | 16-18 Oct | night-before briefings; morning and pre-launch use; live updates | after your race: what you used, what you ignored |
| Retro | 19-20 Oct | what to keep, change, drop; next regatta | the next version's backlog |

**Definition of done for a story:** it works on your phone (or laptop, for decks) with real data,
you have seen it at a demo, and its numbers can be traced to their source.

## 10. Risks

| risk | effect | mitigation |
|---|---|---|
| Signal at the Charles on race day is poor | morning view does not load | page works offline after the first load; the night-before card has the numbers already |
| The line optimiser is slow or unsettled | no line by race day | lines precomputed the night before; arch choices from geometry if the solve fails |
| Live results source unknown or blocked | no day update (R9) | the day index is a "should"; the forecast stands without it |
| Over-precision undermines confidence | the crew doubts the plan | principle 2: clear calls, ranges only in the coach's view |
| Too much information on race morning | ignored | principle 1: a handful of numbers, one decision |

## 11. Answers — round 2 (2026-10-08)

1. **Crews:** Joshua coxes the **women's 60+ four** (Veteran Fours 60+); the eight's coxswain coxes the
   **women's 50+ eight** (Grand Master Eights). Race weekend 17-18 October 2026; the 2026 start
   times are not yet found (2024 had both on Saturday morning) — to confirm from the HOCR schedule.
2. **A race plan holds:** pacing, moves, turns, focuses, landmarks, key decisions (which arch, and
   under what conditions to take the alternative), where the power moves are, stroke rates and splits.
3. **Live results:** to be located (Joshua: "a basic search will tell you"); the 2026 regatta is on
   RegattaCentral (event 10096), whose pages block scripted access, as before. Next step in sprint 1.
4. **Targets:** both the requalify split and the best achievable. **Rowers do not get splits**
   unless the cox chooses to share one.
5. **Decks:** in the browser.
6. **Private link:** Joshua and the eight's coxswain. Coaches and other coxswains see the presentations only.

## 11a. Answers — round 3 feedback on the first card (2026-10-08)

1. **Uncertainty:** wanted. Every goal now has a central time (even odds) and a range (fast end 80%,
   slow end 20%); the forecast has an 80% range; the wind's cost is also given at 30% less and more wind.
2. **The eight:** goals are a medal (top 3), beating last year's place (8th of 34 in 2025, so 7th
   or better) and requalifying (top 15 of 31), with the crew forecast alongside.
3. **Landmarks:** the bridges, Magazine Beach, the mile markers, the wooden pier before the finish,
   and the big turn (which begins near Mile 2).

## 11b. Open questions — round 3 (for the sprint 1 demo)

- Do you want the Lake Sammamish equivalent of the target split as well (for the last practices)?
- Does the eight's coxswain want to be part of the demos?
- ~~Wooden pier, Magazine Beach, big turn~~ answered: see the log, 2026-10-08 (round 3b).

## 11d. Open questions — round 4 (for Saturday's demo, 11 Oct)

1. **Approve the refresh job once.** Its first run (09 Oct 13:04 UTC) is waiting on a PowerShell permission
   prompt; "Run now" on *Charles race card refresh* and approving its tools fixes every later run.
2. **Share** the phone page with the eight's coxswain, and each deck when it is ready (Share menu on each).
3. **Plan letters A/B/C:** right three situations? Rewrite the wording to your words.
4. **Calls:** fill the bracketed call words in both rowers' decks (yours and the eight's coxswain's).
5. **The eight's arches:** centre throughout, or your Cambridge-Cambridge-centre? (About 3 s for her eight.)
6. **The coach:** the rate plan (base rate, moves, finish), and whether the two shallow stretches fit
   how they want the crew to race.
7. **Requalifying range:** the history says the cut moves ~19 s a year; the model's range is narrower.
   Widen it, or keep it and say so?
8. **Your own notes on the phone:** do you want to write focuses or calls per landmark into the page?
9. **The eight's crew list** (weights, heights, 5 k ergs), if she wants her eight modelled rower by rower.

## 11c. Race-week runbook

All commands from the project folder; Python is `.venv/Scripts/python.exe`.

| when | what | how |
|---|---|---|
| any time | refresh both cards by hand | `scripts/race_card.py --crew four --out data/local/raceday/card_four.json` (and `--crew eight ... card_eight.json`), then write them to the page's database (`cards/four`, `cards/eight`) — the scheduled task does exactly this hourly |
| when an arch plan changes | re-optimise the planned line and re-price the alternatives (a few minutes per crew) | edit `arches` in `CREWS` (`scripts/race_card.py`), then run it once with `--plan` |
| when the field changes (withdrawals, a new feeder result) | re-simulate the field | `research/crew/feeders/charles_update_2026.py` (writes `data/local/raceday/field_<crew>.npz`) |
| start times published | mark them confirmed | set `race_start` and `start_confirmed=True` in `CREWS` |
| race morning | the day factor from earlier events | copy `data/local/raceday/day_results_TEMPLATE.csv` to `day_results.csv`, add a row per finished event (`event,today,last_year`); the next refresh uses it; delete the file after the race |
| race morning | 15-minute refresh | scheduled task `charles-race-morning-refresh` (needs the Claude app open and its tools approved once with Run now) |
| night before | decks | coxswains': https://claude.ai/artifact/BcoXi6fXZDCdnVRFD4jadW; coach: https://claude.ai/artifact/QhYVRU6SnTwLn1HgtN8RGz; rowers (four): https://claude.ai/artifact/Ya37KopwH6av76LUn9mkXe; rowers (eight): https://claude.ai/artifact/Vw1JoPHcoiGP7E1z96co9U. Share each from its Share menu; fill the bracketed calls first |
| before sharing a rowers' deck | check it carries no odds or times | `python -m coxswain.raceday.decks data/local/raceday/decks/rowers` (sources kept in `data/local/raceday/decks/`) |
| after the race | stop the jobs | disable both scheduled tasks; note in the log what was used and what was ignored (the retro) |

The phone page: https://claude.ai/artifact/72R1Wr3wYbZ4k9WTQLVRNP (share with the eight's coxswain from its Share menu).

**The self-updating site** (https://josh-porg.github.io/charles-race-card/, repo `josh-porg/charles-race-card`):
the phone fetches the NWS forecast and USGS flow itself and applies precomputed tables, so weather and
flow need nothing running. The passphrase is in `data/local/raceday/site_passphrase.txt` (local only).
After any change to the plan (arches, field, heat sheet), rebuild and push:

| step | command |
|---|---|
| cards | `scripts/race_card.py --crew four ...` and `--crew eight ...` (as above) |
| wind tables (only when a planned line changes) | `scripts/wind_table.py` (~4 min) |
| bundle | `scripts/raceday_site.py` |
| page (after editing `race_card.html` or `raceday_site_loader.js`) | `scripts/build_site_index.py` |
| encrypt | `.venv-tools/Scripts/python.exe scripts/raceday_encrypt.py` |
| publish | in `data/local/raceday/site/`: `git add -A && git commit -m "Update" && git push` (token via `git credential fill`, docs/RELEASING.md) |


## 12. Log

- 2026-10-08 — Sprint 0: discovery round 1 answered (users, moments, decisions, private link); plan written.
- 2026-10-08 — **Sprint 1 started.** Built `coxswain/raceday/` (`forecast.py`: NWS hourly forecast
  for the course; `card.py`: target time + wind → splits by section between bridges, with
  head/tail/cross labels and the wind felt at 1.5 m), `scripts/race_card.py`, 4 unit tests. Race
  card published as a private page with a database: https://claude.ai/artifact/72R1Wr3wYbZ4k9WTQLVRNP
  (card in `cards/four`; the page keeps the last card on the phone for poor signal). First card: the
  four, requalify 21:16 still air / best 22:29, wind observed now (race-time forecast in range from
  about 10 Oct). Hosting decision: published pages cannot fetch outside data, so a scheduled job
  writes the card into the page's database (to build in sprint 2); the page works on whatever it
  last received.
- 2026-10-08 — Round 2 answered (crews, plan contents, targets, decks, access). Existing race-day
  work found and folded in (§5 findings): one line, weather into splits, plans keyed on traffic
  and arch decisions, the day factor for live updates.
- 2026-10-08 — **Round 3 feedback built.** Goals with ranges (`coxswain/raceday/goals.py`, from the
  simulated 2026 fields `data/local/raceday/field_{four,eight}.npz`); the eight (medal, beat 2025,
  requalify, forecast); the course re-sectioned at the coxswains' landmarks (Magazine Beach 535 m,
  River St, Mile 1, Western Ave, Weeks, Anderson, Mile 2, big turn 3369 m from the line's heading,
  Eliot, wooden pier 4658 m to confirm). Railway and BU bridges within 200 m of the start are not
  sections. Page: crew switch, a target selector for the breakdown, a landmark timeline with be-there-by times and
  stretch splits, each with its range. Both cards in the db (`cards/four`, `cards/eight`).
  **Model fix:** the line evaluator charged wind against a club eight's power (3040 W) for every
  boat, cutting the four's wind cost several-fold; the card now sizes the steering model to the
  boat's own resistance by scaling the wind drag area (`card.wind_area`; the steering model and the
  shared default are unchanged, since sizing the model's power to the boat inflated every turn). In
  a 22 mph WSW wind the four's line costs about +20% and the eight's +15% (previously +3% and +6%). The eight is a generic
  masters eight (68 kg, 1.68 m) until its crew list is in. 6 unit tests.
- 2026-10-08 — **Round 3b.** Target selector above the course breakdown (Medal / Last year /
  Requalify / Forecast for the 8+), which stays on screen while scrolling. Magazine Beach replaced by **Powerhouse**, the start of the
  straight at the end of the Magazine bend (983 m, from the line's heading), the point Joshua calls.
  Wooden pier confirmed on the Boston side (kept at 4.66 km, 160 m before the line); big turn
  (3.37 km) confirmed. Calm splits are even along the course (2:11-2:13 for the four), so slow
  stretches on a windy card are the wind, not the steering model.
- 2026-10-08 — **Turning cost from our own boats.** Joshua: at Tail of the Lake a hard rudder visibly
  slowed the split. Measured from the two CBGPS per-stroke files (`research/raceday/rudder_totl.py`):
  speed lost against GPS turn rate, at the crew's pace either side, fits dv/v = k omega^2, k = 0.0071 +-
  0.0009 (four), 0.0099 +- 0.0009 (eight), omega in deg/s; the four's called turn at 1.7 km lost 4% at
  1.8 deg/s. Applied on the Charles from the line's curvature x speed (`card.turn_rate`, heading
  smoothed over 40 m because the chart's centreline has 150-200 m S-wiggles no cox steers). Turning
  costs the four ~3 s and the eight ~4.6 s over the course, most in the big turn to Eliot (1.1 / 1.7 s)
  and after Eliot (0.6 / 0.9 s). Target totals are unchanged; the time moves into the bends. The card
  shows a rudder chip per stretch with how far the split drifts at the hardest point. Because the loss
  goes with the square of the turn rate, a long even arc is cheaper than a late hard one (for the
  race plan, R3/R4). Caveat: two races, and turns there coincided with pressure calls.
- 2026-10-08 — **R3 started: key decisions (arches).** `coxswain/raceday/plan.py` prices every legal
  arch combination at the three bridges with a choice (River Street, Western Avenue, Weeks: centre or
  Cambridge shore; HOCR rules forbid the Boston arch everywhere and the Cambridge arch at the
  trestle, Anderson and Eliot), with the card's evaluator and turning cost, in calm and today's wind.
  Four / eight: **River Street Cambridge arch 1-4 / 0-2 s quicker** (16 m shorter, the inside out of the
  Powerhouse bend; -2 to -5 s for the four across six winds, largest in a northerly); **Western Avenue
  Cambridge +3-4 s**; **Weeks Cambridge +14-20 s** (41 m longer into the bend). Experienced sources:
  Knauth's steering guide ("center arches all the way"; drift wide to port after the blockhouse onto
  the Powerhouse centreline), Heyman 2011 (Powerhouse side "makes little difference", maybe in E/NE
  winds; Weeks and Eliot right-hand arches are "arches of last resort", easing is cheaper). Both agree
  with the model's direction; the model's River Street gain is larger than Heyman's "little".
  So the plan says: centre throughout; River Street's Cambridge arch is a free alternative whenever
  the centre is busy; Western's a cheap one; Weeks' a last resort. The lines compared are arch-pinned,
  not optimised within each choice. Page: a key decisions card with a verdict per bridge. 8 tests.
- 2026-10-08 — **Arch plan from Joshua, and a correction.** Joshua's plan: Cambridge arch at River Street
  and Western Avenue, centre at Weeks. Giving each arch choice its own optimised line
  (`plan.optimised_time`: the arch points fixed, the line free at 1100/1610/2030/2500 m, Nelder-Mead on
  time with the turning cost) **overturns the River Street finding above**: against an optimised
  centre line, the four is centre +0, Cambridge at River St only +1.2 s, Joshua's plan +1.0 s; the
  eight is +0 / +1.0 / +2.7 s. The "1-4 s quicker" came from a poorly drawn centre baseline
  (Heyman's "little difference" was right). Differences under 1.5 s are reported as "either: choose
  by traffic" (`plan.EQUAL_S`). The four's card now follows Joshua's plan; the eight's stays on the
  centre arches until the eight's coxswain says otherwise (her line through both Cambridge arches costs about 3 s).
  The planned line and the priced alternatives are optimised the night before
  (`scripts/race_card.py --plan`, a few minutes) and cached in `data/local/raceday/plan_<crew>.json`.
- 2026-10-08 — **Plans optimised and on the card.** Against the four's planned line (Cambridge, Cambridge,
  centre), each with its own optimised line: River St centre -0.3 s (either), Western Ave centre -1.6 s
  (the plan pays it for traffic and a straight run), Weeks Cambridge +9.8 s (last resort). Eight on
  centre throughout: River St Cambridge +0.9 s (either), Western Cambridge +1.8 s (cheap), Weeks
  Cambridge +10.6 s. Advice is rebuilt from the cached prices on every card run. Fixed: times rounding
  to "13:60.0" (Python and the page).
- 2026-10-09 — **Auto-refresh on** (Joshua said yes). Scheduled tasks `charles-race-card-refresh` (hourly
  05:00-22:00, 9-18 Oct) and `charles-race-morning-refresh` (every 15 min 04:00-11:59 on 17 Oct) run
  `scripts/race_card.py` for both crews (cached plans, never `--plan`) and write `cards/four` and
  `cards/eight` with version pins. They run while the Claude app is open.
- 2026-10-09 — **R6 draft: coxswains' deck** (13 slides, browser): https://claude.ai/artifact/BcoXi6fXZDCdnVRFD4jadW
  (source copied to `data/local/raceday/decks/coxswains/`). Course by landmarks; one line, weather into
  splits; which stretches each wind hits; measured turning cost; the big turn; arches (Powerhouse within
  the model's precision, Weeks Cambridge +10 s); passing rules in open water; push in the shallows;
  same-river day factor; the phone card; a confidence table; questions for the experienced coxswains.
  No other crew's odds. New finding on the way: on the planned lines the two slow stretches in still air
  are Powerhouse to River Street (1.8-2.4 m; four +1.7 s/500) and Anderson to Mile 2 (2.1-2.7 m; four +1.3,
  eight +3.4 s/500): the faster boat pays more in the shallows.
- 2026-10-09 — **R5 traffic, R11 moves, R12 flow, R9 day factor** (while Joshua was away, under /goal).
  *R5* `coxswain/raceday/traffic.py`: bows from the published entries, every other crew's time from
  the field simulation (same day as ours), starts 15 s apart (15-20 s reported: illinirowing.com,
  boston.com); a pair draws level at x = L dt / (T - T_i). Four (bow 21 of 26): passes made 1.2,
  passed by 2.2, clear race 1%; bow 22 passes us 57% (most likely before Powerhouse); a pass within
  100 m of River St / Western / Weeks 17-18% each. Eight (bow 7 of 31): made 0.5, passed by 0.3,
  clear race 41%; we pass bow 6 30% (near Weeks). On the card: a traffic section (neighbours by
  chance, where, and the bridge chances). Constant speed per crew is the simplification.
  *R11* push markers: stretches where the river (depth, current; turning removed) costs >= 1.0 s/500 in
  still air: Powerhouse to River St (four +1.7, eight +1.4) and Anderson to Mile 2 (+1.0, +2.8).
  *R12* `coxswain/raceday/flow.py`: USGS 01104500 live; 134 cfs today (Oct median ~99): +1.7 s (four),
  +1.3 s (eight); 10 m3/s would be +13 s, 30 m3/s +50 s.
  *R9* `coxswain/raceday/dayfactor.py`: no live results service is confirmed (searches found team
  recaps only), so the input is a CSV anyone can fill on the morning (`data/local/raceday/
  day_results.csv`, template beside it): earlier events today against last year; the median ratio
  (bootstrap 80% interval) replaces the wind model's factor on the totals (`card.rescale`). The field
  model's goals rest on 2025 Charles times, so today/last-year is the consistent scaling.
  *Found and logged* (TRACKING, open): `RouteEvaluator.elapsed` carries a feasibility penalty that
  multiplies Charles lines ~40x (5 m-scale yaw on the chart's centreline wiggle); the card never used
  it, the first flow estimate did and was switched. Plans checked steerable at the GPS scale (peak
  <= 3.0 deg/s); `plan.MAX_TURN` now guards the optimiser. 11 unit tests.
- 2026-10-09 — **R7, R8 drafts, plan letters, map (R4).** Coach deck, 12 slides:
  https://claude.ai/artifact/QhYVRU6SnTwLn1HgtN8RGz (plan first; model inputs; targets with ranges; wind with
  the correction; turning chart from the TOTL GPS; arches on optimised lines; shallows; traffic; what
  would change the plan; asks). Rowers' deck, 8 slides: https://claude.ai/artifact/Ya37KopwH6av76LUn9mkXe
  (landmarks, plan, big turn, traffic, wind, calls with bracketed placeholders for Joshua's words);
  `coxswain/raceday/decks.py` checks it has no percentages, chance words, requalify/medal goals or
  times (unit test runs it on `data/local/raceday/decks/rowers` when present). Plan letters A/B/C on the
  card (`plan.variants`): A clear water, B the likeliest traffic (four: passed early by bow 22; eight:
  catching bow 6), C a hard headwind; tap to choose at the start, remembered on the phone; wording is
  the coxswain's to change. R4 map: `coxswain/raceday/map.py` (banks from the channel raster, planned
  line, bridges, push stretches; 6.6 kB in the card, drawn as SVG on the phone); a slide version in
  both decks. Wind validation: TOTL (4 Oct) and the 2 and 7 Oct practices were all near calm (KBFI, KSEA,
  KRNT under 3 m/s), so no on-water wind test is possible yet; station records saved. 2026 start
  times still unpublished (searched; 2024 slots kept, marked assumed). 14 unit tests.

### Backlog status (2026-10-09)

| id | status |
|---|---|
| R1 target splits | done: goals with ranges, splits and be-there-by times per landmark, both crews |
| R2 wind | done: NWS forecast at the start (observation until in range), head/tail/cross per stretch, cost with range |
| R3 race plan | drafted: arch decisions priced on optimised lines; plan letters A/B/C; Joshua's arch plan; the call wording is the coxswains' |
| R4 line and bridges | done: optimised planned line, turn-rate guard, map on the card and in the decks; approach clearance between control points not separately checked |
| R5 traffic | done: neighbours, where passes happen, bridge chances (15 s interval assumed) |
| R6-R8 decks | drafts: coxswains' (14), coach (12), rowers' (8, checked for no odds) |
| R9 day factor | built: CSV of earlier events on the morning; no live results service confirmed |
| R10 poor signal | the page keeps the last card on the phone; no offline install (artifact hosting) |
| R11 moves | done: push stretches from still-air river cost |
| R12 flow | done: live USGS, effect on the line |
| Auto-refresh | scheduled hourly to 18 Oct and every 15 min on race morning; awaiting the first run's tool approval |
- 2026-10-09 — Rowers' deck for the eight (https://claude.ai/artifact/Vw1JoPHcoiGP7E1z96co9U; centre arches,
  the eight's quieter traffic, the eight's coxswain's calls as placeholders; passes the no-odds check). Race-week runbook (§11c).
- 2026-10-09 — **Check on the four's requalifying target** against history (`data/hocr_wvet4_results.csv`,
  60-69 band, place = half the field): cut 21:45.7 (2019), 21:18.1, 21:01.6, 20:58.5, 21:38.9, 21:28.4 (2025);
  mean 21:22, year-to-year SD 19 s (1.5%; cut/winner SD 0.019). The model's 2026 central 21:15.7 sits
  inside that history; its 80-20 range (+-11 s, an SD of about 13 s) is narrower than the raw
  year-to-year spread. Some narrowing is right (the 2026 entries are known, which removes field
  composition), but the range may be a little optimistic. Not changed: flagged for the coach deck's
  confidence slide and for Joshua (YEAR_SD 0.013 in `charles_update_2026.py` is the lever). Note the
  history counts the 60-69 band; the 2026 field model counts all 26 entries (60+ and 70+), as the
  entry list does.
- 2026-10-09 — **Lake Sammamish training splits** (the round 3 question, answered by building it):
  `scripts/lake_equivalents.py` runs `scripts/splits.py` (same power, deep still water; SOURCES §87) for
  every goal and the card shows it under each goal. Four: requalify 2:12.1 on the Charles = 2:09.9 on
  Sammamish; forecast 2:18.5 = 2:16.8 (the Charles 1.7-2.3 s/500 slower). Eight: medal 1:49.8 = 1:43.5,
  beat 2025 1:52.9 = 1:47.4, requalify 1:56.6 = 1:52.0, forecast 1:54.6 = 1:49.5 (4.6-6.3 s/500): the
  faster boat sits near the shallow-water drag rise, so the eight's lake targets must be well quicker
  than its Charles splits. Category boats from `splits.py`, not our crews rower by rower.
- 2026-10-09 — The card is insensitive to the eight's guessed rower mass (60/68/76 kg give identical wind cost and section shares): the wind area is scaled to the boat's own resistance, so the boat cancels; the card depends on the target pace, not crew detail.
- 2026-10-09 — **Page checked in a browser** for the first time since the db version (a local copy with
  the real cards, phone size, both themes): no console errors, no sideways scroll; crew switch, target
  selector and plan letters work. Fixed: times showed in the viewer's own zone (the 10:32 ET start read
  07:32 on a Pacific phone); all times now in Boston time with "ET". Fixed: the push shading on the
  map was near-invisible in dark mode. Local preview: `raceday-preview` in the parent folder's
  `.claude/launch.json` serves `data/local/raceday/` (open `_preview.html`, rebuilt with the cards).
- 2026-10-10 — **Race-day forecast in range** (12-hour period; `forecast.daily_at`, used until the hourly
  reaches Saturday morning): Saturday W 12-15 mph, 36% chance of showers, ~15 C. A westerly is a
  headwind from Weeks to the finish; the card puts it at +12% for the four's line (+8 to +18%) and +9%
  for the eight (+6 to +13%). **Chances on the card** (`goals.chances`; coxswains' page only, never the
  rowers' decks), each goal at our forecast on the same day as the field: eight — win 0.03%, medal 1%,
  beat 2025 (7th or better) 18%, requalify 87%, most likely 11th (6th-16th); four — requalify 4%, most
  likely 21st of 26 (16th-24th). The coach deck's wind slide carries Saturday's forecast. Heat sheet:
  published on RegattaCentral (Joshua), which refuses scripted access and was refused in the browser
  pane; `scripts/heat_sheet.py` imports pasted rows (bows into the entries file in place, the start time
  into `start_times.json`, read by the card). The refresh job's first run (9 Oct) is still waiting on a
  permission prompt. 17 unit tests.
- 2026-10-12 — **Hourly forecast reaches the race.** Saturday 04:00-13:00 ET: northwest 16-18 mph, showers
  clearing overnight (rain 15-25% by the starts), mostly sunny by 10:00. The card (forecast for the
  start, 7.6 m/s from 315): four's line +18% (+11 to +27%), eight's +14% (+8 to +20%). With a
  northwesterly the headwind runs from the start to the big turn (Weeks-Anderson across), and it is
  across from starboard from the big turn to the finish; the Powerhouse stretch is sheltered (1.6 m/s
  felt). Rowers' decks' wind slide rewritten for it (and corrected from the seat: a headwind is at the
  rowers' backs, and "starboard", not "right"). Checking whether the line should change for it.
- 2026-10-12 — **Start order and times confirmed** (Joshua's go-ahead; RegattaCentral read in the browser
  pane): the heat-sheet page is still empty, but the event and competitor pages are published. Women's
  Grand Master Eights (event 17) 9:34 AM and Women's Veteran Fours (event 22) 10:36 AM, Saturday 17 Oct
  (tentative per the regatta). Bow order identical to the entries we model (four bow 21 of 26, eight
  bow 7 of 31; every bow checked), so the traffic analysis stands. `data/local/raceday/start_times.json`
  carries the times; the cards now read "confirmed". **Line in Saturday's wind:** re-optimising each
  planned line for NW 7.6 m/s (warm-started from the planned line) gains 0.9 s (four) and 1.3 s (eight)
  and moves the line under 1 m on average: keep the line. (A cold start from arch-to-arch lines came out
  worse than the planned line, an optimiser artefact; `plan.optimised_time` now takes `start_route`.)
  Plan C now names the forecast's headwind stretches (`plan.headwind_stretches`). 18 unit tests.
- 2026-10-12 — **Self-updating site (free, no laptop, no credits).** Joshua: keep both pages, move to GitHub;
  keep only the race-morning laptop job once the site works. `scripts/wind_table.py` precomputes each
  crew's card against 112 winds (0-12 m/s x 16 directions; factor, section shares, felt wind, turning);
  bilinear interpolation reproduces the card within 0.1% at Saturday's forecast (four +18.07 vs +17.99%,
  eight +13.63 vs +13.57%). `scripts/raceday_site.py` bundles cards, wind and flow tables;
  `scripts/raceday_encrypt.py` (.venv-tools, `cryptography`) encrypts it (AES-256-GCM, PBKDF2-SHA256
  250k); the page (`scripts/raceday_site_index.html`, generated by `build_site_index.py` from the card
  page plus `raceday_site_loader.js`, one renderer for both) unlocks with the passphrase, fetches NWS
  (hourly at the start, else the 12-hour period, else Logan now) and USGS (both allow browser
  requests), and rebuilds the card; race-morning day factor typed into the page (kept on the phone).
  Tested locally: wrong passphrase refused; live forecast (NW 17 mph at 10 ET) and flow (208 cfs,
  +5.5 s after rain); day factor 3% applied exactly; fixed the wind range showing under a measured day
  factor. Published: https://josh-porg.github.io/charles-race-card/ (public repo, encrypted data only).
- 2026-10-15 — **Short course, the river on race day, the field, password.** (1) *Short course* (high
  wind): start off Riverside Boat Club, same finish. Riverside projects to 1107 m on our chart, leaving
  3.73 km (Joshua's estimate 3.2 km: to reconcile; the start landmark is a parameter, `card.SHORT_START`).
  `card.short_course` (Python) and `shortOf` (site) give the short card: sections from Riverside, the
  full course's times from there on, its own traffic (times scaled by the remaining share). Both pages
  have a Full / Short switch. Short targets now: four requalify 18:41; eight medal 14:46, requalify 15:47.
  (2) *River*: no official forecast at Waltham, but the National Water Model forecasts the reach: 181 cfs
  now rising to ~570 cfs (16 m3/s) by Saturday morning after rain. On the planned lines that current
  costs the four +24 s and the eight +18 s (`river_conditions.py`); basin level at the dam +0.06 ft on its
  median (5-95% range +-0.45 ft is worth +-3 s four, +-6 s eight). Both now in the targets (every crew
  pays them), on the card and the site (NWM and USGS are browser-readable). (3) *Field*: the published
  competitor lists (xlsx) match our entries bow for bow. Older Charles results added for crews without a
  2025 time (2021-2024, scaled by the field medians, uncertainty growing with the years): fours
  Saratoga, Minneapolis, Green Lake, Lincoln Park, Newport; eights **Marlow (2024 winners, now favourites
  for the medal)**, DeWolfe, Riverfront, Blood Street, Holyoke. Eight medal chance 0.3%, requalify 88%;
  four requalify 6%. Web search found no autumn 2026 results for the remaining 14 crews. (4) Site
  password set (held in `data/local`, not here); email to the eight's coxswain drafted in `data/local/raceday/`.
  (5) The hourly laptop job now runs (approved).
- 2026-10-15 — **Analysis for the session: the Charles in 2026** (`research/raceday/analysis_2026.py`,
  `scripts/charles_2026_report.py` -> `out/report/charles_2026.html`, https://claude.ai/artifact/5r6MfgCQ43n33WupFUGVbF).
  Saturday on the planned lines at requalifying pace, against still air / slack / deep water: four wind +181 s
  (65 s of it the first km into the north-wester), current +29 s, shallows +16 s, turning +4 s; eight wind
  +107 s, current +22 s, shallows +34 s (twice the four's: nearer the depth drag rise), turning +6 s.
  Re-optimising the line for Saturday's wind and current gains 0.6 s (four) and 1.7 s (eight), moving it a
  metre or so: keep the planned line. The September reports in out/report predate the week's corrections
  (wind area, measured turning, the evaluator's penalised `elapsed`); this page supersedes their numbers.
- 2026-10-15 — **Briefing map (live, interactive)**: https://josh-porg.github.io/charles-race-card/briefing.html
  (`scripts/raceday_briefing.html`, published by `raceday_encrypt.py`; same passphrase). For a laptop or
  projector: the planned line coloured stretch by stretch by Saturday's live wind (NWS for each start),
  live river flow (NWM) and basin level (USGS); "where the time goes" per stretch (wind from the wind
  tables; current from `analysis_2026.json` scaled by the live flow; shallows and turning from the
  analysis); click a stretch for its costs, push note, arch advice and the big-turn note; pan and zoom;
  crew and full/short course switches; **rowers' view by default** (no targets), coxswains' view adds
  splits and be-there-by times. Map data now carries each stretch's outline (`map.section_lines`).
- 2026-10-15 — **Briefing map layers and the possible lines** (`research/raceday/fields_2026.py`). Pace fields
  over the raced water (25 offsets x 161 stations, +-72 m): the loss against the planned line's pace at the
  same station, by cause (depth, current, wind) and total, plus raw depth, current (Saturday's flow) and
  wind felt (the card's forecast when built). Local pace only: getting there costs distance, which the
  line comparison carries. Lines timed on Saturday with a breakdown against the planned line (distance,
  depth, current, wind, turning). **Finding: "inside of the bends" is 25 s (four) / 15 s (eight) faster
  than the planned line**: 83 m shorter, 50 m of it in the big turn, where the planned line went round
  ~6 m to starboard of centre (the outside of a port turn) because its optimiser only had free points
  before Anderson. The inside line is steerable (peak 2.35 deg/s), deep (>= 2.16 m) and arch-legal, but
  the course's buoy lines are not in the model. Re-optimising both planned lines with free points through
  the big turn and after Eliot, kept >= 15 m off the bank as a stand-in for the buoys (`bend_reopt.py`).
  Also: "shortest legal" runs aground (0.95 m) and costs the eight 40 s in the shallows.
- 2026-10-15 — **Bend-optimised lines** (`bend_reopt.py` -> `plan_<crew>_bends.json`; same arches, free points
  through the big turn and after Eliot, kept >= 15 m off the bank where free, ~11 m at the closest between
  points): still air four -11.8 s, eight -7.9 s; on Saturday's forecast four -9.7 s, eight -7.2 s (distance
  -14.8 / -11.4 s, given back: depth +3.2 / +3.7, wind +3.7 / +1.6). 55 m shorter. In the briefing map's
  comparison as "Planned arches, bends optimised (15 m off the bank)". NOT adopted as the planned line:
  the buoy lines are not in the model; Joshua and the coach decide. Straightened course view added to the
  briefing map (station x offset, width x10).
- 2026-10-16 · **Gusts.** The hourly NWS forecast leaves `windGust` empty, so the card never had gusts. The
  gridded forecast has them: Saturday 26 mph at the eight's start, 28 mph at the four's (sustained 15-16 mph
  NW, gust factor ~1.8). `coxswain/raceday/gusts.py` turns them into a drag allowance (turbulence intensity
  (G-1)/3; half the wind table's curvature x the speed variance, so the table's linear interpolation does not
  bias it): four +6 s, eight +3 s. Off by default (the model was calibrated on sustained winds); a "Steady wind
  / Count gusts" switch on the race card (GitHub site and the claude.ai page), the briefing map and the 2026
  report. Every crew pays it, so chances are unchanged. Not modelled: steering corrections in gusts, direction
  swings. A measured day factor already contains gusts; the switch then does nothing.
- 2026-10-16 · **Map wind layers fixed.** `fields_2026.py` evaluated local pace with no steering model, so the
  wind was charged against the evaluator's 260 N default instead of ~840 N: the "from wind" and "total" layers
  were 3.2x too strong (+67 s/500 mean on the centreline vs the card's +21). Also the forecast tuple overwrote
  the `wind` layer key, so "...from wind" drew nothing. Both fixed; a 40 m speed lag added (the boat's momentum
  through shelter patches). The line comparison was never affected (it uses the card's evaluator). On the
  15 mph forecast the bend-optimised line is now -9.3 s (four) / -6.8 s (eight).
- 2026-10-16 · **River forecast fallback.** NOAA's medium-range flow (and its blend) came back empty on 16 Oct
  and the 18 h short range ends Friday 23:00 ET, so the cards fell back to the flow observed now (181 cfs)
  while the short range shows the river rising to ~440 cfs. `flow.forecast_at` (and the site and briefing)
  now use the short range's last hour when the start is within 18 h beyond it: four requalify 24:53, eight
  medal 19:43 / requalify 21:03 on the full course. The short range covers the starts from about 17:00 ET Friday.
- 2026-10-16 · **Both practised arch lines.** Each crew's other line (Cambridge or centre at River Street and
  Western Avenue, centre at Weeks) optimised for that crew: `race_card.py --arches {cambridge,centre}` (cached
  `plan_<crew>_<option>.json`, `card_<crew>_<option>.json`, `wind_table.py four:centre`). An Arches switch on the
  phone card (splits and landmarks per line, and today's difference between them, live) and both lines on the
  briefing map. On Saturday's forecast centre is quicker for both: four -2.7 s, eight -3.9 s (map comparison;
  the phone page -1.9 / -3.8 s live), near the model's precision.
- 2026-10-16 · **Results, live calibration and a backtest.** RegattaCentral's event pages carry each event's
  results (with Riverside / Weld / CBC splits) as JSON; scripts are refused (Cloudflare), so they are read by
  visiting the pages in the browser (`scripts/rc_event_summary.js`; in-page fetch bursts get challenged, and a
  local receiver was dropped). Its timestamps encode the Eastern wall clock as Pacific (ET = epoch - 7 h); the
  pages' own strings say 09:34 / 10:36 US/Eastern, so our starts are right. Summaries for 2024, 2025 and 2026
  Friday: `data/local/raceday/results/hocr_event_summaries.csv`. Backtest (`results/backtest_dayfactor.py`,
  2024->2025 and 2025->2026 Friday): mean absolute error of an event's median, all / with an earlier finished
  event that day: last year as is 1.6 / 1.7%; wind+flow model on Logan's observed wind 3.8 / 3.7%; same-day
  ratio 1.5 / **1.2%**; model corrected by same-day 3.4 / 2.6%. Day pairs, actual vs model: Fri 25->26 -3.5 vs
  -7.3%; Fri 24->25 +3.2 vs -1.1%; Sat 24->25 +0.2 vs +3.3%: the model gets the direction but about twice the
  size (Logan is not the course, and fields change). Not recalibrated on three day pairs; the same-day factor
  replaces it on the morning. Friday 2026 ran ~3.5% faster than Friday 2025 (SSE tailwind).
- 2026-10-16 · **Live day factor tomorrow.** `scripts/day_from_results.py` (`due` / `add` / `build`) turns the
  finished Saturday events (`results/rc_events_2026.json`, events 9-22) into `day_results.csv` against 2025's
  medians / 1.02 (2025 ran 2% slower than the still-air goals: events 17 and 22 medians against the still-air
  requalify times 1.021 / 1.018). The `charles-race-morning-refresh` task now reads the results, builds the
  factor, refreshes all four cards and pushes the site; the site uses the card's day factor unless one is typed in
  on the phone (both now on the same 1.02 base). Expect a large move when it switches on: the wind model puts
  Saturday's 15 mph NW at +15% for the four; the backtest suggests the real effect is nearer half that.
- 2026-10-17 · **Race day retro (first notes).** Both scheduled jobs disabled after the race. The race-morning job
  never ran during racing: the app was not open, and its one run started at 12:32 ET, so the cards stayed on the
  wind model and the live day factor was not used. Lesson: keep the laptop awake with the app open (or start the
  job by hand) on the morning. Race video synced and rendered with the overlay (layout A, title card; CoxBox
  start at video 1:06.5, called splits agree to 1.3 s): data/local/overlay/videos_titled/hocr2026/.
- 2026-10-19 · **Race review** (artifact https://claude.ai/artifact/9HJDekNn7LcGaQ8agY5JE6; data/local/raceday/
  race_review_2026.py, race_review_opt.py, race_review_page.py). Four: 21st of 26, 24:07.7 (GPS 23:54.7), centre
  arches at River Street, Western and Weeks. Eight: 8th of 31, 18:59.1. The day: KBOS 18 mph gusting 26 from 330;
  flow 196 cfs (NWM had forecast ~440); Saturday fields 4.9% slower than 2025 (14 events), about +7% on still
  water, against the card's +15% (forecast wind) and +26% (observed wind) for the four: the wind model overstates
  by 2-3x. Calibrated wind 3.85 m/s (0.40 of the open reference). Our line was 2.9 s off the optimum through the
  same arches (optimum 33 m longer, out of the shallows: depth +12.2, distance -8.7). The day-calibrated card would
  have said requalify 22:53 / forecast 23:51 (actual median 23:20.7; ours 24:07.7). bestline.row3r.com (13 public
  tracks, no .fit): their shortest line is 8.6 s slower than the optimum in our model; only one of their lines
  beats ours (data/local/bestline/). Request for the .fit files drafted: data/local/bestline/email_to_row3r.txt.
- 2026-10-19 · **Race review on the site, encrypted.** https://josh-porg.github.io/charles-race-card/review.html
  (passphrase as the race card). research/raceday/race_review_site.py (.venv-tools): the page is a locked shell;
  text, numbers, GPS track and replay are in review.enc (AES-256-GCM, PBKDF2-SHA256 600k). noindex meta; not linked
  from the race card. Rebuild: race_review_page.py, then race_review_site.py, then push site/.
- 2026-10-21 to 10-23 · **Post-race analyses** (each a TRACKING Done row): the HOCR race added to the coxing paper's
  sessions; line and race reviews for any crew (`REVIEW_CREW`); a blind comparison of six club coxswains' lines with
  intended line vs steering; a Radau-collocation racing line; wind vs steering on one-minute ASOS; skill vs
  externalities; adaptation to conditions (Monte Carlo); coxswains as human operators; HOCR arch rules and the line
  optimiser's bridge points fixed (TRACKING, Fixed); depth flagged as overstated (banner above).
