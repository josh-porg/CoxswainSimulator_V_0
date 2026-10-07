# callmodel — calls and boat as one coupled process

A coxswain's calls and the boat's response are two time series that each carry their
own momentum and may shift each other. This package models them jointly,

    p(C, B) = Π_t  p(C_t | C_<t, B_<t) · p(B_t | B_<t, C_<t),

and measures the two coupling terms on held-out data:

* **C→B**: how much the call history adds to predicting the boat, given the boat's own past
* **B→C**: how much the boat history adds to predicting the next call, given the calls' own past

Two models are fitted to the same folds: a causal transformer, and a linear Hawkes-style
autoregression that serves as the reference.

## Data: one folder per session

```
<root>/sessions/<id>/
    meta.json     required  {"id", "kind", "boat_class", "date", "source", optional "t0", "t1"}
    words.jsonl   optional  {"t": s, "w": word} per line           the call stream
    boat.csv      optional  t,speed,rate  (s, m/s, spm)             the boat stream
    labels.json   optional  [{"t": s, "<field>": label}, ...]       richer call codes
```

Any subset of the optional files is valid:

| session has          | trains                       |
|----------------------|------------------------------|
| transcript only      | the call stream's own dynamics |
| boat log only        | the boat's own dynamics      |
| both, on one clock   | the coupling, and both of the above |

Raw data stays under `data/local/` (gitignored). Nothing in this package holds data.

## Adding data

```bash
# every transcript in a folder of YouTube .vtt files, as call-only sessions
python ingest.py captions --raw <vtt dir> --index channel_index.txt --root <root>

# one synchronised session: transcript plus a boat log (CoxOrb/CoxBox/GPS export)
python ingest.py session --root <root> --id race_04 --vtt race.vtt --boat log.csv --offset 1.0
```

An NK LiNK export (CoxBox Core, SpeedCoach, CBGPS) goes in directly with `--export "file.csv"`
in place of `--boat`; per-stroke Empower oarlock fields (power, catch, slip, finish, force) are
kept when the export has them.

`--offset` is the seconds added to a caption time to reach the boat log's clock. Find it by
sliding spoken split readings against the logged splits.

## Short-horizon responses

`short_horizon.py` asks the coxswain's own question: does the boat change within three to five
strokes of a call? Each call episode is compared with the four strokes before it and with what
the boat's own momentum (an AR(4) on per-stroke speed) predicts, against a circular-shift null.
Sessions add `phrases.json` and `labels_phrases.txt` (function and valence per phrase). A
`control` session — a warm-up of builds, paddles, stops and starts — is the positive control.
`test_short_horizon.py` plants a response in synthetic sessions and checks it is recovered,
that no effect gives chance, and that momentum is not mistaken for a call effect.

## Theory-derived call properties

`call_properties.py` is the measurement model of the review "What makes a call work?". Phrases are
hand-coded from the transcript only (one line per phrase: target, stage, level, action, scaffolding
function, focus, reference, discrepancy, appraisal, form, arousal; the codebook lives with the
data). It builds call episodes (first phrase of a same-target run), computes the relational
properties from the boat (familiarity of the wording, contingency on the speed change before the
call, intensity), delivery from the audio (loudness, pitch, speech rate), and responses beyond an
AR(4) forecast on speed, rate and distance per stroke up to 20 strokes. Inference shifts each
session's call strokes together, globally or locally (10-40 strokes, which keeps place in the
piece); properties defined from the boat's state must be recomputed inside the statistic.
`test_call_properties.py` checks episode unions, recovery of a planted response, chance for a
relational property under no effect, and Holm.

## Running

```bash
python run.py --root <root> --folds 5 --nulls 50 --out results.json
python selftest.py          # positive control on synthetic sessions with known coupling
```

`run.py` reports held-out log-likelihood gains in nats per step: each stream's own-past
gain over no history, and the two coupling gains with circular-shift p-values.

## Design choices

* **Grid.** 2 s steps, about one stroke. Missing data are masked, never filled.
* **Boat scaling.** Speed and rate are z-scored within each session, which removes crew and
  boat-class level differences. No centred detrending is applied: a centred moving average
  uses future samples and inflates apparent predictability.
* **Stream dropout.** The transformer is trained with the whole boat or call input of a window
  removed at random. One set of weights then gives both the own-past and the both-pasts
  predictions, and single-stream sessions train it without special handling.
* **Folds.** Synchronised sessions are split into contiguous blocks; single-stream sessions are
  assigned whole to folds. Held-out rows are never used as training inputs; a training segment
  that follows a test block starts fresh.
* **Null.** The other stream is circularly shifted within each test session. This keeps each
  stream's own structure and destroys only their alignment.

## Modules

| file          | role |
|---------------|------|
| `captions.py` | rolling YouTube captions → de-duplicated (time, word), per-word timing |
| `encode.py`   | words → call channels (keyword categories, or labels from `labels.json`) |
| `sessions.py` | the session format and the common time grid |
| `ingest.py`   | build session folders from raw transcripts and boat logs |
| `model.py`    | causal transformer with Gaussian boat and Bernoulli call heads |
| `train.py`    | folds, segments, training with stream dropout and early stopping |
| `evaluate.py` | linear reference, cross-validation, coupling and null |
| `run.py`      | command line |
| `selftest.py` | synthetic positive control |

## Environment

A separate virtual environment in `research/callmodel/.venv` (torch CPU, numpy, scipy,
scikit-learn). Do not install torch into the game's environment.
