# The coxswain as the performer

Notes on the literature about what a coxswain actually *does*, and what
this project could build to train it.

Everything else in this repository answers "what is true about the boat".
This file is about the gap between that and the job, because the job is a
real-time verbal performance and the physics is only its raw material.

---

## Nugent, De Toledo, Myers & Kearney (2025)
### *What do elite rowing coxswains say during races?*
International Journal of Sports Science & Coaching 20(5), 2109–2117

Thematic analysis of eight elite cox recordings (World Championships,
U23s, World Cup, Henley semis and finals, 2011–2022). Six crews won,
two came second. Five male coxes, three female. Races 5:31–7:36.

### The numbers worth memorising

| | |
|---|---|
| **Call rate** | **32 per minute — one every 1.9 s** |
| Technical calls | 40.4% |
| Motivational calls | 38.6% |
| Tactical calls | 21% |
| Directed at the whole crew | 94% |

The call rate is the striking one. Nugent compares it directly: boxing
coaches manage **8 statements per minute** in the break between rounds;
basketball coaches **2.54 per possession**. A coxswain talks at four
times a boxing corner and does it for six minutes without stopping.
There is essentially **no silence in an elite race**.

That reframes the coxswain's problem. It is not "what should I say", it
is **"I have ~190 slots in a six-minute race and a hard limit on how much
a working crew can absorb"** — Nugent explicitly raises overload as the
tension: enough direction without saturating the athletes.

### Attentional focus — where practice departs from the evidence

This is the finding with the most leverage, and it is a *criticism* of
elite practice rather than an endorsement.

| focus type | definition | examples from the tapes |
|---|---|---|
| **Internal (IF)** | the body movement itself | 'legs down', 'hands up', 'on the heels', 'through the toes' |
| **External (EF)** | the effect of the movement on the environment | 'blades in', 'footplate', 'in and on through the front' |
| **Holistic (HF)** | the general feel | 'stay loose', 'rhythm', 'squeeze', 'long', 'stay clean' |

**Every cox in the study used IF cues heavily. EF cues had limited use.**

And the motor-learning literature (Wulf) is consistent that **EF and HF
outperform IF** across tasks, skill levels and ages. Neumann's rowing
study found the best 2000 m ergometer performance in a group *switching*
between IF and EF every 250 m — better than either alone. Schücker found
IF cues *raised* VO₂ at the same work, i.e. worse movement economy.

Nugent's own words: the coxes' heavy IF use "appears to conflict with
guidance from research."

**This is a directly actionable gap.** Elite coxes are not necessarily
optimal — they are elite at everything else. A cox who deliberately
shifted the IF/EF/HF mix, or who switched systematically the way
Neumann's best group did, would be doing something the tapes show elite
coxes are not doing.

Two honest caveats. Rowing is cyclic and continuous, unlike the discrete
tasks (putting, darts, jumping) most attentional-focus research uses, and
Nugent flags that the evidence may not transfer cleanly. And the IF calls
cluster on the legs — which is where 45% of propulsive force is generated
(32% trunk, 23% arms), so the cox's attention is at least aimed at the
right part of the stroke.

### Delivery — the part that is craft

**Calls are timed to the phase of the stroke.** 'Sharp' at the catch.
'Legs, hips' during the drive. Some calls deliberately span two phases —
'legs' spoken on the drive, 'there' on the finish. This is not decoration;
it is how eight people are made to change something simultaneously.

**Tone is an instrument.** Quiet → loud, elongation, repetition. C5,
transcribed:

> *Coming up on the ¼ mile, stay loose, stay relaxed [quiet], stay
> relaxed, stay relaxed, yeah boys [loud], coming up on the rhythm call
> [quiet], loose [quiet and elongated], there [loud], there, there, there
> [increasing in tone and elongated].*

**Tactical changes are always prepared, never sprung.** The pattern is
consistent across every cox: `'ready?'` → `'in two, in one'` → `'go'` /
`'now'`. Eight people cannot change together off an unannounced call.

**Position is given constantly**, and followed by its trend: 'we're
coming back with the Americans', 'moving out to 2 lengths clear', 'still
sitting on that bow ball'. Raw position without the trend is not what
elite coxes give.

**Boat metrics are quoted directly**: '36 and a half', 'you're on 1:18',
'still on 1:30's'. Rate and split, spoken as numbers.

**Chiding exists but is a minority.** Mostly positive; a few coxes used
'everyone the rate has dropped', 'heads in the boat', 'bow pair you need
to empty it right here'.

---

## What this project can build from it

The physics side of this repository now knows a great deal that a
coxswain could act on. The gap is that **none of it is delivered in the
form the job actually takes**: a call, timed to a stroke phase, chosen
against a 32-per-minute budget.

### 1. Call-value ranking — the one nothing else can do

Every lever in `scripts/time_budget.py` is priced in seconds. Nugent
shows the cox has ~190 slots per race. Nobody has ever asked **which
call is worth the most seconds**, because until now nobody had both
halves.

A first cut, straight from the measured levers:

| what a call could change | worth |
|---|---|
| blade depth (90 mm → optimum) | **57 s** |
| the racing line | 14 s |
| running the boat smoothly | ~18 s |
| taking the Cambridge arch | 1.8 s |
| rate, at constant power | unresolved — see `muscle.py` |
| seat order, rig | **0.03–0.5 s — never worth a call** |

This is not a script to read out. It is a *training* tool: it says which
technical calls have physical consequences worth the airtime, and which
are habits.

### 2. Cue-mix analysis from real recordings

`coxbox_gps manual.pdf` is in the corpus, so recordings exist. Nugent's
coding framework (their Table 1) is directly implementable: classify
calls as IF / EF / HF, technical / motivational / tactical, crew /
individual / section.

Feed in a recording, get back the mix against the elite baseline
(40/39/21) and, more usefully, **the IF:EF ratio against what the motor
learning literature recommends**. That is a measurable, trainable number
that no coxswain currently has.

### 3. Phase-timed call rehearsal

The simulator already produces the stroke cycle and `render3d.py` already
puts the camera at the coxswain's eye. Combining them gives a drill:
the boat runs, and the trainee places calls against the actual catch,
drive and finish. Scored on phase accuracy — which Nugent shows is what
elite coxes do and nobody is taught explicitly.

### 4. Race-day decision rehearsal

Already half-built. `scripts/powerhouse.py`, `scripts/wind_maps.py`,
`scripts/water_level.py` and `coxswain/river/stations.py` between them can
pose a real question — *"KBOS reads 250° at 6 m/s, the river is 0.3 m
low, which arch at Western Avenue?"* — and score the answer against the
optimiser. The scenarios exist; what is missing is the quiz around them.

### 5. The steering-cost feedback loop

`scripts/mpc_tune.py` established that **tighter tracking is slower** —
holding the line to the last centimetre costs more helm than the
centimetres are worth. That is a genuinely counter-intuitive coaching
point and it is measurable per-race from a GPS trace: how much did this
coxswain's steering actually cost, in seconds?

---

## Research gaps worth the project's attention

**Nobody has measured what a call is worth.** Nugent counts and classifies
calls; no study connects a call to a change in boat speed. With
instrumented boats and a physics model this is answerable and would be
genuinely new.

**Attentional focus has not been studied in coxed boats.** Nugent says so
explicitly. Every IF/EF study cited is on ergometers or solo athletes. The
cox is delivering cues to eight people simultaneously who cannot see the
direction of travel — a condition none of the underlying research covers.

**The overload threshold is unquantified.** 32 calls/minute is what elite
coxes do; nobody knows whether it is optimal, or where the crew stops
processing. This is the single most useful thing that could be measured
for coxswain education.

**Steering cost is unmeasured in the field.** This project can now compute
it. No published work quantifies how many seconds a coxswain's line
actually costs against the optimum.

## Still missing from the corpus

**"Coxswains Are Performers Too: Mental Skills Training"** is not in the
rowing directory — `Coaching-the-Coxswain-EXTRACT.pdf` is Dommert's
coach-facing guide and `Coxswain-Evaluations-2.0.pdf` is an evaluation
form. The mental-skills paper needs adding before it can be read.

## What a call is worth — measured, and bounded

The gap above ("nobody has measured what a call is worth") is now partly
closed. One masters head race, 4050 m and 576 strokes, with the coxswain's
speech synchronised to stroke-by-stroke GPS through her own spoken split
callouts. Crews are referred to generically throughout; no raw recording or
sheet data lives in this repository.

**The headline is an upper bound, not an effect.** With theme measures that
pass cross-validation, peak responses are under 0.6 mm/s per unit of theme
intensity, and the model explains 6.0% of stroke-scale speed variance.
Motivational and tactical peak at one stroke with intervals excluding zero;
technical is negative throughout with an interval spanning zero.

**Dropping the three-theme taxonomy changes nothing.** The buckets are
inherited from thematic work and may not be the axis along which calls
differ in effect. An unconstrained elastic net over all 55 unigrams and
bigrams occurring six or more times selected **0 phrases**, out-of-sample
R2 = -0.010. A first run without controls selected "we" (+100 mm/s) and
"at" (+68 mm/s); both are proxies for how much the coxswain is talking at
all, and partialling out speech density removes them. So the finer-grained
question — which specific calls work — is not answerable from one race by
either a theme model or a phrase model.

**Why, and what it would take.** Stroke-scale speed noise is 57.7 mm/s after
detrending, against a theme intensity of 1.35 per stroke: a per-stroke
signal-to-noise ratio of order 1e-2. Power by injection into the observed
design matrix, with the false-positive rate verified at 2.0-4.5% against a
nominal 5%:

| true effect | 1 race | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|
| 0.5 mm/s | 6% | 6% | 12% | 18% | 32% |
| 1.0 mm/s | 8% | 16% | 27% | 44% | 77% |
| 2.0 mm/s | 18% | 46% | 71% | 98% | 100% |
| 4.0 mm/s | 62% | 98% | 100% | 100% | 100% |
| 8.0 mm/s | 100% | 100% | 100% | 100% | 100% |

One race excludes per-unit effects of 8 mm/s and larger. It says almost
nothing about effects the size actually observed: 24 races still reach only
32% power at 0.5 mm/s. **Eighty per cent power needs roughly 2 mm/s at 6-12
races, or 1 mm/s at 24** — and that is optimistic, because the simulation
replicates one design matrix and so carries no between-race heterogeneity.

This is the concrete data ask: 6-24 races with audio synchronised to speed,
rate and position. Cox-box and CoxOrb class devices already log all of it
simultaneously, so a corpus is a retention-policy question rather than a
hardware one. No public dataset pairs the two; it has to come from a squad.

**Four methods that looked fine and were not.** Each produced a confident
wrong answer and none was visible without validation: rolling automatic
captions duplicate 12-39% of words; caption punctuation differs so severely
between corpora (26.2 sentence marks per 100 words versus 0.0-0.6) that
sentence segmentation is not comparable; fixed windows attribute each call
its neighbours' effect, inflating estimates about sixfold; and single-label
coding of overlapping speech is ill-posed, giving a classifier at 0.506
accuracy against a 0.388 majority baseline. Replacing the classifier with
continuous per-theme intensity passed (technical AUC 0.920 by lexicon,
motivational 0.798 and tactical 0.790 learned).

Written up as `paper.tex` for *Int J Sports Sci Coach* (SAGE), six pages.

### Functional categories beat the themes — catch calls

The three themes pool calls that do different jobs; a catch call and a ratio
call are both "technical". Re-cutting the transcript into eight functional
categories a coach distinguishes (power, catch, finish, length, ratio,
rate, motivational, tactical), with terms taken from the race's own
vocabulary, changes the answer.

As an ensemble it still fails: in-sample R2 0.240 against the themes' 0.060,
but blocked out-of-sample R2 **-0.059**, worse than a shifted control at
-0.020. Five categories had pointwise intervals excluding zero and none
survived a family-wise threshold. That in-sample number is the same trap the
unvalidated classifier set.

**One category is real.** Fitted alone, catch calls score out-of-sample
R2 **+0.084**, the only positive of the eight (next best is 0.000). Matched
null of 500 circular shifts maxes at +0.039; corrected for picking the best
of eight, **p = 0.040**. Peak **-9.7 mm/s at lag 21 strokes**, and it is not
a detrending artifact:

| detrend | OOS R2 | null 95th | peak |
|---|---|---|---|
| 31-stroke MA | -0.002 | +0.012 | -7.4 |
| 61-stroke MA | +0.084 | +0.028 | -9.7 |
| 101-stroke MA | +0.103 | +0.019 | -10.9 |
| 151-stroke MA | +0.094 | +0.017 | -11.0 |
| linear | +0.058 | +0.005 | -11.1 |

The 31-stroke window is the only one that removes it, which is what should
happen — that filter attenuates the timescale the effect lives on. The 21
calls fall at 17 strokes in 12 groups across all four quarters of the race,
so it is not one coincidence; but leave-one-out shows a single call carrying
47.5% of the score.

**Do not read the sign causally.** The likelier account is that the coxswain
calls the catch when she can see it going, and what she saw continues for
the next twenty strokes. A marker of trouble, not necessarily a cause.

The methodological point is independent of that: -10 mm/s is an order of
magnitude above the theme peaks and, per the power table above, comfortably
inside what one race resolves. It was invisible at theme level because
averaging it with ratio, rate, length and finish diluted it to the -1.6 mm/s
the technical theme reported with an interval spanning zero. **Where an
effect lives at a finer grain than the coding scheme, the coding scheme
destroys it.**

### Elite n>1: World Rowing broadcast telemetry, partially usable

World Rowing cox recordings are dubbed over broadcast footage carrying a
live overlay: race clock, leader distance, and per-crew speed in km/h to
0.1 (0.028 m/s, whose quantisation noise is 7x below the stroke-scale noise,
so not a limiting factor). The crew being coxed is highlighted. This is real
synchronised speed-plus-audio data.

The limit is coverage. On the 2022 World Cup III women's eight A-final the
SPEED table is present for **18% of the race**, in three runs of 26, 23 and
18 s. Too fragmentary for impulse responses on that video. Other recordings
have not yet been scanned, and the scan is cheap.

Official World Rowing race data (speed and rate every 50 m, rounded to
0.1 m/s) is not a substitute: the rounding is tolerable but 50 m is about
8-10 strokes, which destroys the short lags entirely.

### Unsupervised clustering: null, and the reason is sample size

Hand-built categories risk finding what was put in, so the transcript was
asked to group its own calls: TF-IDF over 8-word windows, k-means at
k = 4, 6, 8, 10, speed playing no part in forming the clusters. Each
cluster's impulse response was then fitted and scored by blocked
out-of-sample R2 against a matched circular-shift null, with the best
cluster compared to the best-of-k null distribution.

**Nothing at any k.** Strongest was k=8, cluster of 203 windows, OOS R2
+0.019 against a best-of-k null 95th of +0.044, p = 0.267.

The diagnostic is the useful part. Catch terms appear in only 24 of 453
windows; at k=8 they scatter across seven clusters and are at most 13% of
any one. **No cluster is a catch cluster.** Clustering groups by dominant
vocabulary, so a call type too rare to form its own cluster is diluted
below detection — which is exactly what happened to the one effect that a
pure hand-built series did find. This is a statement about how much speech
one race contains, not about clustering.

### Elite recordings surveyed — telemetry does not exist at the needed resolution

All six identified elite cox recordings checked:

| recording | footage | telemetry |
|---|---|---|
| 2022 World Cup III W8+ (CAN) | broadcast | speed km/h + clock + distance, **18% of race** |
| U23 2021 GB M8+ | broadcast | race clock only |
| 2011 Worlds LM8+ AUS | broadcast | none |
| Henley Thames v Barge | onboard bow cam | none (course boom visible) |
| Henley 2022 Leander v Yale | onboard + broadcast | none |
| Henley Women's Varsity 8+ | onboard | none |

Only one carries speed, and only in runs of 26, 23 and 18 s. **A fragment
shorter than the impulse response cannot estimate one, however many are
pooled** — so broadcast telemetry cannot supply n>1 for this analysis at
any collection effort. Official 50 m race data averages 8–10 strokes and
removes the short lags. The corpus has to be recorded, not found.

### Can boat speed be recovered from cox-POV video? Tested against GPS: no

If speed could be extracted from onboard video, every cox-POV recording
becomes an n>1 datapoint. The case race is the ideal test bed because it
has **both** the video and CoxBox GPS for the same 576 strokes, so any
optical estimate can be scored against truth rather than assumed.

**Important:** the camera is head-mounted, not boat-mounted. The IMU and
the image therefore describe the coxswain's head, not the hull.

Four approaches, all scored against per-stroke GPS:

| method | best r2 vs GPS | configs tried |
|---|---|---|
| optical flow magnitude, water patches | 0.061 | 15 |
| head IMU surge, rotated to world + integrated | 0.039 | 9 |
| flow de-rotated by IMU angular velocity | 0.000 | 9 |
| flow divergence (rotation-invariant) + low-pass | 0.051 | 20 |

Every figure is the best of many configurations and so is inflated; the
honest expectation is lower. Speed is the *dependent variable* in the
impulse-response analysis, so measurement noise enters directly. Against
effects of 0.5-10 mm/s on a stroke-scale sd of 58 mm/s, an r2 of 0.05
is not usable — it would swamp everything the analysis is looking for.

The DJI Osmo Action files do carry a per-frame telemetry track
(`DJI meta`, protobuf `dvtm_ac203`, 29.97 Hz): unit orientation quaternion
and 3-axis accelerometer, verified by gravity landing on world -Z at
-1.006 g. **No GPS** — the camera was not paired to a GPS source. The IMU
is real and clean; it simply measures the wrong body.

The "reference a position from a frame and integrate" route (tracking a
fixed part of the hull to separate head motion from boat motion) remains
untried and is the one optical avenue not yet closed. It is a harder
computer-vision problem than anything above, and it would still need
ground truth to validate.

**Conclusion: video alone cannot supply n>1.** More cox-POV footage,
however much of it exists, does not help without paired boat data.

### Head IMU: stroke detection works, rate and speed do not

The Osmo telemetry is head motion, but the whole boat surges each stroke,
so the rhythm survives. Scored against the CoxBox for all 576 strokes:

| quantity | result | usable? |
|---|---|---|
| stroke **events** | 577 detected vs 576 true, 97% matched, median timing error **0.127 s** | **yes** |
| stroke **rate**, per stroke | r2 0.040, error sd 2.31 spm | no |
| stroke rate, 11-stroke smoothed | r2 0.337, error sd 0.95 spm vs signal sd 0.87 | no |
| surge **speed** | r2 0.039 (best of 9) | no |

Stroke detection is the one thing that transfers: any cox-POV recording
with this telemetry can be segmented into strokes and its speech aligned
to them, which solves synchronisation for free. But rate carries ±0.13 s
of timing jitter, which on a 2.0 s interval is ±2 spm — larger than the
1.70 spm of real rate variation. Smoothing plateaus with error still above
signal, so rate cannot serve as a response variable either.

Also tried and failed: **OCR of the CoxBox LCD**, which is visible in frame
throughout. The display is washed out by glare at an oblique angle; no
digits are recoverable at 1080p.

**Every route from video to a response variable is now closed**, each
tested against ground truth rather than assumed:

| route | verdict |
|---|---|
| broadcast telemetry overlay | 18% coverage, fragments shorter than the IRF |
| official 50 m race data | averages 8-10 strokes, kills short lags |
| optical flow (5 variants) | r2 <= 0.061 vs GPS |
| head IMU surge | r2 0.039 |
| head IMU stroke rate | r2 0.337, error > signal |
| CoxBox LCD OCR | illegible |
| hull-referenced flow subtraction | r2 0.007 |

On fusing a physics model with optical data via a Kalman filter: a filter
cannot create information the measurements lack. With observations at
r2 ~ 0.05 the posterior is dominated by the prior, and the prior knows
nothing about call-driven deviations — that being the unknown. The output
would be the model's own prediction, and regressing calls against it
recovers our assumptions, not the crew's behaviour.

**n>1 requires instrumented boat data logged at the time.** The audio,
the synchronisation method and the analysis pipeline are all built and
validated; only the boat channel is missing.

## n>1 SOLVED: the CoxBox display is readable in the coxswain's own footage

Every route from video to a *derived* response variable failed. The route
that works reads the instrument directly.

The coxswain's own channel holds ~40 race recordings. In the **stern-loaded**
boats the CoxBox sits facing the coxswain, so the camera sees the display
close to face-on rather than at the oblique, glare-washed angle of the
bow-loaded case race. At 1080p the LCD is legible.

Verified on `Masters National Championship 2025 SRA Womens Club F8+`:

| video t | rate | split | DPS | speed from split |
|---|---|---|---|---|
| 114 s | 39½ | 1:50 | 6.69 | 4.545 m/s |
| 119 s | 39½ | 1:44 | 7.32 | 4.808 |
| 124 s | 39½ | 1:43 | 7.21 | 4.854 |
| 129 s | 39 | 1:46 | 7.18 | 4.717 |
| 134 s | 38½ | 1:45 | 7.44 | 4.762 |

Display layout (confirmed by the coxswain): top-left stroke rate, top-right
split, lower fields distance-per-stroke and distance.

**The reads validate internally.** The three fields must satisfy
DPS = speed x 60 / rate. They do, to a mean error of +1.2% and a maximum of
3.2%. That confirms the digit reads *and* the field semantics without any
external ground truth — which matters, because the one race that has GPS
truth is the bow-loader whose display is illegible.

**Resolution is adequate.** Split at 1 s gives 44 mm/s quantisation, so
13 mm/s of noise against a stroke-scale speed sd of 58 mm/s. Rate reads to
half a stroke per minute — far better than the r2 0.337 the head IMU
managed. Both are usable response variables.

### What remains

Engineering, not discovery:

1. **Tracking.** The camera is head-mounted so the CoxBox moves in frame
   (y range 398-972 px). A rigid template finds it in 28% of samples with
   one clean 22 s run; a scale-invariant or CSRT tracker should do far
   better, and coverage per video is the figure that decides how many
   recordings are usable.
2. **Digit OCR.** Clean fixed-segment LCD, consistent font. Digit templates
   suffice; the DPS identity above gives a per-frame self-check that
   rejects bad reads automatically.
3. **Synchronisation.** Free — the speech and the display share one video
   clock, so the split-callout method is not even needed.

**This is the n>1 route.** It needs no new recording, no instrumentation
and no third party: the corpus already exists and is the coxswain's own.
The power table says 6-24 races; the channel holds roughly 40, of which the
stern-loaded subset is the candidate pool.

## Literature the paper was missing

### Gabana et al. (2015) — the only controlled test of a coxswain

*J Appl Sport Psychol* 27(3): 288–300. 26 female intercollegiate rowers, four
maximal 1000 m ergometer sprints under **music / live coxswain / both /
control**, measuring time to completion, RPE, attentional focus and
motivation. Nugent cites it.

The design is the point: an ergometer has no steering, no boat to balance
and no crew to synchronise, so it strips a coxswain down to the voice
alone. Cleanest available evidence on whether the voice does anything.

**Verified from the publisher's abstract and the paper's own table notes:**
the design above, and that the reported findings concern attentional focus
(external attention to music can coexist with task-relevant thought) and
motivation (no significant difference across conditions). **No performance
benefit of the coxswain condition is reported.**

**Not verified:** several secondary sources state "no significant
difference in performance time between conditions", but that claim does not
appear in the abstract, the key-takeaways, or the table notes that are
publicly accessible. The full text is paywalled. The paper is cited in
PAPER.tex for what was checked, not for the stronger claim.

### Zach & Furman (2022) — the nearest prior art, and it agrees with us

*Int J Sports Sci Coach* 17(6): 1306–1316. Related basketball coaches'
feedback to the outcome of the possession it was given during: 3 coaches,
5 games, **1931 feedbacks over 761 possessions, 2.54 per possession**.

Two things matter for us:

1. They used **multi-label** coding — one utterance could sit in several of
   six categories rather than being forced into one. We reached the same
   conclusion from the other direction: single-label coding cross-validated
   at 0.506 accuracy and had to be replaced with continuous intensity.
2. Of six categories, **only valence (positive/negative) related to
   outcomes; the content categories did not.** That is the same shape as
   our result — the three themes explain almost nothing, and the one thing
   that predicts held-out strokes is a functional category cutting across
   the technical theme.

So "a taxonomy built to describe what is said need not align with what
works" is now supported in two sports by two independent designs, not just
asserted by us.

### What Nugent cites (50 refs, via Semantic Scholar)

The reference list is dominated by four clusters:

- **Attentional focus** — Wulf 2013; Chua et al. 2021 (meta-analysis,
  external focus superior); Neumann & Brown 2022 (rowing-specific:
  *switching* internal/external beats either); Becker 2019; Zhuravleva 2023.
  This is where the technical/motivational split comes from, and Nugent
  flags that elite coxswains use internally focused language the evidence
  base would discourage.
- **Coach speech in competition** — Halperin et al. 2016 (boxing ringside);
  Mason et al. 2020 (AFL in-game, two papers); Zach & Furman 2022; Smith &
  Cushion 2006 (soccer); Havira et al. 2024 (football pregame speeches).
  Almost all descriptive; Zach is the exception.
- **Rowing** — Baudouin & Hawkins 2002 (biomechanics review); Wing &
  Woodburn 1995 (crew coordination); Kleshnev 2000 (power); Nugent 2021
  (low back pain).
- **Qualitative method** — Nowell 2017 (thematic analysis); Smith &
  McGannon 2018 (generalizability); Sui et al. 2022 (**YouTube as a
  research source** — the methodological warrant for both Nugent's corpus
  and ours).

Nugent's corpus is also now pinned down: **16 recordings found, 8 included**
— 2011 World Senior Championships, 2021 World U23, a 2022 World Cup, and
Henley semi-finals and finals 2014–2022. That is the same set our elite
comparison corpus was drawn from, which is worth stating explicitly in the
paper rather than leaving as coincidence.
