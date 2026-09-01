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
