# Sensing, control and actuation of the body: a literature review for the rower model

*Started 2026-11-01, before phase 4.3's multi-clock body (SOURCES §206). SOURCES §207 indexes it. Open papers are
kept privately in `data/local/literature/neuromuscular/<paper>/` (pdf plus extracted text).*

## Why this review, and how to read it

§204–206 found three things the rower model cannot yet explain:

- the trunk needs an *instant* internal model;
- reading the stroke's acceleration even 25 ms late collapses the stroke;
- with an efference copy and 100 ms delays, the trunk wanders 29–39° off its path.

§205 adds that the athletes' own body sequencing closes the speed-fluctuation (IVV) gap. The next build gives the legs,
trunk and arms their own time laws. Before choosing how, this review asks what the field uses to model how a body
**senses itself, estimates its state, decides, and actuates**, and where the field disagrees.

**Read levels.** Every source carries a mark, so a claim can be traced to how closely it was read:

- **F**: full text read for this review;
- **A**: abstract, or an indexed or full-text summary, read;
- **K**: cited from background knowledge, bibliographic details checked, content not re-read. No number in this
  review rests on a K source alone.

Each section gives:

1. the mechanism;
2. the competing models, with their equations where they matter;
3. the evidence for and against, from several sources;
4. what follows for the rower.

Section 11 maps the findings onto §204–206. Section 12 proposes the architecture.

---

## 0. The short answer: what the state of the art puts in a model of self-control

| layer | what most current models use | main alternatives | what is contested | sources |
|---|---|---|---|---|
| actuator | Hill-type muscle–tendon unit (force–length, force–velocity, parallel and series elasticity), with first-order activation dynamics (≈10–15 ms on, 40–50 ms off) | cross-bridge (Huxley; distribution-moment); four-element Hill with serial damping; history-dependent (titin) models | accuracy in submaximal, dynamic tasks (≈54 % of force variance in cycling); history dependence | §1 |
| intrinsic mechanics | the muscle's own stiffness and damping: short-range stiffness, force–velocity, called "preflexes" (zero delay) | — | its size; whether it is under neural control | §2 |
| sensors | spindle (length and velocity, with fusimotor gain), Golgi tendon organ (force), cutaneous load, vestibular, vision | physiological spindle models (bag/chain fibres); spindles that encode force and its rate (yank); spindles as forward models | what spindles actually encode in active muscle | §3 |
| delays | short-latency reflex 20–45 ms, long-latency 50–100 ms, trunk 40 ms (identified loop) to 90–117 ms (EMG onset), visual 100–150 ms | — | per-channel values are well bracketed; the auditory delay is still unsourced | §4 |
| spinal layer | reflexes with task- and phase-dependent gains; a central pattern generator (CPG) in many locomotion models; synergies as a low-dimensional basis | reflex-only models (no CPG); CPG-only models | whether humans have a locomotor CPG; whether synergies are neural or mechanical | §5 |
| state estimation | a Kalman-type observer: a forward model driven by an efference copy, corrected by delayed, noisy sensing | a Smith predictor (a forward model plus a delayed copy for comparison) | where it lives (the cerebellum is the leading candidate) | §6 |
| controller | optimal feedback control (OFC; LQG and its signal-dependent-noise extension); in musculoskeletal practice, PD and reflex ("extended P") controllers dominate | impedance control; intermittent (event-driven) control; equilibrium-point/threshold control; active inference | EP is under serious challenge; active inference vs OFC is unresolved; intermittency is supported for balance | §7 |
| what gets optimised | effort/energy (metabolic) plus accuracy, for practised tasks; minimum variance under signal-dependent noise | minimum jerk; minimum torque change; fatigue | the weighting; task dependence | §8 |
| coordination of segments | emerges from the dynamics plus the cost: proximal-to-distal sequencing, interaction torques (leading joint), biarticular energy transfer | coupled oscillators (HKB) for timing between limbs and people; synergies as fixed modules | how much is planned and how much falls out of the mechanics | §9 |
| simulation practice | predictive simulation by direct collocation for open-loop optima; reflex/PD closed loops with physiological delays; deep RL emerging | — | stability with delays and muscle nonlinearity is itself treated as a validity test | §10 |

**The one sentence the review supports.** The field models self-control as a **hierarchy**:

- **at the bottom**, a zero-delay layer, the muscle's own impedance;
- **above it**, a short-delay spinal layer: reflexes, and possibly a pattern generator;
- **at the top**, a long-delay layer that estimates the state with a forward model fed by an efference copy, and acts on
  that estimate with near-optimal, task-specific feedback.

Our trunk loop collapsed all of this into one 100 ms delayed PD acting on a pure torque generator.

---

## 1. Actuation: muscle–tendon models

### 1.1 The Hill-type unit (the field's default)

The contractile element produces the active force, with a parallel elastic element beside it and a tendon in series:

  F_CE = a(t) · f_L(l̃_M) · f_V(ṽ_M) · F_0,  F_MT = (F_CE + F_PE(l̃_M)) cos α = F_T(l̃_T)

Here *a* is activation (0–1), *f_L* and *f_V* are the normalised force–length and force–velocity curves, *α* is pennation,
and the series-elastic tendon *F_T* closes the length loop (l_MT = l_M cos α + l_T).

- **Origin and standard form:** Hill (1938) [K]; Zajac (1989), *Crit. Rev. Biomed. Eng.* 17:359 [K], which
  dimensionless-ised the unit and set the conventions OpenSim still uses.
- **Numerical forms:** Millard, Uchida, Seth & Delp (2013), *J. Biomech. Eng.* 135:021005 [A]. They compare an
  elastic-tendon equilibrium model, a *damped* equilibrium model and a rigid-tendon model.
  - Against maximally activated muscle, the mean absolute errors are below 8.9 %, 8.9 % and 20.9 % of F_0.
  - At low activation the damped model is ~29× faster with an explicit integrator.
  - A rigid tendon matches the elastic models only for short tendons.
- **Four-element variant:** Haeufle, Günther, Bayer & Schmitt (2014), *J. Biomech.* 47:1531 [F]. It adds a serial
  damping element and a separate eccentric force–velocity branch, which gives human-like elbow flexion, and it is
  published as ready code.

**How good is it outside maximal effort?** Dick, Biewener & Wakeling (2017), *J. Exp. Biol.* 220:1643 [A], compared
Hill-model gastrocnemius forces with ultrasound-based estimates during cycling:

- the models captured on average **54 %** of the time-varying force;
- at low speed and high activation the figure was ~80 %;
- a two-element model (separate slow and fast fibres) did not fix the gap.

Taken with Millard's maximal-effort benchmark, this brackets the model's accuracy: good at full activation, about half
the variance in submaximal cyclic work. **Rowing at race pace is high-activation cyclic work, closer to the good end.**

### 1.2 Activation and excitation–contraction dynamics

The field's default is a first-order lag from neural excitation *u* to activation *a*, with an activation-dependent
time constant:

  ȧ = (u − a) / τ(a, u),  τ ≈ τ_act ≈ 10 ms when u > a, ≈ τ_deact ≈ 40 ms when u < a

- **10 / 40 ms:** Zajac (1989) and Winters (1990), as implemented in OpenSim's first-order activation dynamics [A].
- **15 / 50 ms:** Thelen (2003), *J. Biomech. Eng.* 125:70 [A; the values come through a citing study].
- **Second-order alternative:** a 2nd-order low-pass from command to force, with τ1 = τ2 = 40 ms, is used in the
  optimal-control literature (Todorov 2005 [F via PMC]).

**Electromechanical delay** (EMG onset to force) adds to this lag:

- Cavanagh & Komi (1979), *Eur. J. Appl. Physiol.* 42:159 [A]: ~50–56 ms in the elbow flexors (eccentric 49.5,
  isometric 53.9, concentric 55.5 ms), mostly the slack-uptake of series elasticity.
- Later work reports 8 to >100 ms across muscles and methods [A].
- Ultrasound splits [A] put only ~15 ms of it in the gastrocnemius into muscle and tendon parts.

So the delay from neural command to force onset in a model is **activation lag plus tendon slack, ~20–60 ms**, and it
must be counted on top of every neural delay in §4.

### 1.3 Cross-bridge and distribution-moment models

- **Huxley (1957)** [K]: muscle force as a population of attached cross-bridges. It is a partial differential equation
  in the bond-length distribution, which makes it physiological but expensive.
- **Distribution-moment approximation:** Zahalak (1981), *Math. Biosci.* 55:89 [K]. It tracks the distribution's first
  moments instead (Gaussian assumption; an N-moment generalisation exists, see Donovan [A]).
- **MyoSim:** Campbell's MyoSim adds Ca²⁺ activation and cooperativity [A].

These models produce short-range stiffness and history effects naturally (§2), but they are rarely used in
whole-body simulation because of the cost (Millard 2013 [A]).

### 1.4 History dependence: force enhancement and depression

After active stretch, isometric force stays *above* the force-length value (residual force enhancement). After active
shortening it stays below (force depression).

- **First systematic study:** Abbott & Aubert (1952) [K, via the Herzog reviews A].
- **Current view:** Herzog's reviews [A] attribute enhancement mainly to a passive molecular spring, titin. A 2023
  *J. Biomech.* review [A] concludes that cross-bridge theory cannot explain either effect, and calls for a
  three-filament model that includes titin.
- **Existing models:** models with a parallel elastic element reproduce the main features.

**For rowing:** the drive is concentric, so force depression is the relevant effect. No rowing model includes it.

### 1.5 Fatigue and energetics

- **Fatigue, Xia & Frey-Law (2008):** a three-compartment model (active, fatigued, resting) [A].
  - Frey-Law, Looft & Heitsman (2012), *J. Biomech.* [A], fitted joint-specific parameters to isometric endurance
    curves with low error.
  - An intermittent-contraction extension adds enhanced recovery (Looft et al. 2018) [A].
  - It is *not validated at non-zero joint velocity* [A]. This matters for a 2000 m race.
- **Metabolic cost models:** Umberger, Gerritsen & Martin (2003), *CMBBE* 6:99 [A]; Bhargava, Pandy & Anderson (2004)
  [K].
  - Koelewijn, Heinrich & van den Bogert (2019), *PLoS ONE* [A], compared seven models against gas exchange in
    walking. All correlated at r ≥ 0.9, and the Bhargava and Lichtwark–Wilson models were best (r = 0.95).
  - A twice-differentiable Umberger model exists for gradient-based optimisation (Koelewijn et al. 2018) [A].

### 1.5a Joint-level torque generators (added 2026-11-01, for the build)

A whole-body model rarely carries individual muscles. The standard reduction is a **muscle torque generator** at
each joint, with Hill's relations expressed in joint angle and angular velocity, and with parameters from population
dynamometry.

- **Lower limb:** Anderson, Madigan & Nussbaum (2007), *J. Biomech.* 40:3105 [F].
  - T = C1 cos(C2(θ − C3)) f(ω): a Hill hyperbola written through the speeds at 75 % and 50 % of isometric torque,
    with eccentric torque scaled by (1 + C6|ω|) and exponential passive terms.
  - Six exertions (hip, knee, ankle, each way); 34 adults by sex and age; C1 normalised by weight × height.
  - Tested ranges: hip −0.5 to 1.3 rad, knee 0 to 1.8 rad.
- **Knee and elbow strength surfaces:** Frey-Law et al. (2012), *J. Appl. Biomech.* 28:726 [F], 54 adults:
  - measured surfaces of peak torque against angle and speed;
  - the normalised surfaces do not differ by sex;
  - the knee's torque-velocity decay is nearly linear, unlike Hill's hyperbola;
  - the median coefficient of variation is ~0.3.
- **A rival form:** Yeadon, King & Wilson (2006), *J. Biomech.* 39:476 [A], is a 7–9 parameter generator with
  "differential activation" (voluntary activation falls at eccentric and slow speeds), usually fitted per subject.
- **Deep knee flexion:** Kukić, Mrdaković, Stanković & Ilić (2022), *Biology* 11:1490 [F, table]. Isometric knee extension at interior
  angles 80–130° (flexion 100–50°): 62 % of peak at 100° flexion, 97 % at 70°. Nothing deeper was found, though a
  rower's catch flexes the knee 120–145°.
- **Rowers vs controls:** elite rowers' isometric trunk extension is ~319 N·m in 20 German elite men [A, abstract].
  International rowers are stronger than controls in every back-strength measure [A]. No rower-specific joint torque
  surface was found.

### 1.6 Rowing-specific muscle-driven work

- **Hase, Kaya, Zavatsky & Halliday (2004)**, *J. Appl. Biomech.* 20:317 [A]: a 3-D whole-body musculoskeletal model
  driven by measured ergometer kinematics, forces and EMG, in 5 rowers and 5 non-rowers. Rowers had higher vastus
  forces, foot-stretcher force, and knee and lumbar moments. It is inverse, not predictive. The JSME series (2002) [A]
  is the same group's model.
- **Zarei (2016)**, Cleveland State thesis, *Predictive simulation of rowing exercise* [A]: musculoskeletal optimal
  control of an arm on a rowing machine.
- **Adapted-ergometer studies:** OpenSim models with static optimisation for knee loads (PMC11761565) [A].

**No published predictive, muscle-driven, on-water rowing simulation was found.** The field has inverse analyses of
the ergometer, and optimal control of parts of the body. A forward, closed-loop rower on a moving hull would be new.

---

## 2. Intrinsic mechanics: the zero-delay layer

**Short-range stiffness.** Rack & Westbury (1974), *J. Physiol.* 240:331 [A]:

- an active muscle resists the first ~1 % of a stretch with a high "short-range" stiffness, attributed to attached
  cross-bridges;
- the stiffness collapses into viscous-like behaviour as the stretch continues;
- it resets whenever the muscle is held still [A].

Cui, Perreault and colleagues (2008) model short-range stiffness from muscle geometry and active force [A, via a
citing study]. Across arm postures and forces, the model explains most of the variance in endpoint-stiffness shape and
orientation.

**Preflexes.** Brown & Loeb (2000) [A, secondary] named the zero-delay response of muscle mechanics to a perturbation
the "preflex":

- it acts within milliseconds, an order of magnitude before any reflex;
- in the frog, intrinsic properties stabilise the wiping movement without reflexes (Richardson et al. 2005, via [A]);
- in walking simulations they stabilise in a way delayed neural feedback cannot [A].

**But intrinsic stiffness is not enough on its own.** Two lines of direct measurement agree:

- **Ankle:** Loram & Lakie (2002), *J. Physiol.* 545:1041 [A]. Intrinsic stiffness in quiet standing is
  **91 ± 23 %** of the minimum needed to balance; the remainder comes from neural modulation. The magnitude is
  contested: Peterka's group measures lower values at low frequency.
- **Trunk:** Moorhouse & Granata (2007), *J. Biomech.* 40:1058 [F, via PMC]. In 11 men, with pseudorandom perturbations
  at T10 during isometric extension at 20–50 % of maximum, they identified an intrinsic and a reflex pathway:
  - intrinsic stiffness k_INT was 1280 ± 240 N/m at 20 %, rising to 2116 ± 710 N/m at 50 % (activation-dependent);
  - the gravitational (destabilising) term was 1695 ± 249 N/m;
  - so intrinsic stiffness *minus* gravity was negative in 8 of 11 subjects;
  - the reflex pathway, with a **40 ms** delay, a static gain of 221 ± 84 N·s/m, ω_n ≈ 9 Hz and ζ ≈ 0.58, supplied
    about **42 %** of the total stabilising stiffness.
- **Trunk, modelling:** the same group's linear analyses (Franklin & Granata 2007; the Virginia Tech dissertation [A])
  found a trade-off. A higher proportional reflex gain lowers the co-contraction needed but shrinks the delay margin.
  Without reflexes, no recruitment pattern stabilised the spine under 200 N.
- **Trunk, system identification:** van Drunen et al. (2013), *J. Biomech.* 46:1440 [A]. A model with intrinsic
  impedance, spindle position and velocity feedback, and tendon-organ force feedback explained 90 % of trunk kinematic
  variance. "Resisting" a perturbation cut admittance by 61 % and raised reflex gain by 73 % below 1.1 Hz. A companion
  sagittal study [A] needed spindles to **encode acceleration** as well, and found vestibular feedback did not improve
  the fit.
- **Seated balance:** Reeves, Cholewicki & Narendra (2009), *J. Biomech.* 42:164 [A]. In seated balance, longer
  reflex delays did not destabilise the loop, but they raised trunk displacement and moment.

**What follows (§11):** a stabilising trunk loop has a *zero-delay* intrinsic part, which is activation-dependent and
of the order of the gravity load, plus a ~40 ms reflex part of similar size. It is not a single 100 ms delayed
controller.

---

## 3. The sensors

### 3.1 Muscle spindles

Spindles signal length and its rate. Their sensitivity is set centrally by static and dynamic fusimotor (γ) drive.

**Simple models:** Prochazka & Gorassini (1998), *J. Physiol.* 507:277 and 293 [A]. Fitted to 47 afferents recorded
in walking cats:

- hamstring spindle primaries were predicted from length and velocity alone (a power law in velocity plus length
  terms);
- triceps surae spindles deviated during EMG activity;
- tendon-organ ensembles predicted whole-muscle force well.

Their summary: large afferents signal velocity, length and force "to a first approximation" at locomotor speeds.

**Physiological models:** Mileusnic, Brown, Lan & Loeb (2006), *J. Neurophysiol.* 96:1772 [F]:

- three nonlinear intrafusal fibres (bag1; bag2 plus chain, which share the static fusimotor drive) feed the primary
  and secondary afferents;
- one parameter set fits a range of published fusimotor and afferent data;
- the companion paper models the tendon organ (96:1789) [A].

**What a spindle encodes is contested:**

- **Force and yank:** Blum, Lamotte d'Incamps, Zytnicki & Ting (2017), *PLoS Comput. Biol.* 13:e1005767 [F]. In
  passive muscle, primary afferents fire in proportion to **force and its derivative dF/dt (yank)**, including the
  initial burst, history dependence, and the acceleration-like burst at stretch onset. They attribute the burst to
  short-range stiffness (§2).
- **Context-dependent:** their 2020 *eLife* model [A] shows a spindle can appear to encode force, yank, length,
  stiffness, velocity or acceleration, depending on conditions.
- **Forward models:** Dimitriou & Edin (2010), *Curr. Biol.* 20:1763 [A]. Human spindle discharge leads the
  kinematics, so with fusimotor drive spindles act as *forward sensory models* that partly offset neural delay.

**For the rower:** a length-plus-velocity spindle is the default and suffices for small perturbations. The trunk fits
need an acceleration term (§2), which a force/yank spindle supplies naturally.

### 3.2 Golgi tendon organs, cutaneous and load receptors

- **Tendon organs:** force in series with the muscle (Mileusnic & Loeb 2006 [A]; Prochazka & Gorassini 1998 [A]).
  Positive force feedback from extensor tendon organs is the stance mechanism in reflex walking models (Geyer & Herr
  2010 [A]; Song & Geyer 2015 [F via PMC]).
- **Load and cutaneous receptors:** Zehr & Stein (1999), *Prog. Neurobiol.* 58:185 [A]. These shape stance support
  and step-cycle timing, with task-, phase- and context-dependent gains.

The rower's equivalents are foot-stretcher and handle load, and seat contact, all candidate phase-transition triggers.

### 3.3 Vestibular and visual

- **Vestibular:** galvanic vestibular stimulation evokes a short-latency (~60 ms) and a medium-latency (~110–120 ms)
  postural response of opposite sign (Britton et al. 1993 [K]; Fitzpatrick & Day 2004 [K]; Cathers, Day & Fitzpatrick
  2005 [A]; Day and colleagues [A]). The canals drive the medium-latency response.
- **Visual corrections:** Saunders & Knill (2005), *Exp. Brain Res.* [F]:
  - corrections to perturbed hand feedback appeared within **117 ms**, invariant to movement speed;
  - their earlier study found 160 ms.
- **Visual initiation vs correction:** Franklin (2016), *Motor Control* 20:171 [F]: initiation ≈ 150 ms, correction
  to EMG ≈ 100 ms (citing Franklin & Wolpert 2008; Pruszynski et al. 2010).
- **Rowing:** vestibular and visual input matter for balance and crew timing. They are too slow for the within-stroke
  trunk loop.

### 3.4 Combining the senses

- **Statistically optimal integration:** Ernst & Banks (2002), *Nature* 415:429 [A]. Vision and touch are weighted in
  inverse proportion to their variances (maximum likelihood). Körding & Wolpert (2004), *Nature* 427:244 [K] extend
  this to priors.
- **Posture:**
  - Peterka (2002), *J. Neurophysiol.* 88:1097 [A]: sway is explained by a linear sum of reweighted sensory channels.
  - Kuo (2005), *J. Neural Eng.* 2:S235 [A]: LQ state feedback on an optimal estimator, with an internal model of the
    body and sensors (proprioceptive, vestibular, visual), reproduces sway across sensory conditions with one parameter
    set.
- **Simulation practice:** Shanbhag et al. (2023), *J. NeuroEng. Rehabil.* 20:111 [F, the review's discussion]. Of 102
  postural-control simulation studies, most fuse channels by independent or interacting sums, few use Kalman
  reweighting, and few model more than one sense.

---

## 4. Delays

| pathway | latency | sources |
|---|---|---|
| short-latency stretch reflex (spinal, Ia) | 20–45 ms | Kurtzer 2015 [F, SOURCES "sensory and reflex delays"]; Ia conduction 60–120 m/s, ~85 m/s fastest estimate [A] |
| long-latency reflex (transcortical; goal-dependent) | 50–100 ms | Kurtzer 2015 [F]; Pruszynski & Scott 2012 [A]; Crevecoeur & Scott 2013 [F] |
| trunk, identified reflex loop delay | **40 ms** | Moorhouse & Granata 2007 [F] |
| trunk, EMG onset after a sudden support or load perturbation | 89–117 ms (≈17 ms shorter when anticipated) | Milosevic et al. 2016 [A]; Pedersen et al. 2004 [A] |
| trunk, onsets in sudden-perturbation studies | 60–120 ms | review cited with Cholewicki et al. 2005 [A] |
| vestibular (GVS) | ~60 ms and ~110–120 ms | Britton 1993 [K]; Cathers 2005 [A] |
| visual correction / visual initiation | ~100–117 ms / ~150 ms | Saunders & Knill 2005 [F]; Franklin 2016 [F] |
| command to force: activation lag plus electromechanical delay | ~20–60 ms | §1.2 |
| auditory | no source found | (still open; SOURCES §25's 160 ms remains unsourced) |

**The trunk figures differ for a reason.** The 40 ms *loop delay* is what a system-identification model needs between
stretch and reflex force. The 90–117 ms *EMG onsets* after a step perturbation include detection and the long-latency
burst. A model should use the loop delay for its reflex pathway and keep the longer onsets as a check on the
long-latency layer.

**Delays are not a nuisance term.** Song et al. (2021), *J. NeuroEng. Rehabil.* 18:126 [F]: "a control model that
cannot produce stable walking with physiological properties, such as nonlinear muscle dynamics and neural transmission
delays, is likely missing some important aspects of human control". The trunk collapse in §206 is that test, failed.

---

## 5. The spinal layer: reflexes, pattern generators, synergies

### 5.1 Reflexes as programmable feedback

Reflexes are not fixed. Their gain depends on the task and on the phase of the cycle:

- Zehr & Stein (1999) [A] review the evidence in locomotion;
- tibial-nerve responses reverse sign between early and late swing (Yang & Stein 1990, via [A]);
- the long-latency stretch response carries the same goal-dependence as voluntary control and shares its circuitry,
  including motor cortex (Pruszynski & Scott 2012 [A]).

Pruszynski and Scott frame it as part of an optimal feedback controller (§7.1) rather than a fixed servo.

**Reflex-based walking models** show stable gait needs no central rhythm:

- Geyer & Herr (2010), *IEEE TNSRE* 18:263 [A]: muscle reflexes encoding legged mechanics (positive force feedback,
  length feedback) walk, resist ground disturbances and adapt to slopes without retuning.
- Song & Geyer (2015), *J. Physiol.* 593:3493 [F via PMC]:
  - 3-D, 22 muscles, 10 reflex modules per leg, 82 parameters;
  - all pathways delayed in short/medium/long classes;
  - parameters found by an evolutionary search (CMA-ES) on falls, steady gait and metabolic cost;
  - it reproduces walking at 0.8–1.8 m/s, slopes, stairs up, turning, and most EMG patterns within the inter-subject
    range.
  - The authors argue that "feedback integration" matters more than central pattern generation, with CPGs possibly
    acting as observers of the feedback.

### 5.2 Central pattern generators

**The classic model** is a half-centre: two mutually inhibiting neuron pools (Brown 1911 [K]). Two refinements
followed:

- **Two levels:** McCrea & Rybak (2008), *Brain Res. Rev.* 57:134 [A]. A rhythm generator drives a separate
  pattern-formation network, explaining deletions that a single half-centre cannot. The design is still debated [A].
- **Oscillator models:** Matsuoka (1985) [K] mutually inhibiting oscillators. Taga, Yamaguchi & Shimizu (1991),
  *Biol. Cybern.* 65:147 [A]: one oscillator pair per joint, entrained with the body into a global limit cycle that
  adapts to the environment.

**Reviews:** Ijspeert (2008), *Neural Networks* 21:642 [A]; Grillner (2006) [K].

**In humans the CPG is contested.** Minassian, Hofstoetter, Dzeladini, Guertin & Ijspeert (2017), *Neuroscientist*
23:649 [A]:

- the lumbar cord can generate rhythmic patterns after injury;
- in healthy gait a CPG may shape phase-specific activation and act as a feedforward component for speed;
- but its existence and role are not settled.

Hofstoetter et al. (2017) [A] are more sceptical: assisted stepping may entrain reflex circuits instead.

**Feedforward and feedback trade off.** Kuo (2002), *Motor Control* 6:129 [A]:

- a pure feedforward oscillator is fragile to disturbances;
- pure feedback is fragile to sensor noise;
- the hybrid is best when both are present.

Dzeladini, van den Kieboom & Ijspeert (2014), *Front. Hum. Neurosci.* 8:371 [A], added a CPG to Geyer's reflex model
as a feedforward *predictor of the feedback*: it kept the robustness and gained speed control.

### 5.3 Muscle synergies

**For:** EMG across many muscles is well described by a few activation modules:

- d'Avella, Saltiel & Bizzi (2003), *Nat. Neurosci.* 6:300 [A];
- Ting & Macpherson (2005), *J. Neurophysiol.* 93:609 [A], in cats: synergies that map to ground-reaction-force
  directions for balance;
- Ivanenko, Poppele & Lacquaniti (2004), *J. Physiol.* 556:267 [A]: five patterns cover up to 25 limb and trunk
  muscles in walking.

**Against:**

- Tresch & Jarc (2009), *Curr. Opin. Neurobiol.* 19:601 [A] weigh both sides.
- Kutch & Valero-Cuevas (2012), *PLoS Comput. Biol.* 8:e1002434 [F]: limb biomechanics alone confines muscle-length
  changes to a low-dimensional subspace, so EMG synergies can appear without any neural coupling. A neural origin is
  unproven.
- de Rugy, Loeb & Carroll (2013) [K] argue synergies can be a product of optimisation.

**Rowing:** three synergies at all powers and in experts and novices alike.

- Turpin, Guével, Durand & Hug (2011), *Eur. J. Appl. Physiol.* 111:3017 [A]: 23 muscles; three synergies stable from
  60 to 120 % of 2000 m power; EMG amplitude rose with power, but timing and shape barely changed.
- Turpin et al. (2011), *J. Electromyogr. Kinesiol.* 21:1030 [A]: experts and novices share the same synergies;
  performance lies "in the mechanical output of the synergies" rather than their shape or timing.
- Shaharudin & Agrawal (2016) [A] and Shaharudin, Zanotto & Agrawal (2014) [A]: three synergies explain 93–94 % of
  variance; synergies related to economy in collegiate rowers.

**For the rower:** three activation modules (roughly legs, trunk, arms), robust across power, are the right *basis*
for excitations. Whether they are neural does not matter for a model that only needs the basis.

---

## 6. State estimation, internal models and efference copy

**The problem:** sensing arrives late and noisy, so the controller must act on an *estimate* of the present state.
The standard solution is a Kalman-type observer:

  x̂(t+1) = A x̂(t) + B u(t) + K [ y(t) − C x̂(t − d) ]

The forward model (A, B) is driven by the efference copy of u. It is corrected by the delayed observation y(t), which
is compared against the estimate *for the time the observation describes*, x̂(t − d). This is implemented by
augmenting the state with its last d steps (Todorov & Jordan 2002 supplementary notes [A]), or by a Smith predictor.

**Evidence that the brain does this:**

- **Kalman account of hand estimation:** Wolpert, Ghahramani & Jordan (1995), *Science* 269:1880 [A]. Errors in
  estimating hand position in the dark, under applied forces, follow a Kalman observer that combines efference copy
  with sensory inflow.
- **The cerebellum as a Smith predictor:** Miall, Weir, Wolpert & Stein (1993), *J. Mot. Behav.* 25:203 [A]. It holds
  two internal models: a fast forward model of limb and muscles, and a model of the loop delay that holds the
  prediction back so it can be compared in register with real feedback.
- **Long-latency responses use a delay-aware estimate:** Crevecoeur & Scott (2013), *PLoS Comput. Biol.* 9:e1003177
  [F].
  - Long-latency responses (50–100 ms) reflect the relationship between delayed joint displacement and the limb's
    *current* state.
  - The prior behind them updates trial by trial.
  - A Kalman filter with explicit delays reproduces the modulation.
  - An estimator that "ignor[es] the feedback delay" does not reproduce it.
  - They also state that delays "as short as tens of milliseconds can destabilize" feedback control unless the
    estimate is corrected for them.
- **Corollary discharge across species:** Crapse & Sommer (2008), *Nat. Rev. Neurosci.* 9:587 [A]. Motor-to-sensory
  copies ("corollary discharge", Sperry's term; "efference copy", von Holst & Mittelstaedt [K]) appear across the
  animal kingdom and take several forms (facilitate, inhibit, modulate).
- **Prediction-error learning:** Shadmehr, Smith & Krakauer (2010), *Annu. Rev. Neurosci.* 33:89 [A; text retrieved].
  Forward models predict the sensory consequences of commands, stay calibrated by sensory prediction errors, and so
  offset delay and noise. Error correction (feedback) and adaptation (model update) are distinct processes.
- **Reviews:** Franklin & Wolpert (2011), *Neuron* 72:425 [A]: forward models "counteract the effects of delays and
  noise", one of five mechanisms alongside OFC, impedance control, Bayesian decision theory and learning. McNamee &
  Wolpert (2019), *Annu. Rev. Control Robot. Auton. Syst.* 2:339 [A, bibliographic] is the recent review.
- **Paired forward and inverse models:** Wolpert & Kawato (1998), *Neural Networks* 11:1317, the MOSAIC architecture
  [K]: modules that switch with context.
- **Spindles as forward models:** Dimitriou & Edin (2010) [A] (§3.1). Fusimotor drive may put part of the prediction
  in the periphery.

**For the rower:** the state the controller uses is a *prediction made from the efference copy and corrected by
delayed sensing*. Using a delayed measurement directly, as §206 did when it read τ' and τ'' late, is exactly what
these sources say destabilises a loop.

---

## 7. Controllers

### 7.1 Optimal feedback control (the dominant theory)

**The idea:** the nervous system acts as a stochastic optimal controller with an optimal estimator.

- Todorov & Jordan (2002), *Nat. Neurosci.* 5:1226 [K]; the minimal-intervention NIPS paper [A];
- Todorov (2004), *Nat. Neurosci.* 7:907 [K];
- Scott (2004), *Nat. Rev. Neurosci.* 5:534 [A].

**Its signature prediction is the *minimal-intervention principle*:** deviations are corrected only when they affect
the task, so variability piles up in task-irrelevant directions.

- That matches the uncontrolled-manifold findings (Scholz & Schöner 1999, *Exp. Brain Res.* 126:289 [A]).
- It is supported by structured muscle variability (Valero-Cuevas, Venkadesan & Todorov 2009 [A]).
- It is disputed by Rácz & Valero-Cuevas (2013) [A].

**Signal-dependent noise:** Todorov (2005), *Neural Comput.* 17:1084 [F via PMC]:

- Motor noise grows with the command: u(1 + ε), with σ ∝ |u|.
- With such noise, estimation and control are *no longer separable*: the filter gains become adaptive and the two must
  be solved jointly.
- The reaching example uses a 2nd-order muscle filter (40 ms, 40 ms) and noise scales of ~0.5.
- A 2024 NeurIPS paper [A] finds a bias in that derivation when internal noise is present.

**What OFC needs:**

- a cost function;
- a model of the plant;
- a delay-augmented estimator.

### 7.2 Impedance control

Hogan (1985), *J. Dyn. Syst. Meas. Control* 107:1–24, parts I–III [A]: controlling position or force alone is not
enough; the *dynamic relation* between them (stiffness, damping, inertia) must be controlled.

- **The nervous system tunes impedance selectively:** Burdet, Osu, Franklin, Milner & Kawato (2001), *Nature* 414:446
  [A]. People learn to stabilise an unstable environment by raising stiffness only along the unstable direction, an
  energy-efficient strategy.
- **Why impedance helps:** Franklin & Wolpert (2011) [A] note that higher stiffness also reduces the effect of motor
  noise, and impedance acts with no neural delay (§2).
- **Cost:** co-contraction costs metabolic energy, so its level is itself an optimisation outcome.

### 7.3 Intermittent (event-driven) control

**The idea:** for systems balanced against delay, continuous control is not the only answer.

- **The theory:** Gawthrop, Loram, Lakie & Gollee (2011), *Biol. Cybern.* 104:31 [A] treat human control as
  *intermittent* predictive control: open-loop trajectories between event-triggered updates, with refractory periods.
  It explains phenomena continuous models miss, at lower computational cost.
- **In standing:** Asai et al. (2009), *PLoS ONE* 4:e6169 [F]. With a 200 ms delay, a PD controller that switches off
  near the stable manifold of the inverted pendulum's saddle is robustly stable at *small* gains. A continuous delayed
  PD is stable only in a narrow region of the P–D plane, and needs gains so high they make posture stiff.
- **Acceleration feedback and neutral loops:** Insperger, Milton & Stépán (2013), *J. R. Soc. Interface* 10:20120763
  [A]; Zhang, Stépán & Insperger (2018), *J. R. Soc. Interface* 15:20170771 [F].
  - Adding *acceleration* feedback to delayed PD raises the critical delay by ~40 %.
  - The closed loop is then a **neutral** delay equation: the delay enters the highest derivative, J θ̈ = … − A θ̈(t − τ).
  - Its stability needs the normalised acceleration gain small. They analyse A ∈ [−0.2, 0.8]; a neutral equation
    needs |A| < 1 relative to the inertia.

**For the rower:** this is the mathematics of §206's collapse (§11).

### 7.4 Equilibrium-point (threshold) control

**The model:** Feldman's λ model (1966, 1986 [K]; Feldman & Levin 2009 [A, bibliographic]). The brain sets each
muscle's stretch-reflex threshold λ, activation is ∝ [l − λ(t)]⁺, and movement is the drift to the new equilibrium
("referent configuration").

**Bizzi's α-model** is a related form.

**Challenges:**

- Gomi & Kawato (1996), *Science* 272:117 [A]: arm stiffness measured during multijoint movement implies an
  equilibrium trajectory far from the simple one EP needs. Feldman's camp replied that the estimation procedure was at
  fault [A].
- Mangalam & Stergiou (2026), *Exp. Brain Res.* 244:44 [F, abstract and introduction]:
  - EP is under-specified for multi-joint movement: it leaves inverse dynamics, impedance and timing unresolved;
  - it conflicts with force–length and force–velocity properties and with task-dependent reflex modulation;
  - it needs a "homunculus" to set the referents;
  - it is useful mainly for quasi-static, single-joint settings.

### 7.5 Active inference

Adams, Shipp & Friston (2013), *Brain Struct. Funct.* 218:611 [F]:

- descending motor signals are *proprioceptive predictions*, not commands;
- spinal reflexes are the loop that cancels proprioceptive prediction error, which explains why motor cortex is
  agranular;
- motor control becomes Bayesian inference, with no separate cost function (Friston 2011, *Neuron* 72:488, "What is
  optimal about motor control?" [K]).

**Status:** it predicts the same behaviour as OFC in many cases. Its evidence is mainly anatomical (modulatory
descending connections). It has few quantitative whole-body simulations.

### 7.6 What simulations actually use

Shanbhag et al. (2023) [F] counted 108 control approaches in 102 postural-control simulations:

| controller | count |
|---|---|
| PD | 43 |
| PID | 17 |
| "extended P" (muscle reflex) | 14 |
| P | 9 |
| LQR | 9 |
| MPC | 4 |
| other | 12 |

- In musculoskeletal models, PD (13) and reflex controllers (14) dominate.
- LQR and MPC appear only in simpler models, never combined with musculoskeletal gait except once.

The field's *theory* is OFC. Its *musculoskeletal practice* is reflexes and PD, with delays.

---

## 8. What gets optimised

- **Smoothness:** minimum jerk, Flash & Hogan (1985), *J. Neurosci.* 5:1688 [A], for planar reaching.
- **Torque change:** minimum torque change, Uno, Kawato & Suzuki (1989), *Biol. Cybern.* 61:89 [A], for curved and
  via-point paths. Flash (1990) [A] disputes its inertias.
- **Minimum variance:** Harris & Wolpert (1998), *Nature* 394:780 [A]. Under signal-dependent noise, choose the
  trajectory that minimises endpoint variance. It reproduces saccades, reaching and Fitts' law. Later work argues for
  a compromise with effort [A].
- **Effort and energy for practised tasks:**
  - the consensus behind neuromechanical locomotion (Song et al. 2021 [F]);
  - Ackermann & van den Bogert (2010) [K];
  - Falisse et al. (2019), *J. R. Soc. Interface* 16:20190402 [A]: a combined energy-and-effort cost predicted
    healthy gait, the walk–run transition and pathological gaits, in 3-D muscle models solved in ~36 min by direct
    collocation.
- **Maximal-performance tasks:** maximum jump height reproduces the human proximal-to-distal sequence (§9).

**For rowing** the objective is mixed: speed over 2000 m (maximal performance) under a metabolic and fatigue budget,
with technique constraints (blade, balance).

---

## 9. Coordinating several segments

**Interaction torques and the leading joint.** Dounskaia (2010), *Exerc. Sport Sci. Rev.* 38:201 [F, review text]:

- in multi-joint movements one *leading* joint is driven by simple reciprocal muscle activity, ignoring the rest;
- the *subordinate* joints' muscles regulate the interaction torques it creates;
- the leader is usually proximal ("like a whip handle"), but it depends on the task;
- in horizontal arm swings the trunk led (Kim, Hinrichs & Dounskaia 2009, via [A]).

**Proximal-to-distal sequencing comes out of optimal control:**

- Pandy & Zajac (1991), *J. Biomech.* 24:1 [A]: optimal control of a muscle-driven jump;
- Bobbert & van Soest (2001), *Exerc. Sport Sci. Rev.* 29:95 [A]: the proximo-distal sequence emerges as the optimum
  for jump height.

**Biarticular energy transfer:** biarticular muscles carry energy from proximal to distal joints.

- van Ingen Schenau, Bobbert & Rozendal (1987), *J. Anat.* [K];
- Jacobs, Bobbert & van Ingen Schenau (1996) [A], for explosive leg extension;
- sprint and squat studies [A]: biarticular muscles carry energy rather than generate it.

In rowing, the hamstrings, rectus femoris and gastrocnemius span hip and knee.

**Coupled-oscillator timing.** Haken, Kelso & Bunz (1985), *Biol. Cybern.* 51:347 [A]: relative phase φ follows

  φ̇ = Δω − a sin φ − 2b sin 2φ

which gives in-phase and antiphase attractors, with antiphase lost as frequency rises. Schöner & Kelso (1988) [A]
generalise it. The project already uses the crew-level version (SOURCES §22; de Brouwer et al. 2013; Cuijpers et al.
2015, 2019).

**Rowing sequencing:**

- the legs–trunk–arms order, with overlap at the transitions, is the textbook pattern [A];
- Ertel et al. (2022), ISBS 40:175 [A]: in 16 experts on an ergometer, an *early* trunk extension cut performance and
  raised trunk-extensor activity, so the usual sequence "appears to be more performant";
- Simon, Ertel et al. (2022) [A]: earlier peak trunk power raised trunk power;
- inexperienced rowers start the handle with the arms, experienced ones with the legs [A];
- Kleshnev's power shares (legs 43 %, trunk 33 %, arms 24 %) are coaching analysis, not peer-reviewed [A].

**For the rower:** the sequence should be an *output* of segment dynamics plus cost (the leading-joint and
optimal-control views agree on this). It should not be an imposed clock per segment. Each segment's timing needs its
own state, coupled to the others through interaction torques and the handle.

---

## 10. Simulation practice and tools

- **OpenSim:** Seth et al. (2018), *PLoS Comput. Biol.* 14:e1006223 [F]. The Millard and Thelen muscles,
  computed-muscle control, static optimisation and forward simulation. Delp et al. (2007) [K].
- **Moco:** Dembia, Bianco, Falisse, Hicks & Delp (2020), *PLoS Comput. Biol.* 16:e1008493 [F]. Direct collocation
  for tracking, prediction, parameter fitting and EMG-driven problems. The first such tool to handle kinematic
  constraints, which a rower in a boat has (hands on the handle, feet on the stretcher).
- **SCONE:** Geijtenbeek (2019) [K]. Reflex/CPG controllers optimised by CMA-ES. A 2023 bioRxiv study [A] uses it to
  compare reflexes and CPGs.
- **Deep RL:** Song et al. (2021) [F] review it, including the NeurIPS Learn-to-Move competitions (1300+ teams; quick
  turns and walk-to-stand without reference motion). Controllers of musculoskeletal models trained by RL are
  "rarely" human-plausible yet. DEP-RL (Schumacher et al. 2023, ICLR) [A] addresses exploration in overactuated
  muscle systems. MyoSuite (Caggiano et al. 2022) [K].
- **Validity:** Song et al. (2021) [F] list four tests:
  - stability under physiological delays and muscle nonlinearity;
  - human-like kinematics, kinetics and EMG;
  - realistic responses to unexpected disturbances;
  - adaptation to new conditions.

---

## 11. What the review says about our results (§204–206)

**(a) The collapse when τ'' is read late (§206 row 3) is a neutral delay loop with too large an acceleration gain.**

- Reading the clock's acceleration d seconds late makes the hip torque depend on τ''(t − d), which feeds back into τ''
  through the coupled 2×2 balance.
- That is the neutral form J θ̈ = … − A θ̈(t − d) of §7.3, with an effective normalised gain near the whole coupled
  inertia, |A| ≳ 1.
- Neutral-type theory predicts instability at any delay when |A| ≥ 1 (Zhang, Stépán & Insperger 2018 [F]; Insperger,
  Milton & Stépán 2013 [A]). The model found collapse at 25 ms.
- The literature's acceleration signals are partial: spindle acceleration and yank components (Blum 2017 [F]; van
  Drunen's trunk fits [A]), with gains well below the inertia.
- The physiological alternative §206 also tried is the efference copy (§6), and the sources are unanimous that it is
  the right structure (Wolpert 1995; Miall 1993; Crevecoeur & Scott 2013).

**(b) The 29–39° wander with an efference copy and 100 ms delays has a likely cause: no zero-delay layer, and a
reflex delay ~2.5× the identified trunk loop.**

The model's trunk is a pure torque generator with *all* its stabilising stiffness behind 100 ms. The measured human
trunk (Moorhouse & Granata 2007 [F]) differs:

- intrinsic stiffness acting at zero delay: k_INT ≈ 1.3–2.1 kN/m at T10, rising with activation;
- a reflex pathway at **40 ms**, supplying ~42 % of the total stabilising stiffness.

*An order-of-magnitude check (assumed lever L5/S1 to T10 of 0.25–0.30 m, so k_rot = k h²):*

| term | linear (N/m) | rotational (N·m/rad) |
|---|---|---|
| gravity | 1695 | 106–153 |
| intrinsic | 1280–2116 | 80–190 |
| reflex | ~1400 | 87–126 |

The model's coupled destabilising term λ_c² · I ≈ 24.7 × (4.5–6.4) ≈ 110–160 N·m/rad is the same size as the
gravity term. That is an independent check that the trunk plant in §182/§204 is the right size. The difference is
where the stabilising stiffness sits: in people roughly half acts with *no* delay and the rest at 40 ms.

**Prediction to test:** activation-scaled intrinsic stiffness and damping, plus a 40 ms spinal reflex, should cut the
wander from tens of degrees to a few. The drive's high extensor activation raises intrinsic stiffness exactly when the
load is largest (k_INT +65 % from 20 to 50 % effort). The long-latency layer (90–117 ms) then corrects through the
estimator, not by direct delayed PD.

**(c) The IVV gap is body sequencing (§205), and the review says sequencing should emerge.**

- The leading-joint view (Dounskaia 2010 [F]) and optimal control of multi-joint pushes (Pandy & Zajac 1991; Bobbert &
  van Soest 2001 [A]) both make the proximal-to-distal order an outcome of segment dynamics plus cost.
- Rowing EMG shows three robust activation modules whose *amplitudes*, not shapes or timing, separate performance
  (Turpin 2011ab [A]).
- So the multi-clock body should get its sequence from per-segment dynamics driven by module excitations, not from
  per-segment time laws fitted to athletes.

**(d) The "instant internal model" of §204 is not unphysiological, but it is incomplete.**

- An efference-copy forward model *is* instant: the prediction runs ahead of sensing (Miall 1993; Shadmehr 2010).
- What §204 lacked was the correction by delayed sensing, and the lower layers that keep the error small while that
  correction is in flight.

---

## 12. A proposed architecture for the multi-clock body (for decision)

Every layer is taken from the review. The order is the build order, and each step has a gate.

1. **Actuators: three muscle groups per side.** Leg extensors (knee and hip), trunk extensors (hip and lumbar) and arm
   pullers (elbow flexors and shoulder extensors), each Hill-type (Millard or Haeufle form, rigid tendon where tendons
   are short).
   - Activation dynamics 10–15 / 40–50 ms, plus an electromechanical delay.
   - Strength scaled from sourced maximal joint moments.
   - **Gate:** at the athletes' measured kinematics, the joint powers fall within the inverse-dynamics bands
     (§174–176 targets). No fit.
2. **The zero-delay layer.** Intrinsic stiffness and damping proportional to activation (short-range stiffness and
   force–velocity), with co-contraction as a parameter.
   - **Gate:** the trunk's static stiffness at 20–50 % effort lies in Moorhouse & Granata's range.
3. **The spinal layer.** Spindle (length, velocity, and a yank/acceleration term) and tendon-organ (force) feedback per
   group at a 20–45 ms delay; trunk at 40 ms. Gains are phase-dependent (drive and recovery).
   - **Gate:** a trunk perturbation reproduces the identified admittance and reflex gain (van Drunen 2013;
     Moorhouse & Granata 2007).
4. **Excitations from modules on a phase oscillator.** Three module waveforms (legs, trunk, arms), timed by one phase
   variable entrained to sensory events (handle load at the catch, stretcher load, blade release). This is a CPG used
   as a feedback predictor (Dzeladini 2014; Kuo 2002), so the stroke clock becomes *neural* and the segments' timing
   comes from their own dynamics.
   - **Gate:** the legs→trunk→arms sequence and the hands' time law emerge, and rung 2's IVV improves on the shared
     clock, with nothing fitted to the athletes.
5. **The long-latency layer.** A delay-aware estimator (forward model plus efference copy, corrected by 50–100 ms
   proprioception), adjusting the module amplitudes and the trunk's set-point. This is rung 5's predictor (§182)
   generalised.
   - **Gate:** stable at the sourced delays, and the §206 cases rerun with trunk error under 5°.
6. **Later options:**
   - **Optimal control (OFC/LQG):** time-varying LQG with signal-dependent noise, linearised about the stroke; useful
     to *derive* the module amplitudes and reflex gains, and for the stochastic programme's crew-uncertainty model.
   - **Predictive optimisation:** direct collocation (Moco-style) for the open-loop optimum stroke at a given power.
   - **Fatigue:** a three-compartment fatigue model across the race, flagged as unvalidated at non-zero velocity.

**Not proposed:**

- equilibrium-point control, which is under-specified for multi-joint dynamic tasks (§7.4);
- deep RL as the first controller, which is not yet human-plausible (§10) and is tier 3's job;
- a Huxley-type muscle, whose cost is unjustified at the accuracy whole-body models reach.

**What stays uncertain** (and should be reported as such with any result):

- Hill-model accuracy in submaximal dynamics (~54 % of force variance; Dick 2017);
- whether humans use a CPG at all (§5.2);
- the neural status of synergies (§5.3);
- OFC vs active inference (§7);
- the auditory delay, which is still unsourced.

---

## References

Read levels as above. Full texts that are openly available are in `data/local/literature/neuromuscular/` where the
mark is F.

- Adams RA, Shipp S, Friston KJ (2013). Predictions not commands: active inference in the motor system. *Brain Struct Funct* 218:611–643. [F]
- Asai Y, Tasaka Y, Nomura K, Nomura T, Casadio M, Morasso P (2009). A model of postural control in quiet standing: robust compensation of delay-induced instability using intermittent activation of feedback control. *PLoS ONE* 4:e6169. [F]
- Blum KP, Lamotte D'Incamps B, Zytnicki D, Ting LH (2017). Force encoding in muscle spindles during stretch of passive muscle. *PLoS Comput Biol* 13:e1005767. [F]
- Bobbert MF, van Soest AJ (2001). Why do people jump the way they do? *Exerc Sport Sci Rev* 29:95–102. [A]
- Brown IE, Loeb GE (2000). A reductionist approach to creating and using neuromusculoskeletal models. In Winters & Crago (eds), *Biomechanics and Neural Control of Posture and Movement*. [A, secondary]
- Burdet E, Osu R, Franklin DW, Milner TE, Kawato M (2001). The central nervous system stabilizes unstable dynamics by learning optimal impedance. *Nature* 414:446–449. [A]
- Cavanagh PR, Komi PV (1979). Electromechanical delay in human skeletal muscle under concentric and eccentric contractions. *Eur J Appl Physiol* 42:159–163. [A]
- Cholewicki J, Silfies SP, Shah RA, et al. (2005). Delayed trunk muscle reflex responses increase the risk of low back injuries. *Spine* 30:2614–2620. [A]
- Crapse TB, Sommer MA (2008). Corollary discharge across the animal kingdom. *Nat Rev Neurosci* 9:587–600. [A]
- Crevecoeur F, Scott SH (2013). Priors engaged in long-latency responses to mechanical perturbations suggest a rapid update in state estimation. *PLoS Comput Biol* 9:e1003177. [F]
- d'Avella A, Saltiel P, Bizzi E (2003). Combinations of muscle synergies in the construction of a natural motor behavior. *Nat Neurosci* 6:300–308. [A]
- Dembia CL, Bianco NA, Falisse A, Hicks JL, Delp SL (2020). OpenSim Moco: musculoskeletal optimal control. *PLoS Comput Biol* 16:e1008493. [F]
- Dick TJM, Biewener AA, Wakeling JM (2017). Comparison of human gastrocnemius forces predicted by Hill-type muscle models and estimated from ultrasound images. *J Exp Biol* 220:1643–1653. [A]
- Dimitriou M, Edin BB (2010). Human muscle spindles act as forward sensory models. *Curr Biol* 20:1763–1767. [A]
- Dounskaia N (2010). Control of human limb movements: the leading joint hypothesis and its practical applications. *Exerc Sport Sci Rev* 38:201–208. [F]
- Dzeladini F, van den Kieboom J, Ijspeert A (2014). The contribution of a central pattern generator in a reflex-based neuromuscular model. *Front Hum Neurosci* 8:371. [A]
- Ernst MO, Banks MS (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature* 415:429–433. [A]
- Ertel et al. (2022). Influence of trunk extension technique on performance and core stability during ergometer rowing. *ISBS Proc Arch* 40:175–178. [A]
- Falisse A, Serrancolí G, Dembia CL, Gillis J, Jonkers I, De Groote F (2019). Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. *J R Soc Interface* 16:20190402. [A]
- Feldman AG, Levin MF (2009). The equilibrium-point hypothesis – past, present and future. In *Progress in Motor Control*, Springer. [A, bibliographic]; Feldman AG (1986). *J Mot Behav* 18:17–54. [K]
- Flash T, Hogan N (1985). The coordination of arm movements: an experimentally confirmed mathematical model. *J Neurosci* 5:1688–1703. [A]
- Franklin DW (2016). Rapid feedback responses arise from precomputed gains. *Motor Control* 20:171–176. [F]
- Franklin DW, Wolpert DM (2011). Computational mechanisms of sensorimotor control. *Neuron* 72:425–442. [A]
- Frey-Law LA, Looft JM, Heitsman J (2012). A three-compartment muscle fatigue model accurately predicts joint-specific maximum endurance times for sustained isometric tasks. *J Biomech* 45:1803–1808. [A]; Xia T, Frey-Law LA (2008). *J Biomech* 41:3046–3052. [K]
- Friston K (2011). What is optimal about motor control? *Neuron* 72:488–498. [K]
- Gawthrop P, Loram I, Lakie M, Gollee H (2011). Intermittent control: a computational theory of human control. *Biol Cybern* 104:31–51. [A]
- Geyer H, Herr H (2010). A muscle-reflex model that encodes principles of legged mechanics produces human walking dynamics and muscle activities. *IEEE TNSRE* 18:263–273. [A]
- Gomi H, Kawato M (1996). Equilibrium-point control hypothesis examined by measured arm stiffness during multijoint movement. *Science* 272:117–120. [A]
- Haeufle DFB, Günther M, Bayer A, Schmitt S (2014). Hill-type muscle model with serial damping and eccentric force–velocity relation. *J Biomech* 47:1531–1536. [F]
- Haken H, Kelso JAS, Bunz H (1985). A theoretical model of phase transitions in human hand movements. *Biol Cybern* 51:347–356. [A]
- Harris CM, Wolpert DM (1998). Signal-dependent noise determines motor planning. *Nature* 394:780–784. [A]
- Hase K, Kaya M, Zavatsky AB, Halliday SE (2004). Musculoskeletal loads in ergometer rowing. *J Appl Biomech* 20:317–323. [A]
- Hill AV (1938). The heat of shortening and the dynamic constants of muscle. *Proc R Soc B* 126:136–195. [K]
- Hogan N (1985). Impedance control: an approach to manipulation, parts I–III. *J Dyn Syst Meas Control* 107:1–24. [A]
- Huxley AF (1957). Muscle structure and theories of contraction. *Prog Biophys Biophys Chem* 7:255–318. [K]
- Ijspeert AJ (2008). Central pattern generators for locomotion control in animals and robots: a review. *Neural Netw* 21:642–653. [A]
- Insperger T, Milton J, Stépán G (2013). Acceleration feedback improves balancing against reflex delay. *J R Soc Interface* 10:20120763. [A]
- Ivanenko YP, Poppele RE, Lacquaniti F (2004). Five basic muscle activation patterns account for muscle activity during human locomotion. *J Physiol* 556:267–282. [A]
- Koelewijn AD, Heinrich D, van den Bogert AJ (2019). Metabolic cost calculations of gait using musculoskeletal energy models, a comparison study. *PLoS ONE* 14:e0222037. [A]
- Körding KP, Wolpert DM (2004). Bayesian integration in sensorimotor learning. *Nature* 427:244–247. [K]
- Kuo AD (2002). The relative roles of feedforward and feedback in the control of rhythmic movements. *Motor Control* 6:129–145. [A]
- Kuo AD (2005). An optimal state estimation model of sensory integration in human postural balance. *J Neural Eng* 2:S235–S249. [A]
- Kurtzer IL (2015). Long-latency reflexes account for limb biomechanics through several supraspinal pathways. *Front Integr Neurosci* 8:99. [F; SOURCES]
- Kutch JJ, Valero-Cuevas FJ (2012). Challenges and new approaches to proving the existence of muscle synergies of neural origin. *PLoS Comput Biol* 8:e1002434. [F]
- Loram ID, Lakie M (2002). Direct measurement of human ankle stiffness during quiet standing: the intrinsic mechanical stiffness is insufficient for stability. *J Physiol* 545:1041–1053. [A]
- Mangalam M, Stergiou N (2026). The equilibrium point hypothesis revisited: why threshold control does not explain human movement. *Exp Brain Res* 244:44. [F, abstract and introduction]
- McCrea DA, Rybak IA (2008). Organization of mammalian locomotor rhythm and pattern generation. *Brain Res Rev* 57:134–146. [A]
- McNamee D, Wolpert DM (2019). Internal models in biological control. *Annu Rev Control Robot Auton Syst* 2:339–364. [A, bibliographic]
- Miall RC, Weir DJ, Wolpert DM, Stein JF (1993). Is the cerebellum a Smith predictor? *J Mot Behav* 25:203–216. [A]
- Mileusnic MP, Brown IE, Lan N, Loeb GE (2006). Mathematical models of proprioceptors. I. Control and transduction in the muscle spindle. *J Neurophysiol* 96:1772–1788. [F]; II. Golgi tendon organ, 96:1789–1802. [A]
- Millard M, Uchida T, Seth A, Delp SL (2013). Flexing computational muscle: modeling and simulation of musculotendon dynamics. *J Biomech Eng* 135:021005. [A]
- Minassian K, Hofstoetter US, Dzeladini F, Guertin PA, Ijspeert A (2017). The human central pattern generator for locomotion: does it exist and contribute to walking? *Neuroscientist* 23:649–663. [A]
- Moorhouse KM, Granata KP (2007). Role of reflex dynamics in spinal stability: intrinsic muscle stiffness alone is insufficient for stability. *J Biomech* 40:1058–1065. [F]
- Pandy MG, Zajac FE (1991). Optimal muscular coordination strategies for jumping. *J Biomech* 24:1–10. [A]
- Peterka RJ (2002). Sensorimotor integration in human postural control. *J Neurophysiol* 88:1097–1118. [A]
- Prochazka A, Gorassini M (1998). Models of ensemble firing of muscle spindle afferents recorded during normal locomotion in cats; Ensemble firing of muscle afferents recorded during normal locomotion in cats. *J Physiol* 507:277–291, 293–304. [A]
- Pruszynski JA, Scott SH (2012). Optimal feedback control and the long-latency stretch response. *Exp Brain Res* 218:341–359. [A]
- Rack PMH, Westbury DR (1974). The short range stiffness of active mammalian muscle and its effect on mechanical properties. *J Physiol* 240:331–350. [A]
- Reeves NP, Cholewicki J, Narendra KS (2009). Effects of reflex delays on postural control during unstable seated balance. *J Biomech* 42:164–170. [A]
- Saunders JA, Knill DC (2005). Humans use continuous visual feedback from the hand to control both the direction and distance of pointing movements. *Exp Brain Res* 162:458–473. [F]
- Scholz JP, Schöner G (1999). The uncontrolled manifold concept: identifying control variables for a functional task. *Exp Brain Res* 126:289–306. [A]
- Scott SH (2004). Optimal feedback control and the neural basis of volitional motor control. *Nat Rev Neurosci* 5:534–546. [A]
- Seth A, Hicks JL, Uchida TK, et al. (2018). OpenSim: simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement. *PLoS Comput Biol* 14:e1006223. [F]
- Shadmehr R, Smith MA, Krakauer JW (2010). Error correction, sensory prediction, and adaptation in motor control. *Annu Rev Neurosci* 33:89–108. [A; full text retrieved]
- Shaharudin S, Agrawal S (2016). Muscle synergies during incremental rowing VO2max test of collegiate rowers and untrained subjects. *J Sports Med Phys Fitness*. [A]
- Shanbhag J, Wolf A, Wechsler I, et al. (2023). Methods for integrating postural control into biomechanical human simulations: a systematic review. *J NeuroEng Rehabil* 20:111. [F, discussion]
- Song S, Geyer H (2015). A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion. *J Physiol* 593:3493–3511. [F, via PMC]
- Song S, Kidziński Ł, Peng XB, et al. (2021). Deep reinforcement learning for modeling human locomotion control in neuromechanical simulation. *J NeuroEng Rehabil* 18:126. [F]
- Taga G, Yamaguchi Y, Shimizu H (1991). Self-organized control of bipedal locomotion by neural oscillators in unpredictable environment. *Biol Cybern* 65:147–159. [A]
- Thelen DG (2003). Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older adults. *J Biomech Eng* 125:70–77. [A]
- Ting LH, Macpherson JM (2005). A limited set of muscle synergies for force control during a postural task. *J Neurophysiol* 93:609–613. [A]
- Todorov E (2005). Stochastic optimal control and estimation methods adapted to the noise characteristics of the sensorimotor system. *Neural Comput* 17:1084–1108. [F, via PMC]
- Todorov E, Jordan MI (2002). Optimal feedback control as a theory of motor coordination. *Nat Neurosci* 5:1226–1235. [K]; A minimal intervention principle for coordinated movement, *NIPS* 15. [A]
- Todorov E (2004). Optimality principles in sensorimotor control. *Nat Neurosci* 7:907–915. [K]
- Tresch MC, Jarc A (2009). The case for and against muscle synergies. *Curr Opin Neurobiol* 19:601–607. [A]
- Turpin NA, Guével A, Durand S, Hug F (2011). Effect of power output on muscle coordination during rowing. *Eur J Appl Physiol* 111:3017–3029. [A]; No evidence of expertise-related changes in muscle synergies during rowing. *J Electromyogr Kinesiol* 21:1030–1040. [A]
- Uno Y, Kawato M, Suzuki R (1989). Formation and control of optimal trajectory in human multijoint arm movement: minimum torque-change model. *Biol Cybern* 61:89–101. [A]
- Umberger BR, Gerritsen KGM, Martin PE (2003). A model of human muscle energy expenditure. *Comput Methods Biomech Biomed Eng* 6:99–111. [A]
- van Drunen P, Maaswinkel E, van der Helm FCT, van Dieën JH, Happee R (2013). Identifying intrinsic and reflexive contributions to low-back stabilization. *J Biomech* 46:1440–1446. [A]
- Wolpert DM, Ghahramani Z, Jordan MI (1995). An internal model for sensorimotor integration. *Science* 269:1880–1882. [A]
- Wolpert DM, Kawato M (1998). Multiple paired forward and inverse models for motor control. *Neural Netw* 11:1317–1329. [K]
- Zahalak GI (1981). A distribution-moment approximation for kinetic theories of muscular contraction. *Math Biosci* 55:89–114. [K]
- Zajac FE (1989). Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control. *Crit Rev Biomed Eng* 17:359–411. [K]
- Zehr EP, Stein RB (1999). What functions do reflexes serve during human locomotion? *Prog Neurobiol* 58:185–205. [A]
- Zhang L, Stépán G, Insperger T (2018). Saturation limits the contribution of acceleration feedback to balancing against reaction delay. *J R Soc Interface* 15:20170771. [F]
