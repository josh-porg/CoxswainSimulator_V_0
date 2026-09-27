# Data requests: rower motion through the stroke

## Why this data, specifically

The simulator reproduces boat speed, steering response, directional
stability and within-stroke attitude against independent anchors (15 of
17 checks in `scripts/validate.py`). The one quantity it still gets wrong
is **intracycle velocity variation**: 56% against 37–41% measured.

Section 39 of `SOURCES.md` records six candidate causes that have been
eliminated, including an implementation bug in the crew–hull momentum
coupling (the identity holds to 0.24 m/s², correlation 0.9993). The crew
reaction term dominates the hull's acceleration by nearly three to one,
and every crew kinematic quantity checks out *individually* — trunk swing,
seat travel, segment masses. What cannot be checked is how the segment
excursions **combine through the stroke**, because the driving dataset
gives four instants per cycle.

**So the ask is narrow: densely sampled rower kinematics through the
stroke, ideally synchronised with hull velocity.** Four keyframes cannot
determine the shape of the traverse, and the shape is what sets the boat's
speed variation.

Useful in descending order:

1. Full segment kinematics (seat, hip, shoulder, elbow, hand, knee, ankle)
   at ≥50 Hz through complete strokes, on water.
2. Seat position and trunk angle time series alone — already enough.
3. Hull velocity or acceleration recorded simultaneously with (1) or (2).
4. Ergometer equivalents, as a fallback: the crew-mass motion is the
   quantity of interest and much of it survives the transfer.

## Addresses

**Do not guess these.** Take the corresponding-author address from the
paper itself; several of these people have moved institution since
publishing.

---

## 1. Cloud, Hubbard & Moore — UC Davis / TU Delft

*Adaptive smartphone-based sensor fusion for estimating competitive
rowing kinematic metrics*, PLOS ONE 14(12): e0225690, 2019.

**The strongest contact: their data is already in use here.** This
request is also the most specific and the easiest to answer.

> Subject: Rower-mounted sensor orientation in your PLOS ONE 2019 rowing dataset
>
> Dear Dr Moore,
>
> I'm a coxswain building an open 6-DOF dynamics simulator for racing
> shells, and I've been using the CC0 dataset accompanying your 2019 PLOS
> ONE paper on smartphone sensor fusion. The differential-GPS baseline
> logs have been extremely useful — I've been able to recover your logged
> stroke rates to a mean absolute error of 0.21 spm, and the hull velocity
> traces are the anchor for my whole validation suite.
>
> I have one specific question. Alongside the boat-mounted phone, the club
> session includes a rower-mounted unit (`Pelvis2x-20180420T085631`) and
> the elite session a `Waist` unit. I would like to use these to measure
> the crew's motion relative to the hull, which is the quantity my model
> currently gets wrong. The obstacle is that the device frame rotates with
> the rower — the trunk swings some 50° through the stroke — so I cannot
> reliably resolve the signal onto the boat's surge axis. When I rotate
> using the logged CoreMotion attitude, the hull acceleration I recover
> disagrees with the device-frame figure that I can independently validate
> against your DGPS logs, so I have the rotation wrong somewhere.
>
> Could you tell me how those units were mounted and oriented on the
> athlete, and whether any attitude reference or calibration pose was
> recorded? Even a photograph or a sentence on axis convention would
> resolve it.
>
> If it is useful, I'm glad to share what I find, including two pitfalls in
> the logs that cost me some time: `log_time` is not monotonic (the file
> interleaves streams and must be sorted first), and `motion_user_
> acceleration` is in g rather than m/s².
>
> Sincerely,
> Joshua Poznanski

---

## 2. Valery Kleshnev — BioRow

Author of *Biomechanics of Rowing* and the BioRow newsletters; holds by
far the largest rowing biomechanics dataset in existence. Much of the
model's crew data comes from his published figures.

> Subject: Request: seat and trunk kinematics through the stroke, for an open rowing simulator
>
> Dear Dr Kleshnev,
>
> I'm a coxswain, and over the past months I've built an open-source
> six-degree-of-freedom dynamics simulator for racing shells — full rigid-
> body hull dynamics, a slip-based blade model, and a segment-level rower
> driven by measured joint angles. Your published work underpins several
> parts of it: the segment power shares (legs 43 / trunk 33 / arms 24), the
> trunk swing figure of 50.8°, and most recently the drive force curve,
> which I re-fitted to your reported peak at 40% of the drive length and
> decay to 74% of peak by 60%.
>
> The model now reproduces boat speed by class, steering response and
> within-stroke attitude against independent measurements. One quantity is
> still wrong: the boat's within-stroke speed variation, which comes out at
> 56% against 37–41% measured on differential GPS.
>
> I've eliminated the obvious causes, and what remains is that my rower is
> driven by a four-keyframe dataset. Four instants per stroke fix the
> postures but not the *shape* of the traverse between them, and the shape
> is what sets the hull's speed fluctuation.
>
> Would you be willing to share seat-position and trunk-angle time series
> through complete strokes — even a single athlete at one rate, at 50 Hz or
> better — for validation use? I would cite it however you prefer, and I'm
> happy to send back what the model does with it.
>
> Sincerely,
> Joshua Poznanski

---

## 3. Nick Caplan & Trevor Gardner

*A mathematical model of the oar blade–water interaction in rowing* and
the joint-angle data in J. Sports Sci. 28(3) 263–269, Table II.

**The model's rower is driven by their Table II.** They are the natural
people to ask for the underlying time series.

> Subject: Underlying time series behind the joint angles in your 2010 Table II
>
> Dear Dr Caplan,
>
> I'm building an open six-degree-of-freedom simulator for racing shells,
> and the rower in it is driven by the joint angles in Table II of your
> 2010 J. Sports Sci. paper — shank, knee, hip and trunk at the catch,
> mid-drive, finish and mid-recovery. It has served the model well: seat
> travel and trunk swing both land inside the published ranges.
>
> Its limit is now the binding one. Four instants per stroke determine the
> postures but not the shape of the motion between them, and I've traced my
> model's remaining error — it overstates the boat's within-stroke speed
> variation by about half — to precisely that. Interpolating four points
> cannot recover the traverse, and I've confirmed the error is not in the
> hull dynamics or the crew–hull coupling.
>
> If the original recordings behind Table II still exist as time series
> rather than sampled instants, would you be willing to share them? Even
> one athlete at one rate would let me test whether the traverse shape is
> the explanation. I'd cite it as you prefer and would gladly report back.
>
> I should say the SDs in Table II have been useful in their own right:
> they let me show that a change I was considering stayed well inside your
> measurement uncertainty, which is not something most published tables
> make possible.
>
> Sincerely,
> Joshua Poznanski

---

## 4. Luca Formaggia, Andrea Mola, Nicola Parolini, Edie Miglio — Politecnico di Milano / SISSA

*A model for the dynamics of rowing boats* (2009) and *A three-dimensional
model for the dynamics and hydrodynamics of rowing boats* (2010).

**This simulator's 6-DOF formulation is theirs.**

> Subject: Added mass and damping matrices in your rowing boat dynamics model
>
> Dear Professor Formaggia,
>
> I've built an open-source rowing simulator whose six-degree-of-freedom
> formulation follows your 2009 and 2010 papers, and I wanted first simply
> to say that the papers were clear enough to implement from, which is
> rarer than it should be.
>
> I have a question about the hydrodynamic loads. Your 2006 SIMAI paper
> splits them into a component proportional to the acceleration — the mass
> matrix ℳ — and one proportional to the velocity — the damping matrix 𝒮 —
> both obtained from a potential-flow solve. I have implemented added mass
> by classical strip theory instead, which gives me the mass matrix but
> leaves the velocity-dependent part represented only by an empirically
> scaled Munk moment.
>
> Would you be willing to share representative ℳ and 𝒮 for any of your
> hulls, or the numbers behind them? My immediate difficulty is calibrating
> the destabilising yaw moment: at its full ideal-flow value my eight
> broaches, at zero it becomes insensitive to losing its skeg, which
> contradicts what happens on the water.
>
> More broadly, if you have hull motion time series from those papers I
> would value them for validation — my model currently overstates
> within-stroke speed variation and I am trying to localise why.
>
> Sincerely,
> Joshua Poznanski

---

## 5. Alexander Day, Ian Campbell, David Clelland — University of Strathclyde

Experimental unsteady hydrodynamics of a single scull; drag coefficient of
a men's eight.

> Subject: Unsteady towing-tank data for a rowing shell
>
> Dear Dr Day,
>
> I'm building an open six-degree-of-freedom simulator for racing shells,
> and your experimental work on unsteady hydrodynamics of a single scull is
> directly relevant to a discrepancy I cannot close: my model overstates
> the boat's within-stroke speed variation, 56% against 37–41% measured.
>
> Your finding that acceleration measurably affects viscous drag is
> interesting to me precisely because an unsteady drag term is one of the
> few remaining candidates — my resistance model is quasi-steady, computing
> drag from instantaneous speed with no memory of acceleration.
>
> Would you be willing to share the measured drag-versus-acceleration data
> from those experiments, or the fitted coefficients? I would like to test
> whether an unsteady correction of the size you measured is enough to
> account for what I am seeing, and if it is not, to be able to say so.
>
> Sincerely,
> Joshua Poznanski

---

## 6. Laura Cuijpers, Harjo de Poel, Frank Zaal — University of Groningen

Rowing crew coordination dynamics; antiphase rowing and velocity
fluctuation losses.

> Subject: Ergometer displacement and crew centre-of-mass data
>
> Dear Dr Cuijpers,
>
> I'm building an open dynamics simulator for racing shells, and your work
> on crew coordination — particularly the finding that antiphase rowing
> recovers the 5–6% of power lost to velocity fluctuations — bears directly
> on the problem I'm stuck on.
>
> My model reproduces boat speed, steering behaviour and hull attitude
> against measurement, but overstates within-stroke velocity variation by
> about half. Since that variation is driven almost entirely by the crew's
> centre of mass shuttling along the boat, your mechanically-linked
> ergometer measurements would be an unusually direct test: they isolate
> exactly the coupling I suspect.
>
> Would you be willing to share the ergometer displacement time series, or
> rower centre-of-mass estimates, from those experiments? Even a single
> in-phase condition would let me check my crew model against a measured
> displacement rather than against inferences from hull motion.
>
> Sincerely,
> Joshua Poznanski

---

## 7. Anna Sliasas & Stephen Tullis — McMaster University

Shell-velocity-coupled blade hydrodynamics.

> Subject: Blade force and boat velocity coupling data
>
> Dear Dr Tullis,
>
> I'm building an open six-degree-of-freedom simulator for racing shells.
> Its blade model is slip-based after Cabrera and Ruina, and your
> shell-velocity-coupled work is the closest thing I know of to a proper
> treatment of the interaction I am approximating.
>
> My model currently overstates the boat's within-stroke speed variation,
> and one hypothesis I have not been able to test is that a blade planted
> in the water damps hull surge in a way a prescribed force profile cannot
> represent — that the blade should react to hull motion rather than being
> imposed on it.
>
> Would you be willing to share blade force time series coupled to shell
> velocity from your simulations, or to say whether your results show that
> damping effect at a magnitude that would matter? Either would help me
> decide whether to build a reacting blade model or rule the idea out.
>
> Sincerely,
> Joshua Poznanski

---

## 8. Cooper Knarr & Haley Kwoun

*Using IMU sensors to compare rowing ergometers with rowing on the water*,
Proc. IMechE Part P (with V. Kleshnev).

> Subject: IMU time series from your ergometer-versus-water comparison
>
> Dear Dr Knarr,
>
> I'm building an open dynamics simulator for racing shells. Your IMU
> comparison of ergometer and on-water rowing is relevant to a specific
> gap in it: my rower is driven by a four-keyframe dataset, which fixes the
> postures but not the shape of the motion between them, and I've traced my
> model's remaining error to that.
>
> Would you be willing to share the raw IMU time series — particularly
> anything mounted on the athlete rather than the hull? Seat or trunk
> motion through complete strokes at your sampling rate would let me
> replace an interpolation with a measurement.
>
> I should mention a practical detail in case it saves you a question: I
> have worked with athlete-mounted IMU data before and the sticking point
> was sensor orientation, since the device frame rotates with the rower. If
> your mounting convention is documented, that would be as valuable as the
> data.
>
> Sincerely,
> Joshua Poznanski

---

## 9. Ernst Jan Grift, Mark Tummers & Jerry Westerweel — TU Delft

*Hydrodynamics of rowing propulsion*, J. Fluid Mech. **918** (2021) A32;
*Drag force on an accelerating submerged plate*, J. Fluid Mech. **866**
(2019).

**Two blocked items depend on this**, both recorded in
`docs/PHYSICS_PROGRAMME.md`:

* The 2019 paper measures drag coefficient against immersion depth on an
  aspect-ratio-2 plate chosen to resemble an oar blade — 1.10 at the
  surface, peaking near 1.60 at 20 mm, settling to 1.30 deeper. Our
  `immersion_factor` is a monotone saturating curve whose own docstring
  admits it is "a *shape* chosen … and not a fitted law". **The
  measurement has an optimum and ours does not**, so it is qualitatively
  wrong, and three points read from an abstract are not enough to fit.
* The 2021 paper is the only source found that gives the **tangential**
  blade force through a realistic stroke. Our model's tangential
  component is identically zero. It is the primary validation target for
  the lift-and-drag blade model.

*2026-09-27: partly answered by the literature.* Grift's PhD thesis is open access and
carries both papers; the immersion curve has been read from its Fig. 2.4 and the
lift/drag split from chapter 3 (SOURCES §159). The ask narrows to the tabulated numbers
behind Figs 2.4 and 2.11c and the force traces of Fig. 3.10, and to how the entrainment
rate scales to a full-size blade.

> Subject: Digitised force and drag curves from your rowing-blade work
>
> Dear Dr Grift,
>
> I'm building an open 6-DOF dynamics simulator for racing shells, with
> every empirical constant traced to a source. Two of your results bear
> directly on the part of it I am currently unable to do honestly.
>
> My blade model is [CR06]'s Model 1 — a normal force proportional to the
> square of slip — which has no tangential component at all. Your 2021
> PIV-and-force measurements appear to be the only dataset that resolves
> normal and tangential through a realistic blade path. Would you be
> willing to share the time series behind those figures?
>
> Separately, my ventilation factor is a shape I chose to be monotone in
> immersion depth, which your 2019 plate measurements show is the wrong
> shape — there is an optimum. I have the three points quoted in the
> abstract, which is enough to know I am wrong and not enough to fix it.
> The C_D-against-depth curve would replace a guess with a measurement.
>
> Either would be used with attribution and the provenance recorded in
> the open source tree.
>
> Sincerely,
> Joshua Poznanski

---

## 10. James Hill & Bernhard Fahrig

*The impact of fluctuations in boat velocity during the rowing cycle on
race time*, Scand. J. Med. Sci. Sports **19**(4) 585–594.

Their Table 1 is already the model's independent check on stroke timing.
The ask is the **boat speeds** that go with the drive durations.

> Subject: Boat speeds alongside the drive durations in your Table 1
>
> Dear Dr Hill,
>
> I'm building an open dynamics simulator for racing shells. Your Table 1
> — eight elite coxless pairs at stepped rates, with measured drive
> durations — has been the out-of-sample check on my stroke timing for
> some time.
>
> I've now made the oar angle a dynamic state rather than a prescribed
> schedule, and as a result the drive duration stops being a formula of
> stroke rate and becomes a consequence of how hard the rower pulls and
> how fast the hull is moving. My model already predicts that the drive
> shortens as the boat speeds up. To test that against your measurements
> I need the **mean boat speed at each of the four rates**, which the
> table as published does not carry.
>
> If those are to hand it would turn a plausible prediction into a
> falsifiable one.
>
> Sincerely,
> Joshua Poznanski

---

## 11. Foot-stretcher force components (added 2026-09-14)

**Why.** Phase 4.3's joint moments stall at the knee and ankle. In the
model's plane, the seat's vertical load and the stretcher's vertical force
share one equation, so the split of stretcher force into vertical and
horizontal parts has to be measured (SOURCES, "Foot-force direction on the
water"). Every source found measured it but did not publish it in numbers:
men only, figures only, or horizontal only. **The ask is the same to each
group: vertical and horizontal (or plate-normal and plate-parallel)
stretcher force against stroke time, ideally with a women's or sculling
subset.** Take each address from the paper, as above.

### 11a. Peter Sinclair, Andrew Greene & Richard Smith — University of Sydney

*The effects of horizontal and vertical forces on single scull boat
orientation while rowing*, ISBS 27 (2009).

> Subject: Vertical and horizontal stretcher force from your instrumented single
>
> Dear Dr Sinclair,
>
> Your 2009 ISBS paper on boat orientation used a single scull with a 3-D
> instrumented foot stretcher. Your figures show the vertical stretcher force
> rising into the catch and falling through the drive as pin force takes
> over. I'm building an open simulator for racing shells and working through
> the rower's joint moments on a measured stroke, and the knee and ankle are
> stuck on exactly that split: without it the seat's and the stretcher's
> vertical loads can't be separated.
>
> Would the ensemble-averaged vertical and horizontal (fore-aft) stretcher
> force curves against stroke time from that study, even as numbers read
> off the figures, be something you could share? A few points through the
> drive would already be enough.
>
> Sincerely,
> Joshua Poznanski

### 11b. Arnold Baca, Philipp Kornfeind & Mario Heller — University of Vienna

*Comparison of foot-stretcher force profiles between on-water and ergometer
rowing*, ISBS 24 (2006).

> Subject: Vertical stretcher force from your single-scull dynamometer
>
> Dear Professor Baca,
>
> Your 2006 comparison of stretcher forces on the water and on the Concept2
> measured the plate-normal and plate-parallel components and converted
> them to horizontal and vertical force. Table 1 gives the horizontal peaks.
> I'm modelling the rower's joint moments on a measured single-scull stroke,
> and the knee and ankle need the vertical component alongside the horizontal.
>
> Would the vertical force, or the normal and parallel components with the
> plate angle, be available for the boat trials, as mean curves or a few
> values through the drive? I'd cite the source in the project's documentation.
>
> Sincerely,
> Joshua Poznanski

### 11c. Erica Buckeridge, Anthony Bull & Alison McGregor — Imperial College London

Buckeridge, PhD thesis (2013), doi:10.25560/28699, chapters 5–7; and
*Incremental training intensities increases loads on the lower back of elite
female rowers*, J. Sports Sci. 34(4) (2016).

> Subject: Vertical and horizontal foot force for the heavyweight women
>
> Dear Dr Buckeridge,
>
> Your thesis's instrumented Concept2 recorded vertical and horizontal foot
> force for GB heavyweight women scullers and sweep rowers, shown in Fig. 5.7.
> I'm building an open rowing simulator and have one on-water women's single
> stroke whose hip moment I've computed top-down from the handle. The knee
> and ankle need a foot-force direction for women, and yours is the closest
> measurement I've found.
>
> Would the group-mean vertical and horizontal foot force for the women at
> the catch and at maximum handle force, or the curves behind Fig. 5.7, be
> shareable? I'd be glad to show what the model does with them.
>
> Sincerely,
> Joshua Poznanski

### 11d. Richard Smith & Conny Loschner — University of Sydney *(added 2026-09-18)*

*The relationship between pin forces and individual feet forces applied
during sculling*, 3rd Australasian Biomechanics Conference (2000); and
*Biomechanics feedback for rowing*, J. Sports Sci. 20(10):783–91 (2002).

**Updated the same day: the project owner supplied the 2000 paper, so it is
read ([LO00] in SOURCES) and no longer the thing to ask for.** It measures each
foot separately but *propulsively only*, and its own conclusion defers
"transverse and vertical forces on the pin and stretcher" to a 1999 abstract.
So the ask narrows to two specific things.

Found through the reference list of *Over 50 Years of Researching Force
Profiles in Rowing* ([WA18], 2018), which credits this group with measuring
stretcher and pin force in the **non-propulsive (vertical and transverse)
planes** — the only such citation found anywhere in the search. Smith also
co-authors that review and the 11a paper, so the reviewer, the measurer and
the 11a author are one person; say so in the email.

> Subject: Vertical and transverse stretcher force from your Sydney sculling work
>
> Dear Professor Smith,
>
> I'm building an open simulator for racing shells and working out the rower's
> joint moments on one measured single-scull stroke. The knee and ankle come
> down to a split I have not found published in numbers: how much of the
> stretcher force is vertical and how much horizontal, through the drive.
>
> I've read your ABC3 paper with Conny Loschner on pin forces and individual
> feet forces — it has been genuinely useful, and its peak foot forces for the
> three W1x scullers are the closest match to my athlete I've found. Its
> conclusion points to transverse and vertical forces on the pin and stretcher,
> citing your 1999 IOC congress abstract on three-dimensional pin forces.
>
> Two questions, either of which would help:
>
> 1. Did the same rig record the stretcher in three dimensions as well as the
>    pin, and if so is there a mean vertical (or plate-normal) foot force
>    against stroke time or oar angle that you could share?
> 2. Is the 1999 abstract, or the fuller write-up behind it, available anywhere?
>    I have not been able to find it.
>
> Even a few points through the drive, or values read off your own figures,
> would be enough. I'd cite it in the project's documentation and would be glad
> to show you what the model does with it.
>
> Sincerely,
> Joshua Poznanski

### What not to ask for (2026-09-18)

**Don't ask a Peach PowerLine lab.** [LE26] Legge et al. state that their
gate and stretcher forces "were measured along the boat's longitudinal
axis" — the standard elite system records **one axis at the stretcher**, so
a PowerLine-equipped squad has no vertical component to send, however good
its data otherwise. Every addressee above was picked for custom
instrumentation (Kistler transducers, a footplate dynamometer, an
instrumented Concept2); keep it that way. See SOURCES, "Foot-force
direction on the water".

---

## Status: sent 2026-09-19

The requests in this file were sent by the project owner on **19 September 2026**,
signed Joshua Poznanski. Nothing here is waiting on drafting any more — it is
waiting on replies.

* **Record answers against the section they answer**, and note in SOURCES whether
  the number arrived and what it changed. A reply that declines is still worth
  recording: it closes a line.
* **§11 is the one that blocks live work.** Phase 4.3's knee and ankle need the
  foot-force direction, and the search so far says no one has published it: the
  standard elite instrument ([LE26]'s Peach PowerLine) measures the stretcher on
  one axis only, [LO00] measures each foot but propulsively, and [F09] supplies the
  vertical component by assumption rather than measurement.
* **If nothing comes back on §11**, the fallback is not to guess a direction. It is
  to state in the model that the knee/ankle split is unidentifiable from the
  available on-water data and to carry the hip moment alone, which rung 2 already
  computes and checks.
* Work that does *not* depend on a reply continues meanwhile — 4.3a's catch entry
  is unblocked and is where the effort is going.

## Replies received

### §4 Formaggia — **answered 2026-09-19.** Partly declined, and better than the ask.

Prof. Formaggia replied that the details requested are **unavailable**, but supplied
**Andrea Mola's PhD thesis draft** and referred the project to Mola as its author.
Recorded in SOURCES as **[MOLA]**. The referral is the live thread: Mola is the
person to ask about the rower-motion reconstruction and the marker data behind it.

*What it closed:* the §4 ask as worded is answered — those numbers are not available.
*What it opened:* the thesis carries a **measured hand-marker path** and the full
provenance of [F09]'s rower motion, which is worth more to phase 4.3 than the
original request was. See [MOLA].

### §2 Kleshnev — **answered 2026-09-25. Data received; better than the ask.**

Dr Kleshnev sent a single athlete's **ensemble-averaged stroke (about 20 strokes of
steady-state rowing)**: an elite men's single at **32.4 spm**, rower 1.91 m and 97 kg,
boat 18 kg, inboard 0.875 m, oar 2.885 m, 51 samples across the 1.851 s cycle. Recorded in
SOURCES as **[BR24]** (§156). Stored at `data/local/biorow/` — commercial data shared for
validation, **gitignored and not to be redistributed**.

*What it closed:* the ask was seat-position and trunk time series through complete
strokes. Both are there, plus handle force and velocity for each oar, horizontal and
vertical oar angle for each oar, and the boat's own velocity and acceleration.

*What it opened:* the first **like-for-like** target for the model's largest known
discrepancy. The 37–41% IVV and 8.88 m/s² it had been compared against came from a
club double and a DGPS session, not from the athlete the model was set up as. Here
rig, rate, segment motion and boat response belong to one athlete.

*No longer blocked:* the origins turned out not to matter, and the frame and tracked
point were inferred from the data (SOURCES §156): rower centre-of-mass travel 0.71–0.74 m
relative to the hull. The reply below asks him to confirm that reading rather than to
supply it. *2026-09-27:* question 3 added (handle-force measurement point) after the
blade-at-his-kinematics tests (SOURCES §162) found the propulsive scale of that channel
open. Draft still unsent.

> Subject: Re: seat and trunk kinematics — thank you, and a few questions
>
> Dear Dr Kleshnev,
>
> Thank you — this is exactly what I needed, and more. Having the boat's own velocity
> and acceleration alongside the seat, trunk and handle data for the same athlete makes
> it the first like-for-like target I've had for the model's within-stroke speed
> variation. A first look: this stroke shows a velocity fluctuation of
> 48.9% of the mean and a hull acceleration range of 15.3 m/s², and the drive occupies
> 54% of the cycle by both oar angle and handle force.
>
> A few questions so I use it correctly:
>
> 1. I've read the channels from the data itself and would value a check. **Seat
>    position** appears zeroed at the catch. **Trunk position** appears to be measured
>    relative to the seat rather than the boat, tracking a point near shoulder height: a
>    momentum balance against the boat's acceleration puts it about 0.58 m above the hip,
>    where this athlete's shoulder joint would be. Is that right, and which point is it?
> 2. On that reading the rower's centre of mass travels about 0.72 m relative to the
>    boat. Does that match what you'd expect for him?
> 3. **Handle force**: how is it measured, and at what point along the handle? Is it
>    the component normal to the oar? Taken with the oar angle, the boat's momentum over
>    the cycle wants a propulsive force of about 0.41 of the handle force, which puts the
>    blade's effective centre of pressure near its tip (about 2.07 m from the pin against
>    a blade centre of 1.80). I'd like to know whether that is the blade or my reading of
>    the channel.
> 4. Were the 51 samples resampled from a higher native rate?
> 5. Do you have this athlete, or a comparable one, at **other stroke rates**? How the
>    fluctuation changes with rate is the test the model most needs.
> 6. How would you like the data cited?
>
> I'll send you the model's result against this stroke once it's run.
>
> Sincerely,
> Joshua Poznanski

### §11c Buckeridge — **answered 2026-09-25. Declined; referred on.**

Dr Buckeridge no longer has access to the thesis datasets (PhD completed 2013, since
relocated), and referred the project to her supervisor **Prof. Alison McGregor**
(a.mcgregor@imperial.ac.uk), who remains active in the area and may hold the original
data or have students continuing the work. She asked to hear how the simulator
develops.

*What it closed:* the thesis data are not available from the author.
*What it opened:* the referral is the live thread for the foot-force direction, which
is still the item that blocks phase 4.3's knee and ankle.

> Subject: Foot-force data from the instrumented ergometer work — referred by Erica Buckeridge
>
> Dear Professor McGregor,
>
> Erica Buckeridge kindly suggested I contact you. I'm building an open-source rowing
> simulator, and her thesis's instrumented Concept2 recorded vertical and horizontal
> foot force for GB heavyweight women scullers and sweep rowers (Fig. 5.7). I have one
> on-water women's single stroke whose hip moment I've computed from the handle, and
> the knee and ankle need a foot-force direction for women — hers is the closest
> measurement I've found.
>
> Would the group-mean vertical and horizontal foot force at the catch and at maximum
> handle force, or the curves behind Fig. 5.7, be available from the group? I'd also be
> glad to hear of any students continuing this work.
>
> Sincerely,
> Joshua Poznanski

> Subject: Re: rowing simulator — thank you
>
> Dear Dr Buckeridge,
>
> Thank you for the quick reply and the pointer to Professor McGregor — I'll write to
> her. I'll gladly send you an update as the simulator develops.
>
> Sincerely,
> Joshua Poznanski

## Contact addresses (compiled 2026-09-19)

**Verified means read verbatim out of the paper's own PDF.** Nothing in the
"verified" column was guessed from an institutional naming pattern, and no address
anywhere in this file has been constructed. Where none was found, the row says so
and gives the route instead — a fabricated address either bounces or reaches a
stranger, and neither is acceptable.

| § | who | address | provenance |
|---|---|---|---|
| 3 | Nick Caplan | `nick.caplan@northumbria.ac.uk` | **verified** — Caplan & Gardner (2007) PDF |
| 4 | Luca Formaggia | `luca.formaggia@polimi.it` | **verified** — Formaggia et al. (2009) PDF, and the 3-D companion paper |
| 5 | Alexander "Sandy" Day | `sandy.day@na-me.ac.uk` | **verified** but old — the paper's own address; `na-me.ac.uk` was Strathclyde's Naval Architecture department and may since have folded into `strath.ac.uk` |
| 6 | Laura Cuijpers | `l.s.cuijpers@rug.nl` | **verified** — Cuijpers et al. PDF |
| 11d | Constanze Loschner | `Closchner@dsr.nsw.gov.au` | **verified but almost certainly dead** — printed in the 2000 ABC3 paper; NSW Dept of Sport & Recreation, 26 years old |

**Not found, and not invented.** §2 Kleshnev (BioRow — commercial, contact through
biorow.com), §7 Sliasas & Tullis (McMaster), §8 Knarr & Kwoun, §9 Grift, Tummers &
Westerweel (TU Delft), §10 Hill & Fahrig, §11a Sinclair, §11b Baca, §11c
Buckeridge, Bull & McGregor. For these, take the address from the paper's
corresponding-author footnote, or from the current staff page. Two were checked
directly and carry no public address: Sydney's profile pages render by script
(`profiles.sydney.edu.au/peter.sinclair`) and Vienna's staff page for Baca prints
none.

### A better route to §11 than any of the above

**Dr Conny Draper** is the best-connected person for the foot-force question, on
three separate counts:

* she **co-authors [LE26]** (Legge et al., *Assessment of rowing biomechanics
  during single sculling using functional clustering*) — the Peach PowerLine paper
  whose one-axis statement closed off the standard instrument;
* she **co-authors [WA18]**, the review whose reference list produced §11d;
* she was **Senior Sports Biomechanist for Rowing at the AIS** and now consults to
  national rowing teams — so she knows what instrumentation exists and who holds
  what.

`conny.draper@ausport.gov.au` appears in a 2010 document in the local library.
**Treat it as unverified for current use** — she has since left the AIS for
consulting, so it may be stale.

*One thing worth asking her directly rather than assuming.* §11d's paper is by
Constanze **Loschner** with Richard Smith; [WA18] and [LE26] are by Conny
**Draper** with, in [WA18]'s case, Richard Smith. Same first name, same narrow
field, same collaborator. It is plausible these are one person under a changed
surname, **but a search did not confirm it** and it should not be assumed in
writing. If they are the same person, she is the author of the very paper §11d
asks about, which would make her the single most valuable contact in this file.

The live corresponding author of [LE26] is **`natalie.legge@acu.edu.au`**
(verified, Australian Catholic University), and [WA18]'s is
**`john.warmenhoven@hotmail.com`** (verified). Either is a reasonable route to
Draper and Smith.

## Notes on sending these

* **Send them individually**, not as a mailing list. Each is written to
  the specific work; that is why they are likely to be answered.
* **Attach nothing on the first email.** Offer, don't send.
* **The UC Davis one is the highest-probability answer** — the ask is a
  single factual question about their own instrumentation, and you are
  already using their data productively.
* **Kleshnev's data is commercial.** He may decline, or offer a paid
  arrangement, and either is a reasonable answer to receive.
* If a reply asks what the project is for, the honest answer is a good
  one: a coxswain trying to test steering strategies computationally,
  building in the open, with the validation and the failures both
  published in the repository.
