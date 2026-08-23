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
> With thanks,
> [name]

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
> With thanks and respect for the work,
> [name]

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
> With thanks,
> [name]

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
> With thanks,
> [name]

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
> With thanks,
> [name]

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
> With thanks,
> [name]

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
> With thanks,
> [name]

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
> With thanks,
> [name]

---

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
