import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import io
import shutil
import sys

EDITS = []


def E(old, new):
    EDITS.append((old, new))


E(r"""\noindent One effect survived within-race validation---catch calls, recovered
independently by keyword and by language-model labelling---but the additional races
could not resolve it: their intervals are twenty times wider and contain the
original estimate at every lag. Coding""",
  r"""\noindent One effect survived within-race validation---catch calls, recovered
independently by keyword and by language-model labelling---but it does not survive a
model in which boat speed is also predicted from its own history, and the additional
races could not resolve it in either direction. Treated as one coupled process, with a
causal transformer and a linear autoregression fitted to 38 recordings, neither stream
improved held-out prediction of the other, although both models recover coupling planted
in synthetic data of the same shape. Coding""")

E(r"""other two, together with the validation outcome at each stage.""",
  r"""other two, together with the validation outcome at each stage, and a joint model of
calls and boat that accepts new recordings without modification.""")

E(r"""bootstrap (400 resamples, 40-stroke blocks) that preserves serial correlation.""",
  r"""bootstrap (400 resamples, 40-stroke blocks) that preserves serial correlation. This
model predicts speed from calls alone; Section~\ref{sec:coupled} adds the boat's own
history and models both channels jointly.""")

E(r"""calls at issue differs tenfold between the two event types. The effect is therefore \emph{unreplicated} rather than refuted: robust to how it is
labelled and how the boat data are detrended, and not yet tested at a precision that
could distinguish it from zero.""",
  r"""calls at issue differs tenfold between the two event types. On independent data the
effect is \emph{unreplicated} rather than refuted.

Within race 1, however, it does not survive a stricter model. A regression of speed on
calls alone cannot distinguish a call that changes the boat from a boat that was already
changing when the coxswain spoke. The boat's own past---its last two strokes and
exponentially decayed sums of its history at 2, 8 and 30 strokes---predicts 64.7\% of
held-out stroke-scale variance. Adding the catch series to that model lowers held-out
$R^2$ to 64.5\% (200 circular shifts, $p=0.36$), and with the 25 raw lags of the original
analysis to 62.4\% ($p=0.97$). The centred moving average used for detrending draws on up
to 30 future strokes, which is harmless in a static regression but not in one that
conditions on the past, so the test was repeated under three causal treatments of speed
(150 shifts each): raw speed (0.740 to 0.733, $p=0.37$), a trailing 61-stroke detrend
(0.708 to 0.697, $p=0.57$) and first differences (0.234 to 0.190, $p=0.68$). Under no
treatment does the catch series add predictive information. The race 1 estimate is
accounted for by the boat's own dynamics: catch calls followed boat behaviour that was
already predictable from the preceding strokes, as expected if the coxswain calls what
they can already feel.""")

NEWSEC = r"""\section{Calls and boat as one coupled process}
\label{sec:coupled}
The regressions above treat the calls as a fixed input and speed as the only outcome.
Both are sequences with their own momentum, and each may alter what the other does next.
The appropriate object is the joint process
\begin{equation}
p(C,B)=\prod_t p\bigl(C_t\mid C_{<t},B_{<t}\bigr)\,p\bigl(B_t\mid B_{<t},C_{<t}\bigr),
\end{equation}
whose two factors define two coupling questions, each a Granger test scored by held-out
log-likelihood: whether the call history improves prediction of the boat given the
boat's own past ($C\to B$), and whether the boat history improves prediction of the next
call given the calls' own past ($B\to C$).

\textbf{Data.} Every recording is placed on a grid of 2\,s steps, about one stroke. Speed
and rate are standardised within each recording, which removes crew and boat-class
level differences; the call stream is ten binary channels, the eight functional
categories plus bare counting and any speech. Missing cells are masked, not filled. The
three races are joined by 35 further recordings from the same coxswain that carry
captions but no boat data. These inform only the call process's own dynamics, giving
11\,313 grid steps in all, 647 of them with boat data.

\textbf{Models.} The linear reference summarises each stream's history by exponentially
decayed sums at $\tau=2$, 8 and 30 steps, the linear form of a multivariate Hawkes
process, plus the boat's last two readings; the boat is fitted by ridge and each call
channel by logistic regression, with own-past and both-pasts models fitted separately.
The second model is a causal transformer (two layers, width 32, context of 64 steps)
with a Gaussian output for the boat and a Bernoulli output per call channel. It is
trained with stream dropout: in 30\% of training windows the entire boat input, or the
entire call input, is removed as though it had not been recorded. One set of weights
then yields both own-past and both-pasts predictions, and recordings with a single
stream train the corresponding self-term without special handling.

\textbf{Evaluation.} Five folds; synchronised races are split into contiguous blocks,
other recordings are assigned whole, and held-out steps are never used as training
inputs. The coupling null circularly shifts the other stream within each test race (50
draws), keeping each stream's own structure and destroying their alignment.

\begin{table}[H]
\centering
\small
\caption{Held-out log-likelihood gains, in nats per step (boat: per observed cell; calls:
summed over the ten channels per step). Self-terms are against a model with no history;
coupling terms are both-pasts minus own-past on the synchronised races, with
circular-shift $p$. A second training seed gives transformer coupling of $-0.115$
($p=1.00$) and $-0.040$ ($p=0.28$).}
\label{tab:coupled}
\begin{tabular}{lrr}
\toprule
 & Linear & Transformer\\
\midrule
Boat, own past & $+0.316$ & $+0.283$\\
Calls, own past & $+0.348$ & $+0.323$\\
$C\to B$, calls into boat & $-0.236$ ($p=0.20$) & $-0.063$ ($p=0.67$)\\
$B\to C$, boat into calls & $-0.056$ ($p=0.49$) & $-0.026$ ($p=0.41$)\\
\bottomrule
\end{tabular}
\end{table}

Each stream is predictable from its own past (Table~\ref{tab:coupled}). For the calls
this is largely a matter of timing: bare counting and speech activity carry 0.25 of
the linear model's 0.35 nats, and no single category more than 0.023, consistent with
the absence of theme-level sequence structure noted in Section~\ref{sec:survived}.
Neither direction of coupling improves held-out prediction, and neither is
distinguishable from the shifted null. The negative gains are the cost of the added
parameters when the other stream carries no usable signal, largest for the linear model,
whose joint form adds 33 call features to about 500 boat observations per fold.

\textbf{Positive control.} Synthetic recordings of the same shape (three synchronised
sessions of 600 steps and twenty call-only sessions) were generated from a process with
coupling set by hand. With no coupling, $p$-values ranged from 0.13 to 0.84. With a call
raising speed by one within-session standard deviation over the following three steps,
and low speed raising the probability of a call, both models detected both directions
(gains $+0.044$ to $+0.058$ nats, $p=0.032$, the smallest attainable with 30 draws). At
0.4 standard deviations the shift test still detected coupling ($p=0.032$) while the
held-out gain was not positive, so the permutation test is the more sensitive of the two
readouts. The coupling reported for the races is therefore absent at a size the
pipeline can see, not unmeasurable; effects of the size in Table~\ref{tab:power} lie well
below that.

The transformer does not outperform the linear reference on these data. It is included
because it is the form that scales: it can represent multi-step dependencies,
interactions between call types and per-recording context as data accumulate, and a new
recording enters as a transcript and a boat log on a shared clock, with no change to the
code.

\section{What it would take}"""
E(r"""\section{What it would take}""", NEWSEC)

E(r"""reproduces the category selection reported in
Section~\ref{sec:survived} by a different route.""",
  r"""reproduces the category selection reported in
Section~\ref{sec:survived} by a different route and, like it, takes no account of the
boat's own history.""")

E(r"""requirement is now a specified sample size rather than an open question. One line of""",
  r"""requirement is now a specified sample size rather than an open question.

Treating calls and boat as one coupled process changes what counts as evidence. An
effect of a call estimated without the boat's own history cannot be separated from the
coxswain responding to the boat, and the one candidate effect in these data did not
survive that separation. The same is likely of any call made by feel, which on the
coxswain's own account includes most technical calls. Future corpora should be
analysed through both factors of the joint process, with the coupling null and a
planted-coupling control reported alongside. One line of""")

E(r"""stroke-to-stroke noise, and the one candidate effect we found was not resolvable from
two further races.""",
  r"""stroke-to-stroke noise, and the one candidate effect we found was accounted for by
the boat's own history.""")

E(r"""contamination between themes and the race trend, but not anticipation. The catch result in
particular is at least as consistent with the coxswain calling what they can already see
going wrong as with the call causing what follows.""",
  r"""contamination between themes and the race trend. Conditioning on the boat's own past
(Section~\ref{sec:coupled}) removes anticipation of whatever that past predicts, but not
of what the coxswain perceives and the log does not record.""")

E(r"""The positive findings are limited: one candidate effect that survived every within-race
test but that independent data lacked the precision to confirm or reject, and one
structural finding about the sequence of calls that survived a deliberate attempt to
suppress it. The methods are workable, the results are inconclusive, and the sample size
that would make them conclusive is now stated rather than guessed.""",
  r"""The positive findings are limited: one candidate effect that survived every
within-race test of a calls-only model but is accounted for by the boat's own history,
and one structural finding about the sequence of calls that survived a deliberate
attempt to suppress it. The methods are workable, the results are inconclusive, the
sample size that would make them conclusive is stated, and a joint model validated
against planted coupling is in place for the corpus that sample size requires.""")

for p in ("paper.tex", "paper_sage.tex"):
    s = io.open(p, encoding="utf-8").read()
    for old, new in EDITS:
        n = s.count(old)
        if n != 1:
            sys.exit("%s: %d matches for %r" % (p, n, old[:70]))
        s = s.replace(old, new)
    shutil.copy(p, p.replace(".tex", "_pre_coupled.tex.bak"))
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print(p, "ok")
