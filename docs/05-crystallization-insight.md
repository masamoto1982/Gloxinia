# 05 — Grokking as crystallization: the "insight" model

*Design memo + pre-registration for the crystallization experiment. Held to
`00-charter-and-discipline.md`. Numbers live in `experiments/crystallization/`
once measured.*

## 0. Provenance and the honest boundary

This memo starts from a practitioner's model of how *insight* (閃き) is obtained,
offered by the repository's owner: occupy the "surface mind" with a low-load
repetitive task (chanting, Tetris, restocking a walk-in cooler) so the "deep
mind" — imagined as a solution in which everything seen is dissolved — can stop
being *shaken*, settle, and precipitate a *crystal never seen before*: a good
idea.

**The boundary, stated up front (a cousin of R3):** this is an analogy between an
*inference-time human practice* and a *training-time optimization phenomenon*.
"Consciousness," "toy," "stillness" are metaphors. We do **not** claim grokking
*is* meditation. We claim the *dynamical skeleton* of the practitioner's model
maps, unusually tightly, onto mechanisms this repo has already **measured** — and
that the model makes a sharp, falsifiable prediction we can test. Only the
measured part is load-bearing.

## 1. The mapping (metaphor → measured mechanism)

| practitioner's model | measured mechanism (this repo) |
|---|---|
| the **crystal** (a never-before-seen structure = good idea) | the **low-norm, per-weight-compact structural solution** (M1/M2). A crystal is a low-energy, ordered, compact form precipitating from a disordered solution — a close physical image of "low-norm, per-weight-compact structure." |
| the solution **shaken** into an oil-and-water mess | **memorization drifting to ever-higher norm** without a settling pressure (M2: post-fit the wd=0 norm runs away to ~1000, never reorganizing). |
| **stillness** (安静) → supernatant / precipitate / crystal | **norm reduction** (Phase 3: driving ‖w‖ down by any means precipitates the structural solution). This is annealing — lowering agitation so the system settles into its ordered low-"energy" state. |
| the **toy** (a low-load repetitive task that occupies the surface) | the **already-fit repetitive training data** on the post-memorization plateau: train_loss ≈ 0, so the gradient is small and steady — it *occupies* the network (it must keep fitting train) without *agitating* it. |
| without the toy, **雑念** (itch, hunger) hijack the surface | without the occupying task, only "stillness" (decay) remains and pulls the weights to **zero** — nothing to precipitate, collapse/underfit (cf. M1's underfitting floor at too-low norm). |

Where the mapping is **tight**: crystal↔low-norm structure; shaking↔norm
runaway; stillness↔norm reduction/annealing. Where it is **only metaphor**: the
"two minds," intent, subjective experience. Where it is **newly testable**: the
role of the toy and of *shake-then-still* ordering.

> **Outcome (single seed, see `experiments/crystallization/RESULTS_crystal.md`).**
> **Confirmed.** wd=0 forever never groks (norm runs to 420); memorizing first to
> *any* depth (‖w‖ up to 244 at step 15000) and *then* switching decay on
> **groks every time**, and every run precipitates to the *same* structural norm
> (~122). One sub-prediction was **corrected**: the crystallization time is
> **constant (~3200 steps from the moment stillness begins)**, not longer for
> deeper memorization — the prior shaking duration doesn't matter, only that you
> eventually become still.

## 2. The prediction this model makes (pre-registered)

The practitioner's technique is explicitly *sequential*: **first let the surface
work (agitate), then introduce stillness, and the crystal precipitates from the
already-saturated solution.** Translated:

> **Crystallization prediction:** a network first driven to **memorize** with no
> norm pressure (wd=0 → high-norm "shaken" solution, train=1, val≈0) will, once a
> norm-reducing "stillness" is switched on, **grok belatedly** — the structure
> precipitates *out of the memorized state*. It should grok regardless of how
> long it memorized first; a longer/deeper shake (higher starting norm) may just
> take longer to crystallize (more to shed).

Test: `wd_switch_step` — hold `weight_decay=0` until step *T*, then switch it to
1.0. Sweep *T ∈ {0 (control), 1000, 3000, 8000, 15000}`, plus a `wd=0`-forever
control (shaken forever → never crystallizes).

*Predictions in metric terms:*
- `wd=0` forever: no grok (val stays ≈0, ‖w‖ runs away). The shaken solution.
- every `switch_T`: after *T*, ‖w‖ **turns over and declines**, and val **groks**
  ~a few thousand steps later — the crystal precipitates from the memorized state.
- deeper prior memorization (larger *T*, higher ‖w‖ at switch) ⇒ **longer** delay
  from switch to grok (more norm to shed), but it still crystallizes.

*What would falsify it:* if a network that memorized first **cannot** grok once
decay is switched on (e.g. it's stuck and only training-from-scratch-with-decay
groks), then "the structure is dissolved in the memorized solution, waiting for
stillness" is wrong — grokking would instead require the decay present *during*
fitting, not after. That outcome is equally publishable.

## 3. Why this matters beyond the analogy

If the crystallization prediction holds, it sharpens the whole H-rep/low-norm
arc: the memorized solution is not a dead end to be avoided but a **supersaturated
solution** that already contains the structural circuit dissolved in it; grokking
is the *precipitation* of that circuit under a settling (norm-reducing) pressure,
and the "toy" (steady low-gradient repetition of the mastered task) is what keeps
the system on the train manifold while it settles. It reframes weight decay's role
from "regularizer that prevents overfitting" to "the stillness that lets an
already-present structure crystallize out." Single seed unless RESULTS says
otherwise.
