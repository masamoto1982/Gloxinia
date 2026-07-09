# 03 — Repetition & Forgetting: is grokking induced by emphasis + boredom?

*Pre-registration for the second experiment. Held to `00-charter-and-discipline.md`.
Numbers live in `experiments/repetition_forgetting/` once measured; this doc is
the hypothesis, its operationalization, and the predictions — written before the
runs.*

## 1. The hypothesis (H-rep)

> The inducing factors of grokking are (a) **emphasis** on repeated patterns and
> (b) **boredom & forgetting** of over-repeated patterns.

Restated so it can be measured and can fail. First, where does it sit in the
charter's two-ambiguity split? H-rep is a claim about the **training dynamics**
that cross the underdetermination gap (type (2)); it is not about surface
polysemy (type (1)). It is a proposed *mechanism* for C1's territory.

Mapping the metaphor to mechanism (this mapping is the load-bearing choice, so it
is stated explicitly and can be disputed):

- **"Repeated patterns" that get emphasized** = the *structural rule*
  (`(a+b) mod p`). It recurs across *many distinct* training examples, so its
  gradient is consistent and reinforced. A rule is "repeated" in the productive
  sense: many different data, one law.
- **"Over-repeated patterns" that induce boredom/forgetting** = the *memorized
  specifics*. They come from re-presenting the *identical fixed* training set
  thousands of times. Over-repetition of identical data builds a high-norm
  lookup table, which a decay pressure ("forgetting") then erodes.
- **The synthesis H-rep proposes:** grokking happens when forgetting of the
  over-repeated memorized table lets the emphasized (consistent) rule surface.

## 2. Two phases (per the decision to do both)

### Phase 1 — measure the emergent signatures

No new mechanism; instrument standard training and check three predictions.

- **P1a Over-repetition is necessary.** Compare `data_mode=fixed` (a limited set
  re-presented every step — over-repetition) against `data_mode=online` (fresh
  pairs each step from a large pool — no pair over-repeated). *Prediction:* fixed
  → memorization phase then a step; online → validation tracks train, no
  memorization gap, no step.
  *Honest confound, stated up front:* "no over-repetition" cannot be separated
  from "abundant data" in this design — online *is* the large-data regime. So
  P1a tests the joint claim "over-repetition of limited data induces the
  memorization phase," which is also the known finding that grokking needs
  limited data. We also sweep `train_frac` (pool size) to see the delay move.

- **P1b The forgetting signature.** Log `weight_l2`. *Prediction:* under weight
  decay it *peaks* (memorization) then *declines* (forgetting), and the decline
  coincides with the validation step. The `--weight_decay 0` ablation (already in
  `minimal_grok.py`) shows the counterfactual: no decay → no decline → no grok.
  This is the "forgetting" half made observable.

- **P1c Emphasis needs consistency.** Inject `label_noise` (randomize a fixed
  fraction of train labels): patterns that are *repeated every step* but are
  *inconsistent* with the rule. *Prediction:* they cannot be emphasized as a
  rule, only memorized, so higher noise lengthens/prevents the grok delay and
  caps final val_acc. This probes "emphasis on **consistently** repeated
  patterns."

### Phase 2 — build the mechanism, test induction

Add an explicit per-example **emphasis + boredom** loss reweighting
(`boredom_gamma > 0`): weight example *i* by `(1 − p_correct_i)^gamma` (EMA'd) —
up-weight not-yet-mastered examples (emphasis), down-weight examples the model has
confidently fit for a while (boredom). *Decisive question:* can this **induce
grokking with `weight_decay = 0`** — i.e., can an explicit boredom/forgetting
stand in for weight decay as the engine? *Prediction (H-rep's strong form):* yes,
or at least it shifts the grok delay markedly at fixed weight decay.

## 3. What each outcome means

- **H-rep supported** if: online removes the step (P1a), the weight-norm decline
  coincides with generalization while `wd=0` removes both (P1b), noise delays the
  step (P1c), and the explicit boredom mechanism moves or induces grokking (P2).
- **H-rep weakened/failed** if: the step survives without over-repetition, or the
  forgetting decline does not align with generalization, or the explicit boredom
  mechanism does nothing at `wd=0`. In particular, if boredom cannot replace
  weight decay, then "forgetting" as H-rep means it is not sufficient — we would
  report that weight decay's specific form of forgetting matters, not a generic
  boredom.
- **Reservation (R1 cousin):** even if all four hold, this shows H-rep describes
  the *inducement/timing* of the visible step, not that the resulting solution is
  better. Consistent with C2, the step is a property of the opaque-medium +
  over-repetition regime.

## 4. Relation to prior findings

This is the training-time mechanism behind C1 (grokking lives in the
underdetermination gap) and dovetails with C2 (the visible step needs an opaque
medium): H-rep says the *timing* of crossing the gap is set by over-repetition
(builds the memorized table) and forgetting (erodes it). All single-seed unless
`RESULTS` says otherwise; a seed sweep on the headline P1/P2 contrasts is owed
before anything here is called more than suggestive.
