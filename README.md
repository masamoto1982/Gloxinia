# Gloxinia

*Gloxinia* is a flower whose name rhymes with **grokking**. This repository is a
small, measurement-first enquiry into the relationship between **grokking**
(the delayed-generalization phase transition) and **semantic ambiguity in
language**.

The name is a reminder of the register we want: not grand claims about
intelligence, but a specific, falsifiable question, measured carefully, with the
losses and nulls written down next to the wins.

## The question

Split "ambiguity" into two things and keep them apart at all times:

1. **Surface polysemy** — the same token means different things in different
   contexts.
2. **Underdetermination** — a finite dataset is consistent with *both* a
   memorizing solution and a structural one; the data does not pin down the
   target function.

**Core claim (stated so it can be wrong):** grokking lives in the gap of (2),
not (1). And the *drama* of grokking — the fact that generalization arrives as a
sharp, late step you can *see* — may be an artifact of learning through an
**opaque medium**. If the structure is already *transparent* in the
representation, there is no memorization phase to escape: generalization is
immediate and the step is not visible. **The cliff is a property of the medium,
not of the concept being learned.**

**The AI-first corollary (why we care):** the language that is best for an AI to
*use* is not the language that is best for *making/training* one. A using-agent
wants the cost of resolving ambiguity driven to zero. A training process may
need exposure to ambiguity — not "to induce grokking," but because the target
capability *is itself* the resolution of ambiguity. See
[`docs/01-ai-first-ambiguity.md`](docs/01-ai-first-ambiguity.md).

**The honest reservation we always keep:** ambiguity may only change *when* and
*how visibly* a solution is reached — not the *quality* of the final
representation. A grokked circuit and a smoothly-learned one can be the same
circuit. We do **not** claim "ambiguity makes better AI."

## Where this came from

The enquiry began as a different question — whether an ML capability threshold
is set by a dominant dimensionless number (a Reynolds-number analogy). Five
tiered agent-benchmark sweeps (Haiku/Sonnet/Opus as relative capability tiers)
on the Ajisai language established, among other things, that context that
*lowers ambiguity* sharply lowers the weaker tier's effort. But all of that
measured **inference-time behavior of fixed, pre-trained models**, never
training dynamics. Grokking is a *training-time* phenomenon. That gap is the
reason this repository exists. The prior work is summarized as evidence — and
explicitly bounded as inference-time, not training — in
[`docs/01-ai-first-ambiguity.md`](docs/01-ai-first-ambiguity.md).

## The discipline (non-negotiable)

This repo inherits a research culture. It is enforced, not decorative.

- **Mirage guard** (Schaeffer et al. 2023): never claim a transition from a
  binary / discontinuous metric alone. Always report a continuous quantity from
  the *same* data, and only call something a transition when both move at the
  same point.
- **Proxy-naming discipline:** metrics are named for exactly what they measure,
  never for the conclusion we hope for. Metrics carry a version tag; records
  from different versions are marked incomparable.
- **Publish nulls and losses.** No fabrication. Unmeasured cells are left blank.
- **Mark the unverified**, especially the line between "measured in training"
  and "inferred by analogy from inference-time behavior."
- **Always split ambiguity** into (1) polysemy and (2) underdetermination,
  explicitly, every time.
- **Determinism:** seeds fixed, reproduction steps kept.

## Layout

```
docs/
  00-charter-and-discipline.md    the theses and the discipline, in full
  01-ai-first-ambiguity.md        design memo: the use-language vs make-language split
  02-experiment-design.md         the grokking x encoding experiment, and the
                                  labeling tension it has to resolve
experiments/
  grokking_encoding/              minimal, reproducible modular-arithmetic harness
    encoding.py                   the ONLY thing that varies across arms
    harness.py                    model + training loop + metric logging
    run.py                        CLI for one run
    results/                      committed JSON metric curves (the record)
    RESULTS.md                    measured outcomes, incl. nulls
```

## Status

Early. Two experiment rounds run (single seed, `p=97`, MLP). Full detail and
caveats in
[`RESULTS.md`](experiments/grokking_encoding/RESULTS.md) (v1) and
[`RESULTS_v2.md`](experiments/grokking_encoding/RESULTS_v2.md) (v2):

- **Harness verified:** one-hot reproduces classic grokking, confirmed by both
  the binary metric *and* the continuous co-metric (not a mirage).
- **Supports C1:** injecting underdetermination (irrelevant nuisance dims) more
  than doubled the grokking delay (2800 → 7600 steps) while keeping full
  generalization — a *later*, not absent, step.
- **Supports C2 (v2, after removing a v1 confound):** with everything held fixed
  and *only* the encoding changed, the opaque one-hot shows a memorization phase
  (val_loss *rises* to ~15 while val_acc sits at 0) and a visible step, while a
  transparent unit-norm Fourier encoding shows **no memorization phase and no
  step** (val_loss falls monotonically, val tracks train) — across every `nf`
  and weight-decay tried. v1's "transparent arm can't generalize" was the
  regularizer scale, not the medium. C2 conclusion:
  [`RESULTS_v2.md`](experiments/grokking_encoding/RESULTS_v2.md).
- **Simplest grokking recipe distilled:**
  [`DISTILLED_RECIPE.md`](experiments/grokking_encoding/DISTILLED_RECIPE.md) +
  [`minimal_grok.py`](experiments/grokking_encoding/minimal_grok.py) — four
  load-bearing ingredients (opaque encoding · weight decay · limited data ·
  full-batch Adam); the `--weight_decay 0` ablation kills grokking outright.

- **Seed-robust (v3):** across 5 seeds the C1 delay separation is clean —
  onehot `grok_delay` 2680 ± 98 vs distractor 7200 ± 283, non-overlapping — and
  across 3 seeds the C2 discriminator is unanimous: opaque val_loss humps by
  ~+10.7 while transparent stays monotone (+0.00).
  [`RESULTS_v3_seeds.md`](experiments/grokking_encoding/RESULTS_v3_seeds.md).

- **Second hypothesis tested (H-rep):** grokking as *repetition-emphasis +
  boredom/forgetting*
  ([`docs/03`](docs/03-repetition-forgetting.md),
  [`RESULTS_hrep.md`](experiments/repetition_forgetting/RESULTS_hrep.md)).
  *Descriptively supported* (Phase 1): grok delay falls monotonically with data
  and never groks below a critical fraction; a weight-norm decline ("forgetting")
  accompanies the step in the over-memorized regime; label noise (inconsistent
  repetition) collapses generalization. *Strong mechanism form falsified*
  (Phase 2): an explicit boredom (loss reweighting) or forgetting (weight noise)
  does **not** induce grokking at `weight_decay=0` — the engine is specifically
  **norm-selective** forgetting (weight decay shrinking the high-norm memorized
  table toward the low-norm structural circuit), not boredom/forgetting in
  general. **Pinned down (Phase 3):** driving the weight norm down at
  `weight_decay=0` by *any* means — uniform manual shrink or shrinking a random
  **2%** of weights per step — groks *identically* (delay ~2800), while
  norm-flat (boredom) and norm-up (noise) never grok. **Norm reduction is the
  essence; weight decay is just one instance.**

- **Why norm reduction works — low-norm ⟺ generalization**
  ([`docs/04`](docs/04-low-norm-generalization.md),
  [`RESULTS_low_norm.md`](experiments/low_norm/RESULTS_low_norm.md)). **M1
  (supported):** among train-fitting solutions the generalizing ones are
  low-norm (‖w‖≈120) far below the memorizer (≈324), with an underfitting floor
  below — so norm reduction selects generalization. **M2 (my scaling prediction
  failed, honestly reported):** the memorizer:generalizer gap does *not* widen
  with p; instead the gap is a *post-fit drift* (at fit-onset the norms match;
  unregularized training then inflates norm) and the structural solution is
  *per-weight compact* (‖w‖/√#params ≈ 0.43, ~p-independent).

- **Grokking as crystallization ("insight")**
  ([`docs/05`](docs/05-crystallization-insight.md),
  [`RESULTS_crystal.md`](experiments/crystallization/RESULTS_crystal.md)). From a
  practitioner's model of insight (occupy the surface mind with a low-load
  repetitive task so the deep mind settles and precipitates a *crystal*). Mapped
  to mechanism and tested: memorize at wd=0 (a high-norm "shaken solution"), then
  switch decay on late. **Confirmed:** never-still never groks; memorizing to any
  depth then becoming still **precipitates the same low-norm structural crystal
  (‖w‖≈122) every time**, at a **constant ~3200-step crystallization time** from
  when stillness begins. The memorized state is *supersaturated* — it already
  holds the structure dissolved in it; weight decay is the *stillness* that
  precipitates it, not merely an overfitting preventer. **The "toy" test** (does
  stillness need the right occupation?): during the still phase only *studying the
  problem* crystallizes — *idle* (decay only) collapses the norm to 0 (the task
  gradient does the reorganizing; decay just supplies pressure), and *mantra*
  (chanting a constant) makes the net *forget* the problem. So the occupation must
  BE the problem — a vanilla MLP lacks the human surface/deep memory split, an
  honest dis-analogy.

Owed throughout: seed sweeps on the H-rep / low-norm / crystallization contrasts
(single-seed so far); a cleaner minimum-norm-interpolant measurement than the
runaway wd=0 norm; and an independent test of the "toy" half (removing the data
gradient during the still phase).
