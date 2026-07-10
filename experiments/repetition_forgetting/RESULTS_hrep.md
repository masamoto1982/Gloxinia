# RESULTS — Repetition & Forgetting (H-rep)

Tests the hypothesis (`docs/03-repetition-forgetting.md`) that grokking is
induced by **emphasis on repeated/consistent patterns** and **boredom &
forgetting of over-repeated patterns**. `p=97`, one-hot MLP, full-batch AdamW,
seed 0 unless noted. `metrics_version = rf-metrics-v1`. Data:
`experiments/repetition_forgetting/results/*.json`.

![phase 1](results/hrep_phase1.png)

## Phase 1 — the emergent signatures (all single-seed)

### P1a — over-repetition of limited data lengthens/creates the step ✓

`train_frac` sweep (dataset size = the "how much is each pattern leaned on"
axis), wd=1.0, 20k steps:

| train_frac | grok_delay | final val_acc | |w| peak |
|-----------:|-----------:|--------------:|--------:|
| 0.2 | — (never) | 0.000 | 144 |
| 0.3 | — (never) | 0.291 | 162 |
| 0.4 | 7000 | 1.000 | 161 |
| 0.5 | 2800 | 1.000 | 150 |
| 0.6 | 1600 | 1.000 | 142 |
| 0.8 | 400 | 1.000 | 133 |

The grok delay **falls monotonically as data grows** (7000 → 2800 → 1600 → 400),
and below a critical fraction (~0.3–0.4) it never groks in budget. Less data ⇒
higher memorization (|w| peak) ⇒ longer or absent step. This matches the H-rep
reading "over-repetition of a limited set builds a memorized table whose
forgetting is grokking," **with the standing caveat** (`docs/03`) that on a
finite task "over-repetition of a limited set" and "small dataset" are the same
thing — this is also the known Power-et-al. dependence of grokking on data
fraction, reproduced here.

### P1b — the forgetting signature aligns with generalization ✓ (with a caveat)

`weight_l2` peaks (memorization) then declines (forgetting). Where does
generalization sit relative to the decline?

| frac | |w| peak @ step | |w| at val_generalize | val_gen step |
|-----:|----------------:|----------------------:|-------------:|
| 0.4 | 160.7 @ 2400 | 133.3 | 7600 |
| 0.5 | 149.6 @ 2200 | 138.7 | 3400 |

In the memorization-heavy regime, `|w|` peaks early and generalization arrives
**during the subsequent decline** (frac=0.4: |w| falls 160.7→133.3 before val
generalizes at 7600). The `--weight_decay 0` ablation (in `minimal_grok.py`) is
the counterfactual: no decay → `|w|` keeps rising, no decline, no grok. So the
"forgetting" half is observable and its presence/absence tracks grokking.

**Honest caveat (visible in the figure):** at high data (frac=0.8) generalization
happens *before* the |w| peak — with abundant data the model generalizes fast
and there is little memorization to forget. So "forgetting enables
generalization" is a statement about the **over-memorized (low/moderate data)
regime**, not a universal ordering. Reported, not smoothed over.

### P1c — emphasis needs consistency ✓

`label_noise` (randomize a fixed fraction of train labels: patterns repeated
every step but inconsistent with the rule), frac=0.5:

| label_noise | final val_acc |
|------------:|--------------:|
| 0.0 | 1.000 |
| 0.1 | 0.790 |
| 0.2 | 0.230 |
| 0.3 | 0.028 |

Generalization collapses monotonically with noise. Inconsistent patterns cannot
be captured by the emphasized rule — they can only be memorized — and they
progressively block grokking. Supports "emphasis on **consistently** repeated
patterns."

## Phase 2 — build the mechanism, test induction

_(pending — `results/p2_*.json`)_ Can an explicit **emphasis+boredom** (loss
reweighting, `boredom_gamma`) or **forgetting** (weight noise, `forget_noise`)
**induce grokking with `weight_decay = 0`** — i.e. replace weight decay as the
engine? Controls: `wd=0` plain (no grok), `wd=1.0` plain (groks, delay 2800).

## Reading so far

Phase 1 supports H-rep's descriptive claims on a single seed: over-repetition of
limited data creates the memorization phase, its forgetting (weight-norm decline)
aligns with the step in the over-memorized regime, and consistency is required
for the rule to be emphasized. Whether an *explicit* boredom/forgetting is
*sufficient* to induce grokking without weight decay is the Phase 2 question.
Single seed throughout; a seed check on the headline contrasts is owed.
