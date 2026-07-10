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

## Phase 2 — build the mechanism, test induction → **strong claim FAILS** ✗

![phase 2](results/hrep_phase2.png)

Can an explicit **emphasis+boredom** (loss reweighting) or **forgetting** (weight
noise) **induce grokking at `weight_decay = 0`** — replace weight decay as the
engine? frac=0.5, 20k steps, seed 0. Every config memorizes (train_acc = 1.0):

| config | val_acc | |w| final | grokked? |
|--------|--------:|----------:|:--------:|
| wd=1.0 plain (ref) | 1.000 | 123 (peak 150 → declines) | **yes**, delay 2800 |
| wd=0 plain (ref) | 0.024 | 271 (rises) | no |
| wd=0 + boredom γ=2 | 0.025 | 270 | no |
| wd=0 + boredom γ=4 | 0.021 | 271 | no |
| wd=0 + forget-noise 0.003 | 0.030 | 275 | no |
| wd=0 + forget-noise 0.01 | 0.031 | 494 (rises faster) | no |
| wd=0 + forget-noise 0.03 | 0.002 | 1174 (train breaks: tr 0.83) | no |
| wd=1.0 + boredom γ=2 | 1.000 | 123 | yes, delay **2800** (unchanged) |

**Neither explicit mechanism induces grokking at wd=0, and boredom does not even
change the delay at wd=1.0** (2800, identical to plain). The figure shows why:

- **Grokking rides a weight-norm DECLINE.** The only run that generalizes
  (wd=1.0) is the only one whose `|w|` peaks (~150) and then *falls* (to 123).
  Every wd=0 run's `|w|` only *rises*.
- **Boredom (loss reweighting) is inert.** In full-batch training a mastered
  example already contributes ~zero gradient, so down-weighting it changes
  almost nothing — the boredom curve sits on top of the wd=0-plain curve.
- **Weight noise moves the norm the WRONG way.** Isotropic additive noise is a
  random walk that *inflates* `|w|` (494 at 0.01, 1174 at 0.03), the opposite of
  the reduction grokking needs; large noise just breaks training.

### Refined conclusion on H-rep

The metaphor is **descriptively apt but mechanistically specific**:

- Phase 1 confirms the *signatures* (over-repetition builds memorization; a
  weight-norm decline = "forgetting" accompanies the step in the over-memorized
  regime; consistency is required).
- Phase 2 falsifies the *strong constructive form*: a **generic** boredom or a
  **generic** forgetting does not induce grokking. The "forgetting" that induces
  it is specifically **norm-reducing decay toward the low-norm structural
  solution** — weight decay's particular mechanism, which exploits that the
  memorized table has higher weight norm than the structural circuit. Generic
  boredom (reweighting) and generic forgetting (isotropic noise, which *raises*
  norm) do not have this property.

So H-rep is best stated as: grokking's engine is **norm-selective forgetting**
(weight decay), not boredom/forgetting in general. This is a null for the strong
mechanism claim and is reported as such (single seed; the two mechanisms tested
are not exhaustive — a norm-reducing stochastic forgetting was not built).

## Bottom line

- **Descriptively (Phase 1): H-rep holds.** Over-repetition of limited data
  creates the memorization phase; a weight-norm decline ("forgetting") aligns
  with the step in the over-memorized regime; consistency is required for the
  rule to be emphasized.
- **Mechanistically (Phase 2): the strong form fails.** Explicit boredom (loss
  reweighting) and explicit forgetting (weight noise) do **not** induce grokking
  at wd=0, and boredom does not accelerate it at wd=1.0. The engine is
  **norm-selective** forgetting (weight decay), which shrinks the high-norm
  memorized table toward the low-norm structural solution — a property generic
  boredom/noise lack.
- **Net:** "emphasis + boredom/forgetting" is a good *description* of what
  grokking looks like, but the *cause* is specifically weight-decay-style
  norm reduction, not boredom or forgetting in general.

Single seed throughout; the two Phase-2 mechanisms are not exhaustive (a
norm-reducing stochastic forgetting was not built). A seed check on the headline
Phase-1 contrasts is owed before these are more than suggestive.
