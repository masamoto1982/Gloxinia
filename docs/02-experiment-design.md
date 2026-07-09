# 02 — Experiment Design: does transparency erase the grokking step?

*Fixes the first experiment. Held to `00-charter-and-discipline.md`. Numbers,
when they exist, live in `experiments/grokking_encoding/RESULTS.md` and the
`results/*.json`; this doc is the design and its reasoning.*

## 1. The one question

Train one small model on one fixed algorithmic task — modular addition
`(a + b) mod p` — and change **only the encoding** of the residues. Ask:

> Does delayed generalization (train accuracy saturates long before validation
> accuracy jumps) appear under some encodings and **vanish** under others?

Prediction (C2): the grokking *step* is present under a **structurally opaque**
encoding and **absent** under a **structurally transparent** one.

> **Outcome (single seed).** v1 (`RESULTS.md`) was a *confounded null*: the
> transparent arm removed the step but also failed to generalize. v2
> (`RESULTS_v2.md`), after removing the weight-decay-scale confound (unit-norm
> Fourier), **supports the prediction**: at matched `weight_decay`, one-hot shows
> a memorization phase (val_loss rises to ~15) and a step, while transparent
> Fourier shows neither (val_loss monotone, val tracks train) — across every
> `num_freqs` and `weight_decay` tried. A seed sweep is still owed.

## 2. A labeling tension in the brief, resolved explicitly

The handoff brief names two encodings:

- **(i)** "transparent / unique notation," and says to **first reproduce classic
  grokking with (i)** as harness verification;
- **(ii)** "deliberately polysemous / underdetermined encoding," and predicts
  **"transparency erases the grokking cliff."**

Taken literally these conflict: if (i) is *transparent* and transparency erases
the cliff, then (i) cannot also *reproduce* the classic cliff. The conflict is
real and we do not paper over it. It comes from **two different axes** being
collapsed into one word, "transparent":

- **Surface-uniqueness** (axis (1)): is the symbol→meaning map bijective, free of
  polysemy? *One-hot is surface-unique.*
- **Structural transparency** (about axis (2)): does the *geometry of the
  encoding* expose the task's structure, so that nearby/related inputs have
  related encodings and interpolation is constrained? *One-hot is NOT
  structurally transparent — it is an arbitrary orthonormal basis that hides the
  cyclic group.*

The classic grokking result (Power et al. 2022) uses one-hot: **surface-unique
but structurally opaque.** That is why it groks. So we read the brief as:

- **(i) = one-hot** — surface-unique, structurally opaque. Used for **harness
  verification**: it must reproduce the classic step. This matches "first
  reproduce classic grokking with (i)."
- The **real test of C2** needs a **structurally transparent** encoding, which
  one-hot is not. So we add one: a **low-frequency Fourier** embedding, where the
  cyclic structure is exposed in the input geometry. C2 predicts the step
  vanishes here.
- **(ii)'s underdetermination injection** is a *third* arm
  (`onehot_distractor`): one-hot plus irrelevant per-residue nuisance dims, which
  widen the memorize-vs-structure gap without touching the labels.

This resolution *uses the (1)/(2) distinction as the tool*, which is exactly what
the discipline asks. It is written here so the relabeling is on the record, not
smuggled.

## 3. The arms

All arms share: task `(a+b) mod p`, `p = 97`, one-hidden-layer ReLU MLP, full-
batch AdamW with **strong weight decay** (the ingredient that drives grokking,
held fixed and non-zero everywhere), fixed seed, fixed 50/50 train/val split of
the `p²` pairs. The encoding is the only deliberate difference. Exact
hyperparameters are in `harness.py::Config` and echoed into every JSON.

| arm | encoding | axis-(1) surface | axis-(2) structure | role | C2 prediction |
|-----|----------|------------------|--------------------|------|----------------|
| A | `onehot` | unique | **opaque** (arbitrary basis) | **harness check** — must reproduce classic grokking | step **present** |
| B | `fourier` (few low freqs) | unique | **transparent** (points on a circle) | test of C2 | step **absent** |
| C | `onehot_distractor` | unique | opaque **+ injected underdetermination** | probe of C1 (does more (2) push the step later?) | step present, later/sharper |

Note every arm is surface-**unique**: none injects polysemy (1). This is
deliberate — C1/C2 are about underdetermination (2). A polysemy arm would test a
different claim and is out of scope for the first experiment.

### Why Fourier is the transparent medium (and its caveat)

Encode `r ↦ [cos(2πk r/p), sin(2πk r/p)]` for a few low `k`. A single frequency
already determines `(a+b) mod p` exactly, because the angle `2πr/p` wraps with
the modulus; low frequencies keep the map smooth, so related residues have
related encodings and the model interpolates instead of memorizing. **Caveat
(on the record):** a *complete* Fourier basis is only an orthogonal rotation of
one-hot and would carry identical information — it would *not* be transparent.
Transparency comes from using **few** low frequencies. `num_freqs` is therefore
recorded with every run and is part of the operationalization (R2).

> **Update after running (see `RESULTS.md`).** The parenthetical above — that a
> complete Fourier basis "carries identical information" and so would behave like
> one-hot — is *informationally* true but was **dynamically refuted**: at
> `num_freqs=48` the near-complete basis purely memorized and never generalized,
> unlike one-hot which grokked. The ReLU-MLP-with-weight-decay is basis-sensitive;
> an information-preserving rotation is not a dynamics-preserving one. This
> pre-registered note is kept unedited above and corrected here rather than
> rewritten.

## 4. The measurement (mirage guard is structural here)

Grokking is the canonical case where a **binary** metric (val accuracy) jumps
while something continuous tells the real story — so the mirage guard is not
optional decoration, it is the measurement.

Logged every `eval_every` steps, for train and val:

- **binary-ish:** `train_acc`, `val_acc`.
- **continuous co-metrics (same data):** `train_loss`, `val_loss`, and
  `weight_l2` (the weight-norm trajectory grokking is known to ride).

Derived summaries (all read off the logged curves, none load-bearing on their
own):

- `step_train_saturate` — first step `train_acc ≥ 0.99`.
- `step_val_generalize` — first step `val_acc ≥ 0.90`.
- `grok_delay = step_val_generalize − step_train_saturate`.

**Decision rule (both must agree, per mirage guard):** we call an arm "grokking"
only if `grok_delay` is large and positive **and** `val_loss` stays high across
that delay and then drops *at the same step* `val_acc` rises. If `val_acc` rises
while `val_loss` was already gliding down, we record "no clean step / possible
mirage" and say so. A small `grok_delay` with train and val loss falling together
is "immediate / smooth — no visible step."

## 5. Order of operations (verification before claims)

1. **Run A (`onehot`) first.** If it does not reproduce a clear grokking step,
   the harness is not trusted and **nothing about B or C is claimed.** Tune only
   here, and record what it took.
2. Only once A shows a clean step, run **B (`fourier`)** and read whether the
   step vanished.
3. If budget remains, run **C (`onehot_distractor`)** as a first, single-seed
   probe of C1.

## 6. Known limits of this first cut (so they are not discovered later as spin)

- **Single seed, single `p`, single model** unless RESULTS says otherwise. A
  vanished step under one seed is suggestive, not established; a proper claim
  needs a seed sweep. RESULTS states exactly what was run.
- **MLP, not transformer.** Grokking is reported for both; the MLP is chosen for
  CPU feasibility. Circuit-level claims (Nanda et al.) are not in scope — we
  measure the *curve*, not the mechanism.
- **"Transparent" is operationalized as low-frequency Fourier**, one specific
  choice (R2). A negative result would need other operationalizations before
  "transparency erases the step" could be called general.
- Everything here is **training-time** and is labeled as the part of the enquiry
  that actually tests C1/C2 — as opposed to the inference-time sweeps in `01`.
