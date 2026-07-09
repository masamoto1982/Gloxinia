# RESULTS — grokking × encoding transparency

Measured outcomes for the experiment in
[`docs/02-experiment-design.md`](../../docs/02-experiment-design.md). Numbers are
read from the committed `results/*.json`. **Single seed (0), single `p=97`,
single MLP** unless a row says otherwise — these are first cuts, not sweeps.

`metrics_version = grok-metrics-v1`. All arms share: `(a+b) mod 97`, 50/50
split, one-hidden-layer MLP (hidden 256), full-batch AdamW, `lr=1e-3`,
`weight_decay=1.0`, seed 0. Only the encoding (and, within the fourier sweep,
`num_freqs`) changes.

## Decision rule (mirage guard)

An arm "groks" only if **both**: `grok_delay` is large and positive **and**
`val_loss` stays high through the delay then drops at the same step `val_acc`
rises. Note `grok_delay = None` is **ambiguous** on its own — it can mean "never
generalized" *or* "val tracked train with no late jump." The curve, not the
summary scalar, decides which; each arm below states which.

## Arm A — `onehot` (opaque / canonical): grokking reproduced ✓

Harness verification. **Clean grokking, confirmed by both metrics:**

| quantity | value |
|---|---|
| `step_train_saturate` (train_acc ≥ 0.99) | 600 |
| `step_val_generalize` (val_acc ≥ 0.90) | 3400 |
| `grok_delay` | **2800** |
| final val_acc | 1.000 |
| peak val_loss / step | 8.44 @ 600 |
| peak weight_l2 / step | 149.6 @ 2200 |

Memorization phase (train_acc = 1.0 by step 800, val_acc = 0.000, val_loss
*rising* to 8.4, weights growing), then val_acc 0 → 1.0 over steps ~1200–4000
with **val_loss dropping at the same steps** (8.4 → 0.005) and weight norm
declining from its peak. Binary and continuous metrics move together → the
transition is real, not a mirage. **Harness trusted.**

## Arm B — `fourier` (transparent): the step vanishes, but generalization is incomplete

The headline is split, and both halves are reported:

- **The grokking step vanished.** val_acc rises *with* train_acc from step ~200
  (tr 0.37 / va 0.14) onward; there is no delay and no late jump. val_loss falls
  alongside train_loss. This is consistent with C2's "transparency removes the
  visible step."
- **But generalization did not complete.** At `num_freqs=4`, val_acc plateaus at
  **~0.44** while train_acc → 1.0; val_loss sticks at ~1.47 while train_loss
  keeps falling to ~0.35. So the strong reading "transparency → immediate *full*
  generalization" is **not supported at num_freqs=4.** `grok_delay = None` here
  means "never crossed 0.90," not "late jump."

This is a null for the naive strong prediction and a partial confirmation of the
weaker one (no *step*). The obvious confound: is 0.44 a real property of a
transparent medium, or just too few frequencies to resolve 97 classes? The
`num_freqs` sweep below tests exactly that — and tests the elegant prediction
that a **complete** Fourier basis (num_freqs = (p−1)/2 = 48), being only an
orthogonal rotation of one-hot, should behave like one-hot and bring the step
back.

## Fourier `num_freqs` sweep — transparency as a dial

Prediction under test: few low frequencies = transparent (no step); as
`num_freqs` → complete basis (48), the encoding approaches a rotation of one-hot
and the grokking step should **return**.

| num_freqs | dim/operand | step shape (from curve) | step_val_generalize | grok_delay | final val_acc |
|-----------|-------------|--------------------------|---------------------|------------|----------------|
| 4  | 8   | no step; val tracks train | — (plateau 0.44) | n/a | 0.44 |
| 16 | 32  | _(pending)_ |  |  |  |
| 32 | 64  | _(pending)_ |  |  |  |
| 48 (complete) | 96 | _(pending)_ |  |  |  |
| — (`onehot`, ref) | 97 | grokking step | 3400 | 2800 | 1.00 |

_(Table filled from `results/fourier_nf*.json` once the sweep completes.)_

## Arm C — `onehot_distractor` (opaque + injected underdetermination): probe of C1

_(pending — `results/onehot_distractor.json`.)_ Single-seed probe of whether
adding irrelevant per-residue nuisance dims (widening the memorize-vs-structure
gap) pushes the grokking step later / sharper, as C1 would predict.

## What this does and does not establish (kept honest)

- **Established (this seed/config):** the classic grokking step is present under
  the opaque one-hot encoding and **absent** under the low-frequency transparent
  encoding. That much matches C2's core prediction about *visibility*.
- **Not established:** that transparency yields an *equally good* solution — at
  `num_freqs=4` it plainly did not (0.44 vs 1.00). Whether that is an
  expressivity artifact is what the sweep addresses; see the filled table.
- **Reservation R1 stands:** even where the step vanished, we are describing
  *timing/visibility*, not a quality verdict.
- **All training-time**, single seed. A vanished/returned step under one seed is
  suggestive, not established; a seed sweep is future work.
