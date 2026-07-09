# 00 — Charter and Discipline

This document fixes what Gloxinia is trying to find out and the rules under which
it is allowed to claim anything. It is the contract; the other docs and the
experiments are held to it.

## 1. The thesis, split so it can fail

### 1.1 Two ambiguities, never conflated

- **(1) Surface polysemy.** One surface token, context-dependent meaning. A
  representational fact about the *notation*.
- **(2) Underdetermination.** A finite training set is consistent with more than
  one function extension — in particular with a memorizing table *and* with a
  structural rule. A fact about the *data relative to the hypothesis class*.

Every claim below names which one it is about. A sentence that says "ambiguity"
without a number is a bug in the writing.

### 1.2 The claims

- **C1.** Grokking (delayed generalization: train metric saturates, val metric
  jumps much later) lives in the gap of **(2)**. It is what escaping
  underdetermination *looks like*.
- **C2.** The *visibility* of grokking — that it appears as a sharp, late step
  rather than smooth improvement — is a product of learning through an **opaque
  medium**. When the target structure is **transparent in the representation**
  (nearby inputs map to nearby, structure-respecting encodings), there is no
  memorization plateau to escape, generalization is roughly immediate, and the
  step is not visible.
- **C3 (corollary, AI-first).** The optimal language to *use* an AI with
  differs from the optimal language to *make* one. Using wants (1) and (2) both
  driven toward zero. Making may require exposure to (2) — because many target
  capabilities *are* the resolution of (2). Detailed in `01-ai-first-ambiguity.md`.

### 1.3 The reservations we refuse to drop

- **R1.** C2 is about *timing and visibility*, not final *quality*. A grokked
  solution and a smoothly-learned one may be the identical circuit. We do not
  claim ambiguity yields a *better* solution.
- **R2.** "Opaque" and "transparent" are properties we must *operationalize per
  experiment* (e.g. onehot vs low-frequency Fourier for modular addition). A
  claim about opacity is only as good as the operationalization, which must be
  stated.
- **R3.** All prior Ajisai sweep evidence is **inference-time** behavior of
  fixed pre-trained models. It motivates but does not test C1/C2, which are
  **training-time**. This line is never blurred.

## 2. The discipline

These are inherited and enforced.

### 2.1 Mirage guard (Schaeffer et al. 2023)

A binary or otherwise discontinuous metric can manufacture the *appearance* of a
sharp transition where the underlying learning is smooth. Therefore:

- Every evaluation logs **both** a binary-ish metric (accuracy) **and** a
  continuous co-metric from the same data (cross-entropy loss; also weight
  norm).
- We only write "transition" when the continuous quantity moves at the **same
  place** the accuracy does. If accuracy jumps but loss was already gliding down,
  we say so and we do not call it a transition.

### 2.2 Proxy-naming discipline

Precedent: a score once named `energyProxyScore` was never allowed to
call itself `energyUsed`, because it was a proxy, not the thing. So:

- Metrics are named for what they *are* (`val_acc`, `val_loss`, `weight_l2`),
  never for the conclusion (`generalization`, `capability`).
- Every metrics record carries a `metrics_version`. Records with different
  versions are **not comparable** and must not be plotted on the same axis
  without a restated definition.

### 2.3 Nulls, losses, blanks

- Negative and null results are first-class and get written down.
- Nothing is fabricated. A cell we did not measure is **blank**, not guessed.

### 2.4 Unverified is labeled

- Any inference-time-to-training-time analogy is flagged as such at the point of
  use.
- Any result from a single seed / single config is labeled as such; we do not
  imply a sweep we did not run.

### 2.5 Determinism

- One seed fixes data split and initialization; it is recorded in every output
  JSON. Reproduction is a single command line, kept in the experiment README.

## 3. What would move each claim

- **C1 supported** if delayed generalization tracks the *degree* of
  underdetermination (e.g. train-fraction / injected-nuisance) while holding the
  encoding's surface form fixed.
- **C2 supported** if the grokking *step* (large positive `grok_delay` plus a
  simultaneous val-loss drop) is present under the opaque encoding and *absent*
  under the transparent one, with the model, seed, optimizer and data split held
  fixed.
- **Either falsified** if the step survives transparency, or if the continuous
  co-metric shows the "transition" was a mirage in the first place. Both
  outcomes get published.
