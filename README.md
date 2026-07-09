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

Early. First experiment run (single seed, `p=97`, MLP). Measured so far — full
detail and caveats in
[`experiments/grokking_encoding/RESULTS.md`](experiments/grokking_encoding/RESULTS.md):

- **Harness verified:** one-hot reproduces classic grokking, confirmed by both
  the binary metric *and* the continuous co-metric (not a mirage).
- **Supports C1:** injecting underdetermination (irrelevant nuisance dims) more
  than doubled the grokking delay (2800 → 7600 steps) while keeping full
  generalization — a *later*, not absent, step.
- **C2 not cleanly tested — a null:** the transparent (low-frequency Fourier)
  encoding removed the step but also failed to generalize fully, so the two
  can't be separated. A pre-registered caveat (that a complete Fourier basis
  ≈ one-hot) was *refuted*: the near-complete basis memorized and never grokked,
  because the MLP is basis-sensitive. Both the null and the refutation are on the
  record.
