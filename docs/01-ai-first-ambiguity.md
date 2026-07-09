# 01 — The AI-First Ambiguity: the language you *use* vs the language you *make*

*Design memo. Fixes claim C3 and its evidence. Held to `00-charter-and-discipline.md`.*

> **Verification status, stated up front (R3).** Every empirical result cited
> here is **inference-time** behavior of fixed, pre-trained models on an agent
> benchmark. **None of it measures training dynamics.** Grokking is a
> training-time phenomenon. So this memo *motivates* the training-time claims
> (C1, C2) and *fixes* the using-vs-making split (C3); it does not test C1/C2.
> The test is `02-experiment-design.md` and the harness.

## 1. The split

Two languages, two objectives:

- **The using-language.** The interface a competent agent acts *through*. Here
  every unit of ambiguity — polysemy (1) or underdetermination (2) — is a tax:
  work the agent must spend to resolve which meaning / which extension is
  intended before it can act. The design target is to drive that tax to zero.
  This is the target of **Ajisai**: value-integrity-first, vector-oriented,
  exact — a language engineered so that the dominant error classes are removed
  at the language level (exact arithmetic, a JSON module, native strong-Kleene
  logic).

- **The making-language.** The medium a system is *trained through*. The claim
  of this memo is that this medium must, in general, **expose** ambiguity —
  specifically underdetermination (2) — because a large class of target
  capabilities *is the resolution of (2)*. A trainer that removes (2) from the
  medium removes the very thing the model was supposed to learn to resolve.

The two objectives are opposed on axis (2). They agree on axis (1): surface
polysemy is a tax for the user and (per C1) is *not* where grokking lives, so
removing it costs the trainer nothing of interest. The interesting divergence is
underdetermination.

## 2. The evidence, summarized and bounded

Five tiered sweeps were run on Ajisai's agent benchmark, using Haiku / Sonnet /
Opus as **relative capability tiers within one model family** (T1 / T2 / T3).
Summarized (full numbers live with the original sweeps, not reproduced here):

- **Binary pass/fail floored every time.** All five tasks — including ones made
  deliberately hard or trap-laden — were passed by every tier. The binary metric
  carried no tier signal. *(This is itself a mirage-guard lesson: the
  discontinuous metric said "no difference"; the signal was elsewhere.)*
- **The real signal was continuous:** tool-uses, tokens, wall-time, number of
  revisions. Tiers separated on *effort*, not on *success*.
- **Lowering ambiguity via context (a `SKILL.md`) collapsed the weaker tier's
  effort** and compressed the effort gradient between tiers. This is the direct
  observation behind **H1**: *lowering the ambiguity denominator lowers the work
  a tier must spend.* But the gradient was **task-specific** — on another task it
  did not fall monotonically; it scattered.
- **The conserved-quantity search was null.** The solution space collapsed to a
  point, so there was no "medium with dynamics" in which a soliton-like conserved
  continuous quantity could even be defined, let alone tested.
- **Root cause of the floors.** Ajisai's value-integrity-first design
  *systematically removes the error classes that would separate tiers* — exact
  arithmetic, JSON module, native strong-Kleene logic. Tier discrimination (and
  therefore any capability *cliff*) is suppressed *by design*.

### What this does and does not license

- It **supports C3's using-side**: a language built to zero out ambiguity really
  does flatten the effort an agent must spend, and really does erase the
  differences (cliffs) between capability tiers. Ajisai is a good *using*
  language *because* it removes ambiguity.
- By the same mechanism, read in reverse, it **motivates C3's making-side**: the
  design move that makes Ajisai a good using-language — removing the
  discriminating error classes — is exactly the move that would erase the cliff
  a trainer is trying to study. A medium with no ambiguity has no transition to
  observe. **But this is an analogy across the inference/training line (R3), not
  a measurement.** The cliff we saw suppressed was a *tier-discrimination* cliff
  at inference time, not a *grokking* cliff in training.

## 3. Why "expose ambiguity," stated carefully

The weak version — "expose ambiguity to *induce* grokking" — is not the claim,
and would run straight into reservation **R1** (grokking may be mere timing, not
quality). The stronger, cleaner version:

> Expose underdetermination (2) not to induce a training artifact, but because
> the **target capability is the resolution of (2)**. A model that never met
> underdetermination in its medium was never asked to learn the thing.

Under this version, grokking is a *symptom* that the model is doing the intended
work (escaping (2)), interesting mainly as a **diagnostic** — and, per C2, its
dramatic *visibility* is a property of the medium's opacity, not a measure of
how much was learned. This keeps R1 intact: we are not selling the step as
quality.

## 4. The corollary for tooling

If C3 holds, the tempting unification — "one language for humans, agents, and
training" — is a category error on axis (2):

- Ship the **using**-language (Ajisai-like) at the agent interface: zero-tax,
  structure transparent, ambiguity resolved for the agent.
- Do **not** assume that same language is the right *training* medium. A training
  medium that has already resolved (2) for the model has removed the capability's
  substance. This does **not** mean training should be deliberately obfuscated
  (that would be R1's trap); it means the medium must *contain* the
  underdetermination the capability is *about*.

## 5. What would falsify C3

- If, in the training-time experiment (`02`), grokking / delayed generalization
  turned out to be **independent of underdetermination** — present even when the
  data pins the function down, or absent even when it does not — the "(2) is the
  gap grokking lives in" premise fails, and with it the making-side of C3.
- If a single transparent, ambiguity-zero medium trained models that resolved
  novel (2) *at least as well* as an opaque one, the making/using split would
  collapse into "just use the transparent one," and C3 would be uninteresting
  even if literally true.

Both outcomes are publishable and neither is currently measured.
