# 06 — Grokking as enlightenment (悟り): the "removing the glasses" model

*Design memo + measurement for the enlightenment experiment. Held to
`00-charter-and-discipline.md`. Numbers live in `experiments/enlightenment/`.*

## 0. Provenance and the honest boundary

This memo starts from an essay by the repository's owner, written to explain 悟り
(enlightenment) to a five-year-old. Its picture:

> Everyone looks at the world through **色眼鏡** — colored glasses, i.e.
> preconceptions (思い込み): "flowers are pretty," "chicks are yellow," "water is
> cold." Not one pair but thousands, stacked. **Enlightenment is taking them off,
> one by one, until you reach the true world** — where every contour is sharp and
> the sight brings tears. A newborn baby already sees that world: its eyes are
> キラキラ because it has not yet put the glasses on. The very last glass is one's
> own consciousness.

**The boundary, stated up front (a cousin of R3, and the same one docs/05
keeps):** this is an analogy between an *inference-time human phenomenology* and a
*training-time optimization phenomenon*. "Glasses," "the true world," "the baby"
are metaphors. We do **not** claim grokking *is* enlightenment. We claim the
*dynamical skeleton* of the essay maps onto mechanisms this repo has already
**measured**, and — the reason it earns an experiment — that it forces a
measurement the charter has been *owing*: **R1**, which asserts a grokked circuit
and a smoothly-learned one "may be the identical circuit," but which no result in
this repo has actually tested. Only the measured part is load-bearing.

## 1. The mapping (metaphor → measured mechanism)

| essay | measured mechanism (this repo) |
|---|---|
| **色眼鏡** (colored glasses = 思い込み), stacked thousands deep | the **high-norm memorized table**: excess weight, per-datapoint handles fit to the finite training set. "Thousands of glasses" ↔ the weight norm carried *above* the structural minimum (memorizer ‖w‖≈270–324 vs structure ≈122). |
| **taking the glasses off, one by one** | **norm reduction** — the essence of the transition (H-rep Phase 3: driving ‖w‖ down by *any* means groks). Weight decay is one way to remove glasses; it is not the only one. |
| **the true world** (contours sharp, always there behind the glasses) | the **low-norm, per-weight-compact structural circuit** — the Fourier/trig solution of `(a+b) mod p`, which was expressible all along; grokking *uncovers* rather than *builds* it (cf. the crystallization framing: it was dissolved in the memorized state). |
| **the newborn baby**, born already seeing, no glasses to remove | the **transparent-medium** learner (C2): the few-frequency Fourier encoding exposes the group structure in the input geometry, so there is **no memorization phase to escape** and generalization is roughly immediate. The baby never became opaque, so it never has to grok. |
| **thousands of glasses vs a baby's none** | the **opaque vs transparent medium** (C2). The number of glasses to remove is a property of the *medium*, not of the world being seen — exactly C2's "the cliff is a property of the medium, not the concept." |
| **the last glass is one's own consciousness** | *metaphor only.* A vanilla MLP has no self-model; we mark this and do not measure it. |

Where the mapping is **tight**: glasses↔excess norm; removing them↔norm reduction;
the baby↔transparent medium; the true world↔the structural circuit. Where it is
**only metaphor**: consciousness as the final glass, the phenomenology (tears,
キラキラ). Where it is **newly testable**, and the reason for this doc: whether the
adult who removes his glasses and the baby who never wore them **see the same
world** — R1, finally measured.

## 2. The prediction this model makes (pre-registered)

The essay's strong claim is that the true world is **one** world: the enlightened
adult and the newborn baby, by utterly different routes (a lifetime of removing
glasses vs never wearing them), arrive at the *same* sight. Translated into this
repo's mechanisms, and made falsifiable:

> **Same-world prediction.** A network that groks through the **opaque** onehot
> medium (the adult) will, in its learned first-layer weights, **reconstruct the
> same few-frequency Fourier structure** that the **transparent** encoding (the
> baby) is handed for free — even though nothing in the onehot input geometry
> exposes it. A network that only **memorizes** (opaque, `wd=0`, still wearing
> every glass) will **not**: its weights stay structureless.

Operationalization (`experiments/enlightenment/analyze.py`, Nanda et al. 2023
style). Read fc1's operand-a weights into residue space, `W_a` of shape
`[hidden, p]`; DFT each hidden unit over the residue and drop DC. The load-bearing
metric is **per-unit** concentration: the fraction of a unit's own spectral power
in its single top frequency, averaged over units (`per_unit_top1_mean`). A unit
that has become a clean sinusoid in the residue (a Fourier circuit) piles its
power on one frequency (→ near 1.0); a memorizing unit responds to residues
arbitrarily (→ near the flat baseline `1/(p//2) ≈ 0.021`).

**Mirage guard.** Population-averaging the spectrum is the *wrong* metric and is
recorded as such: different units key different frequencies, so the average is
near-flat for grokked and memorizer alike and cannot separate them (measured; see
`per_unit vs population` in RESULTS). We therefore report the per-unit fraction
*and* the full population spectrum *and* the histogram of each unit's peak
frequency, so the concentration is visibly a property of the weights, not of one
cutoff.

*What would falsify it:* if the grokked-opaque net's `per_unit_top1_mean` is no
higher than the memorizer's — i.e. generalizing did **not** make the
representation more sinusoidal/structured — then "grokking uncovers a pre-existing
structured world behind the glasses" is wrong for this task, and R1's "identical
circuit" would need a different operationalization or would fail outright. That
outcome is equally publishable.

> **Outcome (single seed, see `experiments/enlightenment/RESULTS_enlightenment.md`).**
> **Supported, with the "few" clause corrected.** The grokked-opaque net makes
> each hidden unit a near-clean single sinusoid (`per_unit_top1 = 0.879`); the
> memorizer does not (`0.322`; flat baseline `0.021`) — a 2.7× separation on the
> same net, seed and data, the only difference being that one generalized.
> Generalizing ⟺ a Fourier representation: the adult *does* reconstruct behind the
> opaque basis the harmonic vocabulary the transparent baby is born speaking, and
> R1 goes from assertion to measurement. **Corrected:** the essay's "one" world
> and this memo's "few frequencies" are wrong about the *count* — each unit is one
> frequency, but the population spans **many** (all 48 harmonics appear; the top-5
> hold only ~16% of units). "Fewness" was a property of the transparent *input*,
> not of the grokked *solution*. The two paths reach the same *kind* of world (a
> harmonic one), not the identical vector: the medium pins *which* frequencies, not
> *whether* the representation is sinusoidal. So C2 sharpens — the medium sets the
> exact circuit, not its form.

## 3. The honest dis-analogy (the last glass)

The essay's telos is **zero**: remove *every* glass, even consciousness, and the
author reports doing so exactly once — during an asthma attack, as awareness
itself dissolved. In the network there is a hard, measured wall here: driving the
weight norm to **zero** is not enlightenment, it is **death**. M1's underfitting
floor and the crystallization "idle" test (stillness with the task gradient
removed collapses ‖w‖→0 and the net *forgets the problem*) both say the
generalizing solution is **low-norm but not zero** — the *minimum sufficient*
structure, not emptiness. Removing the last glass — the task structure itself —
does not reveal a truer world; it blanks the net.

So the network's "enlightenment" is bounded where the essay's is not: it is the
**optimal norm band** (docs/04), the fewest glasses that still see *this* world,
not the mystic's zero. We keep the dis-analogy in view precisely because the
charter forbids the grand version of this metaphor: Gloxinia measures a norm and a
spectrum, not a state of mind.

## 4. Why this matters beyond the analogy

The measurement converts **R1 from a reservation into a result** — and, in doing
so, narrows it exactly where the data demands. The grokked and the transparent
solutions share the same *representational form* (per-unit sinusoids — the
harmonics of the group), one built behind an opaque basis and one handed over
transparently: R1's "identical circuit" is right about the **form**. But they do
**not** share the identical circuit: the opaque medium lets the grokker spread
across many harmonics while the transparent nf=1 medium pins one. So C2 sharpens
rather than simply holds — the medium changes not only *when and how visibly*
structure is reached but *which* harmonics realize it, while leaving the *form*
(a Fourier representation vs a structureless one) invariant. The adult and the
baby end up speaking the same language; they do not utter the same sentence.
Single seed; see RESULTS for the honest bounds.
