# 04 — Low norm ⟺ generalization: why norm reduction selects structure

*Pre-registration for the third experiment. Held to `00-charter-and-discipline.md`.
Numbers live in `experiments/low_norm/` once measured; this doc is the claim and
the predictions, written before the runs.*

## 1. The claim this rests on

The whole H-rep arc concluded (PR #5) that grokking's engine is **norm
reduction** — driving the weight norm down by any means selects the generalizing
solution. That conclusion silently assumes:

> **L: the generalizing (structural) solution has lower weight norm than the
> memorizing solution**, so that a norm-reducing pressure moves the network from
> the memorizer to the generalizer.

Phase 3 established *causation* (reduce norm → generalize). This experiment
measures the *substrate* of that causation: is L actually true, how tightly does
generalization track norm, and **why** is structure low-norm?

This is training-time and mechanistic; it is the foundation under C1/C2/H-rep,
not a new phenomenon.

> **Outcome (single seed, see `experiments/low_norm/RESULTS_low_norm.md`).**
> **M1 supports L:** at p=97 the generalizing solutions sit at low norm
> (‖w‖≈118–153, val≈1) far below the memorizer (‖w‖=324, val 0.04), with an
> underfitting floor at too-low norm (wd=5, ‖w‖=53, val 0.51). **M2's scaling
> prediction was NOT supported** (the naive gap ratio shrinks with p and rests on
> the runaway wd=0 norm). Two cleaner facts replaced it: (a) the norm gap is a
> *post-fit drift* — at fit-onset memorizer≈generalizer norm; unregularized
> training then inflates norm — and (b) the structural solution is *per-weight
> compact* (‖w‖/√#params ≈ 0.43, ~p-independent). Refined reading: norm reduction
> keeps the net in the low-norm basin where fitting *requires* the structural
> solution; memorization isn't intrinsically norm-hungry at the fitting point.

## 2. Two measurements

### M1 — the norm ⟺ generalization law (fixed p=97)

Vary `weight_decay` so the network settles at different weight norms; for each,
record final `|w|`, `train_acc`, `val_acc`. Plot `val_acc` against achieved
`|w|`.

*Predictions:*
- Among solutions that **fit** train (`train_acc≈1`), **lower `|w|` ⟺ higher
  `val_acc`** — the generalizing solutions are the low-norm ones.
- At very high weight decay the norm is pushed **too** low to represent the
  function: `train_acc` falls and `val_acc` collapses (underfitting). So there is
  an **optimal norm band**: low enough to be structural, high enough to fit. The
  story is not "smaller is always better," it is "structure sits at a lower norm
  than memorization, above the underfitting floor."
- The `wd=0` point anchors the high-norm/memorizing corner (high `|w|`,
  `train=1`, `val≈0`).

### M2 — why structure is low-norm: scaling with problem size

A memorizer must store all `p²` input→output pairs; a structural solution
computes `(a+b) mod p` from a few Fourier components. So the memorizer's norm
should grow with `p` faster than the structural solution's.

For `p ∈ {17, 31, 47, 59, 97}`, measure:
- **generalizer norm** = final `|w|` of a grokked run (`wd=1.0`, `train=val=1`);
- **memorizer norm** = `|w|` of a `wd=0` run that has fit train but not
  generalized (`train=1`, `val≈0`), read at a fixed budget.

*Prediction:* the **gap / ratio memorizer:generalizer widens with `p`** —
structure is increasingly norm-cheap relative to memorization as the problem
grows. That is the mechanistic reason a norm-reducing pressure selects structure,
and why it should bite harder on larger problems.

*Honest caveat, up front:* the `wd=0` "memorizer norm" is where SGD happens to
sit at a fixed budget, not a well-defined *minimum* norm to memorize; its
magnitude keeps drifting with training time. We read it at a fixed step and say
so; the load-bearing quantity is the **generalizer** norm and how *it* scales,
plus the *direction* of the gap, not the exact memorizer number.

## 3. What each outcome means

- **L supported** if M1 shows val_acc rising as norm falls among fitters (with an
  underfitting floor) and M2 shows the generalizer norm growing slowly while the
  memorizer/gap grows with `p`.
- **L weakened** if generalization does not track norm (e.g. high-norm solutions
  generalize just as well), or if the generalizer norm grows as fast as
  memorization with `p` (then "structure is low-norm" is not the reason norm
  reduction works, and the Phase-3 result would need a different explanation).
- **Reservation:** these are correlational characterizations of the *solutions*
  the optimizer finds, not a proof about minimum-norm interpolants. Phase 3
  already supplied the causal arrow (norm↓ ⇒ generalize); M1/M2 characterize the
  landscape that makes it work. Single seed unless RESULTS says otherwise.
