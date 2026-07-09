# The simplest reliable grokking recipe

Distilled from this repo's experiments (`RESULTS.md`, `RESULTS_v2.md`).
Reference implementation: [`minimal_grok.py`](minimal_grok.py) — ~90 lines,
torch only, no imports from the rest of the harness.

```bash
python minimal_grok.py                    # groks: val 0 -> ~0.97, step near 7k
python minimal_grok.py --weight_decay 0   # ablation: memorizes forever, no grok
```

## The recipe in one line

> Train a small MLP with **full-batch AdamW and weight decay** on **half the
> pairs** of an algebraic task (`(a+b) mod p`) presented through an **opaque
> (one-hot) encoding**. Watch train accuracy hit 1.0 within a few hundred steps
> while validation accuracy stays at zero — then jump, thousands of steps later.

## The four load-bearing ingredients

Everything else (model width, exact `lr`, `p`, number of layers) is a
convenience knob. These four are the mechanism:

1. **An opaque encoding of an algebraic task.**
   One-hot residues hide the cyclic group structure, so the network must
   *discover* it. That discovery is preceded by a memorization phase — which is
   the thing that later gets escaped. *Evidence:* replace the opaque encoding
   with a structurally transparent one (low-frequency Fourier) and the
   memorization phase, and the visible step, disappear entirely (`RESULTS_v2.md`).
   This is why the recipe specifies one-hot, not "any encoding."

2. **Weight decay — the engine.**
   Grokking is driven by weight decay eventually making the structural solution
   cheaper (smaller norm) than the memorized lookup table. It is the single most
   important knob. *Evidence:* `--weight_decay 0` → train accuracy 1.0, validation
   accuracy 0.004, validation loss *diverges* to 33.7. No decay, no grok, ever.

3. **Limited data (a fraction of all pairs).**
   Training on ~half of the `p²` pairs creates the underdetermination gap the
   network memorizes into before it generalizes. With all pairs there is nothing
   to generalize *to*; with too few, the structural solution is not identifiable.
   Half is a robust default.

4. **Full-batch training with Adam.**
   The full-batch gradient keeps the dynamics clean and deterministic so the
   delayed step is crisp rather than smeared by minibatch noise. (Grokking still
   occurs with minibatches; full-batch just makes it legible and reproducible.)

## What the run looks like (defaults: p=31, hidden=256, wd=1.0, lr=3e-3, seed=0)

```
 step  tr_acc  va_acc   tr_loss   va_loss
    0   0.056   0.023    3.4380    3.4462
  600   1.000   0.000    0.0024    7.5356   <- memorized; val_loss HUMPS up
 2000   1.000   0.316    0.0000    2.5055   <- structure forming
 7000   1.000   0.906    0.0000    0.3837   <- grokked (val >= 0.90)
20000   1.000   0.971    0.0000    0.1878
```

The **val_loss hump** (rising while train_loss is already ~0) is the signature to
watch, not the accuracy alone — that is the mirage guard in practice. In the
`--weight_decay 0` ablation the same hump appears and then simply *never comes
back down*: memorization without the engine that would escape it.

## Reproducing the classic `p=97` version

`minimal_grok.py --p 97 --lr 1e-3 --steps 30000` reproduces the standard
Power-et-al.-scale result (this repo's Arm A: train saturates ~600, grokks
~3400). The small-`p` defaults above are just faster to watch on CPU.

## What this recipe is NOT

- Not a claim that grokking requires this exact setup — it is the *simplest* one
  we could reduce to while keeping the step reliable and reproducible.
- Not a statement about solution *quality* (reservation R1): the grokked solution
  and a smoothly-learned one may be the same circuit. This recipe reproduces the
  *phenomenon and its visible step*, which — per `RESULTS_v2.md` — is a property
  of the opaque medium, deliberately kept in the recipe because that is what
  makes grokking visible in the first place.
