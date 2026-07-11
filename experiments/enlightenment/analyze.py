"""
Enlightenment ("悟り") measurement: do the two paths reach the SAME true world?

This is the operationalization of R1 (charter 1.3) -- "a grokked solution and a
smoothly-learned one may be the identical circuit" -- which the repo has asserted
but never measured. The essay in docs/06 gives R1 a face:

  * the ADULT removes his 色眼鏡 (colored glasses = the high-norm memorized table)
    one by one until the true world's contours become sharp -> this is grokking
    through the OPAQUE (onehot) medium under weight decay;
  * the BABY is born already seeing the true world, キラキラ, no glasses to remove
    -> this is the TRANSPARENT (few-frequency Fourier) medium: structure exposed
    in the input geometry, generalization roughly immediate (C2).

"The same true world" is a testable claim about the LEARNED REPRESENTATION, not
the input/output function (both reach ~100% val acc, so as functions they are
trivially equal on the task). We ask: behind the opaque orthonormal basis, does
the grokked network REBUILD the same few-frequency structure the transparent
encoding is handed for free?

Operationalization (Nanda et al. 2023 style, kept honest):
  Take fc1's weights on the operand-a slice: W_a has shape [hidden, p]. Row h is
  how hidden unit h reads residue r in [0, p). DFT each row over r; drop DC
  (freq 0 = the mean, no cyclic structure).

  The load-bearing metric is PER-UNIT concentration: a Fourier/trig circuit has
  each hidden unit responding as (near) a single sinusoid in the residue, so its
  own spectrum piles onto ONE frequency; a memorizer's unit responds to residues
  arbitrarily, so its spectrum is ~flat. Population averaging is the WRONG metric
  (v0 mistake, kept in the record): different units key DIFFERENT frequencies, so
  the population-averaged spectrum washes the per-unit peaks back to near-flat and
  cannot tell a grokked net from a memorizer. We therefore report, per unit, the
  fraction of that unit's power in its own top-1 frequency, then average that over
  units (`per_unit_top1_mean`). Flat gives ~1/(p//2) ~= 0.021; a clean single-
  sinusoid unit gives ~1.0.

MIRAGE GUARD analog: we do not report only the thresholded per-unit fraction. The
full population spectrum (all p//2 frequency powers) AND the distribution of each
unit's peak frequency are written to JSON, so the concentration is visibly a
property of the weights, not an artifact of one cutoff. We also report weight_l2
(the "number of glasses") and val_acc (did it reach the world) for every net,
from the SAME weights.

Determinism: single seed, fixed and recorded. Single seed unless RESULTS says
otherwise.
"""

from __future__ import annotations

import json
import os
import sys

import torch

# reuse the verified model + data utilities; vary nothing about them
_HARNESS = os.path.join(os.path.dirname(__file__), "..", "grokking_encoding")
sys.path.insert(0, os.path.abspath(_HARNESS))
from harness import MLP, make_data, split  # noqa: E402
from encoding import build_encoding  # noqa: E402

METRICS_VERSION = "enlightenment-spectrum-v1"


def train_capture(encoding, weight_decay, steps, p=97, train_frac=0.5,
                  hidden=256, lr=1e-3, seed=0, num_freqs=4,
                  fourier_normalize=False):
    """Train one net and return (model, final metrics). Mirrors harness.train
    exactly (full-batch AdamW, betas (0.9,0.98)); the only reason we re-loop
    here instead of calling harness.train is that we need the final WEIGHTS,
    which the RunResult does not carry."""
    torch.manual_seed(seed)
    enc = build_encoding(encoding, p, num_freqs=num_freqs,
                         fourier_normalize=fourier_normalize, seed=seed)
    ab, y = make_data(p)
    (ab_tr, y_tr), (ab_va, y_va) = split(ab, y, train_frac, seed)
    model = MLP(enc, hidden, p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay, betas=(0.9, 0.98))
    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(ab_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(ab_tr).argmax(1) == y_tr).float().mean().item()
        va_acc = (model(ab_va).argmax(1) == y_va).float().mean().item()
    wl2 = sum((pn.detach() ** 2).sum().item()
              for n, pn in model.named_parameters() if "E" not in n) ** 0.5
    return model, {"train_acc": tr_acc, "val_acc": va_acc, "weight_l2": wl2}


def effective_residue_weights(model, p):
    """The learned per-residue reading of operand a, in residue space.

    fc1 maps concat(E[a], E[b]) -> hidden. The operand-a block of fc1.weight is
    [hidden, d] and acts on E[a] (d = encoding dim). Composing with the fixed
    encoding table E ([p, d]) gives W_a = (fc1_a @ E^T) of shape [hidden, p]:
    how each hidden unit responds to each *residue*, in a basis-independent
    residue space. This is the fair place to compare onehot (d=p) and Fourier
    (d=2*num_freqs) nets -- both become [hidden, p]."""
    E = model.E                       # [p, d]
    d = E.shape[1]
    W1 = model.fc1.weight             # [hidden, 2d]
    W_a = W1[:, :d]                   # operand-a block, [hidden, d]
    return W_a @ E.t()               # [hidden, p]


def summarize(model, p):
    """Per-unit spectral concentration (load-bearing) plus the population
    spectrum and peak-frequency histogram (record / mirage guard)."""
    with torch.no_grad():
        W_a = effective_residue_weights(model, p)          # [hidden, p]
        spec = torch.fft.rfft(W_a, dim=1)                  # [hidden, p//2+1]
        power = (spec.abs() ** 2)[:, 1:]                    # drop DC -> [hidden, p//2]
    nfreq = power.shape[1]
    freqs = list(range(1, nfreq + 1))
    unit_total = power.sum(dim=1, keepdim=True).clamp_min(1e-12)
    unit_norm = power / unit_total                         # each row sums to 1
    # per-unit concentration: fraction of a unit's own power in its top-1 / top-2
    top1 = unit_norm.max(dim=1).values                     # [hidden]
    top2 = unit_norm.topk(2, dim=1).values.sum(dim=1)      # [hidden]
    peak_freq = (unit_norm.argmax(dim=1) + 1)              # 1-indexed key freq/unit
    # population spectrum (mean over units) -- the WRONG metric, kept as record
    pop = power.mean(dim=0)
    pop = (pop / pop.sum().clamp_min(1e-12)).tolist()
    # distinct key frequencies actually used across the population
    hist = {}
    for f in peak_freq.tolist():
        hist[f] = hist.get(f, 0) + 1
    return {
        "per_unit_top1_mean": float(top1.mean()),
        "per_unit_top2_mean": float(top2.mean()),
        "flat_baseline_top1": 1.0 / nfreq,                 # what a memorizer ~ gives
        "n_distinct_key_freqs": len(hist),                 # size of the "world"
        "peak_freq_hist": dict(sorted(hist.items())),      # which freqs, how many units
        "population_spectrum": pop,                         # full curve (record)
        "freqs": freqs,
    }


def main():
    p = 97
    steps = 20000
    seed = 0
    out = {}

    def report(tag, d):
        s = d["spectrum"]
        print(f"    {tag}: val_acc={d['val_acc']:.3f}  |w|={d['weight_l2']:.1f}  "
              f"per_unit_top1={s['per_unit_top1_mean']:.3f} "
              f"(flat={s['flat_baseline_top1']:.3f})  "
              f"distinct_key_freqs={s['n_distinct_key_freqs']}")

    print("[1/3] adult / glasses removed: onehot (opaque) + weight_decay=1.0 -> grok")
    m_grok, met_grok = train_capture("onehot", 1.0, steps, p=p, seed=seed)
    out["grokked_onehot"] = {**met_grok, "spectrum": summarize(m_grok, p),
                             "note": "opaque medium, groks; the adult who removed the glasses"}
    report("grokked  ", out["grokked_onehot"])

    print("[2/3] glasses ON: onehot (opaque) + weight_decay=0.0 -> memorizer")
    m_mem, met_mem = train_capture("onehot", 0.0, steps, p=p, seed=seed)
    out["memorizer_onehot_wd0"] = {**met_mem, "spectrum": summarize(m_mem, p),
                                   "note": "opaque medium, wd=0; still wearing all the glasses"}
    report("memorizer", out["memorizer_onehot_wd0"])

    print("[3/3] baby / born seeing: fourier (transparent nf=1, normalized) + weight_decay=0.1")
    m_baby, met_baby = train_capture("fourier", 0.1, steps, p=p, seed=seed,
                                     num_freqs=1, fourier_normalize=True)
    out["transparent_fourier"] = {**met_baby, "spectrum": summarize(m_baby, p),
                                  "note": "transparent medium (nf=1, the cleanest world); the baby, no memorization phase"}
    report("baby     ", out["transparent_fourier"])

    out["_meta"] = {
        "metrics_version": METRICS_VERSION,
        "p": p, "steps": steps, "seed": seed, "hidden": 256,
        "single_seed": True,
    }
    dst = os.path.join(os.path.dirname(__file__), "results", "spectrum.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
