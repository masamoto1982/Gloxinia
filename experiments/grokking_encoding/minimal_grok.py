"""
minimal_grok.py — the simplest reliable grokking reproduction we found.

Self-contained (only torch). No imports from the rest of the harness. Run:

    python minimal_grok.py                 # groks in a few thousand steps
    python minimal_grok.py --weight_decay 0   # ablation: never groks (memorizes)

WHAT GROKKING NEEDS (distilled from this repo's experiments — see
DISTILLED_RECIPE.md). Only four ingredients are load-bearing:

  1. An OPAQUE encoding of an algebraic task. One-hot residues hide the group
     structure, so the net must *discover* it -> a memorization phase exists to
     be escaped. (A transparent encoding removes the phase and the step; see
     RESULTS_v2.md. That is why one-hot, not Fourier, is the grokking recipe.)
  2. WEIGHT DECAY. This is the engine. It is what eventually makes the
     structural solution cheaper than the memorized table. Set it to 0 and the
     step never comes (the --weight_decay 0 ablation shows train->1.0,
     val stuck). This is the single most important knob.
  3. LIMITED DATA (a fraction of all pairs). This creates the
     underdetermination gap the net memorizes into before generalizing.
  4. FULL-BATCH training with Adam. Keeps the dynamics clean/deterministic so
     the delayed step is crisp rather than smeared by minibatch noise.

Everything else (model size, exact lr, p, hidden width) is a convenience knob,
not an ingredient. Smaller p just makes it grok sooner (fewer steps to watch).

mirage guard: we print train/val ACCURACY and train/val LOSS together. Grokking
= val_acc jumps late while val_loss, having risen during memorization, drops at
the same step. If you only ever look at accuracy you cannot tell a real
transition from a thresholding artifact -- so both are always shown.
"""

import argparse
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=31)             # small => groks sooner
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)  # THE engine
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    p = args.p

    # data: every (a, b) pair, label (a+b) mod p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    ab = torch.stack([a, b], 1)
    y = (a + b) % p

    # deterministic split
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(p * p, generator=g)
    n_tr = int(args.train_frac * p * p)
    tr, va = perm[:n_tr], perm[n_tr:]

    # one-hot encoding table (OPAQUE: arbitrary basis, structure hidden)
    E = torch.eye(p)

    # one hidden layer MLP: [onehot(a) ; onehot(b)] -> hidden -> p logits
    net = torch.nn.Sequential(
        torch.nn.Linear(2 * p, args.hidden), torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, p),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))

    def batch(idx):
        return torch.cat([E[ab[idx, 0]], E[ab[idx, 1]]], 1), y[idx]

    xtr, ytr = batch(tr)
    xva, yva = batch(va)

    print(f"p={p} train_frac={args.train_frac} weight_decay={args.weight_decay} "
          f"seed={args.seed}")
    print(f"{'step':>6} {'tr_acc':>7} {'va_acc':>7} {'tr_loss':>9} {'va_loss':>9}")
    for step in range(args.steps + 1):
        net.train()
        opt.zero_grad()
        loss = F.cross_entropy(net(xtr), ytr)   # FULL BATCH
        loss.backward()
        opt.step()
        if step % 200 == 0:
            net.eval()
            with torch.no_grad():
                tra = (net(xtr).argmax(1) == ytr).float().mean().item()
                vaa = (net(xva).argmax(1) == yva).float().mean().item()
                trl = loss.item()
                val = F.cross_entropy(net(xva), yva).item()
            print(f"{step:>6} {tra:>7.3f} {vaa:>7.3f} {trl:>9.4f} {val:>9.4f}")


if __name__ == "__main__":
    main()
