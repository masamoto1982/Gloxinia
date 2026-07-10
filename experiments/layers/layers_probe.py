"""Per-layer decomposition of grokking: which layer forms the crystal?

Standard onehot MLP (fc1 hidden, fc2 output) on (a+b) mod p. Logs val_acc and the
weight norm of EACH layer separately, to ask where memorization (norm peak-then-
decline) and the low-norm structural solution live: the hidden ("deep") layer, or
the output ("surface") layer.
"""
import argparse, json, torch, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97); ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--wd", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/layers.json")
    a = ap.parse_args(); torch.manual_seed(a.seed); p = a.p
    A = torch.arange(p).repeat_interleave(p); B = torch.arange(p).repeat(p)
    ab = torch.stack([A, B], 1); y = (A + B) % p
    g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(p*p, generator=g)
    ntr = (p*p)//2; tr, va = perm[:ntr], perm[ntr:]; E = torch.eye(p)
    def enc(idx): return torch.cat([E[ab[idx,0]], E[ab[idx,1]]], 1)
    xtr, ytr, xva, yva = enc(tr), y[tr], enc(va), y[va]
    fc1 = torch.nn.Linear(2*p, a.hidden); fc2 = torch.nn.Linear(a.hidden, p)
    opt = torch.optim.AdamW(list(fc1.parameters())+list(fc2.parameters()), lr=a.lr, weight_decay=a.wd, betas=(0.9,0.98))
    def fwd(x): return fc2(F.relu(fc1(x)))
    rec = {"step":[], "val_acc":[], "train_acc":[], "fc1_norm":[], "fc2_norm":[]}
    for s in range(a.steps+1):
        opt.zero_grad(); loss = F.cross_entropy(fwd(xtr), ytr); loss.backward(); opt.step()
        if s % 200 == 0:
            with torch.no_grad():
                va_acc = (fwd(xva).argmax(1)==yva).float().mean().item()
                tr_acc = (fwd(xtr).argmax(1)==ytr).float().mean().item()
                n1 = sum((q**2).sum().item() for q in fc1.parameters())**0.5
                n2 = sum((q**2).sum().item() for q in fc2.parameters())**0.5
            for k,v in zip(rec, [s,va_acc,tr_acc,n1,n2]): rec[k].append(v)
    json.dump(rec, open(a.out,"w")); print("done", a.out, "final val", rec["val_acc"][-1])

if __name__ == "__main__": main()
