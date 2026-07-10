"""
Repetition & forgetting harness (H-rep).

Hypothesis under test (user's words, paraphrased): the inducing factors of
grokking are (a) EMPHASIS on repeated/consistent patterns and (b) BOREDOM &
FORGETTING of over-repeated patterns.

This harness keeps the grokking task fixed ((a+b) mod p, one-hot MLP, full-batch
AdamW) and adds the knobs needed to probe that hypothesis in two phases:

PHASE 1 (measure emergent signatures):
  * data_mode = "fixed"   -> standard grokking: a fixed limited train set is
                             RE-PRESENTED every step (over-repetition of identical
                             data). This is the regime where memorization -> grok.
  * data_mode = "online"  -> each step draws FRESH pairs from a large pool, so no
                             single pair is over-repeated. Tests whether removing
                             over-repetition removes the memorization phase (and
                             the step). NOTE: "no over-repetition" is inseparable
                             from "abundant data" -- we state that plainly rather
                             than pretend we isolated repetition from data volume.
  * label_noise = rho     -> randomize a FIXED fraction of train labels: patterns
                             that are repeated every step but are NOT consistent
                             with the rule. Tests "emphasis on CONSISTENTLY
                             repeated patterns" (inconsistent repetition can only
                             be memorized, then must be forgotten).
  * weight_l2 is logged as the FORGETTING signature: under weight decay it peaks
                             (memorization) then declines (forgetting) -- we check
                             whether that decline coincides with generalization.

PHASE 2 (build the mechanism): an explicit per-example EMPHASIS+BOREDOM loss
reweighting (see boredom_gamma). Emphasis = up-weight not-yet-mastered examples;
boredom = down-weight examples the model has confidently fit for a while. The
decisive question is whether this can INDUCE grokking with weight_decay=0 -- i.e.
whether an explicit "boredom/forgetting" can stand in for weight decay as the
engine. Off by default (boredom_gamma=0 -> uniform weights = standard training).

Discipline (unchanged): mirage guard (accuracy AND loss logged together),
proxy-named metrics, versioned schema, fixed seed, nulls reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

METRICS_VERSION = "rf-metrics-v1"


@dataclass
class RFConfig:
    p: int = 97
    train_frac: float = 0.5          # size of the fixed pool (fraction of p*p)
    data_mode: str = "fixed"         # "fixed" | "online"
    online_batch: int = 512          # fresh pairs per step in online mode
    label_noise: float = 0.0         # fraction of FIXED-train labels randomized
    hidden: int = 256
    lr: float = 1e-3
    weight_decay: float = 1.0
    steps: int = 20000
    eval_every: int = 200
    seed: int = 0
    # Phase-2 explicit emphasis+boredom (0 => uniform weights => off)
    boredom_gamma: float = 0.0       # >0: weight_i ∝ (1 - p_correct_i)^gamma
    boredom_floor: float = 0.05      # keep a floor so bored examples aren't zeroed
    boredom_ema: float = 0.9         # EMA on per-example correctness prob
    # Phase-2 explicit FORGETTING as weight noise (0 => off). Each step adds
    # Gaussian noise ~ N(0, forget_noise^2) to the trainable weights: an active
    # "forgetting" that corrupts memorized specifics without being weight decay
    # (it does not pull toward zero). Tests whether forgetting can replace weight
    # decay as the grokking engine.
    forget_noise: float = 0.0
    # Phase-3 NORM-REDUCING forgetting: does grokking need weight decay
    # specifically, or just *norm reduction in any form*? Two non-AdamW routes,
    # both applied manually each step, both shrink norm but differ in form:
    #   shrink_lambda      -- uniform: w *= (1 - shrink_lambda) every step. This
    #                         is decoupled weight decay done by hand (positive
    #                         control: confirms norm-reduction -> grok, and that
    #                         it isn't something special about AdamW's wd param).
    #   stoch_shrink_frac  -- stochastic: each step a RANDOM subset (this
    #   stoch_shrink_amount   fraction of elements) is shrunk by this amount;
    #                         the rest untouched. Structurally unlike uniform
    #                         decay -- "randomly forget some connections" -- but
    #                         still reduces norm. The decisive test.
    shrink_lambda: float = 0.0
    stoch_shrink_frac: float = 0.0
    stoch_shrink_amount: float = 0.0


class MLP(nn.Module):
    def __init__(self, p: int, hidden: int):
        super().__init__()
        self.register_buffer("E", torch.eye(p))   # one-hot (opaque)
        self.fc1 = nn.Linear(2 * p, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, ab):
        x = torch.cat([self.E[ab[:, 0]], self.E[ab[:, 1]]], 1)
        return self.fc2(F.relu(self.fc1(x)))


def all_pairs(p):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return torch.stack([a, b], 1), (a + b) % p


@dataclass
class RFResult:
    config: dict
    metrics_version: str = METRICS_VERSION
    steps_log: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    weight_l2: list = field(default_factory=list)
    step_train_saturate: int | None = None
    step_val_generalize: int | None = None
    grok_delay: int | None = None
    weight_l2_peak: float | None = None
    step_weight_l2_peak: int | None = None
    final_val_acc: float | None = None
    wall_seconds: float | None = None


@torch.no_grad()
def _eval(model, ab, y):
    logits = model(ab)
    return ((logits.argmax(1) == y).float().mean().item(),
            F.cross_entropy(logits, y).item())


def train(cfg: RFConfig, verbose: bool = True) -> RFResult:
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    ab, y = all_pairs(cfg.p)
    n = cfg.p * cfg.p
    perm = torch.randperm(n, generator=g)
    n_tr = int(round(cfg.train_frac * n))
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    ab_tr, y_tr = ab[tr_idx].clone(), y[tr_idx].clone()
    ab_va, y_va = ab[va_idx], y[va_idx]

    # label noise: corrupt a fixed fraction of the (fixed) train labels
    if cfg.label_noise > 0:
        k = int(round(cfg.label_noise * len(y_tr)))
        noisy = torch.randperm(len(y_tr), generator=g)[:k]
        y_tr[noisy] = torch.randint(0, cfg.p, (k,), generator=g)

    # online mode draws fresh each step from the WHOLE space minus the held-out
    # val set, so val stays an honest generalization measure while no single
    # train pair is over-repeated.
    pool_ab = pool_y = None
    if cfg.data_mode == "online":
        mask = torch.ones(n, dtype=torch.bool)
        mask[va_idx] = False
        pool = torch.arange(n)[mask]
        pool_ab, pool_y = ab[pool], y[pool]

    model = MLP(cfg.p, cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    # Phase-2 per-example correctness EMA (only used if boredom_gamma>0, fixed mode)
    pcorr = torch.full((len(y_tr),), 0.5)

    res = RFResult(config=asdict(cfg))
    t0 = time.time()
    for step in range(cfg.steps + 1):
        model.train()
        opt.zero_grad()
        if cfg.data_mode == "online":
            sel = torch.randint(0, len(pool_y), (cfg.online_batch,), generator=g)
            logits = model(pool_ab[sel])
            loss = F.cross_entropy(logits, pool_y[sel])
        else:
            logits = model(ab_tr)
            if cfg.boredom_gamma > 0:
                with torch.no_grad():
                    p_i = torch.softmax(logits, 1)[torch.arange(len(y_tr)), y_tr]
                    pcorr.mul_(cfg.boredom_ema).add_((1 - cfg.boredom_ema) * p_i)
                    w = (1.0 - pcorr).clamp_min(cfg.boredom_floor) ** cfg.boredom_gamma
                    w = w / w.mean()
                loss = (F.cross_entropy(logits, y_tr, reduction="none") * w).mean()
            else:
                loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()

        # explicit forgetting: corrupt the trainable weights with Gaussian noise
        if cfg.forget_noise > 0:
            with torch.no_grad():
                for nm, pn in model.named_parameters():
                    if "E" not in nm:
                        pn.add_(torch.randn(pn.shape, generator=g) * cfg.forget_noise)

        # norm-reducing forgetting (uniform and/or stochastic-subset shrink)
        if cfg.shrink_lambda > 0 or cfg.stoch_shrink_frac > 0:
            with torch.no_grad():
                for nm, pn in model.named_parameters():
                    if "E" in nm:
                        continue
                    if cfg.shrink_lambda > 0:
                        pn.mul_(1.0 - cfg.shrink_lambda)
                    if cfg.stoch_shrink_frac > 0:
                        mask = torch.rand(pn.shape, generator=g) < cfg.stoch_shrink_frac
                        pn[mask] *= (1.0 - cfg.stoch_shrink_amount)

        if step % cfg.eval_every == 0:
            model.eval()
            # train metric: fixed set in fixed mode; a fresh draw in online mode
            if cfg.data_mode == "online":
                sel = torch.randint(0, len(pool_y), (2048,), generator=g)
                tr_acc, tr_loss = _eval(model, pool_ab[sel], pool_y[sel])
            else:
                tr_acc, tr_loss = _eval(model, ab_tr, y_tr)
            va_acc, va_loss = _eval(model, ab_va, y_va)
            wl2 = sum((pn.detach() ** 2).sum().item()
                      for nm, pn in model.named_parameters() if "E" not in nm) ** 0.5
            res.steps_log.append(step)
            res.train_acc.append(tr_acc); res.val_acc.append(va_acc)
            res.train_loss.append(tr_loss); res.val_loss.append(va_loss)
            res.weight_l2.append(wl2)
            if res.step_train_saturate is None and tr_acc >= 0.99:
                res.step_train_saturate = step
            if res.step_val_generalize is None and va_acc >= 0.90:
                res.step_val_generalize = step
            if verbose:
                print(f"step {step:6d}  tr {tr_acc:.3f}  va {va_acc:.3f}  "
                      f"trL {tr_loss:.4f}  vaL {va_loss:.4f}  |w| {wl2:.1f}", flush=True)

    res.wall_seconds = time.time() - t0
    res.final_val_acc = res.val_acc[-1] if res.val_acc else None
    if res.weight_l2:
        pk = max(range(len(res.weight_l2)), key=lambda i: res.weight_l2[i])
        res.weight_l2_peak = res.weight_l2[pk]
        res.step_weight_l2_peak = res.steps_log[pk]
    if res.step_train_saturate is not None and res.step_val_generalize is not None:
        res.grok_delay = res.step_val_generalize - res.step_train_saturate
    return res
