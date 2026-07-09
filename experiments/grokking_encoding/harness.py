"""
Grokking x encoding-transparency harness (modular addition).

Task:   (a + b) mod p  ->  class in [0, p).
Model:  small MLP (concat of the two operand encodings -> hidden -> p logits).
Recipe: full-batch AdamW with strong weight decay -- the canonical grokking
        recipe (Power et al. 2022; MLP variant per Gromov 2023). Weight decay
        is what drives the memorize -> generalize transition, so it is the one
        ingredient we keep fixed and non-zero across every arm.

We vary ONLY the input encoding (see encodings.py) and ask a single question:
does delayed generalization -- train accuracy saturating long BEFORE validation
accuracy jumps -- appear under some encodings and vanish under others?

DISCIPLINE baked into the logging:
  * mirage guard (Schaeffer et al. 2023): every eval logs a binary-ish signal
    (train_acc / val_acc) AND a continuous co-metric (train_loss / val_loss /
    weight_l2). We only call something a "transition" when the continuous
    quantity moves at the same place the accuracy does.
  * proxy naming: metrics are named for exactly what they are (val_acc is a
    validation accuracy, not a claim about capability). metrics_version tags
    the schema; records from different versions are not comparable.
  * determinism: one seed fixes the data split and the init; it is recorded.

This module is import-safe and side-effect free; run.py is the CLI entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import build_encoding, METRICS_ENCODING_VERSION

METRICS_VERSION = "grok-metrics-v1"


@dataclass
class Config:
    encoding: str = "onehot"
    p: int = 97
    train_frac: float = 0.5
    hidden: int = 256
    lr: float = 1e-3
    weight_decay: float = 1.0
    steps: int = 20000
    eval_every: int = 100
    seed: int = 0
    # encoding-specific knobs (recorded verbatim so runs are self-describing)
    num_freqs: int = 4
    num_distractor: int = 8
    # thresholds used to summarize the curves (reported, not load-bearing)
    train_acc_thresh: float = 0.99
    val_acc_thresh: float = 0.90


class MLP(nn.Module):
    """One hidden layer is enough to grok modular addition (Gromov 2023).

    The encoding table E is a fixed (non-trainable) buffer: we are studying the
    effect of the *input medium*, so the residue representations must not be
    allowed to drift into a learned embedding.
    """

    def __init__(self, enc_table: torch.Tensor, hidden: int, p: int):
        super().__init__()
        self.register_buffer("E", enc_table)
        d = enc_table.shape[1]
        self.fc1 = nn.Linear(2 * d, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, ab: torch.Tensor) -> torch.Tensor:
        # ab: [N, 2] integer residues
        ea = self.E[ab[:, 0]]
        eb = self.E[ab[:, 1]]
        x = torch.cat([ea, eb], dim=1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)


def make_data(p: int):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    ab = torch.stack([a, b], dim=1)
    y = (a + b) % p
    return ab, y


def split(ab, y, train_frac, seed):
    g = torch.Generator().manual_seed(seed)
    n = ab.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    tr, va = perm[:n_train], perm[n_train:]
    return (ab[tr], y[tr]), (ab[va], y[va])


@dataclass
class RunResult:
    config: dict
    metrics_version: str
    encoding_version: str
    steps_log: list = field(default_factory=list)      # step indices at each eval
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    weight_l2: list = field(default_factory=list)
    # summary (all derived from the logged curves; None == not reached)
    step_train_saturate: int | None = None
    step_val_generalize: int | None = None
    grok_delay: int | None = None
    final_val_acc: float | None = None
    wall_seconds: float | None = None


@torch.no_grad()
def _eval(model, ab, y):
    logits = model(ab)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return acc, loss


def train(cfg: Config, verbose: bool = True) -> RunResult:
    torch.manual_seed(cfg.seed)
    enc = build_encoding(
        cfg.encoding, cfg.p, num_freqs=cfg.num_freqs,
        num_distractor=cfg.num_distractor, seed=cfg.seed,
    )
    ab, y = make_data(cfg.p)
    (ab_tr, y_tr), (ab_va, y_va) = split(ab, y, cfg.train_frac, cfg.seed)

    model = MLP(enc, cfg.hidden, cfg.p)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )

    res = RunResult(
        config=asdict(cfg),
        metrics_version=METRICS_VERSION,
        encoding_version=METRICS_ENCODING_VERSION,
    )
    t0 = time.time()
    for step in range(cfg.steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(ab_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0:
            model.eval()
            tr_acc, tr_loss = _eval(model, ab_tr, y_tr)
            va_acc, va_loss = _eval(model, ab_va, y_va)
            wl2 = sum((pn.detach() ** 2).sum().item()
                      for n, pn in model.named_parameters() if "E" not in n) ** 0.5
            res.steps_log.append(step)
            res.train_acc.append(tr_acc)
            res.val_acc.append(va_acc)
            res.train_loss.append(tr_loss)
            res.val_loss.append(va_loss)
            res.weight_l2.append(wl2)

            if res.step_train_saturate is None and tr_acc >= cfg.train_acc_thresh:
                res.step_train_saturate = step
            if res.step_val_generalize is None and va_acc >= cfg.val_acc_thresh:
                res.step_val_generalize = step

            if verbose:
                print(f"step {step:6d}  tr_acc {tr_acc:.3f}  va_acc {va_acc:.3f}  "
                      f"tr_loss {tr_loss:.4f}  va_loss {va_loss:.4f}  |w| {wl2:.1f}",
                      flush=True)

    res.wall_seconds = time.time() - t0
    res.final_val_acc = res.val_acc[-1] if res.val_acc else None
    if res.step_train_saturate is not None and res.step_val_generalize is not None:
        res.grok_delay = res.step_val_generalize - res.step_train_saturate
    return res
