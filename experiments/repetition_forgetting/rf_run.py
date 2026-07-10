"""CLI for one repetition/forgetting run. Writes a self-describing JSON."""

import argparse
import json
import os

from rf_harness import RFConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--data_mode", choices=["fixed", "online"], default="fixed")
    ap.add_argument("--online_batch", type=int, default=512)
    ap.add_argument("--label_noise", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--wd_switch_step", type=int, default=0)
    ap.add_argument("--still_task", choices=["study", "idle", "mantra"], default="study")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boredom_gamma", type=float, default=0.0)
    ap.add_argument("--boredom_floor", type=float, default=0.05)
    ap.add_argument("--boredom_ema", type=float, default=0.9)
    ap.add_argument("--forget_noise", type=float, default=0.0)
    ap.add_argument("--shrink_lambda", type=float, default=0.0)
    ap.add_argument("--stoch_shrink_frac", type=float, default=0.0)
    ap.add_argument("--stoch_shrink_amount", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = RFConfig(**{k: v for k, v in vars(args).items()
                      if k not in ("out", "quiet")})
    res = train(cfg, verbose=not args.quiet)

    print("\n=== summary ===")
    for k in ("data_mode", "train_frac", "label_noise", "weight_decay",
              "boredom_gamma"):
        print(f"{k:18s}: {getattr(cfg, k)}")
    print(f"step_train_saturate: {res.step_train_saturate}")
    print(f"step_val_generalize: {res.step_val_generalize}")
    print(f"grok_delay         : {res.grok_delay}")
    print(f"weight_l2_peak     : {res.weight_l2_peak} @ {res.step_weight_l2_peak}")
    print(f"final_val_acc      : {res.final_val_acc}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res.__dict__, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
