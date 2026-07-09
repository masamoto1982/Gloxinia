"""CLI entry point for a single grokking x encoding run.

Example:
    python run.py --encoding onehot  --steps 20000 --out results/onehot.json
    python run.py --encoding fourier --steps 20000 --out results/fourier.json

Writes a self-describing JSON (config + full metric curves + summary). The JSON
is the record; docs/ analysis reads from it so numbers are never hand-copied.
"""

import argparse
import json
import os

from harness import Config, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="onehot",
                    choices=["onehot", "fourier", "onehot_distractor"])
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_freqs", type=int, default=4)
    ap.add_argument("--num_distractor", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        encoding=args.encoding, p=args.p, train_frac=args.train_frac,
        hidden=args.hidden, lr=args.lr, weight_decay=args.weight_decay,
        steps=args.steps, eval_every=args.eval_every, seed=args.seed,
        num_freqs=args.num_freqs, num_distractor=args.num_distractor,
    )
    res = train(cfg, verbose=not args.quiet)

    print("\n=== summary ===")
    print(f"encoding            : {cfg.encoding}")
    print(f"step_train_saturate : {res.step_train_saturate}")
    print(f"step_val_generalize : {res.step_val_generalize}")
    print(f"grok_delay          : {res.grok_delay}")
    print(f"final_val_acc       : {res.final_val_acc}")
    print(f"wall_seconds        : {res.wall_seconds:.1f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res.__dict__, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
