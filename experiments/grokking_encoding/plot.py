"""Plot metric curves from one or more result JSONs.

Renders, per arm, accuracy (binary-ish) and loss (continuous co-metric) on a
log-x axis so the mirage guard is legible: you can see whether val_loss drops at
the same step val_acc rises, or whether the accuracy step sits on top of a loss
that was already gliding down.

    python plot.py results/onehot.json results/fourier.json --out results/curves.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--out", default="results/curves.png")
    args = ap.parse_args()

    runs = []
    for path in args.jsons:
        with open(path) as f:
            runs.append(json.load(f))

    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7), squeeze=False)

    for j, r in enumerate(runs):
        steps = [max(s, 1) for s in r["steps_log"]]
        enc = r["config"]["encoding"]
        delay = r.get("grok_delay")
        tag = enc
        if enc == "fourier":
            tag = f"fourier(nf={r['config'].get('num_freqs')})"
        elif enc == "onehot_distractor":
            tag = f"onehot+distractor(d={r['config'].get('num_distractor')})"

        ax = axes[0][j]
        ax.plot(steps, r["train_acc"], label="train_acc", color="#1f77b4")
        ax.plot(steps, r["val_acc"], label="val_acc", color="#d62728")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{tag}  (grok_delay={delay}, final_va={r['final_val_acc']:.2f})")
        ax.set_ylabel("accuracy")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1][j]
        ax2.plot(steps, r["train_loss"], label="train_loss", color="#1f77b4")
        ax2.plot(steps, r["val_loss"], label="val_loss", color="#d62728")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("step (log)")
        ax2.set_ylabel("cross-entropy loss (log)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)

    fig.suptitle("grokking × encoding — accuracy (top) vs loss (bottom); "
                 "mirage guard = do they move at the same step?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
