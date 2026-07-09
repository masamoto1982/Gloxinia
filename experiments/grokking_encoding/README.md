# experiment: grokking × encoding transparency

Minimal, reproducible test of claim **C2**: does the grokking *step* vanish when
the task's structure is transparent in the input encoding? Design and the
labeling-tension resolution are in [`docs/02-experiment-design.md`](../../docs/02-experiment-design.md).

Task: `(a + b) mod p`. Model: one-hidden-layer MLP, full-batch AdamW with strong
weight decay. Only the **encoding** changes across arms.

## Files

- `encoding.py` — the only thing that varies. `onehot` (opaque/canonical),
  `fourier` (transparent), `onehot_distractor` (opaque + injected
  underdetermination). The residue table is a *frozen buffer*, never a learned
  embedding, so we study the medium and not a drifting representation.
- `harness.py` — model, full-batch training loop, metric logging (binary +
  continuous, per the mirage guard), summary derivation. `metrics_version`
  tags the schema.
- `run.py` — one run to one JSON.
- `results/*.json` — committed metric curves (the record).
- `RESULTS.md` — measured outcomes, including nulls.

## Reproduce

Deterministic; one seed fixes split and init. Requires `torch` (CPU is fine).

```bash
# Arm A — harness verification: must reproduce classic grokking FIRST.
python run.py --encoding onehot  --steps 30000 --eval_every 200 --seed 0 --out results/onehot.json

# Arm B — transparent medium: does the step vanish?
python run.py --encoding fourier --steps 30000 --eval_every 200 --seed 0 --out results/fourier.json

# Arm C — inject underdetermination (single-seed probe of C1).
python run.py --encoding onehot_distractor --steps 20000 --eval_every 200 --seed 0 --out results/onehot_distractor.json

# Fourier num_freqs sweep (transparency as a dial: few low freqs -> near-complete basis).
for nf in 16 32 48; do
  python run.py --encoding fourier --num_freqs $nf --steps 20000 --eval_every 200 --seed 0 --out results/fourier_nf${nf}.json
done

# Figure (accuracy over loss, log-x) for any set of result JSONs.
python plot.py results/onehot.json results/onehot_distractor.json results/fourier.json results/fourier_nf48.json --out results/curves.png
```

See [`RESULTS.md`](RESULTS.md) for the measured outcomes, including the null and
the refuted caveat.

Default hyperparameters (in `harness.py::Config`, echoed into every JSON):
`p=97`, `train_frac=0.5`, `hidden=256`, `lr=1e-3`, `weight_decay=1.0`,
`num_freqs=4` (fourier), `num_distractor=8` (distractor).

## Reading a result

Grokking is claimed for an arm only when **both** agree (mirage guard):
`grok_delay` is large and positive **and** `val_loss` stays high through the
delay then drops at the same step `val_acc` rises. A small delay with train/val
loss falling together is "no visible step." See `RESULTS.md` for the call on each
arm actually run.
