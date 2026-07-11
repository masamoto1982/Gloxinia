# RESULTS — enlightenment (do the two paths reach the same world?)

Tests the "same-world" prediction in `docs/06-enlightenment.md`, which is the
operationalization of **R1** (charter 1.3: "a grokked solution and a
smoothly-learned one may be the identical circuit" — asserted, never measured).
onehot / fourier MLP, p=97, frac=0.5, hidden=256, seed 0, 20000 steps. Metric:
per-hidden-unit DFT of fc1's operand-a weights in residue space, DC dropped;
`per_unit_top1_mean` = each unit's own spectral power in its single top frequency,
averaged over units (flat baseline `1/(p//2) = 0.021`). Data: `results/spectrum.json`.
Reproduce: `python analyze.py`.

## Result — supported ✓ (with the "few frequencies" clause corrected)

| net | role | val_acc | ‖w‖ | **per_unit_top1** | flat |
|---|---|---:|---:|---:|---:|
| onehot, wd=1.0 (grokked) | adult, glasses removed | **1.000** | 122.8 | **0.879** | 0.021 |
| onehot, wd=0.0 (memorizer) | glasses still on | 0.024 | 270.5 | **0.322** | 0.021 |
| fourier nf=1, wd=0.1 (transparent) | baby, born seeing | 0.896 | 289.5 | 1.000† | 0.021 |

† trivially 1.000: the nf=1 input *is* a single sinusoid, so every unit can only
be that one frequency. It is the reference point ("born speaking pure sinusoids"),
not evidence.

**The core prediction holds.** The network that groks through the **opaque**
onehot medium — where nothing in the input geometry exposes the group — makes each
hidden unit a **near-clean single sinusoid in the residue** (`per_unit_top1 =
0.879`), i.e. it rebuilds a **Fourier/trig circuit** from scratch. The memorizer,
holding train at `wd=0` but never generalizing, does **not** (`0.322`): its units
stay diffuse in frequency. A 2.7× separation on the same architecture, same seed,
same data — the only difference is whether it generalized. **Generalizing ⟺ the
representation becomes sinusoidal.** R1 goes from assertion to measurement: the
grokked circuit and the transparent one share the same *representational currency*
— the harmonics of Z/pZ — the adult reconstructing behind the glasses the
vocabulary the baby is born speaking.

**The "few frequencies" clause is corrected (pre-registered wrong, reported).**
`docs/06` predicted the grokked net would reconstruct the same **few**-frequency
structure. It does not. Each unit is *individually* one clean frequency, but the
population spreads across **many**: the busiest single frequency claims only ~10
of 256 units, the top-5 frequencies only ~16% of units, and all 48 frequencies
appear as some unit's peak (`n_distinct_key_freqs = 48` for grokked *and*
memorizer alike — which is why that count is **not** the discriminator, and the
per-unit metric is). "Fewness" was a property of the transparent *input* (nf=1),
not of the grokked *solution*: given an opaque basis that exposes every harmonic,
the net helps itself to a spread of them. So the two paths reach the same *kind*
of world — a harmonic one — but not the identical *vector*: the transparent medium
pins the frequency, the opaque medium lets the grokker pick a spread.

**Mirage guard.** The population-averaged spectrum is near-flat for grokked and
memorizer alike (top-5 population power 0.15 vs 0.14) and **cannot** tell them
apart — the discrimination lives entirely in the *per-unit* concentration, which
is the load-bearing metric. Both the population spectrum and the per-unit peak
histogram are in `spectrum.json`; the separation (0.879 vs 0.322) is a property of
the weights, not of the cutoff.

## What this establishes

- **R1 measured, not assumed.** Generalization and the sinusoidal (Fourier)
  representation move together; memorization keeps the representation
  structureless. The opaque grokker and the transparent learner speak the same
  harmonic vocabulary. Weight norm tracks it too: the grokked net sits at the
  structural crystal norm (‖w‖ 122.8, cf. `low_norm/M1`, `crystallization`), the
  memorizer high (270.5).
- **The medium sets the exact frequencies, not the form.** This *sharpens* C2:
  the opaque vs transparent medium changes not only *when/how visibly* structure
  is reached but *which* harmonics are used — while leaving the structure's *form*
  (per-unit sinusoids) the same. C2's "same circuit, different visibility" is
  right about the form and too strong about the literal circuit; the honest
  statement is "same harmonic vocabulary, medium-dependent frequency choice."

## Honest bounds

- **Single seed, single config** (p=97, hidden=256, frac=0.5). The 0.879 vs 0.322
  gap is large but one draw; a seed sweep is owed before calling the separation a
  law rather than a wide single-seed gap.
- **`per_unit_top1` is a proxy for "is a sinusoid," not a proof of a working
  circuit.** A unit can be sinusoidal without the *pair* of units implementing the
  trig identity that computes `(a+b) mod p`. Confirming the actual circuit
  (operand-a and operand-b units sharing a frequency, the output layer reading the
  product) is the Nanda-style follow-up, not done here.
- **The dis-analogy in `docs/06` §3 stands and is not re-measured here:** driving
  ‖w‖→0 (removing the *last* glass) is death, not enlightenment — see the
  crystallization "idle" test. This experiment only measures the *form* of the
  low-norm solution, not the floor below it.
