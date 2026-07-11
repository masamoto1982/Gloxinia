//! Gloxinia — an in-browser grokking demonstrator.
//!
//! This is a faithful, dependency-free port of the repository's
//! `minimal_grok.py`: a one-hidden-layer MLP trained full-batch with AdamW on
//! modular addition `(a + b) mod p`. The one knob that matters for the story is
//! the *input encoding*:
//!
//!   * `OneHot`  — the OPAQUE medium. Residues are an arbitrary basis; the group
//!     structure is hidden, so the net memorises first and generalisation
//!     arrives as a late, visible step (classic grokking). Weight decay is the
//!     engine that eventually makes the low-norm structural circuit cheaper than
//!     the high-norm memorised table.
//!   * `Fourier` — the TRANSPARENT medium. Residues are unit-norm cos/sin
//!     features, so the structure is already in the representation: there is no
//!     memorisation phase to escape and (with the right regulariser) no step.
//!
//! Everything is written in plain `f32` loops so it compiles to a small,
//! fast WASM module with no numeric dependencies.

use wasm_bindgen::prelude::*;

/// Deterministic SplitMix64 PRNG — seeds reproduce exactly across runs/platforms.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        // 24 random bits -> [0,1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Uniform in [-a, a).
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    /// Fisher–Yates shuffle.
    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq)]
pub enum Encoding {
    OneHot = 0,
    Fourier = 1,
}

/// One dense layer: `out = in @ w^T + b`, with AdamW moments carried alongside.
struct Linear {
    fan_in: usize,
    fan_out: usize,
    w: Vec<f32>, // [fan_out * fan_in]
    b: Vec<f32>, // [fan_out]
    // grads
    gw: Vec<f32>,
    gb: Vec<f32>,
    // adam moments
    mw: Vec<f32>,
    vw: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Linear {
    fn new(fan_in: usize, fan_out: usize, rng: &mut Rng) -> Self {
        // PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        let bound = 1.0 / (fan_in as f32).sqrt();
        let w = (0..fan_in * fan_out).map(|_| rng.sym(bound)).collect();
        let b = (0..fan_out).map(|_| rng.sym(bound)).collect();
        Linear {
            fan_in,
            fan_out,
            w,
            b,
            gw: vec![0.0; fan_in * fan_out],
            gb: vec![0.0; fan_out],
            mw: vec![0.0; fan_in * fan_out],
            vw: vec![0.0; fan_in * fan_out],
            mb: vec![0.0; fan_out],
            vb: vec![0.0; fan_out],
        }
    }

    /// Forward for a batch. `x` is [n * fan_in]; writes [n * fan_out] into `out`.
    fn forward(&self, x: &[f32], n: usize, out: &mut [f32]) {
        for r in 0..n {
            let xr = &x[r * self.fan_in..(r + 1) * self.fan_in];
            let outr = &mut out[r * self.fan_out..(r + 1) * self.fan_out];
            for o in 0..self.fan_out {
                let wrow = &self.w[o * self.fan_in..(o + 1) * self.fan_in];
                let mut acc = self.b[o];
                for i in 0..self.fan_in {
                    acc += wrow[i] * xr[i];
                }
                outr[o] = acc;
            }
        }
    }

    /// AdamW parameter update from accumulated grads. `t` is the 1-based step.
    fn adamw_step(&mut self, lr: f32, wd: f32, b1: f32, b2: f32, eps: f32, t: u64) {
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        upd(&mut self.w, &self.gw, &mut self.mw, &mut self.vw, lr, wd, b1, b2, eps, bc1, bc2);
        upd(&mut self.b, &self.gb, &mut self.mb, &mut self.vb, lr, wd, b1, b2, eps, bc1, bc2);
    }
}

#[allow(clippy::too_many_arguments)]
fn upd(
    p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32],
    lr: f32, wd: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32,
) {
    for i in 0..p.len() {
        // Decoupled weight decay (AdamW): shrink the parameter itself.
        p[i] *= 1.0 - lr * wd;
        let gi = g[i];
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        p[i] -= lr * mhat / (vhat.sqrt() + eps);
    }
}

/// Snapshot of metrics after an evaluation, laid out for the JS side.
#[wasm_bindgen]
pub struct Metrics {
    pub step: u32,
    pub train_acc: f32,
    pub val_acc: f32,
    pub train_loss: f32,
    pub val_loss: f32,
    pub weight_norm: f32,
}

#[wasm_bindgen]
pub struct Trainer {
    p: usize,
    hidden: usize,
    in_dim: usize,
    lr: f32,
    wd: f32,

    l1: Linear,
    l2: Linear,

    xtr: Vec<f32>,
    ytr: Vec<usize>,
    n_tr: usize,
    xva: Vec<f32>,
    yva: Vec<usize>,
    n_va: usize,

    step: u64,
}

#[wasm_bindgen]
impl Trainer {
    /// Build a fresh trainer. `nf` is the number of Fourier frequencies (ignored
    /// for one-hot).
    #[wasm_bindgen(constructor)]
    pub fn new(
        p: usize,
        train_frac: f32,
        hidden: usize,
        lr: f32,
        weight_decay: f32,
        encoding: Encoding,
        nf: usize,
        seed: u32,
    ) -> Trainer {
        let mut rng = Rng::new(seed as u64);

        // Per-operand feature dimension for the chosen encoding.
        let per = match encoding {
            Encoding::OneHot => p,
            Encoding::Fourier => 2 * nf.max(1),
        };
        let in_dim = 2 * per;

        // Encode a single residue value into `dst` (length `per`).
        let encode_val = |v: usize, dst: &mut [f32]| match encoding {
            Encoding::OneHot => {
                for x in dst.iter_mut() {
                    *x = 0.0;
                }
                dst[v] = 1.0;
            }
            Encoding::Fourier => {
                let nf = nf.max(1);
                let scale = 1.0 / (nf as f32).sqrt(); // unit-norm features
                for k in 0..nf {
                    let ang = 2.0 * std::f32::consts::PI * ((k + 1) * v) as f32 / p as f32;
                    dst[2 * k] = ang.cos() * scale;
                    dst[2 * k + 1] = ang.sin() * scale;
                }
            }
        };

        // All (a, b) pairs, deterministic split.
        let total = p * p;
        let mut perm: Vec<usize> = (0..total).collect();
        rng.shuffle(&mut perm);
        let n_tr = ((train_frac * total as f32) as usize).clamp(1, total - 1);

        let build = |idxs: &[usize]| -> (Vec<f32>, Vec<usize>) {
            let n = idxs.len();
            let mut x = vec![0.0f32; n * in_dim];
            let mut y = vec![0usize; n];
            let mut abuf = vec![0.0f32; per];
            let mut bbuf = vec![0.0f32; per];
            for (r, &idx) in idxs.iter().enumerate() {
                let a = idx / p;
                let b = idx % p;
                encode_val(a, &mut abuf);
                encode_val(b, &mut bbuf);
                let row = &mut x[r * in_dim..(r + 1) * in_dim];
                row[..per].copy_from_slice(&abuf);
                row[per..].copy_from_slice(&bbuf);
                y[r] = (a + b) % p;
            }
            (x, y)
        };

        let (xtr, ytr) = build(&perm[..n_tr]);
        let (xva, yva) = build(&perm[n_tr..]);
        let n_va = yva.len();

        let l1 = Linear::new(in_dim, hidden, &mut rng);
        let l2 = Linear::new(hidden, p, &mut rng);

        Trainer {
            p,
            hidden,
            in_dim,
            lr,
            wd: weight_decay,
            l1,
            l2,
            xtr,
            ytr,
            n_tr,
            xva,
            yva,
            n_va,
            step: 0,
        }
    }

    /// Run one full-batch AdamW training step on the training set.
    fn train_step(&mut self) {
        let n = self.n_tr;
        let h = self.hidden;
        let p = self.p;

        // ---- forward ----
        let mut z1 = vec![0.0f32; n * h];
        self.l1.forward(&self.xtr, n, &mut z1);
        // a1 = relu(z1)
        let mut a1 = z1.clone();
        for a in a1.iter_mut() {
            if *a < 0.0 {
                *a = 0.0;
            }
        }
        let mut z2 = vec![0.0f32; n * p];
        self.l2.forward(&a1, n, &mut z2);

        // ---- softmax + cross-entropy grad wrt z2 ----
        // dz2 = (softmax(z2) - onehot(y)) / n
        let mut dz2 = vec![0.0f32; n * p];
        for r in 0..n {
            let row = &z2[r * p..(r + 1) * p];
            let mut mx = f32::NEG_INFINITY;
            for &v in row {
                if v > mx {
                    mx = v;
                }
            }
            let mut sum = 0.0f32;
            let drow = &mut dz2[r * p..(r + 1) * p];
            for c in 0..p {
                let e = (row[c] - mx).exp();
                drow[c] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for c in 0..p {
                drow[c] = drow[c] * inv / n as f32;
            }
            drow[self.ytr[r]] -= 1.0 / n as f32;
        }

        // ---- backward layer 2: gw2 = dz2^T @ a1, gb2 = sum dz2 ----
        for g in self.l2.gw.iter_mut() {
            *g = 0.0;
        }
        for g in self.l2.gb.iter_mut() {
            *g = 0.0;
        }
        // da1 = dz2 @ w2  -> [n * h]
        let mut da1 = vec![0.0f32; n * h];
        for r in 0..n {
            let drow = &dz2[r * p..(r + 1) * p];
            let a1row = &a1[r * h..(r + 1) * h];
            let da1row = &mut da1[r * h..(r + 1) * h];
            for o in 0..p {
                let d = drow[o];
                if d != 0.0 {
                    let wrow = &self.l2.w[o * h..(o + 1) * h];
                    let gwrow = &mut self.l2.gw[o * h..(o + 1) * h];
                    for k in 0..h {
                        gwrow[k] += d * a1row[k];
                        da1row[k] += d * wrow[k];
                    }
                }
                self.l2.gb[o] += d;
            }
        }

        // ---- backward relu: dz1 = da1 * (z1 > 0) ----
        let mut dz1 = da1;
        for k in 0..dz1.len() {
            if z1[k] <= 0.0 {
                dz1[k] = 0.0;
            }
        }

        // ---- backward layer 1: gw1 = dz1^T @ xtr, gb1 = sum dz1 ----
        for g in self.l1.gw.iter_mut() {
            *g = 0.0;
        }
        for g in self.l1.gb.iter_mut() {
            *g = 0.0;
        }
        let d = self.in_dim;
        for r in 0..n {
            let dz1row = &dz1[r * h..(r + 1) * h];
            let xrow = &self.xtr[r * d..(r + 1) * d];
            for o in 0..h {
                let g = dz1row[o];
                if g != 0.0 {
                    let gwrow = &mut self.l1.gw[o * d..(o + 1) * d];
                    for i in 0..d {
                        gwrow[i] += g * xrow[i];
                    }
                    self.l1.gb[o] += g;
                }
            }
        }

        // ---- AdamW update ----
        self.step += 1;
        let t = self.step;
        self.l1.adamw_step(self.lr, self.wd, 0.9, 0.98, 1e-8, t);
        self.l2.adamw_step(self.lr, self.wd, 0.9, 0.98, 1e-8, t);
    }

    /// Run `n` training steps.
    pub fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.train_step();
        }
    }

    /// Forward pass returning (accuracy, mean cross-entropy loss) on a dataset.
    fn eval(&self, x: &[f32], y: &[usize], n: usize) -> (f32, f32) {
        if n == 0 {
            return (0.0, 0.0);
        }
        let h = self.hidden;
        let p = self.p;
        let mut z1 = vec![0.0f32; n * h];
        self.l1.forward(x, n, &mut z1);
        for a in z1.iter_mut() {
            if *a < 0.0 {
                *a = 0.0;
            }
        }
        let mut z2 = vec![0.0f32; n * p];
        self.l2.forward(&z1, n, &mut z2);

        let mut correct = 0usize;
        let mut loss = 0.0f32;
        for r in 0..n {
            let row = &z2[r * p..(r + 1) * p];
            let mut mx = f32::NEG_INFINITY;
            let mut arg = 0usize;
            for (c, &v) in row.iter().enumerate() {
                if v > mx {
                    mx = v;
                    arg = c;
                }
            }
            if arg == y[r] {
                correct += 1;
            }
            let mut sum = 0.0f32;
            for &v in row {
                sum += (v - mx).exp();
            }
            // -log softmax at the true label
            loss += -((row[y[r]] - mx) - sum.ln());
        }
        (correct as f32 / n as f32, loss / n as f32)
    }

    /// Total L2 norm of the two weight matrices (the "excess norm" that weight
    /// decay drives down as the structural circuit precipitates).
    fn weight_norm(&self) -> f32 {
        let mut s = 0.0f32;
        for &w in self.l1.w.iter() {
            s += w * w;
        }
        for &w in self.l2.w.iter() {
            s += w * w;
        }
        s.sqrt()
    }

    /// Evaluate on train + val and return a metrics snapshot.
    pub fn metrics(&self) -> Metrics {
        let (train_acc, train_loss) = self.eval(&self.xtr, &self.ytr, self.n_tr);
        let (val_acc, val_loss) = self.eval(&self.xva, &self.yva, self.n_va);
        Metrics {
            step: self.step as u32,
            train_acc,
            val_acc,
            train_loss,
            val_loss,
            weight_norm: self.weight_norm(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn step_count(&self) -> u32 {
        self.step as u32
    }
    #[wasm_bindgen(getter)]
    pub fn n_train(&self) -> usize {
        self.n_tr
    }
    #[wasm_bindgen(getter)]
    pub fn n_val(&self) -> usize {
        self.n_va
    }
}
