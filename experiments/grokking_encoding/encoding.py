"""
Input encodings for the modular-addition grokking harness.

The whole experiment varies ONLY this file's output: how a residue r in [0, p)
is turned into an input vector. The task, model, optimizer, data split and seed
are held fixed across encodings so that any difference in *when* / *whether*
generalization arrives is attributable to the encoding (the "medium"), not to
anything else.

We deliberately keep two kinds of ambiguity distinct (this distinction is the
spine of the whole Gloxinia enquiry):

  (1) surface polysemy      -- same surface token, context-dependent meaning.
  (2) underdetermination    -- finite data consistent with both a memorizing
                               solution and a structural one.

None of the encodings below inject (1). `onehot` is surface-*unique*
(bijective symbol -> residue) yet structurally *opaque*: the cyclic group
Z/pZ is hidden behind an arbitrary orthonormal basis, so the network must
discover the structure. `fourier` is structurally *transparent*: nearby
residues map to nearby points on a circle, exposing the group structure in the
input geometry. `onehot_distractor` keeps onehot's structure but appends
irrelevant, per-example nuisance dimensions to inject (2).

Each encoding returns a table E of shape [p, d]; row r is the input vector for
residue r. A model input for the pair (a, b) is concat(E[a], E[b]), dim 2*d.
"""

import math
import torch

METRICS_ENCODING_VERSION = "enc-v1"


def onehot_encoding(p: int) -> torch.Tensor:
    """Canonical grokking encoding. d = p.

    Surface-unique (one token per residue, no polysemy) but structurally
    OPAQUE: the p residues sit on an arbitrary orthonormal basis, so the
    cyclic structure of addition is invisible in the geometry. This is the
    Power et al. 2022 / classic setup and is used to VERIFY that the harness
    reproduces delayed generalization before anything else is claimed.
    """
    return torch.eye(p)


def fourier_encoding(p: int, num_freqs: int = 4, normalize: bool = False) -> torch.Tensor:
    """Structurally TRANSPARENT encoding. d = 2 * num_freqs.

    Residue r -> [cos(2*pi*k*r/p), sin(2*pi*k*r/p) for k in 1..num_freqs].
    A single frequency already determines (a+b) mod p exactly, because the
    angle 2*pi*r/p wraps with the modulus; low frequencies keep the map smooth
    (nearby residues -> nearby points), so the group structure is exposed in
    the input geometry and interpolation is constrained.

    `normalize` scales each row to unit L2 norm. This matters because the raw
    rows have norm ~sqrt(num_freqs), so under a FIXED weight_decay the effective
    regularization/input scale drifts with num_freqs and differs from onehot
    (whose rows have unit norm). The v1 sweep left this UNCONTROLLED and so could
    not tell "transparency removes the step" apart from "mis-scaled regularizer
    broke the arm". `normalize=True` removes that confound; it is the v2 setting.

    CAVEAT (kept honest): a *complete* Fourier basis (num_freqs = (p-1)//2) is
    merely an orthogonal rotation of onehot and carries identical information,
    so it would NOT be transparent. Transparency comes from using FEW low
    frequencies. `num_freqs` is recorded with every run for this reason. (v1
    also showed the near-complete basis does NOT behave like onehot dynamically
    -- the MLP is basis-sensitive -- so "few low freqs" is what carries the
    transparency, not the information content.)
    """
    r = torch.arange(p, dtype=torch.float32)
    feats = []
    for k in range(1, num_freqs + 1):
        feats.append(torch.cos(2 * math.pi * k * r / p))
        feats.append(torch.sin(2 * math.pi * k * r / p))
    E = torch.stack(feats, dim=1)  # [p, 2*num_freqs]
    if normalize:
        E = E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return E


def onehot_distractor_encoding(
    p: int, num_distractor: int = 8, seed: int = 0
) -> torch.Tensor:
    """onehot + injected UNDERDETERMINATION (type (2)). d = p + num_distractor.

    Appends `num_distractor` fixed random dims to each residue's onehot row.
    These dims carry no information about (a+b) mod p, but because they are
    fixed per residue they give the network an alternative, spurious handle to
    fit the finite training set (a memorization affordance). This widens the
    gap between memorizing and structural solutions without touching the
    task's label function.

    NOTE: this manipulates the encoding, not the labels; it does not inject
    surface polysemy (type (1)). It is defined here for completeness; whether
    it is actually run is recorded in the results (empty cell == not run).
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.eye(p)
    distract = torch.randn(p, num_distractor, generator=g) * 0.5
    return torch.cat([base, distract], dim=1)


def build_encoding(name: str, p: int, **kwargs) -> torch.Tensor:
    if name == "onehot":
        return onehot_encoding(p)
    if name == "fourier":
        return fourier_encoding(p, num_freqs=kwargs.get("num_freqs", 4),
                                normalize=kwargs.get("fourier_normalize", False))
    if name == "onehot_distractor":
        return onehot_distractor_encoding(
            p,
            num_distractor=kwargs.get("num_distractor", 8),
            seed=kwargs.get("seed", 0),
        )
    raise ValueError(f"unknown encoding: {name}")
