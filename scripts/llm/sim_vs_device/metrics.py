# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Per-tensor divergence metrics and the localized-vs-shared-op verdict.

SQNR here is expressed in dB and computed via :func:`compute_psnr` with the
reference tensor's max-abs as the data range -- i.e. it *is* signal (reference)
over quantization/runtime noise (reference - actual), in dB. Rough intuition:
40 dB ~= 1% relative error, 20 dB ~= 10%, 10 dB ~= 32%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qai_hub_models.utils.compare import compute_max_abs_diff, compute_psnr


def kv_layer_index(name: str) -> int:
    """Decoder-layer index from a past_key_/past_value_ tensor name.

    e.g. 'past_key_13_out' -> 13. Returns -1 if not parseable (sorts first).
    """
    for tok in name.replace("/", "_").split("_"):
        if tok.isdigit():
            return int(tok)
    return -1


def sqnr_db(reference: np.ndarray, actual: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB (reference is golden)."""
    return float(compute_psnr(actual, reference))


def cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    """Cosine similarity over the flattened tensors.

    Catches directional error that a magnitude-based SQNR can mask (e.g. a
    tensor scaled or rotated but with similar energy).
    """
    a = reference.astype(np.float64).flatten()
    b = actual.astype(np.float64).flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class TensorMetric:
    """Divergence of one named tensor, device vs golden-sim."""

    name: str
    sqnr_db: float
    cosine: float
    max_abs_diff: float

    @classmethod
    def compare(
        cls, name: str, reference: np.ndarray, actual: np.ndarray
    ) -> TensorMetric:
        return cls(
            name=name,
            sqnr_db=sqnr_db(reference, actual),
            cosine=cosine_similarity(reference, actual),
            max_abs_diff=compute_max_abs_diff(reference, actual),
        )


@dataclass
class PartMetrics:
    """All tensor metrics for one split Part, averaged over samples.

    ``hidden`` is the residual-stream output that feeds the next Part -- the
    load-bearing localization signal. ``kv`` holds the free per-layer
    ``past_key_*`` / ``past_value_*`` taps (already graph outputs, no surgery).
    """

    part_name: str
    num_layers: int
    hidden: TensorMetric | None = None
    kv: list[TensorMetric] = field(default_factory=list)

    @property
    def hidden_sqnr(self) -> float:
        return self.hidden.sqnr_db if self.hidden is not None else float("nan")

    @property
    def per_layer_hidden_sqnr(self) -> float:
        """Hidden SQNR normalized by the layer count in this Part.

        A shared per-block op degrades every layer, so a Part with more layers
        accumulates more error; dividing by ``num_layers`` makes Parts of
        different sizes (e.g. 10/10/8) comparable so uniform depression is
        distinguishable from a genuine per-Part outlier.
        """
        if self.hidden is None or self.num_layers == 0:
            return float("nan")
        return self.hidden.sqnr_db / self.num_layers

    @property
    def min_kv_sqnr(self) -> float:
        return min((m.sqnr_db for m in self.kv), default=float("nan"))

    @property
    def mean_kv_sqnr(self) -> float:
        vals = [m.sqnr_db for m in self.kv]
        return float(np.mean(vals)) if vals else float("nan")


@dataclass
class Verdict:
    """The Phase-A conclusion: where and what kind of divergence."""

    kind: str  # "localized" | "shared_op" | "clean" | "inconclusive"
    detail: str
    suspect_parts: list[str]
    kv_path_implicated: bool


def summarize(
    parts: list[PartMetrics],
    good_sqnr_db: float = 25.0,
    outlier_margin_db: float = 12.0,
    kv_clean_margin_db: float = 8.0,
    kv_bad_db: float = 25.0,
    good_cosine: float = 0.99,
) -> Verdict:
    """Classify Phase-A results.

    Order matters: the KV path is checked FIRST, because a KV-cache divergence
    produces garbage *generation* (bad KV is re-read every decode step and
    compounds) even when the single-pass prefill *residual* still looks OK on
    SQNR. Ranking hidden-SQNR first would mislabel that as "clean".

    Verdicts:
    * ``kv_divergence`` -- a Part's per-layer KV taps drop below ``kv_bad_db``
      while its hidden output does not proportionally (KV is the weak spot).
      This is the prime suspect for generation garbage. Names the Part(s).
    * ``localized`` -- one Part's hidden SQNR sits ``outlier_margin_db`` below
      the peer median (a bug in that ~10-layer band).
    * ``shared_op`` -- all Parts uniformly depressed (an op every block).
    * ``clean`` -- hidden SQNR healthy AND cosine high AND KV healthy. Only then
      is the divergence genuinely elsewhere (decode loop / sampling / harness).

    ``cosine`` is weighed alongside SQNR: a Part can pass the SQNR gate yet have
    a low cosine (directional error the magnitude ratio misses), which is not
    "clean".
    """
    # Only transformer Parts carry a hidden-state transfer function. Part 1
    # (embedding) has no meaningful residual-output SQNR here.
    tparts = [p for p in parts if p.hidden is not None and p.num_layers > 0]
    if not tparts:
        return Verdict(
            "inconclusive", "No transformer Parts had a hidden tap.", [], False
        )

    hidden_sqnrs = {p.part_name: p.hidden_sqnr for p in tparts}
    median = float(np.median(list(hidden_sqnrs.values())))
    worst_hidden = min(hidden_sqnrs.values())
    worst_kv = min((p.min_kv_sqnr for p in tparts), default=float("nan"))

    # --- KV path first (the generation-garbage suspect) ---
    kv_bad_parts = [
        p.part_name
        for p in tparts
        if not np.isnan(p.min_kv_sqnr) and p.min_kv_sqnr < kv_bad_db
    ]
    # KV is the *weak spot* if it's materially worse than the hidden output --
    # i.e. the residual survived but the cache didn't.
    kv_is_weak_spot = (
        not np.isnan(worst_kv) and worst_kv < worst_hidden - kv_clean_margin_db
    )
    kv_implicated = bool(kv_bad_parts)

    if kv_bad_parts and kv_is_weak_spot:
        return Verdict(
            "kv_divergence",
            (
                f"KV cache diverges on {kv_bad_parts} (min KV "
                f"{worst_kv:.1f} dB vs hidden {worst_hidden:.1f} dB) -- the "
                "residual survives but the cache does not. This is the prime "
                "suspect for generation garbage (KV is re-read every decode "
                "step). Inspect the auto-expanded per-layer KV table for the "
                "onset layer, then check KV-cache quantization there."
            ),
            kv_bad_parts,
            True,
        )

    # --- hidden-state health: SQNR AND cosine ---
    def hidden_healthy(p: PartMetrics) -> bool:
        cos_ok = p.hidden is None or p.hidden.cosine >= good_cosine
        return p.hidden_sqnr >= good_sqnr_db and cos_ok

    if all(hidden_healthy(p) for p in tparts) and not kv_implicated:
        return Verdict(
            "clean",
            (
                f"All Parts track device: hidden >= {good_sqnr_db:.0f} dB "
                f"(min {worst_hidden:.1f} dB), cosine >= {good_cosine}, KV "
                "healthy. Prefill matches; look at the decode loop, sampling, "
                "or the eval harness."
            ),
            [],
            False,
        )

    # A Part that passes SQNR but fails cosine -> flag it as localized-ish.
    low_cos = [
        p.part_name
        for p in tparts
        if p.hidden is not None and p.hidden.cosine < good_cosine
    ]

    outliers = [
        name for name, v in hidden_sqnrs.items() if v < median - outlier_margin_db
    ]
    # A low-cosine Part is a hidden-side suspect even if its SQNR passed.
    outliers = list(dict.fromkeys(outliers + low_cos))
    spread = max(hidden_sqnrs.values()) - min(hidden_sqnrs.values())

    # NOTE: we deliberately do NOT infer which op family (attention vs MLP) is at
    # fault from the KV signal. Degraded KV can be the CAUSE (a KV/attention-side
    # issue) or merely an EFFECT (KV read from an already-diverged residual whose
    # real fault is downstream, e.g. down_proj). Phase A cannot tell these apart
    # -- that is exactly what the per-layer taps in Phase B are for. So we report
    # KV degradation as an observation only, without guessing the op.
    kv_note = (
        f" (KV outputs also diverge, min {worst_kv:.1f} dB -- could be the cause "
        "or a downstream effect; Phase B taps determine which op.)"
        if kv_implicated
        else ""
    )

    if outliers and (spread >= outlier_margin_db or low_cos):
        return Verdict(
            "localized",
            (
                f"Part(s) {outliers} diverge (peer median {median:.1f} dB; "
                f"flagged by SQNR outlier and/or cosine < {good_cosine}).{kv_note} "
                "Run Phase B (per-layer taps) on the flagged Part(s) to localize "
                "the op."
            ),
            outliers,
            kv_implicated,
        )

    # Low spread + depressed => uniform, i.e. a per-block shared op.
    return Verdict(
        "shared_op",
        (
            f"All transformer Parts uniformly depressed (spread {spread:.1f} dB, "
            f"min {worst_hidden:.1f} dB) -> an op repeated EVERY decoder "
            f"block.{kv_note} Phase B on any single Part will show the same "
            "per-layer cliff in all its layers."
        ),
        [p.part_name for p in tparts],
        kv_implicated,
    )
