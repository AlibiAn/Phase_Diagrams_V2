"""
Gated curve model (M2 + M3): separate tree-ensemble experts for systems with
and without a liquid miscibility gap.

Why
---
In monotectic systems the recorded "liquidus" is the top of the two-liquid
dome, a flat monotectic line or a line cut off at the chart edge, and it can
run hundreds of K above both melting points. One forest averages those
curves together with ordinary eutectic curves, so it predicts in-between
shapes: domes too low, and bumps in systems that have none.

How
---
    gap expert    TreeCurveModel trained on the systems with a miscibility
                  gap (miscibility_gap target, or monotectic topology)
    rest expert   TreeCurveModel trained on all the others
    gate          ExtraTrees classifier: probability p that a system has a gap

    curve = p · gap expert + (1 - p) · rest expert          (gate="soft")
    curve = gap expert if p >= 0.5 else rest expert         (gate="hard")

The gate is fitted on natural class frequencies (no class balancing), so p is
an estimate of the actual chance of a gap, which is the weight that keeps
the expected error lowest when mixing the two. Both experts end at Tm and
keep the solidus below the liquidus, so the mix does too. With fewer than
`min_gap_systems` gap systems in training, a single TreeCurveModel is used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from .base import PhaseDataset, SpecialistModel, curve_metrics
from .tree_curves import TreeCurveConfig, TreeCurveModel

logger = logging.getLogger(__name__)


@dataclass
class GatedCurveConfig:
    curves: TreeCurveConfig = field(default_factory=TreeCurveConfig)   # both experts
    gate: str = "soft"                  # "soft" (mix by p) or "hard" (pick one expert)
    min_gap_systems: int = 20
    gate_n_estimators: int = 500
    seed: int = 0


def has_gap(ds: PhaseDataset, rows: np.ndarray) -> np.ndarray:
    """True for systems with a liquid miscibility gap (or monotectic topology)."""
    mono = np.array([ds.topology_name(i) == "monotectic" for i in rows], dtype=bool)
    return ds.miscibility_gap[rows].astype(bool) | mono


class GatedCurveModel(SpecialistModel):
    """Miscibility-gap expert + everything-else expert, mixed by a gate (module docstring)."""

    name = "M2_M3_gated_curves"

    def __init__(self, config: GatedCurveConfig | None = None):
        self.cfg = replace(config) if config else GatedCurveConfig()
        if self.cfg.gate not in ("soft", "hard"):
            raise ValueError("gate must be 'soft' or 'hard'")
        self.gap_expert: TreeCurveModel | None = None
        self.rest_expert: TreeCurveModel | None = None
        self.single: TreeCurveModel | None = None
        self.gate_model = None

    def fit(self, ds: PhaseDataset) -> "GatedCurveModel":
        from sklearn.ensemble import ExtraTreesClassifier
        tr = ds.train_idx
        gap = has_gap(ds, tr)
        if gap.sum() < self.cfg.min_gap_systems or (~gap).sum() < self.cfg.min_gap_systems:
            logger.warning("gated curves: only %d gap systems in training; using one model", gap.sum())
            self.single = TreeCurveModel(self.cfg.curves).fit(ds)
            return self
        self.gap_expert = TreeCurveModel(self.cfg.curves).fit(replace(ds, train_idx=tr[gap]))
        self.rest_expert = TreeCurveModel(self.cfg.curves).fit(replace(ds, train_idx=tr[~gap]))
        self.gate_model = ExtraTreesClassifier(
            n_estimators=self.cfg.gate_n_estimators, min_samples_leaf=2, n_jobs=-1,
            random_state=self.cfg.seed).fit(ds.X[tr], gap.astype(int), sample_weight=ds.weights[tr])
        logger.info("gated curves: gap expert from %d systems, rest expert from %d, %s gate",
                    gap.sum(), (~gap).sum(), self.cfg.gate)
        return self

    # ---- inference ---------------------------------------------------

    def gap_probability(self, X: np.ndarray) -> np.ndarray:
        """Gate weight per system: the probability of a gap (0 or 1 with gate="hard")."""
        p = self.gate_model.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(float) if self.cfg.gate == "hard" else p

    def predict(self, X: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.single is not None:
            return self.single.predict(X)
        if self.gate_model is None:
            raise RuntimeError("call fit() first")
        p = self.gap_probability(X)[:, None]
        lg, sg = self.gap_expert.predict(X)
        lr, sr = self.rest_expert.predict(X)
        return ((p * lg + (1 - p) * lr).astype(np.float32),
                (p * sg + (1 - p) * sr).astype(np.float32))

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int | None = None,
                                 clamp: bool = True):
        """Mean and std of the two-expert mixture: each expert's spread over its
        trees plus how far apart the experts are."""
        if self.single is not None:
            return self.single.predict_with_uncertainty(X)
        p = self.gap_probability(X)[:, None]
        g = self.gap_expert.predict_with_uncertainty(X)
        r = self.rest_expert.predict_with_uncertainty(X)
        out = []
        for k in (0, 2):                                   # liquidus, solidus
            mean = p * g[k] + (1 - p) * r[k]
            var = p * g[k + 1] ** 2 + (1 - p) * r[k + 1] ** 2 + p * (1 - p) * (g[k] - r[k]) ** 2
            out += [mean.astype(np.float32), np.sqrt(var).astype(np.float32)]
        return tuple(out)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        raw_liq, raw_sol = self.predict(ds.X[idx])
        return {"split": split, "n": int(len(idx)), **curve_metrics(ds, idx, raw_liq, raw_sol)}

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path, compress=3)
