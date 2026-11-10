"""
Curve ensemble (M2 + M3): a weighted average of the CurveNet MLP and a
tree-ensemble curve model.

The two make different mistakes (on the MPDS test split the trees were
better on intermetallic and isomorphous systems, CurveNet on simple
eutectics and monotectics), so their average is usually better than either:

    T(x) = w · T_trees(x) + (1 - w) · T_curvenet(x)

`weight` w is chosen on the validation split (the grid 0, 0.1, ..., 1 that
gives the lowest mean of liquidus and solidus MAE) unless it is set. Both
members keep the solidus at or below the liquidus, so their average does too.

Saved as the members' own files plus m23_ensemble.json with the weight:
m23_curves.pt (CurveNet), m23_curves.joblib (trees), m23_ensemble.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .base import PhaseDataset, SpecialistModel, curve_metrics
from .deep_curve_net import CurveNetConfig, LiquidusSolidusModel
from .tree_curves import TreeCurveConfig, TreeCurveModel

logger = logging.getLogger(__name__)

WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


@dataclass
class CurveEnsembleConfig:
    curvenet: CurveNetConfig = field(default_factory=CurveNetConfig)
    trees: TreeCurveConfig = field(default_factory=TreeCurveConfig)
    weight: float | None = None         # weight of the tree model; None = tune on val


class CurveEnsemble(SpecialistModel):
    """Weighted average of a CurveNet and a TreeCurveModel (see module docstring)."""

    name = "M2_M3_curve_ensemble"

    def __init__(self, config: CurveEnsembleConfig | None = None):
        self.cfg = replace(config) if config else CurveEnsembleConfig()
        self.curvenet = LiquidusSolidusModel(self.cfg.curvenet)
        self.trees = TreeCurveModel(self.cfg.trees)
        self.weight = 0.5 if self.cfg.weight is None else float(self.cfg.weight)

    def fit(self, ds: PhaseDataset) -> "CurveEnsemble":
        # A set weight of 0 or 1 leaves one member out entirely: skip training it
        if self.cfg.weight is None or self.weight < 1:
            self.curvenet.fit(ds)
        if self.cfg.weight is None or self.weight > 0:
            self.trees.fit(ds)
        if self.cfg.weight is None:
            self.weight = self._tune_weight(ds)
        logger.info("curve ensemble: tree weight %.1f", self.weight)
        return self

    def _tune_weight(self, ds: PhaseDataset) -> float:
        val = ds.val_idx
        if len(val) == 0:
            logger.warning("curve ensemble: no validation split, using weight 0.5")
            return 0.5
        nets, trees = self.curvenet.predict(ds.X[val]), self.trees.predict(ds.X[val])
        scores = []
        for w in WEIGHT_GRID:
            m = curve_metrics(ds, val, *self._mix(nets, trees, w))
            scores.append(np.nanmean([m["liquidus_MAE_K"], m["solidus_MAE_K"]]))
        logger.info("curve ensemble: val MAE by tree weight %s",
                    ", ".join(f"{w:.1f}: {s:.1f}" for w, s in zip(WEIGHT_GRID, scores)))
        return float(WEIGHT_GRID[int(np.nanargmin(scores))])

    @staticmethod
    def _mix(nets, trees, w: float):
        return (w * trees[0] + (1 - w) * nets[0], w * trees[1] + (1 - w) * nets[1])

    # ---- inference ---------------------------------------------------

    def predict(self, X: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if self.weight >= 1:
            return self.trees.predict(X)
        if self.weight <= 0:
            return self.curvenet.predict(X, clamp=clamp)
        liq, sol = self._mix(self.curvenet.predict(X, clamp=clamp), self.trees.predict(X), self.weight)
        return liq, (np.minimum(sol, liq) if clamp else sol)

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 30, clamp: bool = True):
        """Mean and std of the two-member mixture: each member's own spread
        plus how far apart the members are."""
        w = self.weight
        if w >= 1:
            return self.trees.predict_with_uncertainty(X, clamp=clamp)
        if w <= 0:
            return self.curvenet.predict_with_uncertainty(X, n_samples=n_samples, clamp=clamp)
        net = self.curvenet.predict_with_uncertainty(X, n_samples=n_samples, clamp=clamp)
        tree = self.trees.predict_with_uncertainty(X, clamp=clamp)
        out = []
        for k in (0, 2):                                   # liquidus, solidus
            m_n, s_n, m_t, s_t = net[k], net[k + 1], tree[k], tree[k + 1]
            mean = w * m_t + (1 - w) * m_n
            var = w * s_t ** 2 + (1 - w) * s_n ** 2 + w * (1 - w) * (m_t - m_n) ** 2
            out += [mean, np.sqrt(var)]
        return tuple(out)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        raw_liq, raw_sol = self.predict(ds.X[idx], clamp=False)
        return {"split": split, "n": int(len(idx)), "tree_weight": self.weight,
                **curve_metrics(ds, idx, raw_liq, raw_sol)}

    # ---- save / load (a folder: the members' files + the weight) -------

    def save(self, model_dir: str | Path) -> None:
        p = Path(model_dir)
        self.curvenet.save(p / "m23_curves.pt")
        self.trees.save(p / "m23_curves.joblib")
        (p / "m23_ensemble.json").write_text(json.dumps({"tree_weight": self.weight}))

    @classmethod
    def load(cls, model_dir: str | Path) -> "CurveEnsemble":
        import joblib
        p = Path(model_dir)
        m = cls.__new__(cls)
        m.curvenet = LiquidusSolidusModel.load(p / "m23_curves.pt")
        m.trees = joblib.load(p / "m23_curves.joblib")
        m.weight = float(json.loads((p / "m23_ensemble.json").read_text())["tree_weight"])
        m.cfg = CurveEnsembleConfig(m.curvenet.cfg, m.trees.cfg, m.weight)
        return m
