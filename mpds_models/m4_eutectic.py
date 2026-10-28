"""
M4 — Eutectic Detector + Position Regressor.

Three tree-ensemble sub-models on the same features (XGBoost by default;
`backend` swaps in a random forest or extra trees):
  M4a : classifier — does a eutectic exist?
  M4b : regressor  — eutectic composition (at% B)
  M4c : regressor  — eutectic temperature (K)

M4b/M4c are trained only on systems that have a eutectic and are reported
only where M4a predicts one. For systems with several eutectics the
preprocessing keeps the deepest (lowest-temperature) one as the target.

With `comp_from_lower_tm`, M4b learns the composition measured from the
lower-melting element (c if Tm(A) <= Tm(B), else 100 - c) and converts its
prediction back. Which element is A is just the order of the pair id, while
a eutectic tends to sit towards the lower-melting element, so this lets
every system teach the same pattern instead of two mirror-image ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .base import (TREE_BACKENDS, PhaseDataset, SpecialistModel, make_tree_classifier,
                   make_tree_regressor, precision_recall_f1, tune_threshold)

logger = logging.getLogger(__name__)


@dataclass
class EutecticDetectorConfig:
    # "xgboost" (v6), "random_forest" or "extra_trees"; the forests grow full
    # trees, so they use only the n_estimators settings below
    backend: str = "xgboost"
    # Settings of the v6 models
    cls_n_estimators: int = 1000
    cls_max_depth: int = 5
    cls_learning_rate: float = 0.02
    reg_n_estimators: int = 1000
    reg_max_depth: int = 5
    reg_learning_rate: float = 0.02
    decision_threshold: float = 0.5      # used as-is unless tuning succeeds
    # Pick the M4a threshold maximising F1 on out-of-fold training predictions
    tune_threshold: bool = True
    # M4b predicts the composition measured from the lower-melting element
    comp_from_lower_tm: bool = False
    seed: int = 0


def comp_from_lower_tm(comp: np.ndarray, tm_a: np.ndarray, tm_b: np.ndarray) -> np.ndarray:
    """at% B <-> at% measured from the lower-melting element (its own inverse)."""
    return np.where(tm_a <= tm_b, comp, 100.0 - comp)


class EutecticDetector(SpecialistModel):
    name = "M4_eutectic"

    def __init__(self, config: EutecticDetectorConfig | None = None):
        # Copy: fit() may overwrite decision_threshold
        self.cfg = replace(config) if config else EutecticDetectorConfig()
        if self.cfg.backend not in TREE_BACKENDS:
            raise ValueError(f"backend must be one of {TREE_BACKENDS}")
        self.cls = None
        self.reg_comp = None
        self.reg_temp = None
        self._tm_cols: tuple[int, int] | None = None

    def fit(self, ds: PhaseDataset) -> "EutecticDetector":
        idx = ds.train_idx
        X = ds.X[idx]
        y_exists = ds.eutectic_exists[idx].astype(np.int32)
        w = ds.weights[idx]

        # M4a: classifier
        n_pos = int(y_exists.sum())
        n_neg = int(len(y_exists) - n_pos)
        scale_pos = max(n_neg / max(n_pos, 1), 1.0)

        backend, seed = self.cfg.backend, self.cfg.seed

        def make_cls():
            return make_tree_classifier(
                backend, n_estimators=self.cfg.cls_n_estimators,
                max_depth=self.cfg.cls_max_depth, learning_rate=self.cfg.cls_learning_rate,
                scale_pos_weight=scale_pos, seed=seed)

        if getattr(self.cfg, "tune_threshold", False):
            t = tune_threshold(make_cls, X, y_exists, w)
            if t is not None:
                self.cfg.decision_threshold = t
        self.cls = make_cls()
        self.cls.fit(X, y_exists, sample_weight=w)

        # M4b / M4c: regressors on eutectic systems with a known target value
        comp = ds.eutectic_comp[idx]
        if self.cfg.comp_from_lower_tm:
            names = list(ds.feature_names)
            self._tm_cols = (names.index("phys__melting_point__A"),
                             names.index("phys__melting_point__B"))
            comp = comp_from_lower_tm(comp, *self._melting_points(X))
        n_reg = {}
        for attr, target in (("reg_comp", comp), ("reg_temp", ds.eutectic_temp[idx])):
            rows = ds.eutectic_exists[idx] & np.isfinite(target)
            n_reg[attr] = int(rows.sum())
            if n_reg[attr] < 5:
                logger.warning("M4: only %d training examples for %s — it will be unreliable",
                               n_reg[attr], attr)
            if n_reg[attr] >= 3:
                reg = make_tree_regressor(
                    backend, n_estimators=self.cfg.reg_n_estimators,
                    max_depth=self.cfg.reg_max_depth,
                    learning_rate=self.cfg.reg_learning_rate, seed=seed)
                reg.fit(X[rows], target[rows], sample_weight=w[rows])
                setattr(self, attr, reg)

        logger.info("M4 (%s): classifier trained on %d (%d positive, threshold %.2f); "
                    "regressors on %d (comp) / %d (temp)", backend, len(X), n_pos,
                    self.cfg.decision_threshold, n_reg["reg_comp"], n_reg["reg_temp"])
        return self

    def _melting_points(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[:, self._tm_cols[0]], X[:, self._tm_cols[1]]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cls.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Returns arrays {'exists', 'exists_proba', 'comp', 'temp'};
        comp and temp are NaN where no eutectic is predicted."""
        proba = self.predict_proba(X)
        exists = proba >= self.cfg.decision_threshold
        comp = np.full(len(X), np.nan, dtype=np.float32)
        temp = np.full(len(X), np.nan, dtype=np.float32)
        if exists.any():
            if self.reg_comp is not None:
                c = np.clip(self.reg_comp.predict(X[exists]), 0.0, 100.0)
                if self.cfg.comp_from_lower_tm:
                    c = comp_from_lower_tm(c, *self._melting_points(X[exists]))
                comp[exists] = c
            if self.reg_temp is not None:
                temp[exists] = self.reg_temp.predict(X[exists])
        return {"exists": exists, "exists_proba": proba, "comp": comp, "temp": temp}

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        out = self.predict(ds.X[idx])
        y_exists = ds.eutectic_exists[idx]
        acc = float((out["exists"] == y_exists).mean())
        prec, rec, f1 = precision_recall_f1(y_exists, out["exists"])

        # Position errors where both truth and prediction have a eutectic
        both_pos = y_exists & out["exists"]
        if both_pos.sum() > 0:
            comp_err = np.abs(out["comp"][both_pos] - ds.eutectic_comp[idx][both_pos])
            temp_err = np.abs(out["temp"][both_pos] - ds.eutectic_temp[idx][both_pos])
            comp_mae = float(np.nanmean(comp_err))
            temp_mae = float(np.nanmean(temp_err))
        else:
            comp_mae = temp_mae = float("nan")

        return {
            "split": split, "n": int(len(idx)),
            "exists_accuracy": acc, "exists_precision": prec,
            "exists_recall": rec, "exists_f1": f1,
            "comp_MAE_at_pct": comp_mae,
            "temp_MAE_K": temp_mae,
        }
