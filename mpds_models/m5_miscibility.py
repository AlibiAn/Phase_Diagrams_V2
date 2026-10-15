"""
M5 — Liquid Miscibility Gap classifier.

Does the system show a liquid-liquid miscibility gap (monotectic behaviour)?
Only ~5-10 % of MPDS binaries do, so XGBoost uses scale_pos_weight to
compensate for the imbalance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .base import PhaseDataset, SpecialistModel, precision_recall_f1, xgb_device_kwargs

logger = logging.getLogger(__name__)


@dataclass
class MiscibilityGapConfig:
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    decision_threshold: float = 0.5


class MiscibilityGapDetector(SpecialistModel):
    name = "M5_miscibility"

    def __init__(self, config: MiscibilityGapConfig | None = None):
        self.cfg = config or MiscibilityGapConfig()
        self.clf: xgb.XGBClassifier | None = None

    def fit(self, ds: PhaseDataset) -> "MiscibilityGapDetector":
        idx = ds.train_idx
        y = ds.miscibility_gap[idx].astype(np.int32)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0:
            logger.warning("M5: training set contains 0 miscibility-gap examples — "
                           "model will trivially always predict False")
        scale_pos = max(n_neg / max(n_pos, 1), 1.0)
        self.clf = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            scale_pos_weight=scale_pos,
            objective="binary:logistic",
            eval_metric="aucpr",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        self.clf.fit(ds.X[idx], y, sample_weight=ds.weights[idx])
        logger.info("M5: trained on %d systems (%d positive miscibility-gap examples)",
                    len(y), n_pos)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.cfg.decision_threshold).astype(bool)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.miscibility_gap[idx]
        proba = self.predict_proba(ds.X[idx])
        y_pred = proba >= self.cfg.decision_threshold
        prec, rec, f1 = precision_recall_f1(y_true, y_pred)
        return {
            "split": split, "n": int(len(idx)),
            "accuracy": float((y_pred == y_true).mean()),
            "precision": prec, "recall": rec, "f1": f1,
            "mean_proba_pos": float(proba[y_true].mean()) if y_true.any() else float("nan"),
            "mean_proba_neg": float(proba[~y_true].mean()) if (~y_true).any() else float("nan"),
        }
