"""
M6 — Topology Classifier (5-class).

Predicts the overall topology of the binary phase diagram:
  isomorphous, simple_eutectic, eutectic_with_intermetallics,
  monotectic, complex.

This standalone version uses only the engineered features; the stacked
version that also consumes M1-M5 outputs lives in meta_learner.py.
Comparing the two measures what stacking adds: following Derennes et al.,
the stacked model should win by ~3-8 accuracy points. If it doesn't,
suspect the stacking pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .base import (PhaseDataset, SpecialistModel, pad_class_proba,
                   precision_recall_f1, xgb_device_kwargs)

logger = logging.getLogger(__name__)


@dataclass
class TopologyClassifierConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05


class TopologyClassifier(SpecialistModel):
    name = "M6_topology"

    def __init__(self, config: TopologyClassifierConfig | None = None):
        self.cfg = config or TopologyClassifierConfig()
        self.clf: xgb.XGBClassifier | None = None
        self.n_classes: int = 0

    def fit(self, ds: PhaseDataset) -> "TopologyClassifier":
        idx = ds.train_idx
        y = ds.topology[idx].astype(np.int32)
        self.n_classes = ds.n_topology_classes
        # XGBoost only learns the classes present in y; track them for padding
        self.seen_classes_ = np.unique(y)
        self.clf = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            objective="multi:softprob",
            num_class=int(self.seen_classes_.size),
            eval_metric="mlogloss",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        self.clf.fit(ds.X[idx], y, sample_weight=ds.weights[idx])
        logger.info("M6: trained on %d systems; saw %d/%d topology classes",
                    len(y), self.seen_classes_.size, self.n_classes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities over all n_classes (zero for classes unseen in training)."""
        return pad_class_proba(self.clf.predict_proba(X), self.seen_classes_, self.n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int32)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.topology[idx]
        y_pred = self.predict(ds.X[idx])

        per_class = {}
        for c, name in enumerate(ds.topology_classes):
            prec, rec, f1 = precision_recall_f1(y_true == c, y_pred == c)
            per_class[name] = {"support": int((y_true == c).sum()),
                               "precision": prec, "recall": rec, "f1": f1}
        return {
            "split": split, "n": int(len(idx)),
            "accuracy": float((y_pred == y_true).mean()),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class,
        }
