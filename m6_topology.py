"""
M6 — Topology Classifier (5-class).

Predicts the overall topology of the binary phase diagram:
  isomorphous, simple_eutectic, eutectic_with_intermetallics,
  monotectic, complex.

This is the STANDALONE version — it uses only the original engineered
features. The stacked-ensemble version that consumes M1-M5 outputs lives
in meta_learner.py.

Both versions exist because the stacked ensemble shouldn't be the only
way to predict topology — comparing the two tells you how much the
stacking step adds over a single XGBoost on the raw features. In the
Derennes et al. spirit, the stacked model should outperform the
standalone by 3-8 percentage points of accuracy. If it doesn't, that's
a signal something is off in the stacking pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .base import PhaseDataset, SpecialistModel, xgb_device_kwargs

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
        # XGBoost only learns classes it actually sees in y. Track them.
        self.seen_classes_ = np.unique(y)
        # XGBoost handles multiclass natively
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
        """Returns probabilities padded to ALL self.n_classes (zero for unseen)."""
        raw = self.clf.predict_proba(X)
        if raw.shape[1] == self.n_classes:
            return raw
        full = np.zeros((X.shape[0], self.n_classes), dtype=raw.dtype)
        for col, cls in enumerate(self.seen_classes_):
            full[:, cls] = raw[:, col]
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int32)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.topology[idx]
        y_pred = self.predict(ds.X[idx])

        # Per-class precision/recall
        per_class = {}
        for c, name in enumerate(ds.topology_classes):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            per_class[name] = {
                "support": int((y_true == c).sum()),
                "precision": prec, "recall": rec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            }
        return {
            "split": split, "n": int(len(idx)),
            "accuracy": float((y_pred == y_true).mean()),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "per_class": per_class,
        }
