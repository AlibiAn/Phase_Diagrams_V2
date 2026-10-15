"""
M1 — Phase Classifier (multi-label).

Predicts which phases exist anywhere in the binary system. Output is a
multi-hot vector over the global phase vocabulary built during
preprocessing (typically a few hundred entries after rare-phase pruning).

Why one binary XGBoost per phase rather than one multi-label deep net
--------------------------------------------------------------------
Phase occurrence is severely imbalanced: 'Liquid' appears in every system,
a given intermetallic in perhaps 1-2. Independent classifiers get their own
`scale_pos_weight`, which gives much better recall on the rare intermetallics
that matter most for topology reconstruction.

Phases with fewer than `min_positive_count` training positives get no model
and are always predicted False: a handful of positives carries almost no
signal and a model trained on it memorises noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from joblib import Parallel, delayed

from .base import PhaseDataset, SpecialistModel, precision_recall_f1, xgb_device_kwargs

logger = logging.getLogger(__name__)


@dataclass
class PhaseClassifierConfig:
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    min_positive_count: int = 3      # below this, phase is always predicted False
    decision_threshold: float = 0.5
    # Hundreds of small models: training many at once on one CPU thread each
    # beats giving each model all cores (or paying GPU overhead per model).
    n_parallel: int = -1             # models trained at once; -1 = all CPU cores
    device: str = "cpu"              # "cpu" (fastest for these tiny models) or "auto"


class PhaseClassifier(SpecialistModel):
    """M1 — multi-label phase occurrence."""

    name = "M1_phases"

    def __init__(self, config: PhaseClassifierConfig | None = None):
        self.cfg = config or PhaseClassifierConfig()
        # One classifier per phase, or None for always-true / always-false phases
        self.models: list[xgb.XGBClassifier | None] = []
        self.phase_vocab: list[str] = []
        self.always_false_mask: np.ndarray = np.array([], dtype=bool)
        self.always_true_mask: np.ndarray = np.array([], dtype=bool)

    def fit(self, ds: PhaseDataset) -> "PhaseClassifier":
        self.phase_vocab = list(ds.phase_vocab)
        X_tr = ds.X[ds.train_idx]
        Y_tr = ds.Y_phases[ds.train_idx]
        w_tr = ds.weights[ds.train_idx]

        V = Y_tr.shape[1]
        self.models = [None] * V
        self.always_false_mask = np.zeros(V, dtype=bool)
        self.always_true_mask = np.zeros(V, dtype=bool)

        todo: list[tuple[int, float]] = []
        for j in range(V):
            y = Y_tr[:, j]
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos
            if n_neg == 0:                   # present everywhere, e.g. 'Liquid'
                self.always_true_mask[j] = True
                continue
            if n_pos < self.cfg.min_positive_count:
                self.always_false_mask[j] = True
                continue
            todo.append((j, max(n_neg / max(n_pos, 1), 1.0)))

        # getattr: configs pickled before these fields existed lack them
        if getattr(self.cfg, "device", "cpu") == "cpu":
            dev = {"device": "cpu", "tree_method": "hist", "n_jobs": 1}
        else:
            dev = xgb_device_kwargs()
            if dev["device"] == "cpu":
                dev["n_jobs"] = 1

        def fit_one(j: int, scale_pos: float):
            clf = xgb.XGBClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate,
                scale_pos_weight=scale_pos,
                objective="binary:logistic",
                eval_metric="aucpr",
                verbosity=0,
                **dev,
            )
            clf.fit(X_tr, Y_tr[:, j].astype(np.int32), sample_weight=w_tr)
            return j, clf

        n_par = getattr(self.cfg, "n_parallel", -1) if dev["device"] == "cpu" else 1
        # Threads suffice: XGBoost releases the GIL while training, and threads
        # share X_tr instead of copying it to every worker.
        results = Parallel(n_jobs=n_par, backend="threading")(
            delayed(fit_one)(j, s) for j, s in todo
        )
        for j, clf in results:
            self.models[j] = clf

        logger.info("M1: trained %d/%d phase classifiers "
                    "(%d always-false for sparsity, %d always-true for ubiquity)",
                    len(results), V, int(self.always_false_mask.sum()),
                    int(self.always_true_mask.sum()))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-phase positive-class probability. Shape (N, V)."""
        P = np.zeros((X.shape[0], len(self.models)), dtype=np.float32)
        P[:, self.always_true_mask] = 1.0            # always-false phases stay 0
        for j, m in enumerate(self.models):
            if m is not None:
                P[:, j] = m.predict_proba(X)[:, 1]
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.cfg.decision_threshold).astype(np.int8)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.Y_phases[idx]
        y_pred = self.predict(ds.X[idx])
        # Micro-averaged over every (system, phase) cell
        precision, recall, f1 = precision_recall_f1(y_true, y_pred)
        return {
            "split": split, "n": int(len(idx)),
            "micro_precision": float(precision),
            "micro_recall": float(recall),
            "micro_f1": float(f1),
            # System level: every phase correct
            "exact_match_accuracy": float((y_pred == y_true).all(axis=1).mean()),
            "n_phases_predicted_active": int((y_pred.sum(0) > 0).sum()),
        }