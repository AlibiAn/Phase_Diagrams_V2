"""
M1 — Phase Classifier (multi-label).

Predicts which phases exist anywhere in the binary system. Output is a
multi-hot vector over the global phase vocabulary built during
preprocessing (typically a few hundred entries after rare-phase pruning).

Why one-binary-XGBoost-per-phase rather than a single multi-label deep net
-------------------------------------------------------------------------
Phase occurrence is severely class-imbalanced: 'Liquid' appears in 100%
of systems, while a specific intermetallic like 'Ag0.75Al0.25 ht' appears
in maybe 1-2 systems. A single deep model with sigmoid heads handles
this poorly without careful per-output reweighting. Independent XGBoost
classifiers per phase let us tune `scale_pos_weight` per phase, which
empirically gives much better recall on the rare intermetallics that
matter most for diagram-topology reconstruction.

For very rare phases (positive count < 3 in training set), we don't fit
a per-phase model — we just always predict `False`. This is a deliberate
choice: a one-shot positive example carries near-zero signal and a
trained model on such data tends to memorise its noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .base import PhaseDataset, SpecialistModel, xgb_device_kwargs

logger = logging.getLogger(__name__)


@dataclass
class PhaseClassifierConfig:
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    min_positive_count: int = 3      # below this, phase is always predicted False
    decision_threshold: float = 0.5
    # Patched — speed. M1 trains one small XGBoost per phase (hundreds). One
    # model on ~1,000 rows can't use many cores, so the old loop left most of
    # the CPU idle (and on a GPU machine paid GPU overhead per tiny model).
    # Now many phase models train at once, each on 1 CPU thread.
    n_parallel: int = -1             # models trained at once; -1 = all CPU cores
    device: str = "cpu"              # "cpu" (fastest for these tiny models) or "auto"


class PhaseClassifier(SpecialistModel):
    """M1 — multi-label phase occurrence."""

    name = "M1_phases"

    def __init__(self, config: PhaseClassifierConfig | None = None):
        self.cfg = config or PhaseClassifierConfig()
        # one classifier per phase ID, or None if always-false
        self.models: list[xgb.XGBClassifier | None] = []
        self.phase_vocab: list[str] = []
        self.always_false_mask: np.ndarray = np.array([], dtype=bool)
        # Patched: was only created inside fit(), so predict_proba on an
        # unfitted model failed with a confusing AttributeError.
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
            # Always-positive (like 'Liquid' which appears in every system)
            if n_neg == 0:
                self.always_true_mask[j] = True
                continue
            # Too rare to learn reliably
            if n_pos < self.cfg.min_positive_count:
                self.always_false_mask[j] = True
                continue
            todo.append((j, max(n_neg / max(n_pos, 1), 1.0)))

        device_choice = getattr(self.cfg, "device", "cpu")
        if device_choice == "cpu":
            dev = {"device": "cpu", "tree_method": "hist", "n_jobs": 1}
        else:
            dev = xgb_device_kwargs()
            if dev.get("device") == "cpu":
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

        n_par = getattr(self.cfg, "n_parallel", -1)
        if dev.get("device") != "cpu":
            n_par = 1                    # GPU: one model at a time
        # Threads are enough: XGBoost releases the GIL while training, and
        # threads share X_tr instead of copying it to every worker.
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_par, backend="threading")(
            delayed(fit_one)(j, s) for j, s in todo
        )
        for j, clf in results:
            self.models[j] = clf

        n_trained = sum(m is not None for m in self.models)
        n_false = int(self.always_false_mask.sum())
        n_true = int(self.always_true_mask.sum())
        logger.info("M1: trained %d/%d phase classifiers "
                    "(%d always-false for sparsity, %d always-true for ubiquity)",
                    n_trained, V, n_false, n_true)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-phase positive-class probability. Shape (N, V)."""
        N = X.shape[0]
        V = len(self.models)
        P = np.zeros((N, V), dtype=np.float32)
        for j, m in enumerate(self.models):
            if m is not None:
                P[:, j] = m.predict_proba(X)[:, 1]
            elif self.always_true_mask[j]:
                P[:, j] = 1.0
            # else always_false_mask -> leave as 0
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.cfg.decision_threshold).astype(np.int8)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.Y_phases[idx]
        y_pred = self.predict(ds.X[idx])
        # micro F1 across all phase × system pairs
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        # exact-match accuracy at the system level (all phases correct)
        exact = float(((y_pred == y_true).all(axis=1)).mean())
        return {
            "split": split, "n": int(len(idx)),
            "micro_precision": float(precision),
            "micro_recall": float(recall),
            "micro_f1": float(f1),
            "exact_match_accuracy": exact,
            "n_phases_predicted_active": int((y_pred.sum(0) > 0).sum()),
        }