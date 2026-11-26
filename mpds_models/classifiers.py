"""
The scalar models, all XGBoost (settings of v6; extended features since v14):

  PhaseClassifier          which phases occur (one classifier per phase)
  EutecticDetector         does a eutectic exist, and at which composition and
                           temperature (the temperature regressor keeps the 207
                           processed columns: the extended ones made it worse)
  MiscibilityGapDetector   do the two liquids separate
  TopologyClassifier       diagram type from the features alone
  StackedTopologyClassifier
                           diagram type from the features plus out-of-fold
                           predictions of the other models (the one used)

XGBoost builds different trees on the CPU and the GPU; MPDS_DEVICE=cpu forces
the CPU (as for every version since v13), MPDS_XGB_THREADS sets its threads
(default 6: faster than 20 on ~1,200 rows, and the trees are the same).
"""
from __future__ import annotations

import logging
import os
from copy import copy

import numpy as np
import xgboost as xgb

from .curves import TreeCurves
from .data import PhaseDataset, precision_recall_f1

logger = logging.getLogger(__name__)


def xgb_device() -> dict:
    forced = os.environ.get("MPDS_DEVICE", "").strip().lower()
    if forced != "cpu":
        try:
            import torch
            if forced == "cuda" or torch.cuda.is_available():
                return {"device": "cuda", "tree_method": "hist"}
        except ImportError:
            pass
    return {"device": "cpu", "tree_method": "hist", "n_jobs": int(os.environ.get("MPDS_XGB_THREADS", 6))}


def _binary(n_estimators=1000, max_depth=5, learning_rate=0.02, scale_pos_weight=1.0, **kw):
    return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             scale_pos_weight=scale_pos_weight, objective="binary:logistic", eval_metric="aucpr",
                             verbosity=0, **kw)


def _multiclass(n_classes: int):
    return xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, objective="multi:softprob",
                             num_class=n_classes, eval_metric="mlogloss", verbosity=0, **xgb_device())


def _pos_weight(y) -> float:
    n_pos = int(np.sum(y))
    return max((len(y) - n_pos) / max(n_pos, 1), 1.0)


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights giving every class the same total weight (mean 1)."""
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return (len(y) / (len(counts) * counts))[inverse].astype(np.float32)


def tune_threshold(make_clf, X, y, w, n_folds: int = 5, seed: int = 0) -> float | None:
    """The decision threshold maximising F1 on out-of-fold training predictions
    (ties: the one closest to 0.5); None with too few positives."""
    from sklearn.model_selection import StratifiedKFold
    if min(int(y.sum()), int(len(y) - y.sum())) < n_folds:
        return None
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        oof[va] = make_clf().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[va])[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = np.array([precision_recall_f1(y, oof >= t)[2] for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


class PhaseClassifier:
    """One binary XGBoost per phase of the vocabulary; phases with fewer than 3
    training positives are always False, phases present everywhere always True."""

    def fit(self, ds: PhaseDataset) -> "PhaseClassifier":
        from joblib import Parallel, delayed
        X, Y, w = ds.X[ds.train_idx], ds.Y_phases[ds.train_idx], ds.weights[ds.train_idx]
        n_pos = Y.sum(axis=0)
        self.always_true = n_pos == len(Y)
        todo = [j for j in range(Y.shape[1]) if not self.always_true[j] and n_pos[j] >= 3]

        def fit_one(j):            # many small models: one CPU thread each, run side by side
            return _binary(400, 5, 0.05, _pos_weight(Y[:, j]), device="cpu", tree_method="hist", n_jobs=1).fit(
                X, Y[:, j].astype(np.int32), sample_weight=w)

        self.models = dict(zip(todo, Parallel(n_jobs=-1, backend="threading")(delayed(fit_one)(j) for j in todo)))
        self.n_phases = Y.shape[1]
        logger.info("phases: %d classifiers, %d always present", len(todo), int(self.always_true.sum()))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((len(X), self.n_phases), dtype=np.float32)
        P[:, self.always_true] = 1.0
        for j, m in self.models.items():
            P[:, j] = m.predict_proba(X)[:, 1]
        return P


class EutecticDetector:
    """Existence (threshold tuned for F1), composition (at% B) and temperature (K)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, ds: PhaseDataset) -> "EutecticDetector":
        idx = ds.train_idx
        X, w = ds.X[idx], ds.weights[idx]
        y = ds.eutectic_exists[idx].astype(np.int32)
        make = lambda: _binary(scale_pos_weight=_pos_weight(y), random_state=self.seed, **xgb_device())
        self.threshold = tune_threshold(make, X, y, w) or 0.5
        self.cls = make().fit(X, y, sample_weight=w)
        self.n_temp_features = ds.n_base_features
        self.reg = {}
        for name, target, Xr in (("comp", ds.eutectic_comp[idx], X), ("temp", ds.eutectic_temp[idx], X[:, :self.n_temp_features])):
            rows = ds.eutectic_exists[idx] & np.isfinite(target)
            self.reg[name] = xgb.XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.02,
                                              objective="reg:squarederror", eval_metric="mae", verbosity=0,
                                              random_state=self.seed, **xgb_device()).fit(
                Xr[rows], target[rows], sample_weight=w[rows])
        logger.info("eutectic: %d systems (%d with a eutectic), threshold %.2f", len(y), y.sum(), self.threshold)
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """exists, exists_proba, comp, temp (comp / temp NaN where no eutectic is predicted)."""
        proba = self.cls.predict_proba(X)[:, 1]
        exists = proba >= self.threshold
        comp = np.full(len(X), np.nan, dtype=np.float32)
        temp = np.full(len(X), np.nan, dtype=np.float32)
        if exists.any():
            comp[exists] = np.clip(self.reg["comp"].predict(X[exists]), 0.0, 100.0)
            temp[exists] = self.reg["temp"].predict(X[exists][:, :self.n_temp_features])
        return {"exists": exists, "exists_proba": proba, "comp": comp, "temp": temp}


class MiscibilityGapDetector:
    def fit(self, ds: PhaseDataset) -> "MiscibilityGapDetector":
        idx = ds.train_idx
        X, w, y = ds.X[idx], ds.weights[idx], ds.miscibility_gap[idx].astype(np.int32)
        make = lambda: _binary(scale_pos_weight=_pos_weight(y), **xgb_device())
        self.threshold = tune_threshold(make, X, y, w) or 0.35
        self.clf = make().fit(X, y, sample_weight=w)
        logger.info("miscibility gap: %d systems (%d with a gap), threshold %.2f", len(y), y.sum(), self.threshold)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class _Multiclass:
    """XGBoost over the diagram types seen in training, with balanced class weights."""

    def _fit(self, X, y, w, n_classes):
        self.n_classes, self.seen = n_classes, np.unique(y)
        self.clf = _multiclass(len(self.seen)).fit(X, np.searchsorted(self.seen, y),
                                                     sample_weight=w * balanced_class_weights(y))
        return self

    def _proba(self, X) -> np.ndarray:
        raw = self.clf.predict_proba(X)
        full = np.zeros((len(X), self.n_classes), dtype=raw.dtype)
        full[:, self.seen] = raw
        return full


class TopologyClassifier(_Multiclass):
    def fit(self, ds: PhaseDataset) -> "TopologyClassifier":
        idx = ds.train_idx
        return self._fit(ds.X[idx], ds.topology[idx].astype(np.int32), ds.weights[idx], len(ds.topology_classes))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)


def _curve_summary(C: np.ndarray) -> np.ndarray:
    """min, max, range, mean, the two end slopes and the sign-change roughness of each curve."""
    lo, hi = C.min(1), C.max(1)
    roughness = np.sign(np.diff(C, axis=1)).astype(np.int8).var(axis=1)
    return np.column_stack([lo, hi, hi - lo, C.mean(1), C[:, 1] - C[:, 0], C[:, -1] - C[:, -2],
                            np.trunc(roughness)]).astype(np.float32)


def stacked_features(X, phase_proba, liq, sol, eutectic: dict, gap_proba) -> np.ndarray:
    return np.hstack([X, _curve_summary(liq), _curve_summary(sol),
                      eutectic["exists_proba"].reshape(-1, 1),
                      np.nan_to_num(eutectic["comp"], nan=-1.0).reshape(-1, 1),
                      np.nan_to_num(eutectic["temp"], nan=-1.0).reshape(-1, 1),
                      gap_proba.reshape(-1, 1),
                      (phase_proba > 0.5).sum(axis=1, keepdims=True).astype(np.float32),
                      phase_proba.max(axis=1, keepdims=True), phase_proba.mean(axis=1, keepdims=True)])


class StackedTopologyClassifier(_Multiclass):
    """Diagram type from the features plus the predictions of the phase,
    curve (trees only), eutectic and gap models. Its training inputs are
    out of fold: 5 folds of the training systems, each predicted by models
    fitted on the other four."""

    def fit(self, ds: PhaseDataset) -> "StackedTopologyClassifier":
        from sklearn.model_selection import KFold
        tr = ds.train_idx
        parts = []
        oof = np.zeros(len(tr), dtype=int)
        rows = []
        for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
            sub = copy(ds)
            sub.train_idx, sub.val_idx, sub.test_idx = tr[a].astype(np.int64), np.array([], np.int64), np.array([], np.int64)
            logger.info("stacked type classifier: fold %d", f)
            Xv = ds.X[tr[b]]
            parts.append(self._inputs(Xv, PhaseClassifier().fit(sub), TreeCurves().fit(sub),
                                      EutecticDetector().fit(sub), MiscibilityGapDetector().fit(sub)))
            rows.append(b)
        order = np.concatenate(rows)
        meta_X = np.empty((len(tr), parts[0].shape[1]), dtype=parts[0].dtype)
        meta_X[order] = np.concatenate(parts)
        return self._fit(meta_X, ds.topology[tr].astype(np.int32), ds.weights[tr], len(ds.topology_classes))

    @staticmethod
    def _inputs(X, phases, trees, eutectic, gap) -> np.ndarray:
        return stacked_features(X, phases.predict_proba(X), *trees.predict(X), eutectic.predict(X), gap.predict_proba(X))

    def predict_proba(self, X: np.ndarray, phases, trees, eutectic, gap) -> np.ndarray:
        return self._proba(self._inputs(X, phases, trees, eutectic, gap))
