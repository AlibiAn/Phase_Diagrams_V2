"""
M4 — Eutectic Detector + Position Regressor.

Three XGBoost sub-models on the same features:
  M4a : classifier — does a eutectic exist?
  M4b : regressor  — eutectic composition (at% B)
  M4c : regressor  — eutectic temperature (K)

M4b/M4c are trained only on systems that have a eutectic and are reported
only where M4a predicts one. For systems with several eutectics the
preprocessing keeps the deepest (lowest-temperature) one as the target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import xgboost as xgb

from .base import (PhaseDataset, SpecialistModel, precision_recall_f1, tune_threshold,
                   xgb_device_kwargs)

logger = logging.getLogger(__name__)


@dataclass
class EutecticDetectorConfig:
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


class EutecticDetector(SpecialistModel):
    name = "M4_eutectic"

    def __init__(self, config: EutecticDetectorConfig | None = None):
        # Copy: fit() may overwrite decision_threshold
        self.cfg = replace(config) if config else EutecticDetectorConfig()
        self.cls: xgb.XGBClassifier | None = None
        self.reg_comp: xgb.XGBRegressor | None = None
        self.reg_temp: xgb.XGBRegressor | None = None

    def fit(self, ds: PhaseDataset) -> "EutecticDetector":
        idx = ds.train_idx
        X = ds.X[idx]
        y_exists = ds.eutectic_exists[idx].astype(np.int32)
        w = ds.weights[idx]

        # M4a: classifier
        n_pos = int(y_exists.sum())
        n_neg = int(len(y_exists) - n_pos)
        scale_pos = max(n_neg / max(n_pos, 1), 1.0)

        def make_cls():
            return xgb.XGBClassifier(
                n_estimators=self.cfg.cls_n_estimators,
                max_depth=self.cfg.cls_max_depth,
                learning_rate=self.cfg.cls_learning_rate,
                scale_pos_weight=scale_pos,
                objective="binary:logistic",
                eval_metric="aucpr",
                verbosity=0,
                **xgb_device_kwargs(),
            )

        if getattr(self.cfg, "tune_threshold", False):
            t = tune_threshold(make_cls, X, y_exists, w)
            if t is not None:
                self.cfg.decision_threshold = t
        self.cls = make_cls()
        self.cls.fit(X, y_exists, sample_weight=w)

        # M4b / M4c: regressors on eutectic systems with a known target value
        common_reg_kwargs = dict(
            n_estimators=self.cfg.reg_n_estimators,
            max_depth=self.cfg.reg_max_depth,
            learning_rate=self.cfg.reg_learning_rate,
            objective="reg:squarederror",
            eval_metric="mae",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        n_reg = {}
        for attr, target in (("reg_comp", ds.eutectic_comp[idx]), ("reg_temp", ds.eutectic_temp[idx])):
            rows = ds.eutectic_exists[idx] & np.isfinite(target)
            n_reg[attr] = int(rows.sum())
            if n_reg[attr] < 5:
                logger.warning("M4: only %d training examples for %s — it will be unreliable",
                               n_reg[attr], attr)
            if n_reg[attr] >= 3:
                reg = xgb.XGBRegressor(**common_reg_kwargs)
                reg.fit(X[rows], target[rows], sample_weight=w[rows])
                setattr(self, attr, reg)

        logger.info("M4: classifier trained on %d (%d positive, threshold %.2f); "
                    "regressors on %d (comp) / %d (temp)", len(X), n_pos,
                    self.cfg.decision_threshold, n_reg["reg_comp"], n_reg["reg_temp"])
        return self

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
                comp[exists] = np.clip(self.reg_comp.predict(X[exists]), 0.0, 100.0)
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
