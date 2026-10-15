"""
M4 — Eutectic Detector + Position Regressor.

Three sub-models sharing the same features:
  M4a : binary classifier — does a eutectic exist?            (XGBoost)
  M4b : regressor — composition of the eutectic               (XGBoost)
  M4c : regressor — temperature of the eutectic               (XGBoost)

M4b and M4c are trained ONLY on systems where a eutectic exists in
the training data (masked regression). At inference, M4b/M4c outputs
are valid only when M4a returns positive.

If multiple eutectics exist in one system, the preprocessing stage
returned the deepest one (lowest temperature) as the canonical target,
so that's what these regressors learn to recover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .base import PhaseDataset, SpecialistModel, xgb_device_kwargs

logger = logging.getLogger(__name__)


@dataclass
class EutecticDetectorConfig:
    cls_n_estimators: int = 400
    cls_max_depth: int = 5
    cls_learning_rate: float = 0.05
    reg_n_estimators: int = 400
    reg_max_depth: int = 5
    reg_learning_rate: float = 0.05
    decision_threshold: float = 0.5


class EutecticDetector(SpecialistModel):
    name = "M4_eutectic"

    def __init__(self, config: EutecticDetectorConfig | None = None):
        self.cfg = config or EutecticDetectorConfig()
        self.cls: xgb.XGBClassifier | None = None
        self.reg_comp: xgb.XGBRegressor | None = None
        self.reg_temp: xgb.XGBRegressor | None = None

    def fit(self, ds: PhaseDataset) -> "EutecticDetector":
        idx = ds.train_idx
        X = ds.X[idx]
        y_exists = ds.eutectic_exists[idx].astype(np.int32)
        w = ds.weights[idx]

        # M4a — classifier
        n_pos = int(y_exists.sum())
        n_neg = int(len(y_exists) - n_pos)
        scale_pos = max(n_neg / max(n_pos, 1), 1.0)
        self.cls = xgb.XGBClassifier(
            n_estimators=self.cfg.cls_n_estimators,
            max_depth=self.cfg.cls_max_depth,
            learning_rate=self.cfg.cls_learning_rate,
            scale_pos_weight=scale_pos,
            objective="binary:logistic",
            eval_metric="aucpr",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        self.cls.fit(X, y_exists, sample_weight=w)

        # M4b / M4c — regressors on the positive subset only
        pos_mask = ds.eutectic_exists[idx] & np.isfinite(ds.eutectic_comp[idx])
        X_pos = X[pos_mask]
        w_pos = w[pos_mask]
        comp_pos = ds.eutectic_comp[idx][pos_mask]
        temp_pos = ds.eutectic_temp[idx][pos_mask]

        if len(X_pos) < 5:
            logger.warning("M4: only %d positive training examples — regressors will be unreliable",
                           len(X_pos))

        common_reg_kwargs = dict(
            n_estimators=self.cfg.reg_n_estimators,
            max_depth=self.cfg.reg_max_depth,
            learning_rate=self.cfg.reg_learning_rate,
            objective="reg:squarederror",
            eval_metric="mae",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        if len(X_pos) >= 3:
            self.reg_comp = xgb.XGBRegressor(**common_reg_kwargs)
            self.reg_comp.fit(X_pos, comp_pos, sample_weight=w_pos)
            self.reg_temp = xgb.XGBRegressor(**common_reg_kwargs)
            self.reg_temp.fit(X_pos, temp_pos, sample_weight=w_pos)

        logger.info("M4: classifier trained on %d (%d positive); regressors trained on %d",
                    len(X), n_pos, len(X_pos))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cls.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Returns {'exists': bool, 'exists_proba': float, 'comp': float|NaN, 'temp': float|NaN}.

        comp and temp are NaN for systems predicted to have no eutectic.
        """
        proba = self.predict_proba(X)
        exists = proba >= self.cfg.decision_threshold
        comp = np.full(len(X), np.nan, dtype=np.float32)
        temp = np.full(len(X), np.nan, dtype=np.float32)
        if self.reg_comp is not None and exists.any():
            X_pos = X[exists]
            comp[exists] = self.reg_comp.predict(X_pos)
            temp[exists] = self.reg_temp.predict(X_pos)
            # clamp predicted comp to physically valid [0,100]
            comp = np.clip(comp, 0.0, 100.0)
        return {"exists": exists, "exists_proba": proba, "comp": comp, "temp": temp}

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        out = self.predict(ds.X[idx])
        y_exists = ds.eutectic_exists[idx]
        acc = float((out["exists"] == y_exists).mean())
        tp = int(((out["exists"] == 1) & (y_exists == 1)).sum())
        fp = int(((out["exists"] == 1) & (y_exists == 0)).sum())
        fn = int(((out["exists"] == 0) & (y_exists == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        # regression metrics on systems where both true and predicted say "exists"
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
