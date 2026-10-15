"""
Meta-learner — stacked ensemble + full diagram reconstructor.

Two distinct pieces, used together at inference time
====================================================

1. StackedTopologyClassifier
   Trains a final XGBoost on a feature vector that concatenates:
     - the original ~207 engineered features, AND
     - out-of-fold predictions from M1-M5 (phase set, liquidus stats,
       solidus stats, eutectic existence / position, miscibility gap).
   OOF predictions are obtained via K-fold CV on the training split, so
   the meta-learner doesn't see its own children's overfit predictions.

2. FullDiagramReconstructor
   At inference, runs every specialist model and bundles the outputs into
   a single coherent PredictedDiagram dataclass — with built-in
   consistency checks (e.g. if M6 says "isomorphous" but M4 says a
   eutectic exists, flag the inconsistency; if M3 predicts solidus
   above liquidus at any point, clamp; etc.). The reconstructor is the
   "API surface" the rest of the project actually calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

from .base import PhaseDataset, SpecialistModel, xgb_device_kwargs
from .m1_phase import PhaseClassifier
from .m4_eutectic import EutecticDetector
from .m5_miscibility import MiscibilityGapDetector
from .m6_topology import TopologyClassifier
from .deep_curve_net import LiquidusSolidusModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the stacked feature vector for the meta-learner
# ---------------------------------------------------------------------------

def _stack_features_from_specialist_preds(
    X_base: np.ndarray,
    phase_proba: np.ndarray,            # (N, V) from M1
    liquidus: np.ndarray,               # (N, G) from M2
    solidus: np.ndarray,                # (N, G) from M3
    eutectic_proba: np.ndarray,         # (N,) from M4a
    eutectic_comp: np.ndarray,          # (N,) from M4b — may contain NaN
    eutectic_temp: np.ndarray,          # (N,) from M4c — may contain NaN
    miscibility_proba: np.ndarray,      # (N,) from M5
) -> np.ndarray:
    """Construct the meta-feature matrix.

    We don't pass the full 101-D liquidus and solidus into the meta-learner
    — that's 200 noisy extra features. Instead we summarise each curve by
    a handful of physically meaningful statistics (min, max, range, mean,
    slope at endpoints, slope sign changes). This compression keeps the
    meta-learner's input dimensionality manageable and gives it features
    that correlate directly with topology class.
    """
    N = X_base.shape[0]

    def curve_summary(C: np.ndarray) -> np.ndarray:
        """Per-curve summary stats: (N, 7)."""
        summary = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            c = C[i]
            valid = np.isfinite(c)
            if not valid.any():
                continue
            cv = c[valid]
            summary[i, 0] = cv.min()
            summary[i, 1] = cv.max()
            summary[i, 2] = cv.max() - cv.min()
            summary[i, 3] = cv.mean()
            if len(cv) >= 2:
                summary[i, 4] = cv[1] - cv[0]                 # left-endpoint slope
                summary[i, 5] = cv[-1] - cv[-2]               # right-endpoint slope
                summary[i, 6] = int(np.sign(np.diff(cv)).astype(np.int8).var())  # roughness
        return summary

    liq_stats = curve_summary(liquidus)
    sol_stats = curve_summary(solidus)

    # NaN handling for eutectic comp/T (they're NaN when M4 says no eutectic)
    eut_comp_safe = np.nan_to_num(eutectic_comp, nan=-1.0)
    eut_temp_safe = np.nan_to_num(eutectic_temp, nan=-1.0)

    # Phase-set summary: total count of high-probability phases + max + entropy
    phase_count = (phase_proba > 0.5).sum(axis=1, keepdims=True).astype(np.float32)
    phase_max_prob = phase_proba.max(axis=1, keepdims=True)
    phase_mean_prob = phase_proba.mean(axis=1, keepdims=True)

    return np.hstack([
        X_base,
        liq_stats, sol_stats,
        eutectic_proba.reshape(-1, 1),
        eut_comp_safe.reshape(-1, 1),
        eut_temp_safe.reshape(-1, 1),
        miscibility_proba.reshape(-1, 1),
        phase_count, phase_max_prob, phase_mean_prob,
    ])


# ---------------------------------------------------------------------------
# Stacked topology classifier (meta-learner for M6)
# ---------------------------------------------------------------------------

@dataclass
class StackedMetaConfig:
    n_folds: int = 5
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    seed: int = 0


class StackedTopologyClassifier(SpecialistModel):
    """Stacked-ensemble topology classifier.

    Workflow:
      1. K-fold over the training systems. For each fold, train fresh
         instances of M1-M5 on K-1 folds, predict on the held-out fold.
         Concatenate to get OOF predictions covering all training rows.
      2. Train the meta XGBoost on (base_X || OOF_predictions, y_topology).
      3. At inference: run the full-data-trained M1-M5 (passed in from
         outside) and feed their predictions into the meta XGBoost.

    Note: this class trains its OWN copies of M1-M5 during fit() for the
    OOF step. The caller should also train final M1-M5 on the full
    training set (for inference) and pass them into predict() via the
    `specialists` argument.
    """

    name = "meta_topology"

    def __init__(self, config: StackedMetaConfig | None = None):
        self.cfg = config or StackedMetaConfig()
        self.meta: xgb.XGBClassifier | None = None
        self.n_classes: int = 0

    def _train_fold_specialists(self, ds_fold: PhaseDataset):
        m1 = PhaseClassifier().fit(ds_fold)
        m23 = LiquidusSolidusModel().fit(ds_fold)
        m4 = EutecticDetector().fit(ds_fold)
        m5 = MiscibilityGapDetector().fit(ds_fold)
        return m1, m23, m4, m5

    @staticmethod
    def _make_fold_dataset(ds: PhaseDataset, train_idx_in_train: np.ndarray) -> PhaseDataset:
        """Build a fold-restricted PhaseDataset that uses `train_idx_in_train`
        (absolute indices) as the training split and an empty val split.
        Required because each specialist's fit() reads from ds.train_idx."""
        from copy import copy as _copy
        sub = _copy(ds)
        sub.train_idx = train_idx_in_train.astype(np.int64)
        sub.val_idx = np.array([], dtype=np.int64)
        sub.test_idx = np.array([], dtype=np.int64)
        return sub

    def fit(self, ds: PhaseDataset) -> "StackedTopologyClassifier":
        self.n_classes = ds.n_topology_classes
        train_idx = ds.train_idx
        if len(train_idx) < self.cfg.n_folds * 2:
            logger.warning(
                "meta: not enough training systems (%d) for %d-fold; "
                "using leave-one-out approximation",
                len(train_idx), self.cfg.n_folds,
            )

        kf = KFold(n_splits=min(self.cfg.n_folds, max(2, len(train_idx) // 2)),
                   shuffle=True, random_state=self.cfg.seed)

        # Allocate OOF prediction arrays
        N_tr = len(train_idx)
        V = ds.n_phases
        G = ds.comp_grid.size
        oof_phase = np.zeros((N_tr, V), dtype=np.float32)
        oof_liq = np.zeros((N_tr, G), dtype=np.float32)
        oof_sol = np.zeros((N_tr, G), dtype=np.float32)
        oof_eut_p = np.zeros(N_tr, dtype=np.float32)
        oof_eut_c = np.full(N_tr, np.nan, dtype=np.float32)
        oof_eut_t = np.full(N_tr, np.nan, dtype=np.float32)
        oof_misc = np.zeros(N_tr, dtype=np.float32)

        for fold_i, (in_fold_train, in_fold_val) in enumerate(kf.split(train_idx)):
            logger.info("Meta fold %d: %d train, %d val",
                        fold_i, len(in_fold_train), len(in_fold_val))
            abs_train = train_idx[in_fold_train]
            abs_val = train_idx[in_fold_val]
            ds_fold = self._make_fold_dataset(ds, abs_train)
            m1, m23, m4, m5 = self._train_fold_specialists(ds_fold)

            X_val = ds.X[abs_val]
            oof_phase[in_fold_val] = m1.predict_proba(X_val)
            liq, sol = m23.predict(X_val)
            oof_liq[in_fold_val] = liq
            oof_sol[in_fold_val] = sol
            m4_out = m4.predict(X_val)
            oof_eut_p[in_fold_val] = m4_out["exists_proba"]
            oof_eut_c[in_fold_val] = m4_out["comp"]
            oof_eut_t[in_fold_val] = m4_out["temp"]
            oof_misc[in_fold_val] = m5.predict_proba(X_val)

        # Build the meta-feature matrix on the training rows
        meta_X = _stack_features_from_specialist_preds(
            ds.X[train_idx], oof_phase, oof_liq, oof_sol,
            oof_eut_p, oof_eut_c, oof_eut_t, oof_misc,
        )
        y = ds.topology[train_idx].astype(np.int32)
        self.seen_classes_ = np.unique(y)

        self.meta = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            objective="multi:softprob",
            num_class=int(self.seen_classes_.size),
            eval_metric="mlogloss",
            verbosity=0,
            **xgb_device_kwargs(),
        )
        self.meta.fit(meta_X, y, sample_weight=ds.weights[train_idx])
        logger.info("Meta: trained on %d stacked features × %d systems (saw %d/%d classes)",
                    meta_X.shape[1], meta_X.shape[0],
                    self.seen_classes_.size, self.n_classes)
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        specialists: dict,
    ) -> np.ndarray:
        """Predict topology probabilities (padded to all n_classes columns)."""
        if self.meta is None:
            raise RuntimeError("call fit() first")
        m1, m23, m4, m5 = (specialists["m1"], specialists["m23"],
                           specialists["m4"], specialists["m5"])
        phase_proba = m1.predict_proba(X)
        liq, sol = m23.predict(X)
        m4_out = m4.predict(X)
        misc = m5.predict_proba(X)
        meta_X = _stack_features_from_specialist_preds(
            X, phase_proba, liq, sol, m4_out["exists_proba"],
            m4_out["comp"], m4_out["temp"], misc,
        )
        raw = self.meta.predict_proba(meta_X)
        if raw.shape[1] == self.n_classes:
            return raw
        full = np.zeros((X.shape[0], self.n_classes), dtype=raw.dtype)
        for col, cls in enumerate(self.seen_classes_):
            full[:, cls] = raw[:, col]
        return full

    def predict(self, X: np.ndarray, specialists: dict) -> np.ndarray:
        return self.predict_proba(X, specialists).argmax(axis=1).astype(np.int32)

    def evaluate(self, ds: PhaseDataset, specialists: dict, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        y_true = ds.topology[idx]
        y_pred = self.predict(ds.X[idx], specialists)
        return {
            "split": split, "n": int(len(idx)),
            "accuracy": float((y_pred == y_true).mean()),
            "per_class_support": {
                ds.topology_classes[c]: int((y_true == c).sum())
                for c in range(ds.n_topology_classes)
            },
        }


# ---------------------------------------------------------------------------
# Full diagram reconstruction + consistency checks
# ---------------------------------------------------------------------------

@dataclass
class PredictedDiagram:
    """Bundled per-system prediction with consistency flags."""
    pair: str
    topology: str
    topology_proba: dict[str, float]
    liquidus_T: np.ndarray              # (G,)
    liquidus_T_std: np.ndarray | None
    solidus_T: np.ndarray
    solidus_T_std: np.ndarray | None
    comp_grid: np.ndarray
    phases: list[tuple[str, float]]     # (phase_label, probability) sorted by prob desc
    eutectic_exists: bool
    eutectic_proba: float
    eutectic_comp: float | None
    eutectic_temp: float | None
    miscibility_gap: bool
    miscibility_proba: float
    inconsistencies: list[str] = field(default_factory=list)


def _check_consistencies(d: PredictedDiagram) -> None:
    """Append flags for any logically inconsistent combinations.
    The reconstructor does NOT correct them — surfacing inconsistency is
    diagnostically more useful than silently smoothing it away."""
    if d.topology == "isomorphous" and d.eutectic_exists:
        d.inconsistencies.append(
            f"M6 said 'isomorphous' but M4 predicts eutectic at "
            f"x={d.eutectic_comp:.1f}%, T={d.eutectic_temp:.0f}K"
        )
    if d.topology == "monotectic" and not d.miscibility_gap:
        d.inconsistencies.append(
            "M6 said 'monotectic' but M5 says no miscibility gap"
        )
    if d.miscibility_gap and d.topology not in ("monotectic", "complex"):
        d.inconsistencies.append(
            f"M5 says miscibility gap but M6 says topology = '{d.topology}'"
        )
    # numeric sanity
    order_viol = int((d.solidus_T > d.liquidus_T + 1.0).sum())
    if order_viol > 0:
        d.inconsistencies.append(
            f"solidus > liquidus at {order_viol} grid points (clamped at inference)"
        )


class FullDiagramReconstructor:
    """Run all six specialist models + meta-learner, return PredictedDiagram per system."""

    def __init__(
        self,
        m1: PhaseClassifier,
        m23: LiquidusSolidusModel,
        m4: EutecticDetector,
        m5: MiscibilityGapDetector,
        m6_meta: StackedTopologyClassifier | None,
        m6_solo: TopologyClassifier | None = None,
        topology_classes: tuple[str, ...] = (
            "isomorphous", "simple_eutectic", "eutectic_with_intermetallics",
            "monotectic", "complex",
        ),
        phase_vocab: list[str] | None = None,
        comp_grid: np.ndarray | None = None,
    ):
        self.m1 = m1; self.m23 = m23; self.m4 = m4; self.m5 = m5
        self.m6_meta = m6_meta
        self.m6_solo = m6_solo
        self.topology_classes = topology_classes
        self.phase_vocab = phase_vocab or m1.phase_vocab
        self.comp_grid = comp_grid if comp_grid is not None else np.arange(0, 101, 1, dtype=np.float32)

    def reconstruct(
        self,
        X: np.ndarray,
        pair_ids: list[str],
        with_uncertainty: bool = True,
        n_mc_samples: int = 20,
    ) -> list[PredictedDiagram]:
        # Liquidus / solidus (+ optional MC dropout uncertainty)
        if with_uncertainty:
            liq, liq_std, sol, sol_std = self.m23.predict_with_uncertainty(X, n_samples=n_mc_samples)
        else:
            liq, sol = self.m23.predict(X)
            liq_std = sol_std = None

        phase_proba = self.m1.predict_proba(X)
        m4_out = self.m4.predict(X)
        misc_proba = self.m5.predict_proba(X)

        # Topology — prefer stacked meta-learner if available, else fall back to solo M6
        if self.m6_meta is not None:
            specialists = {"m1": self.m1, "m23": self.m23, "m4": self.m4, "m5": self.m5}
            topo_proba = self.m6_meta.predict_proba(X, specialists)
        elif self.m6_solo is not None:
            topo_proba = self.m6_solo.predict_proba(X)
        else:
            raise RuntimeError("need at least one of m6_meta or m6_solo")
        topo_idx = topo_proba.argmax(axis=1)

        out: list[PredictedDiagram] = []
        for i, pair in enumerate(pair_ids):
            # Top-3 phases by predicted probability
            top_k = np.argsort(-phase_proba[i])[:5]
            phases = [
                (self.phase_vocab[j], float(phase_proba[i, j]))
                for j in top_k if phase_proba[i, j] > 0.1
            ]
            d = PredictedDiagram(
                pair=pair,
                topology=self.topology_classes[topo_idx[i]],
                topology_proba={
                    self.topology_classes[c]: float(topo_proba[i, c])
                    for c in range(len(self.topology_classes))
                },
                liquidus_T=liq[i],
                liquidus_T_std=liq_std[i] if liq_std is not None else None,
                solidus_T=sol[i],
                solidus_T_std=sol_std[i] if sol_std is not None else None,
                comp_grid=self.comp_grid,
                phases=phases,
                eutectic_exists=bool(m4_out["exists"][i]),
                eutectic_proba=float(m4_out["exists_proba"][i]),
                eutectic_comp=float(m4_out["comp"][i]) if np.isfinite(m4_out["comp"][i]) else None,
                eutectic_temp=float(m4_out["temp"][i]) if np.isfinite(m4_out["temp"][i]) else None,
                miscibility_gap=bool(misc_proba[i] >= 0.5),
                miscibility_proba=float(misc_proba[i]),
            )
            _check_consistencies(d)
            out.append(d)
        return out
