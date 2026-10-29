"""
Meta-learner: stacked topology classifier + full diagram reconstructor.

1. StackedTopologyClassifier
   A final XGBoost on the engineered features concatenated with out-of-fold
   (OOF) M1-M5 predictions: phase set, liquidus / solidus summaries, eutectic
   existence and position, miscibility gap. The OOF predictions come from
   K-fold CV on the training split, so the meta-learner never trains on
   predictions its specialists made for their own training rows.

2. FullDiagramReconstructor
   Runs every specialist and bundles the outputs into one PredictedDiagram
   per system, flagging contradictions between models (e.g. M6 says
   "isomorphous" while M4 predicts a eutectic). This is the inference API the
   rest of the project calls.
"""
from __future__ import annotations

import inspect
import logging
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

from .base import (TOPOLOGY_CLASSES, PhaseDataset, SpecialistModel, balanced_class_weights,
                   pad_class_proba, xgb_device_kwargs)
from .m1_phase import PhaseClassifier
from .m4_eutectic import EutecticDetector, EutecticDetectorConfig
from .m5_miscibility import MiscibilityGapDetector
from .m6_topology import TopologyClassifier
from .deep_curve_net import CurveNetConfig, LiquidusSolidusModel
from .tree_curves import TreeCurveConfig, TreeCurveModel
from .curve_ensemble import CurveEnsemble, CurveEnsembleConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the stacked feature vector for the meta-learner
# ---------------------------------------------------------------------------

def _curve_summary(C: np.ndarray) -> np.ndarray:
    """Seven per-curve statistics, (N, G) -> (N, 7): min, max, range, mean,
    left / right endpoint slope, and sign-change roughness. NaN points are
    ignored; an all-NaN curve gives zeros."""
    summary = np.zeros((C.shape[0], 7), dtype=np.float32)

    def stats(cv: np.ndarray, axis: int) -> list:
        lo, hi = cv.min(axis), cv.max(axis)
        roughness = np.sign(np.diff(cv, axis=axis)).astype(np.int8).var(axis=axis)
        return [lo, hi, hi - lo, cv.mean(axis),
                np.take(cv, 1, axis) - np.take(cv, 0, axis),
                np.take(cv, -1, axis) - np.take(cv, -2, axis),
                np.trunc(roughness)]

    # Fast path: complete curves (always the case for CurveNet output)
    full = np.isfinite(C).all(axis=1)
    if C.shape[1] >= 2 and full.any():
        summary[full] = np.column_stack(stats(C[full], axis=1))
    for i in np.flatnonzero(~full):
        cv = C[i][np.isfinite(C[i])]
        if cv.size >= 2:
            summary[i] = stats(cv, axis=0)
        elif cv.size == 1:
            summary[i, :4] = [cv[0], cv[0], 0.0, cv[0]]
    return summary


def _stack_features_from_specialist_preds(
    X_base: np.ndarray,
    phase_proba: np.ndarray,            # (N, V) from M1
    liquidus: np.ndarray,               # (N, G) from M2
    solidus: np.ndarray,                # (N, G) from M3
    eutectic_proba: np.ndarray,         # (N,) from M4a
    eutectic_comp: np.ndarray,          # (N,) from M4b, NaN where no eutectic
    eutectic_temp: np.ndarray,          # (N,) from M4c, NaN where no eutectic
    miscibility_proba: np.ndarray,      # (N,) from M5
) -> np.ndarray:
    """Build the meta-feature matrix.

    The 101-point curves are compressed into a few physically meaningful
    summary statistics rather than passed raw (200 noisy extra features).
    """
    phase_count = (phase_proba > 0.5).sum(axis=1, keepdims=True).astype(np.float32)
    return np.hstack([
        X_base,
        _curve_summary(liquidus), _curve_summary(solidus),
        eutectic_proba.reshape(-1, 1),
        np.nan_to_num(eutectic_comp, nan=-1.0).reshape(-1, 1),
        np.nan_to_num(eutectic_temp, nan=-1.0).reshape(-1, 1),
        miscibility_proba.reshape(-1, 1),
        phase_count,
        phase_proba.max(axis=1, keepdims=True),
        phase_proba.mean(axis=1, keepdims=True),
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
    balance_classes: bool = True        # see TopologyClassifierConfig


class StackedTopologyClassifier(SpecialistModel):
    """Stacked-ensemble topology classifier.

    1. K-fold over the training systems: fit fresh M1-M5 on K-1 folds and
       predict the held-out fold, giving OOF predictions for every row.
    2. Fit the meta XGBoost on [base X ∥ OOF features] -> topology.
    3. At inference, feed predictions from the final M1-M5 (trained on the
       full training split and passed in via `specialists`) to the meta model.

    The fold copies of M1-M5 are discarded after fit(); the caller must
    supply the final specialists separately. Pass the final curve and
    eutectic models' `curve_config` / `eutectic_config` so the fold models
    (and hence the meta-features the meta model learns from) match the
    predictions it sees at inference. The config's type picks the fold curve
    model: CurveNetConfig, TreeCurveConfig or CurveEnsembleConfig.
    """

    name = "meta_topology"

    def __init__(self, config: StackedMetaConfig | None = None,
                 curve_config: CurveNetConfig | TreeCurveConfig | CurveEnsembleConfig | None = None,
                 eutectic_config: EutecticDetectorConfig | None = None):
        self.cfg = config or StackedMetaConfig()
        self.curve_config = curve_config
        self.eutectic_config = eutectic_config
        self.meta: xgb.XGBClassifier | None = None
        self.n_classes: int = 0

    def _train_fold_specialists(self, ds_fold: PhaseDataset):
        m1 = PhaseClassifier().fit(ds_fold)
        # getattr: models pickled before curve_config / eutectic_config existed
        curve_cfg = getattr(self, "curve_config", None)
        curve_cls = {TreeCurveConfig: TreeCurveModel,
                     CurveEnsembleConfig: CurveEnsemble}.get(type(curve_cfg), LiquidusSolidusModel)
        m23 = curve_cls(curve_cfg).fit(ds_fold)
        m4 = EutecticDetector(getattr(self, "eutectic_config", None)).fit(ds_fold)
        m5 = MiscibilityGapDetector().fit(ds_fold)
        return m1, m23, m4, m5

    @staticmethod
    def _make_fold_dataset(ds: PhaseDataset, fold_train_idx: np.ndarray) -> PhaseDataset:
        """Shallow copy of `ds` whose train split is `fold_train_idx` (absolute
        row indices) with empty val / test splits; specialists fit on ds.train_idx."""
        sub = copy(ds)
        sub.train_idx = fold_train_idx.astype(np.int64)
        sub.val_idx = np.array([], dtype=np.int64)
        sub.test_idx = np.array([], dtype=np.int64)
        return sub

    def fit(self, ds: PhaseDataset) -> "StackedTopologyClassifier":
        self.n_classes = ds.n_topology_classes
        train_idx = ds.train_idx
        if len(train_idx) < self.cfg.n_folds * 2:
            logger.warning(
                "meta: only %d training systems for %d-fold CV; using fewer folds",
                len(train_idx), self.cfg.n_folds,
            )

        kf = KFold(n_splits=min(self.cfg.n_folds, max(2, len(train_idx) // 2)),
                   shuffle=True, random_state=self.cfg.seed)

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
        w = ds.weights[train_idx]
        if getattr(self.cfg, "balance_classes", False):
            w = w * balanced_class_weights(y)
        self.meta.fit(meta_X, np.searchsorted(self.seen_classes_, y), sample_weight=w)
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
        liq, sol = specialists["m23"].predict(X)
        m4_out = specialists["m4"].predict(X)
        meta_X = _stack_features_from_specialist_preds(
            X, specialists["m1"].predict_proba(X), liq, sol,
            m4_out["exists_proba"], m4_out["comp"], m4_out["temp"],
            specialists["m5"].predict_proba(X),
        )
        return pad_class_proba(self.meta.predict_proba(meta_X), self.seen_classes_, self.n_classes)

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
    phases: list[tuple[str, float]]     # (phase, probability), most probable first
    eutectic_exists: bool
    eutectic_proba: float
    eutectic_comp: float | None
    eutectic_temp: float | None
    miscibility_gap: bool
    miscibility_proba: float
    n_order_violations: int = 0         # grid points with raw solidus > liquidus
    inconsistencies: list[str] = field(default_factory=list)


def _check_consistencies(d: PredictedDiagram) -> None:
    """Flag logically inconsistent model outputs. They are reported, not
    corrected: surfacing a contradiction is more useful than hiding it."""
    if d.topology == "isomorphous" and d.eutectic_exists:
        where = []
        if d.eutectic_comp is not None:
            where.append(f"x={d.eutectic_comp:.1f}%")
        if d.eutectic_temp is not None:
            where.append(f"T={d.eutectic_temp:.0f}K")
        d.inconsistencies.append(
            "M6 said 'isomorphous' but M4 predicts eutectic"
            + (f" at {', '.join(where)}" if where else "")
        )
    if d.topology == "monotectic" and not d.miscibility_gap:
        d.inconsistencies.append(
            "M6 said 'monotectic' but M5 says no miscibility gap"
        )
    if d.miscibility_gap and d.topology not in ("monotectic", "complex"):
        d.inconsistencies.append(
            f"M5 says miscibility gap but M6 says topology = '{d.topology}'"
        )
    if d.n_order_violations > 0:
        d.inconsistencies.append(
            f"solidus > liquidus at {d.n_order_violations} grid points (clamped at inference)"
        )


class FullDiagramReconstructor:
    """Runs every specialist + the meta-learner; returns one PredictedDiagram per system."""

    def __init__(
        self,
        m1: PhaseClassifier,
        m23: LiquidusSolidusModel | TreeCurveModel | CurveEnsemble,
        m4: EutecticDetector,
        m5: MiscibilityGapDetector,
        m6_meta: StackedTopologyClassifier | None,
        m6_solo: TopologyClassifier | None = None,
        topology_classes: tuple[str, ...] = TOPOLOGY_CLASSES,
        phase_vocab: list[str] | None = None,
        comp_grid: np.ndarray | None = None,
    ):
        self.m1, self.m23, self.m4, self.m5 = m1, m23, m4, m5
        self.m6_meta = m6_meta
        self.m6_solo = m6_solo
        self.topology_classes = topology_classes
        self.phase_vocab = phase_vocab or m1.phase_vocab
        self.comp_grid = comp_grid if comp_grid is not None else np.arange(0, 101, 1, dtype=np.float32)

    @classmethod
    def from_dir(cls, model_dir: str | Path, ds: PhaseDataset | None = None) -> "FullDiagramReconstructor":
        """Load the models saved by train_all from `model_dir`.

        The stacked M6 is optional (solo M6 is used without it). The curve
        model is the ensemble if m23_ensemble.json is present, else
        m23_curves.joblib (tree ensemble) if present, else m23_curves.pt
        (CurveNet). Passing `ds` takes the phase vocabulary, composition grid
        and topology classes from it.
        """
        import joblib
        try:
            from .curves_io import load_curves_model
        except ImportError:
            load_curves_model = LiquidusSolidusModel.load

        p = Path(model_dir)
        stacked = p / "m6_topology_stacked.joblib"
        trees = p / "m23_curves.joblib"
        if (p / "m23_ensemble.json").exists():
            m23 = CurveEnsemble.load(p)
        elif trees.exists():
            m23 = joblib.load(trees)
        else:
            m23 = load_curves_model(p / "m23_curves.pt")
        recon = cls(
            m1=joblib.load(p / "m1_phase.joblib"),
            m23=m23,
            m4=joblib.load(p / "m4_eutectic.joblib"),
            m5=joblib.load(p / "m5_miscibility.joblib"),
            m6_meta=joblib.load(stacked) if stacked.exists() else None,
            m6_solo=joblib.load(p / "m6_topology_solo.joblib"),
        )
        if ds is not None:
            recon.phase_vocab = ds.phase_vocab
            recon.comp_grid = ds.comp_grid
            recon.topology_classes = ds.topology_classes
        return recon

    def reconstruct(
        self,
        X: np.ndarray,
        pair_ids: list[str],
        with_uncertainty: bool = True,
        n_mc_samples: int = 20,
    ) -> list[PredictedDiagram]:
        liq_std = sol_std = None
        if with_uncertainty:
            try:
                liq, liq_std, sol, sol_std = self.m23.predict_with_uncertainty(X, n_samples=n_mc_samples)
            except NotImplementedError as e:            # e.g. XGBoost tree curves
                logger.warning("no curve uncertainty: %s", e)
                with_uncertainty = False
        if not with_uncertainty:
            liq, sol = self.m23.predict(X)

        # Order violations come from the raw (unclamped) curves; a curve model
        # without a `clamp` option (e.g. from curves_io) reports none
        n_viol = np.zeros(len(X), dtype=int)
        if "clamp" in inspect.signature(self.m23.predict).parameters:
            raw_liq, raw_sol = self.m23.predict(X, clamp=False)
            n_viol = (raw_sol > raw_liq + 1.0).sum(axis=1)

        phase_proba = self.m1.predict_proba(X)
        m4_out = self.m4.predict(X)
        misc_proba = self.m5.predict_proba(X)
        misc_threshold = getattr(getattr(self.m5, "cfg", None), "decision_threshold", 0.5)

        # Topology: stacked meta-learner if available, else solo M6
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
            # Up to 5 most probable phases above 10 %
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
                miscibility_gap=bool(misc_proba[i] >= misc_threshold),
                miscibility_proba=float(misc_proba[i]),
                n_order_violations=int(n_viol[i]),
            )
            _check_consistencies(d)
            out.append(d)
        return out
