"""
Whole-curve correction of the liquidus with tabular foundation models (v18).

The ExtraTrees curve model averages the curves of similar training systems.
TabPFN v2 and TabICL v2 learn a different, smoother function of the inputs; in
v17 they were used for one number per system (the mean height of the curve).
Here they predict the leading principal components of the whole deviation
curve (liquidus minus the line between the two melting points), which
describe 99 % of its variation with 6 components. The final liquidus is

    liquidus = (1 - blend) * v17_liquidus + blend * foundation_liquidus

where v17_liquidus is the tree curve after the height correction
(height_correction.py). Both curves end exactly at the two melting points, so
the blend does too. The solidus is the tree model's, capped at the liquidus.
In 5-fold CV on processed_v8 (explore_v18.py): v17 91.7 K, blend 0.5 with
6 components 89.7 K, better in every fold.

As in height_correction.py the foundation models learn in context: fitting
stores the training rows, and the networks are fitted again when the model
first predicts. Licences: TabPFN v2 Prior Labs Licence (Apache 2.0 +
"Built with TabPFN"), TabICL v2 BSD-3.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .base import PhaseDataset, SpecialistModel, curve_metrics, swap_features, swap_rules
from .height_correction import (FOUNDATION_MODELS, HeightCorrectedCurves, _device, fitted_models, make_regressor,
                                reset_store, save_fitted)
from .tree_curves import TreeCurveModel, _fill_curves

logger = logging.getLogger(__name__)
KNOTS = np.arange(0, 101, 2)


class FoundationCurves(SpecialistModel):
    """Tree curves -> height correction -> blend with a foundation-model curve."""

    name = "M2_M3_foundation_curves"

    def __init__(self, base: TreeCurveModel, height_strength: float = 0.5, blend: float = 0.5,
                 n_components: int = 6, models: tuple[str, ...] = FOUNDATION_MODELS, seed: int = 0):
        if not 0.0 < blend <= 1.0:
            raise ValueError("blend must be in (0, 1]")
        self.height = HeightCorrectedCurves(base, strength=height_strength, models=models, seed=seed)
        self.blend, self.n_components, self.models, self.seed = float(blend), int(n_components), tuple(models), seed
        self._X = self._S = self._pca = None
        self._fitted: list | None = None
        self._store = None            # folder of the fitted foundation models, set by save()

    @property
    def base(self) -> TreeCurveModel:          # the stacked type classifier reads the uncorrected curves
        return self.height.base

    @property
    def cfg(self):
        return self.base.cfg

    def fit(self, ds: PhaseDataset) -> "FoundationCurves":
        from sklearn.decomposition import PCA
        self.height.fit(ds)
        b = self.base
        idx = np.setdiff1d(ds.train_idx, ds.extra_idx) if ds.extra_idx.size else ds.train_idx
        frac = b._comp_grid.astype(np.float64) / 100.0
        tm_a = ds.X[:, b._melt_a_idx].astype(np.float64)
        tm_b = ds.X[:, b._melt_b_idx].astype(np.float64)
        line = (1 - frac)[None] * tm_a[:, None] + frac[None] * tm_b[:, None]
        seen = ds.T_liquidus_mask.copy()
        seen[:, [0, -1]] = False
        dev = _fill_curves(ds.T_liquidus, seen, frac, tm_a, tm_b, 3) - line
        idx = idx[np.isfinite(dev[idx]).all(axis=1) & (tm_a[idx] > 0) & (tm_b[idx] > 0)]
        Y = np.concatenate([dev[idx][:, KNOTS], dev[idx][:, ::-1][:, KNOTS]])
        self._pca = PCA(n_components=self.n_components).fit(Y)
        X_swap = ds.X_swap if ds.X_swap.size else swap_features(ds.X, list(ds.feature_names), swap_rules(ds))
        self._X = np.concatenate([ds.X[idx], X_swap[idx]]).astype(np.float32)
        self._S = self._pca.transform(Y).astype(np.float32)
        self._fitted = None
        logger.info("foundation curves: %d components (%.3f of the variance), blend %.2f, %d training systems",
                    self.n_components, self._pca.explained_variance_ratio_.sum(), self.blend, len(idx))
        return self

    def _ensure_fitted(self) -> list:
        if self._fitted is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k = len(self.models)
                flat = fitted_models(getattr(self, "_store", None),
                                     [f"component{c}_{m}" for c in range(self.n_components) for m in self.models],
                                     lambda i: make_regressor(self.models[i % k], self.seed).fit(self._X, self._S[:, i // k]))
                self._fitted = [flat[c * k:(c + 1) * k] for c in range(self.n_components)]
            logger.info("foundation curves: fitted %d x %d models on %s", self.n_components, len(self.models), _device())
        return self._fitted

    def foundation_liquidus(self, X: np.ndarray) -> np.ndarray:
        X32 = np.asarray(X, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = np.column_stack([np.mean([m.predict(X32) for m in comp], axis=0)
                                      for comp in self._ensure_fitted()])
        at_knots = self._pca.inverse_transform(scores)
        grid = np.arange(101)
        dev = np.stack([np.interp(grid, KNOTS, r) for r in at_knots])
        return self.base._base_line(X) + dev

    def _combine(self, X, liq, sol):
        liq = (1 - self.blend) * liq + self.blend * self.foundation_liquidus(X)
        return liq.astype(np.float32), np.minimum(sol, liq).astype(np.float32)

    def predict(self, X: np.ndarray, clamp: bool = True):
        return self._combine(X, *self.height.predict(X))

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int | None = None, clamp: bool = True):
        liq, liq_std, sol, sol_std = self.height.predict_with_uncertainty(X, n_samples, clamp)
        liq, sol = self._combine(X, liq, sol)
        return liq, liq_std, sol, sol_std

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        liq, sol = self.predict(ds.X[idx])
        return {"split": split, "n": int(len(idx)), **curve_metrics(ds, idx, liq, sol)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fitted"] = None
        return state

    def save(self, path) -> None:
        """The wrapper to `path`, every fitted foundation model next to it in
        <path stem>_foundation/ (fitted now if they are not yet)."""
        import joblib
        from pathlib import Path
        store = str(Path(path).with_name(Path(path).stem + "_foundation"))
        self._store = self.height._store = store
        names = [f"component{c}_{m}" for c in range(self.n_components) for m in self.models]
        flat = [m for comp in self._fitted for m in comp] if self._fitted else None
        reset_store(store, names, flat)
        if self.height._fitted:
            for m, fitted in zip(self.height.models, self.height._fitted):
                save_fitted(fitted, Path(store) / f"height_{m}")
        self.height._ensure_fitted()
        self._ensure_fitted()
        joblib.dump(self, path, compress=3)
