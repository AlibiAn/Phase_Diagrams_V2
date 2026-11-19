"""
Height correction of the predicted liquidus with tabular foundation models (v17).

About a third of the liquidus error is one number per system: the mean height
of the curve above the straight line between the two melting points. TabPFN v2
and TabICL v2, trained to predict that number alone, predict it better than the
curve model's own curve does (72.7 and 74.0 K against 76.1 K in 5-fold CV;
71.4 K averaged). HeightCorrectedCurves wraps a TreeCurveModel and moves each
predicted liquidus `strength` of the way to the averaged height:

    liquidus'(x) = liquidus(x) + strength * (h_pred - h_curve) * w(x),
    w(x) = 6 x (1 - x)  (mean 1 over the composition range, 0 at both ends)

so the curve still ends exactly at both melting points. The solidus is the
curve model's, capped at the corrected liquidus. With strength 0.5 the
liquidus improves from 96.1 to 91.7 K in 5-fold CV on processed_v8 (every
fold, every data group) and from 100.4 to 96.1 K on the test split
(results/v17, compare_foundation_models.py).

The foundation models learn in context: "fitting" them only stores the
training rows, so this class saves those rows (not the fitted networks) and
fits the models again, in seconds, the first time it predicts. They run on the
GPU when one is visible.

Licences: TabPFN v2 weights under the Prior Labs Licence (Apache 2.0 with an
attribution requirement: "Built with TabPFN"; Hollmann et al., Nature 637
(2025) 319); TabICL v2 BSD-3 (Inria Soda team).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .base import PhaseDataset, SpecialistModel, curve_metrics, swap_features, swap_rules
from .tree_curves import TreeCurveModel, _fill_curves

logger = logging.getLogger(__name__)
FOUNDATION_MODELS = ("tabpfn", "tabicl")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_regressor(name: str, seed: int = 0):
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=_device(), random_state=seed,
                                                          ignore_pretraining_limits=True)
    if name == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device=_device(), random_state=seed)
    raise ValueError(f"unknown foundation model {name!r}; choose from {FOUNDATION_MODELS}")


class HeightCorrectedCurves(SpecialistModel):
    """A TreeCurveModel whose liquidus height is corrected (module docstring)."""

    name = "M2_M3_height_corrected"

    def __init__(self, base: TreeCurveModel, strength: float = 0.5,
                 models: tuple[str, ...] = FOUNDATION_MODELS, seed: int = 0):
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        self.base, self.strength, self.models, self.seed = base, float(strength), tuple(models), seed
        self._X = self._y = None
        self._fitted: list | None = None

    # The stacked topology classifier was trained on uncorrected curves and
    # keeps using them (meta_learner reads .base); everything else reads this.
    @property
    def cfg(self):
        return self.base.cfg

    def _window(self) -> np.ndarray:
        frac = self.base._comp_grid.astype(np.float64) / 100.0
        w = frac * (1.0 - frac)
        return w / w.mean()

    def fit(self, ds: PhaseDataset) -> "HeightCorrectedCurves":
        """Store the height targets of the training systems of `ds` (without
        extra curve-only systems), in both A/B orders."""
        b = self.base
        idx = np.setdiff1d(ds.train_idx, ds.extra_idx) if ds.extra_idx.size else ds.train_idx
        frac = b._comp_grid.astype(np.float64) / 100.0
        tm_a = ds.X[:, b._melt_a_idx].astype(np.float64)
        tm_b = ds.X[:, b._melt_b_idx].astype(np.float64)
        line = (1 - frac)[None] * tm_a[:, None] + frac[None] * tm_b[:, None]
        seen = ds.T_liquidus_mask.copy()
        seen[:, [0, -1]] = False
        target = (_fill_curves(ds.T_liquidus, seen, frac, tm_a, tm_b, 3) - line).mean(axis=1)
        idx = idx[np.isfinite(target[idx]) & (tm_a[idx] > 0) & (tm_b[idx] > 0)]
        X_swap = ds.X_swap if ds.X_swap.size else swap_features(ds.X, list(ds.feature_names), swap_rules(ds))
        self._X = np.concatenate([ds.X[idx], X_swap[idx]]).astype(np.float32)
        self._y = np.tile(target[idx], 2).astype(np.float32)
        self._fitted = None
        logger.info("height correction (strength %.2f, %s): %d training systems",
                    self.strength, " + ".join(self.models), len(idx))
        return self

    def _ensure_fitted(self) -> list:
        if self._fitted is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._fitted = [make_regressor(m, self.seed).fit(self._X, self._y) for m in self.models]
            logger.info("height correction: fitted %s on %s", ", ".join(self.models), _device())
        return self._fitted

    def predict_height(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean([m.predict(np.asarray(X, dtype=np.float32)) for m in self._ensure_fitted()], axis=0)

    def _correct(self, X, liq, sol):
        line = self.base._base_line(X)
        shift = self.strength * (self.predict_height(X) - (liq - line).mean(axis=1))
        liq = liq + shift[:, None] * self._window()[None]
        return liq.astype(np.float32), np.minimum(sol, liq).astype(np.float32)

    def predict(self, X: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
        liq, sol = self.base.predict(X)
        return self._correct(X, liq, sol)

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int | None = None, clamp: bool = True):
        liq, liq_std, sol, sol_std = self.base.predict_with_uncertainty(X, n_samples, clamp)
        liq, sol = self._correct(X, liq, sol)
        return liq, liq_std, sol, sol_std

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        liq, sol = self.predict(ds.X[idx])
        return {"split": split, "n": int(len(idx)), **curve_metrics(ds, idx, liq, sol)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fitted"] = None                       # refitted on load: saves only the training rows
        return state

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path, compress=3)
