"""
Corrections of the tree liquidus by tabular foundation models (v17, v18).

TabPFN v2 and TabICL v2 are networks pre-trained on millions of synthetic
tables; they learn a new table "in context" (fitting stores the rows). They
predict single numbers per system better than the trees do, so:

1. Height (v17). About a third of the liquidus error is one number per
   system: the mean height of the curve above the melting-point line. The
   two models predict it (averaged), and the tree liquidus is moved
   `height` (0.5) of the way to it with the window 6 x (1 - x), which is 0
   at both ends and averages 1:
       liquidus += height * (h_pred - h_trees) * 6 x (1 - x)
   5-fold CV on processed_v8: 96.1 -> 91.7 K.
2. Whole curve (v18). The training deviation curves (both A/B orders, every
   2 at%) are compressed by PCA into `n_components` (10) numbers, 99.6 % of
   their variance; the two models predict each number; the rebuilt curve is
   blended `blend` (0.5) with the height-corrected liquidus:
       liquidus = (1 - blend) * liquidus + blend * (line + PCA^-1(scores))
   91.7 -> 89.2 K, every fold.
The solidus stays the trees', capped at the liquidus. Training rows are the
labelled training systems in both A/B orders (not the extra systems).

The fitted models are stored next to the saved curve model in
<name>_foundation/ (TabPFN without its network weights, ~25 MB each; TabICL
~10 MB) and fitted again from the stored rows if that folder is missing.
Predictions are remembered per input array, so scoring the same rows twice
costs nothing (each of the 22 models takes 5-10 s per call).

Licences: TabPFN v2 weights, Prior Labs Licence (Apache 2.0 + attribution:
"Built with TabPFN"; Hollmann et al., Nature 637 (2025) 319); TabICL v2,
BSD-3 (Inria Soda team).
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np

from .curves import TreeCurves, fill_curves, melting_point_line
from .data import PhaseDataset

logger = logging.getLogger(__name__)
FOUNDATION_MODELS = ("tabpfn", "tabicl")
KNOTS = np.arange(0, 101, 2)


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_regressor(name: str, seed: int = 0):
    if name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=device(), random_state=seed,
                                                          ignore_pretraining_limits=True)
    if name == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device=device(), random_state=seed)
    raise ValueError(f"unknown foundation model {name!r}")


def _save_fitted(model, path: Path) -> None:
    if type(model).__name__.startswith("TabPFN"):
        from tabpfn.model_loading import save_fitted_tabpfn_model
        save_fitted_tabpfn_model(model, path.with_suffix(".tabpfn_fit"))
    else:
        import joblib
        joblib.dump(model, path.with_suffix(".joblib"))


def _load_fitted(path: Path):
    if path.with_suffix(".tabpfn_fit").exists():
        from tabpfn.model_loading import load_fitted_tabpfn_model
        return load_fitted_tabpfn_model(path.with_suffix(".tabpfn_fit"), device=device())
    if path.with_suffix(".joblib").exists():
        import joblib
        return joblib.load(path.with_suffix(".joblib"))
    return None


class FoundationCurves:
    """Tree curves -> height correction (height > 0) -> blend with the
    foundation-model curve (blend > 0). See the module docstring."""

    def __init__(self, trees: TreeCurves, height: float = 0.5, blend: float = 0.5, n_components: int = 10,
                 models: tuple[str, ...] = FOUNDATION_MODELS, seed: int = 0):
        if not (0 < height <= 1 and 0 <= blend <= 1):
            raise ValueError("need 0 < height <= 1 and 0 <= blend <= 1")
        self.trees, self.height, self.blend, self.n_components = trees, float(height), float(blend), int(n_components)
        self.models, self.seed = tuple(models), seed
        self.store: str | None = None
        self._fitted = None
        self._memo: dict = {}

    def fit(self, ds: PhaseDataset) -> "FoundationCurves":
        """Store the training rows and targets (call after trees.fit on the same `ds`)."""
        from sklearn.decomposition import PCA
        idx = np.setdiff1d(ds.train_idx, ds.extra_idx)
        frac = np.asarray(ds.comp_grid, dtype=np.float64) / 100.0
        tm_a, tm_b = ds.melting_points()
        seen = ds.T_liquidus_mask.copy()
        seen[:, [0, -1]] = False
        dev = fill_curves(ds.T_liquidus, seen, frac, tm_a, tm_b, 3) - melting_point_line(tm_a, tm_b, frac)
        idx = idx[np.isfinite(dev[idx]).all(axis=1) & (tm_a[idx] > 0) & (tm_b[idx] > 0)]
        self._X = np.concatenate([ds.X[idx], ds.X_swap[idx]]).astype(np.float32)
        targets = [np.tile(dev[idx].mean(axis=1), 2)]                     # the mean height, same in both orders
        if self.blend:
            Y = np.concatenate([dev[idx][:, KNOTS], dev[idx][:, ::-1][:, KNOTS]])
            self._pca = PCA(n_components=self.n_components).fit(Y)
            targets += list(self._pca.transform(Y).T)
        self._Y = np.stack(targets, axis=1).astype(np.float32)
        w = frac * (1.0 - frac)
        self._window = w / w.mean()
        self._fitted, self._memo = None, {}
        logger.info("foundation models: %d training rows, height %.2f, blend %.2f with %d components%s", len(self._X),
                    self.height, self.blend, self.n_components,
                    f" ({self._pca.explained_variance_ratio_.sum():.3f} of the variance)" if self.blend else "")
        return self

    # ---- the fitted foundation models ------------------------------------
    def _names(self) -> list[str]:
        targets = ["height"] + ([f"component{c}" for c in range(self.n_components)] if self.blend else [])
        return [f"{t}_{m}" for t in targets for m in self.models]

    def _ensure_fitted(self) -> list:
        """One fitted model per (target, model): loaded from the store if there, else fitted."""
        if self._fitted is None:
            k = len(self.models)
            self._fitted = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, name in enumerate(self._names()):
                    m = _load_fitted(Path(self.store) / name) if self.store else None
                    if m is None:
                        m = make_regressor(self.models[i % k], self.seed).fit(self._X, self._Y[:, i // k])
                        if self.store:
                            Path(self.store).mkdir(parents=True, exist_ok=True)
                            _save_fitted(m, Path(self.store) / name)
                    self._fitted.append(m)
            logger.info("foundation models: %d ready on %s", len(self._fitted), device())
        return self._fitted

    def _targets(self, X: np.ndarray) -> np.ndarray:
        """(N, 1 + n_components): predicted height and component scores (mean of the models)."""
        k = len(self.models)
        X32 = np.asarray(X, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = self._ensure_fitted()
            return np.column_stack([np.mean([m.predict(X32) for m in fitted[t * k:(t + 1) * k]], axis=0)
                                    for t in range(len(fitted) // k)])

    # ---- prediction --------------------------------------------------------
    def predict_stages(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """(liquidus, solidus) after each stage: "trees" (v16), "height" (v17)
        and, with blend > 0, "blend" (v18)."""
        key = hashlib.sha1(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()
        if key not in self._memo:
            liq, sol = self.trees.predict(X)
            out = {"trees": (liq, sol)}
            line = self.trees.line(X)
            t = self._targets(X)
            shift = self.height * (t[:, 0] - (liq - line).mean(axis=1))
            liq = (liq + shift[:, None] * self._window[None]).astype(np.float32)
            sol = np.minimum(sol, liq).astype(np.float32)
            out["height"] = (liq, sol)
            if self.blend:
                at_knots = self._pca.inverse_transform(t[:, 1:])
                dev = np.stack([np.interp(np.arange(101), KNOTS, r) for r in at_knots])
                liq = ((1 - self.blend) * liq + self.blend * (line + dev)).astype(np.float32)
                out["blend"] = (liq, np.minimum(sol, liq).astype(np.float32))
            self._memo[key] = out
        return self._memo[key]

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stages = self.predict_stages(X)
        return stages["blend" if self.blend else "height"]

    # ---- saving --------------------------------------------------------------
    def __getstate__(self):
        return {**self.__dict__, "_fitted": None, "_memo": {}}

    def save(self, path) -> None:
        """This object to `path`, the fitted foundation models to <path stem>_foundation/."""
        import joblib
        path = Path(path)
        store = path.with_name(path.stem + "_foundation")
        if store.exists():
            shutil.rmtree(store)                   # a stale model must never be loaded
        store.mkdir(parents=True)
        self.store = str(store)
        if self._fitted:
            for name, m in zip(self._names(), self._fitted):
                _save_fitted(m, store / name)
        else:
            self._ensure_fitted()
        joblib.dump(self, path, compress=3)
