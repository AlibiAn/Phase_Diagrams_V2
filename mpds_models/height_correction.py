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


def save_fitted(model, path) -> None:
    """Store a fitted foundation model: TabPFN without its network weights
    (save_fitted_tabpfn_model, about 25 MB), TabICL with joblib (about 10 MB)."""
    from pathlib import Path
    path = Path(path)
    if type(model).__name__.startswith("TabPFN"):
        from tabpfn.model_loading import save_fitted_tabpfn_model
        save_fitted_tabpfn_model(model, path.with_suffix(".tabpfn_fit"))
    else:
        import joblib
        joblib.dump(model, path.with_suffix(".joblib"))


def load_fitted(path):
    """The model stored by save_fitted at `path` (either suffix), or None."""
    from pathlib import Path
    path = Path(path)
    if path.with_suffix(".tabpfn_fit").exists():
        from tabpfn.model_loading import load_fitted_tabpfn_model
        return load_fitted_tabpfn_model(path.with_suffix(".tabpfn_fit"), device=_device())
    if path.with_suffix(".joblib").exists():
        import joblib
        return joblib.load(path.with_suffix(".joblib"))
    return None


def reset_store(store, names: list[str], fitted: list | None) -> None:
    """Empty the folder `store` (a stale copy must never be loaded) and write
    the models already fitted in memory, if any, under `names`."""
    import shutil
    from pathlib import Path
    if Path(store).exists():
        shutil.rmtree(store)
    if fitted:
        Path(store).mkdir(parents=True)
        for name, m in zip(names, fitted):
            save_fitted(m, Path(store) / name)


def fitted_models(store, names: list[str], fit) -> list:
    """Load each named model from the folder `store` if it is there, else fit it
    with fit(i) (and keep it in `store`, when given). Loading takes about half
    the time of fitting; TabPFN predictions after loading differ by < 0.01 K."""
    from pathlib import Path
    out = []
    for i, name in enumerate(names):
        m = load_fitted(Path(store) / name) if store else None
        if m is None:
            m = fit(i)
            if store:
                Path(store).mkdir(parents=True, exist_ok=True)
                save_fitted(m, Path(store) / name)
        out.append(m)
    return out


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
        self._store = None            # folder of the fitted foundation models, set by save()

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
                self._fitted = fitted_models(getattr(self, "_store", None), [f"height_{m}" for m in self.models],
                                             lambda i: make_regressor(self.models[i], self.seed).fit(self._X, self._y))
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
        """The wrapper to `path`, the fitted foundation models next to it in
        <path stem>_foundation/ (fitted now if they are not yet)."""
        import joblib
        from pathlib import Path
        self._store = str(Path(path).with_name(Path(path).stem + "_foundation"))
        reset_store(self._store, [f"height_{m}" for m in self.models], self._fitted)
        self._ensure_fitted()
        joblib.dump(self, path, compress=3)
