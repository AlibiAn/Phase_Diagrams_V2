"""
Tree-ensemble curve model for liquidus (M2) and solidus (M3): an
alternative to the CurveNet MLP with the same predict / evaluate interface.

Two multi-output tree ensembles predict whole curves on the composition grid:

    liquidus:  T_liq(x) = line(Tm_A, Tm_B)(x) + d(x)     target="deviation"
    solidus:   T_sol(x) = T_liq(x) - g(x),  g ≥ 0

`d` is the bend away from the straight line between the two melting points
and `g` the gap below the liquidus.

Why the forests get the physics for free
----------------------------------------
A random-forest / extra-trees prediction is a weighted average of training
curves (each leaf stores the mean target curve of its training systems).
With `pin_ends`, every training deviation curve is 0 at both ends, and
every gap curve is ≥ 0, so their averages are too: the predicted liquidus
passes exactly through Tm(A) and Tm(B), and the solidus never rises above the
liquidus, with no penalty terms. XGBoost sums boosted leaf values instead of
averaging, so it keeps neither property exactly (the gap is clipped at 0).

XGBoost would need one tree per output point per round (101 × 1000 trees),
so it predicts the leading principal components of the training curves
instead (`xgb_n_components`, typically > 99 % of the curve variance).

target="absolute" predicts T_liq(x) directly, for comparison: its end points
are then averages of other systems' melting points.

Targets with gaps
-----------------
The ensembles need a value at every output point. Each training curve is
filled by linear interpolation between its observed points, anchored at
T = Tm (deviation 0) at an unobserved end. With `pin_ends` (default) the ends
are set to Tm even where the data has a value there: a pure element melts at
its melting point, and an observed end that disagrees is a data error (see
check_endpoints.py) that would otherwise leak into every curve averaged
with it. Rows with fewer than `min_points`
observed liquidus points are left out. The curves are fitted at every
`knot_step` at% and interpolated back to the full grid, which keeps the
saved forests small. Evaluation uses observed points only, as for CurveNet.

Uncertainty
-----------
For the forests, `predict_with_uncertainty` returns the spread over the
individual trees (the counterpart of CurveNet's MC dropout).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .base import PhaseDataset, SpecialistModel, curve_metrics, xgb_device_kwargs
from .deep_curve_net import _melting_point_columns

logger = logging.getLogger(__name__)

CURVE_BACKENDS = ("extra_trees", "random_forest", "xgboost")
CURVE_TARGETS = ("deviation", "absolute")


@dataclass
class TreeCurveConfig:
    backend: str = "extra_trees"        # "extra_trees", "random_forest" or "xgboost"
    target: str = "deviation"           # "deviation" (exact end points) or "absolute"
    knot_step: int = 2                  # fit every knot_step at%, interpolate the rest
    pin_ends: bool = True               # training curves end exactly at Tm(A), Tm(B)
    min_points: int = 3                 # observed liquidus points a training row needs
    # Forests
    n_estimators: int = 300
    min_samples_leaf: int = 2
    max_features: float = 1.0           # fraction of features tried per split
    # XGBoost (one multi-output tree per round; v6 classifier settings)
    xgb_n_estimators: int = 1000
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.02
    xgb_n_components: int = 12          # principal components of the curves it predicts
    seed: int = 0


class _PCARegressor:
    """Fits `model` to the leading principal components of a multi-output
    target and maps predictions back to the full output."""

    def __init__(self, model, n_components: int):
        self.model = model
        self.n_components = n_components
        self.pca = None

    def fit(self, X, Y, sample_weight=None):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=min(self.n_components, *Y.shape)).fit(Y)
        self.model.fit(X, self.pca.transform(Y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        scores = self.model.predict(X).reshape(len(X), -1)
        return self.pca.inverse_transform(scores)


def _fill_curves(values: np.ndarray, observed: np.ndarray, frac: np.ndarray,
                 left: np.ndarray, right: np.ndarray, min_points: int) -> np.ndarray:
    """Fill every grid point of each row by linear interpolation of its
    observed points, adding the `left` / `right` anchor values at an
    unobserved end (NaN = no anchor). Rows with fewer than `min_points`
    observed points come back all-NaN."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for i in range(len(values)):
        m = observed[i]
        if m.sum() < min_points:
            continue
        xs, ys = frac[m], values[i, m]
        if not m[0] and np.isfinite(left[i]):
            xs, ys = np.r_[0.0, xs], np.r_[left[i], ys]
        if not m[-1] and np.isfinite(right[i]):
            xs, ys = np.r_[xs, 1.0], np.r_[ys, right[i]]
        out[i] = np.interp(frac, xs, ys)
    return out


class TreeCurveModel(SpecialistModel):
    """M2 + M3 from tree ensembles (see the module docstring)."""

    name = "M2_M3_tree_curves"

    def __init__(self, config: TreeCurveConfig | None = None):
        self.cfg = replace(config) if config else TreeCurveConfig()
        if self.cfg.backend not in CURVE_BACKENDS:
            raise ValueError(f"backend must be one of {CURVE_BACKENDS}")
        if self.cfg.target not in CURVE_TARGETS:
            raise ValueError(f"target must be one of {CURVE_TARGETS}")
        self.liq_model = None
        self.gap_model = None
        self._melt_a_idx: int | None = None
        self._melt_b_idx: int | None = None
        self._comp_grid: np.ndarray | None = None
        self._knots: np.ndarray | None = None
        self._tm_fill: tuple[float, float] = (1000.0, 1000.0)

    @property
    def _deviation(self) -> bool:
        return self.cfg.target == "deviation"

    def _make(self):
        c = self.cfg
        if c.backend == "xgboost":
            import xgboost as xgb
            reg = xgb.XGBRegressor(
                n_estimators=c.xgb_n_estimators, max_depth=c.xgb_max_depth,
                learning_rate=c.xgb_learning_rate, random_state=c.seed, verbosity=0,
                **xgb_device_kwargs())
            return _PCARegressor(reg, c.xgb_n_components)
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        forest = ExtraTreesRegressor if c.backend == "extra_trees" else RandomForestRegressor
        return forest(n_estimators=c.n_estimators, min_samples_leaf=c.min_samples_leaf,
                      max_features=c.max_features, n_jobs=-1, random_state=c.seed)

    def _melting_points(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tm(A), Tm(B); a missing value (stored as 0) becomes the training mean."""
        tm_a = X[:, self._melt_a_idx].astype(np.float64)
        tm_b = X[:, self._melt_b_idx].astype(np.float64)
        n_missing = int((tm_a <= 0).sum() + (tm_b <= 0).sum())
        if n_missing:
            logger.warning("tree curves: %d missing melting points replaced by the training "
                           "mean; those curve ends will be off", n_missing)
        return (np.where(tm_a > 0, tm_a, self._tm_fill[0]),
                np.where(tm_b > 0, tm_b, self._tm_fill[1]))

    def _line(self, tm_a: np.ndarray, tm_b: np.ndarray, frac: np.ndarray) -> np.ndarray:
        return (1.0 - frac) * tm_a[:, None] + frac * tm_b[:, None]

    def _to_grid(self, at_knots: np.ndarray) -> np.ndarray:
        """(N, n_knots) -> (N, n_grid) by linear interpolation along composition."""
        grid = self._comp_grid
        if len(self._knots) == len(grid):
            return at_knots
        return np.stack([np.interp(grid, grid[self._knots], row) for row in at_knots])

    # ---- training ----------------------------------------------------

    def fit(self, ds: PhaseDataset) -> "TreeCurveModel":
        idx = ds.train_idx
        if len(idx) == 0:
            raise RuntimeError("training split is empty")
        self._melt_a_idx, self._melt_b_idx = _melting_point_columns(ds.feature_names)
        self._comp_grid = np.asarray(ds.comp_grid, dtype=np.float32)
        frac = self._comp_grid.astype(np.float64) / 100.0
        on_step = np.flatnonzero(np.isclose(self._comp_grid % self.cfg.knot_step, 0))
        self._knots = np.unique(np.r_[on_step, 0, len(frac) - 1])

        X, w = ds.X[idx], ds.weights[idx]
        tm_a_raw, tm_b_raw = X[:, self._melt_a_idx], X[:, self._melt_b_idx]
        has_tm = (tm_a_raw > 0) & (tm_b_raw > 0)
        self._tm_fill = (float(tm_a_raw[tm_a_raw > 0].mean()) if (tm_a_raw > 0).any() else 1000.0,
                         float(tm_b_raw[tm_b_raw > 0].mean()) if (tm_b_raw > 0).any() else 1000.0)
        left = np.where(tm_a_raw > 0, tm_a_raw, np.nan)
        right = np.where(tm_b_raw > 0, tm_b_raw, np.nan)

        liq_obs = ds.T_liquidus_mask[idx].copy()
        if self.cfg.pin_ends:                               # anchors replace observed ends
            liq_obs[:, 0] &= ~np.isfinite(left)
            liq_obs[:, -1] &= ~np.isfinite(right)
        liq = _fill_curves(ds.T_liquidus[idx], liq_obs, frac, left, right, self.cfg.min_points)
        gap_obs = np.clip(liq - ds.T_solidus[idx], 0.0, None)   # NaN where either is missing
        gap_seen = np.isfinite(gap_obs)
        if self.cfg.pin_ends:
            gap_seen[:, [0, -1]] = False
        zeros = np.zeros(len(idx))
        gap = _fill_curves(gap_obs, gap_seen, frac, zeros, zeros, 1)

        liq_rows = np.isfinite(liq).all(1)
        if self._deviation:
            liq_rows &= has_tm                            # no line without both Tm
            liq_target = liq - self._line(tm_a_raw, tm_b_raw, frac)
        else:
            liq_target = liq
        gap_rows = np.isfinite(gap).all(1)
        if liq_rows.sum() < 5 or gap_rows.sum() < 5:
            raise RuntimeError(f"too few training curves: {liq_rows.sum()} liquidus, "
                               f"{gap_rows.sum()} solidus")

        k = self._knots
        self.liq_model = self._make().fit(X[liq_rows], liq_target[liq_rows][:, k],
                                          sample_weight=w[liq_rows])
        self.gap_model = self._make().fit(X[gap_rows], gap[gap_rows][:, k],
                                          sample_weight=w[gap_rows])
        logger.info("tree curves (%s, %s): liquidus from %d systems, solidus gap from %d, "
                    "%d knots", self.cfg.backend, self.cfg.target, liq_rows.sum(),
                    gap_rows.sum(), len(k))
        return self

    # ---- inference ---------------------------------------------------

    def _base_line(self, X: np.ndarray) -> np.ndarray | float:
        """What the liquidus model's output is added to: the melting-point line
        (deviation target) or nothing (absolute target)."""
        if not self._deviation:
            return 0.0
        frac = self._comp_grid.astype(np.float64) / 100.0
        return self._line(*self._melting_points(X), frac)

    def _assemble(self, base, d: np.ndarray, g: np.ndarray):
        """Knot-level liquidus target `d` and gap `g` -> (liquidus, solidus) on the grid."""
        liq = base + self._to_grid(d)
        sol = liq - np.clip(self._to_grid(g), 0.0, None)
        return liq.astype(np.float32), sol.astype(np.float32)

    def predict(self, X: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (T_liquidus, T_solidus), each (N, n_grid). The solidus is
        never above the liquidus, so `clamp` (kept for the CurveNet
        interface) changes nothing."""
        if self.liq_model is None:
            raise RuntimeError("call fit() first")
        return self._assemble(self._base_line(X), self.liq_model.predict(X),
                              self.gap_model.predict(X))

    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int | None = None,
                                 clamp: bool = True):
        """Mean and spread over the forest's trees: (liq_mean, liq_std, sol_mean,
        sol_std). `n_samples` is ignored (every tree is used)."""
        if self.cfg.backend == "xgboost":
            raise NotImplementedError("per-tree uncertainty needs a forest backend")
        X32 = np.asarray(X, dtype=np.float32)
        base = self._base_line(X)
        liqs, sols = [], []
        for t_liq, t_gap in zip(self.liq_model.estimators_, self.gap_model.estimators_):
            liq, sol = self._assemble(base, t_liq.predict(X32).reshape(len(X), -1),
                                      t_gap.predict(X32).reshape(len(X), -1))
            liqs.append(liq)
            sols.append(sol)
        liq_mean, sol_mean = self._assemble(base, self.liq_model.predict(X),
                                            self.gap_model.predict(X))
        return liq_mean, np.std(liqs, axis=0), sol_mean, np.std(sols, axis=0)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        raw_liq, raw_sol = self.predict(ds.X[idx])
        return {"split": split, "n": int(len(idx)), **curve_metrics(ds, idx, raw_liq, raw_sol)}

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path, compress=3)
