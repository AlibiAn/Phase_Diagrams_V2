"""
Liquidus and solidus curves from ExtraTrees (the curve model since v10; the
settings of v14).

Two multi-output ExtraTrees predict whole curves on the composition grid:

    liquidus(x) = line(x) + d(x)
    solidus(x)  = min(line(x) + s(x), liquidus(x))

where line(x) is the straight line between the two melting points and d, s
are the curves' bends away from it. A forest's prediction is an average of
training curves, and every training deviation is 0 at both ends, so the
predicted curves end exactly at Tm(A) and Tm(B) with no penalty terms.

- Each curve is fitted at every 2 at% (51 knots) and interpolated back.
  Training curves are filled by linear interpolation between their data
  points, anchored at Tm at the ends (observed end values are replaced by
  Tm: a disagreeing end is a data error). Liquidus rows need >= 3 points.
- A/B swap: every training system is also used with A and B exchanged
  (features from PhaseDataset.X_swap, curves mirrored); -8 K liquidus.
- Each grid point is the MEDIAN over the trees, not the mean: it keeps the
  level most similar systems share where the mean smears steps into slopes,
  and minimises the absolute error we score; -4.5 K.
"""
from __future__ import annotations

import logging

import numpy as np

from .data import PhaseDataset

logger = logging.getLogger(__name__)


def fill_curves(values: np.ndarray, observed: np.ndarray, frac: np.ndarray, left: np.ndarray, right: np.ndarray,
                min_points: int) -> np.ndarray:
    """Each row filled at every grid point by linear interpolation of its
    observed points, with the `left` / `right` anchors added at an unobserved
    end (NaN = no anchor); rows with fewer than `min_points` points are NaN."""
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


def melting_point_line(tm_a: np.ndarray, tm_b: np.ndarray, frac: np.ndarray) -> np.ndarray:
    return (1.0 - frac) * tm_a[:, None] + frac * tm_b[:, None]


class TreeCurves:
    """Liquidus + solidus from two ExtraTrees (module docstring)."""

    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 2, knot_step: int = 2, seed: int = 0):
        self.n_estimators, self.min_samples_leaf, self.knot_step, self.seed = n_estimators, min_samples_leaf, knot_step, seed
        self.liq_model = self.sol_model = None

    def _forest(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                   max_features=1.0, n_jobs=-1, random_state=self.seed)

    def fit(self, ds: PhaseDataset) -> "TreeCurves":
        tr = ds.train_idx
        # Every training system twice: as (A, B) and as (B, A)
        X = np.concatenate([ds.X[tr], ds.X_swap[tr]])
        w = np.concatenate([ds.weights[tr], ds.weights[tr]])
        both = lambda a: np.concatenate([a[tr], a[tr][:, ::-1]])
        T_liq, seen, T_sol = both(ds.T_liquidus), both(ds.T_liquidus_mask), both(ds.T_solidus)

        self._a, self._b = ds.column("phys__melting_point__A"), ds.column("phys__melting_point__B")
        self._grid = np.asarray(ds.comp_grid, dtype=np.float32)
        frac = self._grid.astype(np.float64) / 100.0
        self._knots = np.unique(np.r_[np.flatnonzero(np.isclose(self._grid % self.knot_step, 0)), 0, len(frac) - 1])
        tm_a, tm_b = X[:, self._a], X[:, self._b]
        has_tm = (tm_a > 0) & (tm_b > 0)
        self._tm_fill = (float(tm_a[tm_a > 0].mean()), float(tm_b[tm_b > 0].mean()))
        left, right = np.where(tm_a > 0, tm_a, np.nan), np.where(tm_b > 0, tm_b, np.nan)
        line = melting_point_line(tm_a, tm_b, frac)

        liq_seen = seen.copy()                        # the Tm anchors replace observed end values
        liq_seen[:, 0] &= ~np.isfinite(left)
        liq_seen[:, -1] &= ~np.isfinite(right)
        liq = fill_curves(T_liq, liq_seen, frac, left, right, 3) - line
        sol_seen = np.isfinite(T_sol)
        sol_seen[:, [0, -1]] = False
        sol = fill_curves(T_sol, sol_seen, frac, left, right, 1) - line
        sol[~has_tm] = np.nan
        liq_rows = np.isfinite(liq).all(1) & (w > 0) & has_tm
        sol_rows = np.isfinite(sol).all(1) & (w > 0)
        k = self._knots
        self.liq_model = self._forest().fit(X[liq_rows], liq[liq_rows][:, k], sample_weight=w[liq_rows])
        self.sol_model = self._forest().fit(X[sol_rows], sol[sol_rows][:, k], sample_weight=w[sol_rows])
        logger.info("tree curves: liquidus from %d curves, solidus from %d (both A/B orders), %d knots",
                    liq_rows.sum(), sol_rows.sum(), len(k))
        return self

    def line(self, X: np.ndarray) -> np.ndarray:
        """The melting-point line of each row (a missing Tm, stored as 0, becomes the training mean)."""
        tm_a, tm_b = X[:, self._a].astype(np.float64), X[:, self._b].astype(np.float64)
        tm_a, tm_b = np.where(tm_a > 0, tm_a, self._tm_fill[0]), np.where(tm_b > 0, tm_b, self._tm_fill[1])
        return melting_point_line(tm_a, tm_b, self._grid.astype(np.float64) / 100.0)

    def _median(self, model, X32: np.ndarray) -> np.ndarray:
        at_knots = np.median([t.predict(X32).reshape(len(X32), -1) for t in model.estimators_], axis=0)
        return np.stack([np.interp(self._grid, self._grid[self._knots], row) for row in at_knots])

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(liquidus, solidus), each (N, 101) in K; solidus <= liquidus."""
        X32 = np.asarray(X, dtype=np.float32)
        line = self.line(X)
        liq = line + self._median(self.liq_model, X32)
        sol = np.minimum(line + self._median(self.sol_model, X32), liq)
        return liq.astype(np.float32), sol.astype(np.float32)
