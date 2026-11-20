"""
Equilibrium liquidus of a regular-solution binary with pure solids — a
thermodynamic estimate that, unlike the Schroder-van Laar curves of
element_features.py, can produce a liquid miscibility gap and its flat
monotectic line.

Model, per mole of atoms, energies relative to the pure LIQUID elements:
    liquid(x, T) = R T [x ln x + (1-x) ln(1-x)] + omega x (1-x)
    solid A      = -H_A (1 - T / Tm_A)      at x = 0
    solid B      = -H_B (1 - T / Tm_B)      at x = 1
At each temperature the stable state at composition x lies on the lower
convex hull of these three. The liquid alone is stable at x when the hull
touches the liquid curve there; the liquidus T_L(x) is the lowest temperature
above which that holds. With omega > 2 R T the liquid curve has two minima,
the hull bridges them, and T_L is flat over the gap: the monotectic line.

omega = 4 x Miedema's tabulated enthalpy of mixing of the equiatomic liquid
(element_features.interaction_parameter); H = heat of fusion, Tm = melting
point (Magpie tables). Nothing is fitted to phase diagrams.

Features (prefix "rs__") of the ordered pair (A, B): the liquidus minus the
melting-point line at 7 compositions, its mean, the share of the composition
range that is flat (two liquids), the monotectic temperature minus the higher
melting point (0 if none), and the lowest liquidus temperature and its
composition (the eutectic of the model).
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)
R = 8.314462618e-3                 # kJ / (mol K)
POINTS = (10, 25, 40, 50, 60, 75, 90)
GRID = np.linspace(0.0, 1.0, 101)


def _lower_hull(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of (x, g), x increasing."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (g[i] - g[a]) - (g[b] - g[a]) * (x[i] - x[a]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull)


def equilibrium_state(tm_a: float, tm_b: float, h_a: float, h_b: float, omega: float,
                      t_step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(T_L, two_liquid) on the 101-point grid.

    T_L(x): the lowest temperature above which no solid is stable at x (K),
    the liquidus in the convention of processed_v8. two_liquid(x): two
    liquids coexist at x just above T_L (the monotectic stretch)."""
    n = len(GRID)
    xs = GRID[1:-1]
    mix = xs * np.log(xs) + (1 - xs) * np.log(1 - xs)
    temps = np.arange(max(tm_a, tm_b) + t_step, 20.0, -t_step)        # descending
    solid = np.zeros((len(temps), n), dtype=bool)
    two_liq = np.zeros((len(temps), n), dtype=bool)
    idx = np.arange(n)
    for k, T in enumerate(temps):
        g_a, g_b = -h_a * (1 - T / tm_a), -h_b * (1 - T / tm_b)
        g = np.r_[min(0.0, g_a), R * T * mix + omega * xs * (1 - xs), min(0.0, g_b)]
        hull = _lower_hull(GRID, g)
        seg = np.clip(np.searchsorted(hull, idx, side="right") - 1, 0, len(hull) - 2)
        left, right = hull[seg], hull[seg + 1]
        on_vertex = np.isin(idx, hull)
        uses_solid_a = (left == 0) & (g_a < 0) & ~(on_vertex & (idx != 0))
        uses_solid_b = (right == n - 1) & (g_b < 0) & ~(on_vertex & (idx != n - 1))
        solid[k] = uses_solid_a | uses_solid_b
        two_liq[k] = ~on_vertex & (left > 0) & (right < n - 1) & (right - left > 1)
    T_L = np.empty(n)
    gap = np.zeros(n, dtype=bool)
    for j in range(n):
        hits = np.flatnonzero(solid[:, j])
        k = hits[0] if hits.size else len(temps) - 1
        T_L[j] = temps[k] + t_step / 2 if hits.size else temps[-1]
        gap[j] = bool(two_liq[max(k - 1, 0), j])
    T_L[0], T_L[-1] = tm_a, tm_b
    return T_L, gap


def equilibrium_liquidus(tm_a, tm_b, h_a, h_b, omega, t_step: float = 2.0) -> np.ndarray:
    return equilibrium_state(tm_a, tm_b, h_a, h_b, omega, t_step)[0]


@lru_cache(maxsize=None)
def pair_features(a: str, b: str) -> dict[str, float]:
    from .element_features import element_table, interaction_parameter
    t = element_table()
    tm_a, tm_b = float(t.at[a, "MeltingT"]), float(t.at[b, "MeltingT"])
    h_a, h_b = float(t.at[a, "HeatFusion"]), float(t.at[b, "HeatFusion"])
    om = interaction_parameter(a, b)
    om = float(om) if np.isfinite(om) else 0.0
    names = [f"rs__dev_{k}" for k in POINTS] + ["rs__mean_dev", "rs__flat_share", "rs__mono_minus_tmax",
                                                "rs__min_T_minus_tmin", "rs__x_at_min"]
    if not all(np.isfinite(v) and v > 0 for v in (tm_a, tm_b, h_a, h_b)):
        return {n: 0.0 for n in names}
    liq, gap = equilibrium_state(tm_a, tm_b, h_a, h_b, om)
    line = (1 - GRID) * tm_a + GRID * tm_b
    flat_share = float(gap.mean())
    mono = float(np.median(liq[gap]) - max(tm_a, tm_b)) if gap.sum() >= 3 else 0.0
    j = int(np.argmin(liq))
    f = {f"rs__dev_{k}": float(liq[k] - line[k]) for k in POINTS}
    f.update({"rs__mean_dev": float((liq - line).mean()), "rs__flat_share": flat_share,
              "rs__mono_minus_tmax": mono, "rs__min_T_minus_tmin": float(liq[j] - min(tm_a, tm_b)),
              "rs__x_at_min": float(j)})
    return f


CACHE = None        # env MPDS_FEATURE_CACHE: folder for a json cache of computed pairs


def _cached(a: str, b: str, store: dict) -> dict:
    key = f"{a}_{b}"
    if key not in store:
        store[key] = pair_features(a, b)
    return store[key]


def build_features(pair_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import json
    import os
    from pathlib import Path
    folder = os.environ.get("MPDS_FEATURE_CACHE")
    path = Path(folder) / "regular_solution.json" if folder else None
    store = json.loads(path.read_text()) if path and path.exists() else {}
    n0 = len(store)
    rows, rows_swap = [], []
    for pid in pair_ids:
        a, b = pid.split("_")[:2]
        rows.append(_cached(a, b, store))
        rows_swap.append(_cached(b, a, store))
    if path and len(store) > n0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(store))
        tmp.rename(path)
    names = list(rows[0])
    mat = lambda rs: np.nan_to_num(np.array([[r[n] for n in names] for r in rs], dtype=np.float32))
    logger.info("regular-solution equilibrium features for %d systems", len(pair_ids))
    return mat(rows), mat(rows_swap), names
