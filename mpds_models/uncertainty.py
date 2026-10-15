"""
Uncertainty sampling for active learning.

Two questions, one function each:

  rank_systems_for_acquisition(reconstructor, X_candidates, pair_ids)
      Which uncharacterised binary should be assessed next? Ranks systems by
      combined predictive uncertainty across M1-M6 (topology entropy, eutectic
      and miscibility ambiguity, MC-Dropout curve std, model inconsistencies).

  rank_grid_points_within_system(diagram)
      Where inside one predicted diagram should the next measurement go?
      Following Terayama et al. (2019), targets (x, T) points close to the
      predicted liquidus / solidus, scaled by the local MC-Dropout std.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meta_learner import PredictedDiagram, FullDiagramReconstructor


def shannon_entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """H(p) = -Σ p log p, with safe handling of 0·log0 = 0."""
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


# ---------------------------------------------------------------------------
# System-level acquisition
# ---------------------------------------------------------------------------

@dataclass
class SystemAcquisitionScore:
    pair: str
    total_score: float                # higher = more uncertain = better candidate
    topology_entropy: float
    eutectic_ambiguity: float         # 1 - 2|p(eutectic) - 0.5|; 1 = maximally ambiguous
    miscibility_ambiguity: float      # same, for p(miscibility gap)
    mean_liquidus_std: float
    mean_solidus_std: float
    n_inconsistencies: int


def score_system_uncertainty(d: PredictedDiagram) -> SystemAcquisitionScore:
    """Aggregate all per-system uncertainty signals into one score."""
    topo_probs = np.asarray(list(d.topology_proba.values()), dtype=np.float64)
    topo_ent = float(shannon_entropy(topo_probs))

    eut_amb = 1.0 - 2.0 * abs(d.eutectic_proba - 0.5)
    misc_amb = 1.0 - 2.0 * abs(d.miscibility_proba - 0.5)

    def mean_std(std: np.ndarray | None) -> float:
        return float(np.nan_to_num(std).mean()) if std is not None else 0.0

    mean_liq_std = mean_std(d.liquidus_T_std)
    mean_sol_std = mean_std(d.solidus_T_std)

    # Heuristic weights; tune to taste. Terayama et al. find boundary-region
    # uncertainty is the primary signal, hence the curve-std terms.
    n_topo = max(len(topo_probs), 2)
    score = (
        topo_ent / np.log(n_topo)                # normalised to [0, 1]
        + 0.5 * eut_amb
        + 0.5 * misc_amb
        + 0.005 * mean_liq_std                   # 200 K of std ≈ 1 point
        + 0.005 * mean_sol_std
        + 0.1 * len(d.inconsistencies)
    )

    return SystemAcquisitionScore(
        pair=d.pair,
        total_score=float(score),
        topology_entropy=topo_ent,
        eutectic_ambiguity=float(eut_amb),
        miscibility_ambiguity=float(misc_amb),
        mean_liquidus_std=mean_liq_std,
        mean_solidus_std=mean_sol_std,
        n_inconsistencies=len(d.inconsistencies),
    )


def rank_systems_for_acquisition(
    reconstructor: FullDiagramReconstructor,
    X_candidates: np.ndarray,
    pair_ids: list[str],
    top_k: int | None = 25,
    n_mc_samples: int = 30,
) -> list[SystemAcquisitionScore]:
    """Rank candidate systems by overall predictive uncertainty, most uncertain
    first: candidates for the next CALPHAD assessment, DFT run or experiment."""
    diagrams = reconstructor.reconstruct(
        X_candidates, pair_ids, with_uncertainty=True, n_mc_samples=n_mc_samples,
    )
    scores = [score_system_uncertainty(d) for d in diagrams]
    scores.sort(key=lambda s: -s.total_score)
    return scores[:top_k] if top_k is not None else scores


# ---------------------------------------------------------------------------
# Intra-system grid point ranking (Terayama-style)
# ---------------------------------------------------------------------------

def rank_grid_points_within_system(
    diagram: PredictedDiagram,
    *,
    T_grid: np.ndarray | None = None,
    top_k: int = 20,
) -> list[tuple[float, float, float]]:
    """Rank (x, T) points inside one system by closeness to a phase boundary.

    Each point scores a Gaussian kernel of its distance to the predicted
    liquidus plus one for the solidus, each distance measured in units of
    the local std (floored at 1 K; 20 K when no uncertainty is available).
    Points on a boundary score highest, points deep inside a single-phase
    region lowest.

    `T_grid` sets the temperature axis; by default 40 points spanning
    100 K below the lowest solidus to 100 K above the highest liquidus.
    Returns the top_k (x, T, score) tuples, highest score first.
    """
    G = len(diagram.comp_grid)
    if T_grid is None:
        Tmin = float(np.nanmin(diagram.solidus_T[np.isfinite(diagram.solidus_T)]) - 100)
        Tmax = float(np.nanmax(diagram.liquidus_T) + 100)
        T_grid = np.linspace(Tmin, Tmax, 40)
    T_grid = np.asarray(T_grid, dtype=np.float64)

    liq_std = diagram.liquidus_T_std if diagram.liquidus_T_std is not None else np.full(G, 20.0)
    sol_std = diagram.solidus_T_std if diagram.solidus_T_std is not None else np.full(G, 20.0)

    # (G, nT) distances to each boundary, in local-std units
    T = T_grid[None, :]
    d_liq = np.abs(T - diagram.liquidus_T[:, None]) / np.maximum(liq_std, 1.0)[:, None]
    d_sol = np.abs(T - diagram.solidus_T[:, None]) / np.maximum(sol_std, 1.0)[:, None]
    score = (np.exp(-0.5 * d_liq ** 2) + np.exp(-0.5 * d_sol ** 2)).ravel()

    best = np.argsort(-score, kind="stable")[:top_k]
    xs = np.repeat(np.asarray(diagram.comp_grid, dtype=np.float64), len(T_grid))
    Ts = np.tile(T_grid, G)
    return [(float(xs[k]), float(Ts[k]), float(score[k])) for k in best]
