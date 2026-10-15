"""
Uncertainty sampling for active learning.

Two kinds of uncertainty are exposed
====================================

  1. Per-system aleatoric uncertainty
     How uncertain are we about ONE specific system's full diagram?
     Surfaced as: liquidus / solidus std from MC-Dropout, topology class
     entropy from the meta-learner, M4 existence probability close to 0.5,
     etc.

  2. Per-(composition, temperature) uncertainty inside ONE system
     Where should the next experiment / DFT calculation be placed on the
     existing diagram? This is the Terayama et al. (2019) uncertainty
     sampling approach: target the (x, T) where the phase-classifier
     posterior is most ambiguous.

The active learning interface
-----------------------------

  rank_systems_for_acquisition(reconstructor, X_candidates, ...) -> List
      Returns candidates ranked by total predictive uncertainty across
      M1-M6. Use this when deciding which uncharacterised binary to
      target next.

  rank_grid_points_within_system(diagram, ...) -> ndarray
      Returns (x, T) grid points ranked by entropy of phase predictions.
      Use this when planning where to focus experiments inside an
      already-predicted diagram.
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
# System-level acquisition score
# ---------------------------------------------------------------------------

@dataclass
class SystemAcquisitionScore:
    pair: str
    total_score: float                # higher = more uncertain = better candidate
    topology_entropy: float
    eutectic_ambiguity: float         # |0.5 - p(eutectic)| ; lower = more ambiguous
    miscibility_ambiguity: float
    mean_liquidus_std: float
    mean_solidus_std: float
    n_inconsistencies: int


def score_system_uncertainty(d: PredictedDiagram) -> SystemAcquisitionScore:
    """Aggregate all per-system uncertainty signals into one score."""
    topo_probs = np.asarray(list(d.topology_proba.values()), dtype=np.float64)
    topo_ent = float(shannon_entropy(topo_probs))

    eut_amb = 1.0 - 2.0 * abs(d.eutectic_proba - 0.5)   # in [0,1], 1 = max ambiguity
    misc_amb = 1.0 - 2.0 * abs(d.miscibility_proba - 0.5)

    mean_liq_std = float(np.nan_to_num(d.liquidus_T_std).mean()) if d.liquidus_T_std is not None else 0.0
    mean_sol_std = float(np.nan_to_num(d.solidus_T_std).mean()) if d.solidus_T_std is not None else 0.0

    # Composite (heuristic weights — tune to taste). The Terayama paper
    # establishes that boundary-region uncertainty is the primary signal,
    # so we lean on the curve stds and topology entropy.
    n_topo = max(len(topo_probs), 2)
    score = (
        topo_ent / np.log(n_topo)                # normalise to [0, 1]
        + 0.5 * eut_amb
        + 0.5 * misc_amb
        + 0.005 * mean_liq_std                   # K-scale → /200
        + 0.005 * mean_sol_std
        + 0.1 * len(d.inconsistencies)           # inconsistencies = unreliability
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
    """Rank candidate uncharacterised systems by overall predictive uncertainty.

    Use this to choose which binary to target with the next CALPHAD
    assessment, DFT calculation, or experimental campaign.
    """
    diagrams = reconstructor.reconstruct(
        X_candidates, pair_ids, with_uncertainty=True, n_mc_samples=n_mc_samples,
    )
    scores = [score_system_uncertainty(d) for d in diagrams]
    scores.sort(key=lambda s: -s.total_score)
    if top_k is not None:
        scores = scores[:top_k]
    return scores


# ---------------------------------------------------------------------------
# Intra-system grid point ranking (Terayama-style)
# ---------------------------------------------------------------------------

def rank_grid_points_within_system(
    diagram: PredictedDiagram,
    *,
    T_grid: np.ndarray | None = None,
    top_k: int = 20,
) -> list[tuple[float, float, float]]:
    """Rank (x, T) points by predictive uncertainty WITHIN a single system.

    The uncertainty at point (x, T) is approximated by the absolute
    distance from the predicted liquidus / solidus boundary, normalised
    by the local std. Points near boundaries get the highest scores;
    points deep inside a single-phase region get low scores.

    Returns list of (x, T, score), sorted descending by score.

    `T_grid` lets the caller specify the temperature axis; if omitted,
    we sample 40 points between the lowest solidus and highest liquidus.
    """
    G = len(diagram.comp_grid)
    if T_grid is None:
        Tmin = float(np.nanmin(diagram.solidus_T[np.isfinite(diagram.solidus_T)]) - 100)
        Tmax = float(np.nanmax(diagram.liquidus_T) + 100)
        T_grid = np.linspace(Tmin, Tmax, 40)

    # std per grid point — fall back to a small constant when uncertainty not available
    liq_std = diagram.liquidus_T_std if diagram.liquidus_T_std is not None else np.full(G, 20.0)
    sol_std = diagram.solidus_T_std if diagram.solidus_T_std is not None else np.full(G, 20.0)

    pts: list[tuple[float, float, float]] = []
    for i, x in enumerate(diagram.comp_grid):
        T_liq = diagram.liquidus_T[i]
        T_sol = diagram.solidus_T[i]
        for T in T_grid:
            d_liq = abs(T - T_liq) / max(liq_std[i], 1.0)
            d_sol = abs(T - T_sol) / max(sol_std[i], 1.0)
            # Closer to either boundary -> higher uncertainty score.
            # Use a Gaussian kernel so the score peaks AT the boundary.
            score = np.exp(-0.5 * d_liq ** 2) + np.exp(-0.5 * d_sol ** 2)
            pts.append((float(x), float(T), float(score)))
    pts.sort(key=lambda t: -t[2])
    return pts[:top_k]
