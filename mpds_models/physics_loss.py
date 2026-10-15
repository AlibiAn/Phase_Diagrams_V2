"""
Physics-informed differentiable loss penalties.

Used by the joint liquidus (M2) + solidus (M3) network. Every penalty is a
torch operation, so gradients flow back to the network weights.

Constraints
-----------
  1. Order:      solidus ≤ liquidus at every composition
  2. Endpoints:  both curves meet Tm(A) at x = 0 and Tm(B) at x = 100
  3. Smoothness: curves are piecewise smooth (kinks only at invariant points)
  4. Bounds:     temperatures stay within a physically plausible range

The model predicts phase boundaries, not a Gibbs energy surface, so
convexity-of-G penalties don't apply; the smoothness term is the practical
surrogate. `lever_rule_penalty` (mass balance inside two-phase regions) is
provided for future models that predict phase fractions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constraint 1 — solidus ≤ liquidus
# ---------------------------------------------------------------------------

def solidus_le_liquidus_penalty(
    T_solidus: torch.Tensor,
    T_liquidus: torch.Tensor,
    *,
    margin: float = 0.0,
    T_ref: float = 100.0,
) -> torch.Tensor:
    """Squared hinge ReLU(T_sol - T_liq + margin) / T_ref, averaged to a scalar.

    `margin` > 0 requires the solidus to sit at least that many K below the
    liquidus; the default 0 allows them to touch (as at the endpoints).
    """
    violation = torch.relu(T_solidus - T_liquidus + margin) / T_ref
    return violation.pow(2).mean()


# ---------------------------------------------------------------------------
# Constraint 2 — endpoint consistency at pure-element compositions
# ---------------------------------------------------------------------------

def endpoint_consistency_penalty(
    T_curve: torch.Tensor,            # (..., n_grid)
    Tm_A: torch.Tensor,               # (...,)
    Tm_B: torch.Tensor,
    *,
    weight_endpoints: float = 1.0,
    T_ref: float = 100.0,
) -> torch.Tensor:
    """Squared error of T(x=0) vs Tm(A) and T(x=100) vs Tm(B), in units of
    T_ref (a 100 K miss contributes 1.0, comparable to the data loss).

    A melting point ≤ 0 means it was missing (load_dataset zero-fills NaN
    features), so that end is skipped rather than pulled towards 0 K.
    """
    err0 = ((T_curve[..., 0] - Tm_A) / T_ref).pow(2) * (Tm_A > 0)
    err1 = ((T_curve[..., -1] - Tm_B) / T_ref).pow(2) * (Tm_B > 0)
    return weight_endpoints * (err0 + err1).mean()


# ---------------------------------------------------------------------------
# Constraint 3 — smoothness
# ---------------------------------------------------------------------------

def smoothness_penalty(T_curve: torch.Tensor, T_ref: float = 100.0) -> torch.Tensor:
    """Mean squared second difference (T_{i+1} − 2T_i + T_{i−1}) / T_ref.

    Gentle on the few genuine kinks (eutectics, peritectics), harsh on noise.
    """
    if T_curve.shape[-1] < 3:
        return torch.zeros((), device=T_curve.device)
    d2 = (T_curve[..., 2:] - 2 * T_curve[..., 1:-1] + T_curve[..., :-2]) / T_ref
    return d2.pow(2).mean()


# ---------------------------------------------------------------------------
# Constraint 4 — bounded magnitude
# ---------------------------------------------------------------------------

def bounded_magnitude_penalty(
    T_curve: torch.Tensor,
    *,
    T_min: float = 100.0,
    T_max: float = 4000.0,
) -> torch.Tensor:
    """Squared penalty for temperatures outside [T_min, T_max]."""
    below = torch.relu(T_min - T_curve)
    above = torch.relu(T_curve - T_max)
    return (below.pow(2) + above.pow(2)).mean()


# ---------------------------------------------------------------------------
# Lever rule (for models that predict phase fractions explicitly)
# ---------------------------------------------------------------------------

def lever_rule_penalty(
    x_overall: torch.Tensor,           # (B,) — system composition
    x_alpha: torch.Tensor,             # (B,) — composition of α in two-phase region
    x_beta: torch.Tensor,
    f_alpha: torch.Tensor,             # (B,) — fraction of α, in [0, 1]
) -> torch.Tensor:
    """Mass balance inside a two-phase region: x = f_α·x_α + (1 − f_α)·x_β.
    Not used by M2/M3; provided for models that predict phase fractions."""
    predicted = f_alpha * x_alpha + (1 - f_alpha) * x_beta
    return (x_overall - predicted).pow(2).mean()


# ---------------------------------------------------------------------------
# Bundled loss for joint M2 + M3 training
# ---------------------------------------------------------------------------

class CurvePhysicsLoss(nn.Module):
    """Composite physics + data loss for the joint liquidus/solidus network.

    Loss = data Huber                          (masked, MAD- and sample-weighted)
         + λ₁ · solidus_le_liquidus
         + λ₂ · endpoint_consistency
         + λ₃ · smoothness
         + λ₄ · bounded_magnitude

    The lambdas are constructor args so they can be tuned per dataset; the
    defaults come from rough scale-matching of the terms on synthetic data.
    """

    def __init__(
        self,
        lambda_order: float = 1.0,
        lambda_endpoints: float = 0.5,
        lambda_smooth: float = 1e-3,
        lambda_bounds: float = 1e-3,
        huber_delta: float = 50.0,
    ):
        super().__init__()
        self.lam_order = lambda_order
        self.lam_endpoints = lambda_endpoints
        self.lam_smooth = lambda_smooth
        self.lam_bounds = lambda_bounds
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)

    def forward(
        self,
        T_liq_pred: torch.Tensor,           # (B, G)
        T_sol_pred: torch.Tensor,           # (B, G)
        T_liq_true: torch.Tensor,
        T_sol_true: torch.Tensor,
        liq_mask: torch.Tensor,             # (B, G) bool
        sol_mask: torch.Tensor,
        Tm_A: torch.Tensor,                 # (B,)
        Tm_B: torch.Tensor,
        liq_mad: torch.Tensor | None = None,
        sol_mad: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,    # (B,)
    ) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, per-component values for logging)."""
        loss_liq = self.huber(T_liq_pred, T_liq_true)
        loss_sol = self.huber(T_sol_pred, T_sol_true)
        # Down-weight grid points where the sources disagree (high MAD)
        if liq_mad is not None:
            loss_liq = loss_liq / (liq_mad + 25.0)
        if sol_mad is not None:
            loss_sol = loss_sol / (sol_mad + 25.0)

        # Masked mean over observed grid points, optionally weighted per system
        w_liq, w_sol = liq_mask.float(), sol_mask.float()
        if sample_weights is not None:
            sw = sample_weights.unsqueeze(-1)
            w_liq, w_sol = w_liq * sw, w_sol * sw
        eps = 1e-3
        data_loss = ((loss_liq * w_liq).sum() / (w_liq.sum() + eps)
                     + (loss_sol * w_sol).sum() / (w_sol.sum() + eps))

        # Physics
        p_order = solidus_le_liquidus_penalty(T_sol_pred, T_liq_pred)
        p_ends = (endpoint_consistency_penalty(T_liq_pred, Tm_A, Tm_B)
                  + endpoint_consistency_penalty(T_sol_pred, Tm_A, Tm_B))
        p_smooth = smoothness_penalty(T_liq_pred) + smoothness_penalty(T_sol_pred)
        p_bounds = bounded_magnitude_penalty(T_liq_pred) + bounded_magnitude_penalty(T_sol_pred)

        total = (
            data_loss
            + self.lam_order * p_order
            + self.lam_endpoints * p_ends
            + self.lam_smooth * p_smooth
            + self.lam_bounds * p_bounds
        )

        return total, {
            "data": data_loss.detach().item(),
            "order": p_order.detach().item(),
            "endpoints": p_ends.detach().item(),
            "smooth": p_smooth.detach().item(),
            "bounds": p_bounds.detach().item(),
            "total": total.detach().item(),
        }
