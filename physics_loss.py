"""
Physics-informed differentiable loss penalties.

Used by the deep curve models (M2 liquidus, M3 solidus) and the joint
liquidus+solidus network. Every penalty here is a torch operation, so
gradients flow back to the network weights.

The four constraint families
----------------------------
  1.  Solidus ≤ Liquidus at every composition
        (hard physical constraint — a liquid can't be below its solidus)

  2.  Endpoint consistency at x = 0 and x = 100
        Liquidus(x=0) ≈ Tm(element A); same at x=100 for B
        Solidus and Liquidus must meet at x=0 and x=100 for pure elements

  3.  Curve smoothness (second-difference penalty)
        Penalises spurious sawtooth predictions. Real liquidus / solidus
        curves are piecewise smooth (kinks only at invariant points).

  4.  Bounded magnitude
        Liquidus must lie between the lowest reasonable T and the
        highest elemental Tm + safety margin. Prevents unphysical blow-ups.

Gibbs phase rule and convexity
------------------------------
For binary at fixed P, F = 3 − P. At an invariant point F = 0 (a unique
(x, T)), at a phase boundary F = 1, in a single-phase region F = 2. The
model outputs phase boundaries — it does NOT output a Gibbs energy
surface, so direct convexity-of-G penalties aren't applicable to this
output. The closest practical surrogate is to require liquidus / solidus
to be ≤ C² piecewise smooth, which the smoothness penalty enforces.

The lever rule applies inside two-phase regions: the proportions of two
phases at composition x must obey x = f·x_α + (1−f)·x_β. We expose this
as a separate `lever_rule_penalty` for downstream models that explicitly
predict phase fractions (M1 with proportions).
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
    """Hinge penalty: ReLU(T_sol - T_liq + margin), normalised by T_ref.

    Shape: (..., n_grid). Returns scalar.
    `margin` lets you require the solidus to lie at least `margin` K below
    the liquidus (default 0 = touching is allowed, as at endpoints).
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
    """At x=0 the curve must equal Tm(A), at x=100 it must equal Tm(B).

    Errors are normalised by T_ref so the penalty is dimensionless and
    comparable in scale to the (Huber / MAD-rescaled) data loss. With
    T_ref=100 K, an endpoint deviation of 100 K contributes 1.0 to the
    loss — roughly matching the data-loss magnitude in practice.
    """
    err0 = ((T_curve[..., 0] - Tm_A) / T_ref).pow(2)
    err1 = ((T_curve[..., -1] - Tm_B) / T_ref).pow(2)
    return weight_endpoints * (err0 + err1).mean()


# ---------------------------------------------------------------------------
# Constraint 3 — smoothness
# ---------------------------------------------------------------------------

def smoothness_penalty(T_curve: torch.Tensor, T_ref: float = 100.0) -> torch.Tensor:
    """L2 of the second difference along the composition axis, normalised by
    T_ref so the penalty is dimensionless and comparable to the data loss.

    For an n_grid-length curve, second difference d²T_i = T_{i+1} − 2 T_i + T_{i−1}.
    Real liquidus curves have a small handful of sharp kinks (at eutectics
    and peritectics) but are otherwise smooth, so a quadratic penalty here
    works well — it's gentle on isolated kinks and harsh on noise.
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
    """Penalty for any predicted temperature outside a reasonable range
    (below ambient or above the highest known elemental melting point + margin).
    """
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
    """x_overall = f_α · x_α + (1 − f_α) · x_β  must hold to within rounding.

    This is the mass-balance constraint inside any two-phase region.
    Not used by M2/M3 directly; exposed for the future M1-with-fractions
    model or the meta-learner's diagram reconstruction step.
    """
    predicted = f_alpha * x_alpha + (1 - f_alpha) * x_beta
    return (x_overall - predicted).pow(2).mean()


# ---------------------------------------------------------------------------
# Bundled loss for joint M2 + M3 training
# ---------------------------------------------------------------------------

class CurvePhysicsLoss(nn.Module):
    """Composite physics + data loss for the joint liquidus/solidus network.

    Loss = data MSE                            (weighted by mask + uncertainty)
         + λ₁ · solidus_le_liquidus
         + λ₂ · endpoint_consistency
         + λ₃ · smoothness
         + λ₄ · bounded_magnitude

    The lambda values are deliberately exposed as constructor args so
    they can be tuned per dataset. Defaults are reasonable starting points
    derived from rough scale matching of the individual terms on synthetic data.
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
        """Returns (total_loss, components_dict for logging)."""
        eps = 1e-3

        # Data loss — Huber, masked by which grid points have observed data
        loss_liq = self.huber(T_liq_pred, T_liq_true)
        loss_sol = self.huber(T_sol_pred, T_sol_true)
        if liq_mad is not None:
            loss_liq = loss_liq / (liq_mad + 25.0)         # downweight noisy points
        if sol_mad is not None:
            loss_sol = loss_sol / (sol_mad + 25.0)
        data_liq = (loss_liq * liq_mask.float()).sum() / (liq_mask.float().sum() + eps)
        data_sol = (loss_sol * sol_mask.float()).sum() / (sol_mask.float().sum() + eps)
        data_loss = data_liq + data_sol

        if sample_weights is not None:
            # Coarse per-system weighting on the data loss only
            sw = sample_weights.unsqueeze(-1)
            data_loss = (
                (loss_liq * liq_mask.float() * sw).sum() / ((liq_mask.float() * sw).sum() + eps)
                + (loss_sol * sol_mask.float() * sw).sum() / ((sol_mask.float() * sw).sum() + eps)
            )

        # Physics
        p_order = solidus_le_liquidus_penalty(T_sol_pred, T_liq_pred)
        p_ends = endpoint_consistency_penalty(T_liq_pred, Tm_A, Tm_B) + \
                 endpoint_consistency_penalty(T_sol_pred, Tm_A, Tm_B)
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
