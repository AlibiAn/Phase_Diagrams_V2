"""
Joint deep model for liquidus (M2) and solidus (M3) curve regression.

Architecture
------------
An MLP maps the system feature vector to TWO curves (liquidus + solidus) on
the fixed composition grid. The heads share one trunk: the solidus tracks the
liquidus physically, so a shared representation is more sample-efficient.

An MLP rather than CGCNN/MEGNet is deliberate: the preprocessing produces
system-level features only (composition, elemental properties, thermodynamic
descriptors), not crystal graphs, and a GNN on such input is just an MLP with
extra machinery. Once MPDS crystal structures enter the pipeline, a structure
GNN embedding can be concatenated to these features ("hybrid GNN").

Uncertainty
-----------
MC Dropout: `predict_with_uncertainty` keeps dropout active at inference and
returns the mean and std over N stochastic forward passes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import PhaseDataset, SpecialistModel, torch_device
from .physics_loss import CurvePhysicsLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CurveNet(nn.Module):
    """Shared-trunk MLP that outputs liquidus + solidus curves at a fixed grid."""

    def __init__(
        self,
        n_features: int,
        n_grid: int = 101,
        hidden: tuple[int, ...] = (512, 512, 256, 256),
        dropout: float = 0.15,
    ):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_liquidus = nn.Linear(prev, n_grid)
        self.head_solidus = nn.Linear(prev, n_grid)
        # Start near a typical melting point so early training is stable
        with torch.no_grad():
            self.head_liquidus.bias.fill_(1000.0)
            self.head_solidus.bias.fill_(950.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_liquidus(h), self.head_solidus(h)


# ---------------------------------------------------------------------------
# Trainer implementing the SpecialistModel interface
# ---------------------------------------------------------------------------

def _melting_point_columns(feature_names: list[str]) -> tuple[int, int]:
    """Column indices of the elemental melting points Tm(A), Tm(B), needed by
    the endpoint constraint."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    try:
        return name_to_idx["phys__melting_point__A"], name_to_idx["phys__melting_point__B"]
    except KeyError as e:
        raise RuntimeError(
            "Expected feature 'phys__melting_point__A/B' from preprocessing; "
            "endpoint physics constraint requires it."
        ) from e


@dataclass
class CurveNetConfig:
    hidden: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 200
    early_stop_patience: int = 20
    lambda_order: float = 1.0
    lambda_endpoints: float = 0.5
    lambda_smooth: float = 1e-3
    lambda_bounds: float = 1e-3


class LiquidusSolidusModel(SpecialistModel):
    """M2 + M3, trained jointly so the solidus ≤ liquidus constraint can act
    on both curves at once."""

    name = "M2_M3_liquidus_solidus"

    def __init__(self, config: CurveNetConfig | None = None):
        self.cfg = config or CurveNetConfig()
        self.net: CurveNet | None = None
        self.device = torch_device()
        # Feature standardisation stats, set by fit()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._melt_a_idx: int | None = None
        self._melt_b_idx: int | None = None
        self._n_grid: int = 101

    # ---- training ----------------------------------------------------

    def _tensors(self, ds: PhaseDataset, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        """Training tensors for rows `idx`, in the argument order of CurvePhysicsLoss.

        Missing targets are filled with placeholders; the masks keep them out
        of the data loss. Tm(A), Tm(B) stay in raw (unstandardised) units.
        """
        sol = ds.T_solidus[idx]
        arrays = (
            np.nan_to_num(ds.T_liquidus[idx], nan=1000.0),
            np.nan_to_num(sol, nan=900.0),
            ds.T_liquidus_mask[idx],
            ~np.isnan(sol),
            ds.X[idx, self._melt_a_idx].astype(np.float32),
            ds.X[idx, self._melt_b_idx].astype(np.float32),
            np.nan_to_num(ds.T_liquidus_mad[idx], nan=0.0),
            np.nan_to_num(ds.T_solidus_mad[idx], nan=0.0),
            ds.weights[idx].astype(np.float32),
        )
        return (self._standardise(ds.X[idx]), *map(torch.from_numpy, arrays))

    def fit(self, ds: PhaseDataset) -> "LiquidusSolidusModel":
        train, val = ds.train_idx, ds.val_idx
        if len(train) == 0:
            raise RuntimeError("training split is empty")

        # Standardise X with train statistics only
        self._mu = ds.X[train].mean(0)
        self._sd = ds.X[train].std(0) + 1e-6
        self._melt_a_idx, self._melt_b_idx = _melting_point_columns(ds.feature_names)
        self._n_grid = ds.comp_grid.size

        train_loader = DataLoader(TensorDataset(*self._tensors(ds, train)),
                                  batch_size=self.cfg.batch_size, shuffle=True)
        val_batch = [t.to(self.device) for t in self._tensors(ds, val)] if len(val) else None

        self.net = CurveNet(
            n_features=ds.n_features, n_grid=self._n_grid,
            hidden=self.cfg.hidden, dropout=self.cfg.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.epochs)
        loss_fn = CurvePhysicsLoss(
            lambda_order=self.cfg.lambda_order,
            lambda_endpoints=self.cfg.lambda_endpoints,
            lambda_smooth=self.cfg.lambda_smooth,
            lambda_bounds=self.cfg.lambda_bounds,
        )

        def batch_loss(batch):
            x, *targets, w = batch
            pred_l, pred_s = self.net(x)
            return loss_fn(pred_l, pred_s, *targets, sample_weights=w)

        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(self.cfg.epochs):
            self.net.train()
            train_total, n_batches = 0.0, 0
            for batch in train_loader:
                loss, _ = batch_loss([t.to(self.device) for t in batch])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                train_total += loss.item()
                n_batches += 1
            sched.step()
            train_avg = train_total / max(n_batches, 1)

            # Early stopping tracks val loss, or train loss if there is no val split
            if val_batch is not None:
                self.net.eval()
                with torch.no_grad():
                    vloss, comps = batch_loss(val_batch)
                v = vloss.item()
                log_extras = (
                    f"val={v:.3f}  [data={comps['data']:.2f} order={comps['order']:.2f} "
                    f"end={comps['endpoints']:.2f} smooth={comps['smooth']:.4f}]"
                )
            else:
                v = train_avg
                log_extras = "(no val split — using train loss for early stop)"

            if v < best_val:
                best_val = v
                best_state = {k: t.detach().cpu().clone() for k, t in self.net.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if epoch % 10 == 0 or epoch == self.cfg.epochs - 1:
                logger.info("CurveNet ep %3d  train=%.3f  %s", epoch, train_avg, log_extras)
            if patience >= self.cfg.early_stop_patience:
                logger.info("CurveNet early stop at epoch %d (best %.3f)", epoch, best_val)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ---- inference ---------------------------------------------------

    def _standardise(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((X - self._mu) / self._sd).astype(np.float32))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (T_liquidus, T_solidus), each shape (N, n_grid)."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        with torch.no_grad():
            x = self._standardise(X).to(self.device)
            liq, sol = self.net(x)
            sol = torch.minimum(sol, liq)          # enforce solidus ≤ liquidus
        return liq.cpu().numpy(), sol.cpu().numpy()

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty. Returns (liq_mean, liq_std, sol_mean, sol_std)."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.train()                       # keep dropout active
        with torch.no_grad():
            x = self._standardise(X).to(self.device)
            liqs, sols = [], []
            for _ in range(n_samples):
                liq, sol = self.net(x)
                liqs.append(liq)
                sols.append(torch.minimum(sol, liq))
        self.net.eval()
        liq_stack = torch.stack(liqs).cpu().numpy()
        sol_stack = torch.stack(sols).cpu().numpy()
        return liq_stack.mean(0), liq_stack.std(0), sol_stack.mean(0), sol_stack.std(0)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        T_liq_pred, T_sol_pred = self.predict(ds.X[idx])
        liq_true = ds.T_liquidus[idx]
        sol_true = ds.T_solidus[idx]
        liq_mask = ds.T_liquidus_mask[idx]
        sol_mask = ~np.isnan(sol_true)
        liq_err = np.abs(T_liq_pred - liq_true)[liq_mask]
        sol_err = np.abs(T_sol_pred - sol_true)[sol_mask]

        def mae(e):
            return float(e.mean()) if e.size else float("nan")

        def rmse(e):
            return float(np.sqrt((e ** 2).mean())) if e.size else float("nan")

        return {
            "split": split, "n": int(len(idx)),
            "liquidus_MAE_K": mae(liq_err), "liquidus_RMSE_K": rmse(liq_err),
            "solidus_MAE_K": mae(sol_err), "solidus_RMSE_K": rmse(sol_err),
            "order_violation_fraction": float((T_sol_pred > T_liq_pred + 1.0).mean()),
        }

    # ---- save / load -------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("nothing to save")
        torch.save({
            "state_dict": self.net.state_dict(),
            "cfg": self.cfg.__dict__,
            "mu": self._mu, "sd": self._sd,
            "melt_a_idx": self._melt_a_idx, "melt_b_idx": self._melt_b_idx,
            "n_grid": self._n_grid,
            "n_features": self._mu.shape[0],
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LiquidusSolidusModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(CurveNetConfig(**blob["cfg"]))
        m._mu, m._sd = blob["mu"], blob["sd"]
        m._melt_a_idx, m._melt_b_idx = blob["melt_a_idx"], blob["melt_b_idx"]
        m._n_grid = blob["n_grid"]
        m.net = CurveNet(n_features=blob["n_features"], n_grid=m._n_grid,
                         hidden=m.cfg.hidden, dropout=m.cfg.dropout).to(m.device)
        m.net.load_state_dict(blob["state_dict"])
        return m
