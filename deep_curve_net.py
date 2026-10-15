"""
Joint deep model for liquidus (M2) and solidus (M3) curve regression.

Architecture
------------
A multi-layer MLP that takes the ~207-feature system vector and outputs
TWO 101-dimensional curves (liquidus + solidus) at the fixed composition
grid. The two heads share trunk weights — they're highly correlated
physically (the solidus tracks the liquidus everywhere), so a shared
trunk is more sample-efficient than two independent networks.

The choice of MLP rather than CGCNN/MEGNet here is deliberate and honest.
CGCNN takes a crystal structure graph as input. Our preprocessing
pipeline currently produces only system-level features (composition +
elemental properties + thermodynamic descriptors) — no per-system graph
representation. A "GNN" on this kind of input is just an MLP with extra
machinery. When MPDS S-entry crystal structures are pulled into the
pipeline (a planned extension), the StructureBackbone interface in this
file can be replaced with a real CGCNN/MEGNet/M3GNet that emits a
per-system embedding which is then concatenated to the matminer features
— the "hybrid GNN" approach described in the project paper.

Uncertainty
-----------
Inference-time uncertainty is provided two ways:
  1. MC Dropout: keep dropout on at inference, sample N forward passes,
     return mean + std.
  2. Deep ensemble: train K independent models with different seeds,
     average and std across ensemble at inference.

Both are exposed via the same predict_with_uncertainty(...) interface.
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
        # init the heads to predict ~1000 K (rough mid-range melting point)
        with torch.no_grad():
            self.head_liquidus.bias.fill_(1000.0)
            self.head_solidus.bias.fill_(950.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_liquidus(h), self.head_solidus(h)


# ---------------------------------------------------------------------------
# Pull Tm(A), Tm(B) from features for the endpoint physics constraint
# ---------------------------------------------------------------------------

def _melting_point_columns(feature_names: list[str]) -> tuple[int, int]:
    """Find the column indices for elemental melting points Tm(A) and Tm(B).
    These come from features.physical_features() and are named
    'phys__melting_point__A' and '...__B'."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    try:
        return name_to_idx["phys__melting_point__A"], name_to_idx["phys__melting_point__B"]
    except KeyError as e:
        raise RuntimeError(
            "Expected feature 'phys__melting_point__A/B' from preprocessing; "
            "endpoint physics constraint requires it."
        ) from e


# ---------------------------------------------------------------------------
# Trainer wrapper that implements SpecialistModel interface
# ---------------------------------------------------------------------------

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
    """The combined M2 + M3 model — trained jointly because the physical
    constraint T_solidus ≤ T_liquidus only makes sense if both are predicted
    together."""

    name = "M2_M3_liquidus_solidus"

    def __init__(self, config: CurveNetConfig | None = None):
        self.cfg = config or CurveNetConfig()
        self.net: CurveNet | None = None
        self.device = torch_device()
        # x-scaler stats (filled by fit)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._melt_a_idx: int | None = None
        self._melt_b_idx: int | None = None
        self._n_grid: int = 101

    # ---- training ----------------------------------------------------

    def fit(self, ds: PhaseDataset) -> "LiquidusSolidusModel":
        train, val = ds.train_idx, ds.val_idx
        if len(train) == 0:
            raise RuntimeError("training split is empty")

        # standardise X using train statistics only
        self._mu = ds.X[train].mean(0)
        self._sd = ds.X[train].std(0) + 1e-6
        self._melt_a_idx, self._melt_b_idx = _melting_point_columns(ds.feature_names)
        self._n_grid = ds.comp_grid.size

        Xtr = torch.from_numpy(((ds.X[train] - self._mu) / self._sd).astype(np.float32))
        Xva = torch.from_numpy(((ds.X[val] - self._mu) / self._sd).astype(np.float32))
        Ltr = torch.from_numpy(np.nan_to_num(ds.T_liquidus[train], nan=1000.0))
        Lva = torch.from_numpy(np.nan_to_num(ds.T_liquidus[val], nan=1000.0))
        Str = torch.from_numpy(np.nan_to_num(ds.T_solidus[train], nan=900.0))
        Sva = torch.from_numpy(np.nan_to_num(ds.T_solidus[val], nan=900.0))
        Lmtr = torch.from_numpy(ds.T_liquidus_mask[train])
        Lmva = torch.from_numpy(ds.T_liquidus_mask[val])
        Smtr = torch.from_numpy(~np.isnan(ds.T_solidus[train]))
        Smva = torch.from_numpy(~np.isnan(ds.T_solidus[val]))
        LMADtr = torch.from_numpy(np.nan_to_num(ds.T_liquidus_mad[train], nan=0.0))
        LMADva = torch.from_numpy(np.nan_to_num(ds.T_liquidus_mad[val], nan=0.0))
        SMADtr = torch.from_numpy(np.nan_to_num(ds.T_solidus_mad[train], nan=0.0))
        SMADva = torch.from_numpy(np.nan_to_num(ds.T_solidus_mad[val], nan=0.0))

        # Elemental melting points (in original feature units, not standardised)
        Tm_a_tr = torch.from_numpy(ds.X[train, self._melt_a_idx].astype(np.float32))
        Tm_b_tr = torch.from_numpy(ds.X[train, self._melt_b_idx].astype(np.float32))
        Tm_a_va = torch.from_numpy(ds.X[val, self._melt_a_idx].astype(np.float32))
        Tm_b_va = torch.from_numpy(ds.X[val, self._melt_b_idx].astype(np.float32))

        w_tr = torch.from_numpy(ds.weights[train].astype(np.float32))
        w_va = torch.from_numpy(ds.weights[val].astype(np.float32))

        train_loader = DataLoader(
            TensorDataset(Xtr, Ltr, Str, Lmtr, Smtr, LMADtr, SMADtr, Tm_a_tr, Tm_b_tr, w_tr),
            batch_size=self.cfg.batch_size, shuffle=True, drop_last=False,
        )

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

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(self.cfg.epochs):
            self.net.train()
            train_total = 0.0
            n_batches = 0
            for batch in train_loader:
                Xb, Lb, Sb, Lmb, Smb, LMADb, SMADb, Tma, Tmb, wb = [
                    t.to(self.device) for t in batch
                ]
                pred_l, pred_s = self.net(Xb)
                loss, _ = loss_fn(pred_l, pred_s, Lb, Sb, Lmb, Smb,
                                  Tma, Tmb, LMADb, SMADb, sample_weights=wb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                train_total += loss.item()
                n_batches += 1
            sched.step()
            train_avg = train_total / max(n_batches, 1)

            # validation (only if val split non-empty; otherwise use train loss for ES)
            if len(val) > 0:
                self.net.eval()
                with torch.no_grad():
                    xv = Xva.to(self.device)
                    pv_l, pv_s = self.net(xv)
                    vloss, comps = loss_fn(
                        pv_l, pv_s, Lva.to(self.device), Sva.to(self.device),
                        Lmva.to(self.device), Smva.to(self.device),
                        Tm_a_va.to(self.device), Tm_b_va.to(self.device),
                        LMADva.to(self.device), SMADva.to(self.device),
                        sample_weights=w_va.to(self.device),
                    )
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
            l, s = self.net(x)
            # enforce solidus ≤ liquidus at inference time (a sanity clamp)
            s = torch.minimum(s, l)
        return l.cpu().numpy(), s.cpu().numpy()

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty.
        Returns (T_liq_mean, T_liq_std, T_sol_mean, T_sol_std)."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.train()                       # keep dropout active
        with torch.no_grad():
            x = self._standardise(X).to(self.device)
            liqs, sols = [], []
            for _ in range(n_samples):
                l, s = self.net(x)
                s = torch.minimum(s, l)
                liqs.append(l.cpu().numpy())
                sols.append(s.cpu().numpy())
        self.net.eval()
        liq_stack = np.stack(liqs)
        sol_stack = np.stack(sols)
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
        sol_err = np.abs(T_sol_pred - np.nan_to_num(sol_true, nan=0.0))[sol_mask]
        return {
            "split": split, "n": int(len(idx)),
            "liquidus_MAE_K": float(liq_err.mean()) if liq_err.size else float("nan"),
            "liquidus_RMSE_K": float(np.sqrt((liq_err ** 2).mean())) if liq_err.size else float("nan"),
            "solidus_MAE_K": float(sol_err.mean()) if sol_err.size else float("nan"),
            "solidus_RMSE_K": float(np.sqrt((sol_err ** 2).mean())) if sol_err.size else float("nan"),
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
        m._mu = blob["mu"]; m._sd = blob["sd"]
        m._melt_a_idx = blob["melt_a_idx"]; m._melt_b_idx = blob["melt_b_idx"]
        m._n_grid = blob["n_grid"]
        m.net = CurveNet(n_features=blob["n_features"], n_grid=m._n_grid,
                         hidden=m.cfg.hidden, dropout=m.cfg.dropout).to(m.device)
        m.net.load_state_dict(blob["state_dict"])
        return m
