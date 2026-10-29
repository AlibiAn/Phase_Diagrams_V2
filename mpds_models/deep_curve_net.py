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

Output scaling
--------------
The heads output O(1) numbers that are converted to kelvin, instead of raw
kelvin. Adam moves each weight by roughly `lr` per step, so raw-kelvin heads
could only drift a fraction of a kelvin per step and stayed close to their
initial guess plus a crude shared shape.

    penalty mode:  T = output_offset_K + output_scale_K · head
    hard mode:     T = line(Tm_A, Tm_B) + 4x(1-x) · output_scale_K · head

Checkpoints saved before output scaling existed load with offset 0 and
scale 1, so they behave exactly as when they were trained.

Endpoints
---------
  endpoint_mode="penalty"  (default)
      A loss term pulls the first / last grid point towards Tm(A) / Tm(B).
      Every grid point has its own output weights, so a strong penalty can
      move only the end points, showing up as a jump at x = 0 / 100 when the
      data and the melting points disagree.

  endpoint_mode="hard"
      Curves always pass through the melting points: the network learns only
      the bend away from the straight line, which vanishes at both ends.
      Use it only when targets are in kelvin and the A/B axis is right.

  mask_end_points=True
      The data values at x = 0 and 100 are left out of the data loss, so only
      the endpoint constraint sets the ends. A third of the MPDS solidus ends
      carry the eutectic temperature instead of Tm (check_endpoints.py), and
      without this the data loss and the endpoint penalty pull those ends in
      opposite directions.

Solidus
-------
  solidus_mode="gap"  (default)
      The solidus head predicts the gap below the liquidus:
          T_sol = T_liq - output_scale_K · softplus(head)
      softplus is never negative, so solidus ≤ liquidus holds by construction
      instead of relying on the order penalty, which narrow-gap (isomorphous)
      systems routinely violated.

  solidus_mode="free"
      Both curves are predicted independently (all models before the gap mode
      existed; checkpoints without the setting load in this mode).

Model selection
---------------
Early stopping keeps the epoch with the lowest validation data error (mean
|pred - true| in K over observed points), not the lowest total loss, which
includes the physics penalties and so rewards hitting the endpoints rather
than fitting the curves.

Uncertainty
-----------
MC Dropout: `predict_with_uncertainty` keeps dropout active at inference and
returns the mean and std over N stochastic forward passes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import PhaseDataset, SpecialistModel, curve_metrics, swap_augment, torch_device
from .physics_loss import CurvePhysicsLoss

logger = logging.getLogger(__name__)

ENDPOINT_MODES = ("penalty", "hard")
SOLIDUS_MODES = ("gap", "free")
INITIAL_GAP_K = 50.0                    # starting solidus offset below the liquidus


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CurveNet(nn.Module):
    """Shared-trunk MLP that outputs liquidus + solidus curves at a fixed grid.

    The head biases set the starting curves (after the wrapper's output
    scaling): liquidus near 1000 K or the melting-point line, solidus ~50 K below.
    """

    def __init__(
        self,
        n_features: int,
        n_grid: int = 101,
        hidden: tuple[int, ...] = (512, 512, 256, 256),
        dropout: float = 0.15,
        init_bias_liquidus: float = 1000.0,
        init_bias_solidus: float = 950.0,
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
        with torch.no_grad():
            self.head_liquidus.bias.fill_(init_bias_liquidus)
            self.head_solidus.bias.fill_(init_bias_solidus)

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
    endpoint_mode: str = "penalty"      # "penalty" or "hard"
    output_offset_K: float = 1000.0     # T = offset + scale · head (penalty mode)
    output_scale_K: float = 500.0       # 1 head unit = 500 K
    solidus_mode: str = "gap"           # "gap" (solidus ≤ liquidus by construction) or "free"
    mask_end_points: bool = False       # leave the x = 0 / 100 data values out of the data loss
    swap_augment: bool = False          # also train on every system with A and B swapped


class LiquidusSolidusModel(SpecialistModel):
    """M2 + M3, trained jointly so the solidus ≤ liquidus constraint can act
    on both curves at once."""

    name = "M2_M3_liquidus_solidus"

    def __init__(self, config: CurveNetConfig | None = None):
        self.cfg = config or CurveNetConfig()
        if self.cfg.endpoint_mode not in ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {ENDPOINT_MODES}")
        if self.cfg.solidus_mode not in SOLIDUS_MODES:
            raise ValueError(f"solidus_mode must be one of {SOLIDUS_MODES}")
        self.net: CurveNet | None = None
        self.device = torch_device()
        # Feature standardisation stats, set by fit()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._melt_a_idx: int | None = None
        self._melt_b_idx: int | None = None
        self._n_grid: int = 101
        self._comp_grid: np.ndarray = np.linspace(0.0, 100.0, 101, dtype=np.float32)

    # ---- curve construction -------------------------------------------

    @property
    def _hard(self) -> bool:
        return self.cfg.endpoint_mode == "hard"

    @property
    def _gap(self) -> bool:
        return self.cfg.solidus_mode == "gap"

    def _new_net(self, n_features: int) -> CurveNet:
        scale = float(self.cfg.output_scale_K)
        offset = 0.0 if self._hard else float(self.cfg.output_offset_K)
        if self._gap:
            # liquidus at offset / line; softplus(bias_s) · scale = INITIAL_GAP_K
            bias_l = 0.0
            bias_s = float(np.log(np.expm1(INITIAL_GAP_K / scale)))
        elif scale == 1.0 and offset == 0.0 and not self._hard:
            bias_l, bias_s = 1000.0, 950.0       # raw-kelvin heads (old checkpoints)
        else:
            bias_l, bias_s = 0.0, -INITIAL_GAP_K / scale
        return CurveNet(n_features=n_features, n_grid=self._n_grid,
                        hidden=self.cfg.hidden, dropout=self.cfg.dropout,
                        init_bias_liquidus=bias_l, init_bias_solidus=bias_s).to(self.device)

    def _curves(self, x_std: torch.Tensor, tm_a: torch.Tensor,
                tm_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Network output -> (liquidus, solidus) in K."""
        h_liq, h_sol = self.net(x_std)
        scale = float(self.cfg.output_scale_K)
        if self._hard:
            frac = torch.as_tensor(self._comp_grid / 100.0, dtype=h_liq.dtype, device=h_liq.device)
            line = (1.0 - frac) * tm_a.unsqueeze(-1) + frac * tm_b.unsqueeze(-1)
            bend = 4.0 * frac * (1.0 - frac) * scale
            liq, sol = line + bend * h_liq, line + bend * h_sol
        else:
            offset = float(self.cfg.output_offset_K)
            liq, sol = offset + scale * h_liq, offset + scale * h_sol
        if self._gap:
            sol = liq - scale * nn.functional.softplus(h_sol)
        return liq, sol

    def _inputs(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(standardised X, Tm_A, Tm_B) on the model's device."""
        tm_a = torch.from_numpy(np.asarray(X[:, self._melt_a_idx], dtype=np.float32))
        tm_b = torch.from_numpy(np.asarray(X[:, self._melt_b_idx], dtype=np.float32))
        return (self._standardise(X).to(self.device),
                tm_a.to(self.device), tm_b.to(self.device))

    # ---- training ----------------------------------------------------

    def _tensors(self, ds: PhaseDataset, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
        """(X, then the CurvePhysicsLoss target arguments, then weights) for rows `idx`.

        Missing targets are filled with placeholders; the masks keep them out
        of the data loss. Tm(A), Tm(B) stay in raw (unstandardised) units.
        """
        sol = ds.T_solidus[idx]
        liq_mask, sol_mask = ds.T_liquidus_mask[idx].copy(), ~np.isnan(sol)
        if self.cfg.mask_end_points:
            liq_mask[:, [0, -1]] = False
            sol_mask[:, [0, -1]] = False
        arrays = (
            np.nan_to_num(ds.T_liquidus[idx], nan=1000.0),
            np.nan_to_num(sol, nan=900.0),
            liq_mask,
            sol_mask,
            ds.X[idx, self._melt_a_idx].astype(np.float32),
            ds.X[idx, self._melt_b_idx].astype(np.float32),
            np.nan_to_num(ds.T_liquidus_mad[idx], nan=0.0),
            np.nan_to_num(ds.T_solidus_mad[idx], nan=0.0),
            ds.weights[idx].astype(np.float32),
        )
        return (self._standardise(ds.X[idx]), *map(torch.from_numpy, arrays))

    def fit(self, ds: PhaseDataset) -> "LiquidusSolidusModel":
        if self.cfg.swap_augment:
            ds = swap_augment(ds)
        train, val = ds.train_idx, ds.val_idx
        if len(train) == 0:
            raise RuntimeError("training split is empty")

        # Standardise X with train statistics only
        self._mu = ds.X[train].mean(0)
        self._sd = ds.X[train].std(0) + 1e-6
        self._melt_a_idx, self._melt_b_idx = _melting_point_columns(ds.feature_names)
        self._n_grid = ds.comp_grid.size
        self._comp_grid = np.asarray(ds.comp_grid, dtype=np.float32)

        if self._hard:
            n_missing = int(((ds.X[train, self._melt_a_idx] <= 0) |
                             (ds.X[train, self._melt_b_idx] <= 0)).sum())
            if n_missing:
                logger.warning("hard endpoints: %d training systems have a missing Tm "
                               "(stored as 0); their curve ends will be pinned to 0 K", n_missing)

        train_loader = DataLoader(TensorDataset(*self._tensors(ds, train)),
                                  batch_size=self.cfg.batch_size, shuffle=True)
        val_batch = [t.to(self.device) for t in self._tensors(ds, val)] if len(val) else None

        self.net = self._new_net(ds.n_features)
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
            tm_a, tm_b = targets[4], targets[5]
            pred_l, pred_s = self._curves(x, tm_a, tm_b)
            loss, comps = loss_fn(pred_l, pred_s, *targets, sample_weights=w)
            return loss, comps, pred_l, pred_s

        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(self.cfg.epochs):
            self.net.train()
            train_total, n_batches = 0.0, 0
            for batch in train_loader:
                loss, *_ = batch_loss([t.to(self.device) for t in batch])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                train_total += loss.item()
                n_batches += 1
            sched.step()
            train_avg = train_total / max(n_batches, 1)

            # Early stopping tracks the val data error in K (train loss if no val split)
            v = train_avg
            log_extras = "(no val split — using train loss for early stop)"
            if val_batch is not None:
                self.net.eval()
                with torch.no_grad():
                    _, comps, pv_l, pv_s = batch_loss(val_batch)
                    _, L, S, Lm, Sm = val_batch[:5]
                    liq_err = (pv_l - L).abs()[Lm]
                    sol_err = (pv_s - S).abs()[Sm]
                    errs = [e.mean().item() for e in (liq_err, sol_err) if e.numel()]
                if errs:
                    v = float(np.mean(errs))
                log_extras = (
                    f"val_data_MAE={v:.1f}K  [loss parts: data={comps['data']:.2f} "
                    f"order={comps['order']:.2f} end={comps['endpoints']:.2f} "
                    f"smooth={comps['smooth']:.4f}]"
                )

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

    def predict(self, X: np.ndarray, clamp: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Returns (T_liquidus, T_solidus), each shape (N, n_grid). With `clamp`
        the solidus is capped at the liquidus; clamp=False gives the raw output."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        with torch.no_grad():
            liq, sol = self._curves(*self._inputs(X))
            if clamp:
                sol = torch.minimum(sol, liq)
        return liq.cpu().numpy(), sol.cpu().numpy()

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 30, clamp: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty. Returns (liq_mean, liq_std, sol_mean, sol_std)."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.train()                       # keep dropout active
        try:
            with torch.no_grad():
                inputs = self._inputs(X)
                liqs, sols = [], []
                for _ in range(n_samples):
                    liq, sol = self._curves(*inputs)
                    liqs.append(liq)
                    sols.append(torch.minimum(sol, liq) if clamp else sol)
        finally:
            self.net.eval()
        liq_stack = torch.stack(liqs).cpu().numpy()
        sol_stack = torch.stack(sols).cpu().numpy()
        return liq_stack.mean(0), liq_stack.std(0), sol_stack.mean(0), sol_stack.std(0)

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        idx = ds.split(split)
        if len(idx) == 0:
            return {"split": split, "n": 0}
        raw_liq, raw_sol = self.predict(ds.X[idx], clamp=False)
        return {"split": split, "n": int(len(idx)), **curve_metrics(ds, idx, raw_liq, raw_sol)}

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
            "comp_grid": self._comp_grid,
            "n_features": self._mu.shape[0],
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LiquidusSolidusModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        saved = dict(blob["cfg"])
        if "output_scale_K" not in saved:
            # Saved before output scaling existed: the heads output kelvin directly
            saved.update(output_scale_K=1.0, output_offset_K=0.0)
        # Saved before the gap mode existed: independent solidus head
        saved.setdefault("solidus_mode", "free")
        known = {f.name for f in fields(CurveNetConfig)}
        m = cls(CurveNetConfig(**{k: v for k, v in saved.items() if k in known}))
        m._mu, m._sd = blob["mu"], blob["sd"]
        m._melt_a_idx, m._melt_b_idx = blob["melt_a_idx"], blob["melt_b_idx"]
        m._n_grid = blob["n_grid"]
        m._comp_grid = np.asarray(blob.get("comp_grid", np.linspace(0.0, 100.0, m._n_grid)),
                                  dtype=np.float32)
        m.net = m._new_net(blob["n_features"])
        m.net.load_state_dict(blob["state_dict"])
        return m
