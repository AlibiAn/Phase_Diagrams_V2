"""
Shared foundations: dataset loader, device detection, metrics helpers and the
SpecialistModel base class.

`load_dataset` reads the artefacts written by `mpds_preprocess.pipeline` into a
single PhaseDataset whose target arrays are all row-aligned with X.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """'cuda' if a CUDA device is available, else 'cpu' (valid for XGBoost ≥ 2.0 and torch)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def xgb_device_kwargs() -> dict:
    """Device kwargs for XGBoost constructors."""
    if detect_device() == "cuda":
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist", "n_jobs": -1}


def torch_device():
    import torch
    return torch.device(detect_device())


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Binary precision / recall / F1 from boolean (or 0/1) arrays of any shape."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def pad_class_proba(raw: np.ndarray, seen_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand XGBoost probabilities over the classes seen in training to all
    `n_classes` columns (zero for classes absent from the training set)."""
    if raw.shape[1] == n_classes:
        return raw
    full = np.zeros((raw.shape[0], n_classes), dtype=raw.dtype)
    full[:, seen_classes] = raw
    return full


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------

# Canonical topology class order (must match mpds_preprocess/targets.py)
TOPOLOGY_CLASSES = (
    "isomorphous",
    "simple_eutectic",
    "eutectic_with_intermetallics",
    "monotectic",
    "complex",
)


@dataclass
class PhaseDataset:
    """All preprocessing outputs, row-aligned and ready for training."""

    # Features
    X: np.ndarray                       # (N, F) float32
    feature_names: list[str]
    pair_ids: list[str]                 # e.g. 'Ag_Al'

    # M1: phase multi-hot
    Y_phases: np.ndarray                # (N, V) int8
    phase_vocab: list[str]

    # M2: liquidus
    T_liquidus: np.ndarray              # (N, 101) float32
    T_liquidus_mad: np.ndarray
    T_liquidus_mask: np.ndarray         # which grid points have data
    comp_grid: np.ndarray               # (101,) float32

    # M3: solidus
    T_solidus: np.ndarray
    T_solidus_mad: np.ndarray

    # M4: eutectic
    eutectic_exists: np.ndarray         # (N,) bool
    eutectic_comp: np.ndarray           # (N,) float, NaN where exists=False
    eutectic_temp: np.ndarray
    eutectic_temp_std: np.ndarray

    # M5: miscibility gap
    miscibility_gap: np.ndarray         # (N,) bool

    # M6: topology (as integer class index)
    topology: np.ndarray                # (N,) int8 index into topology_classes
    topology_classes: tuple[str, ...] = TOPOLOGY_CLASSES

    # Sample weights and provenance
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    n_sources: np.ndarray = field(default_factory=lambda: np.array([]))
    high_disagreement: np.ndarray = field(default_factory=lambda: np.array([]))

    # Splits
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


    @property
    def n_systems(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_phases(self) -> int:
        return len(self.phase_vocab)

    @property
    def n_topology_classes(self) -> int:
        return len(self.topology_classes)

    def split(self, name: str) -> np.ndarray:
        """Return the row indices for 'train' / 'val' / 'test'."""
        return {"train": self.train_idx, "val": self.val_idx,
                "test": self.test_idx}[name]

    def topology_name(self, i: int) -> str:
        return self.topology_classes[int(self.topology[i])]

    def indices_of(self, pairs: list[str]) -> np.ndarray:
        """Row indices of the named pairs; unknown pairs are skipped with a warning."""
        row = {p: i for i, p in enumerate(self.pair_ids)}
        missing = [p for p in pairs if p not in row]
        if missing:
            logger.warning("pairs not in dataset, skipping: %s", ", ".join(missing))
        return np.array([row[p] for p in pairs if p in row], dtype=np.int64)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(processed_dir: str | Path) -> PhaseDataset:
    """Load the preprocessing artefacts into one PhaseDataset, with every
    target realigned to the row order of features.parquet."""
    p = Path(processed_dir)

    X_df = pd.read_parquet(p / "features.parquet")
    pair_ids = X_df.index.tolist()
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
    # Trees tolerate NaN but the torch model does not, so zero-fill everywhere.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scalar_df = pd.read_parquet(p / "targets_scalar.parquet").reindex(pair_ids)
    cls_to_idx = {c: i for i, c in enumerate(TOPOLOGY_CLASSES)}
    topology = np.asarray(
        [cls_to_idx.get(t, cls_to_idx["complex"]) for t in scalar_df["m6_topology"]],
        dtype=np.int8,
    )

    liq = np.load(p / "targets_liquidus.npz", allow_pickle=False)
    sol = np.load(p / "targets_solidus.npz", allow_pickle=False)
    ph = np.load(p / "targets_phases.npz", allow_pickle=True)
    for name, arrs in (("targets_liquidus.npz", liq), ("targets_solidus.npz", sol),
                       ("targets_phases.npz", ph)):
        if "pair_ids" in arrs.files and list(arrs["pair_ids"]) != pair_ids:
            raise ValueError(f"{name} rows are not aligned with features.parquet")

    weights = pd.read_parquet(p / "weights.parquet").reindex(pair_ids)["weight"].to_numpy(np.float32)
    with (p / "splits.json").open() as f:
        splits = json.load(f)

    ds = PhaseDataset(
        X=X,
        feature_names=feature_names,
        pair_ids=pair_ids,
        Y_phases=ph["Y"].astype(np.int8),
        phase_vocab=list(ph["vocab"]),
        T_liquidus=liq["T"].astype(np.float32),
        T_liquidus_mad=liq["MAD"].astype(np.float32),
        T_liquidus_mask=liq["mask"].astype(bool) & np.isfinite(liq["T"]),
        comp_grid=liq["comp_grid"].astype(np.float32),
        T_solidus=sol["T"].astype(np.float32),
        T_solidus_mad=sol["MAD"].astype(np.float32),
        eutectic_exists=scalar_df["m4_eutectic_exists"].to_numpy(bool),
        eutectic_comp=scalar_df["m4_eutectic_comp"].to_numpy(np.float32),
        eutectic_temp=scalar_df["m4_eutectic_temp"].to_numpy(np.float32),
        eutectic_temp_std=scalar_df["m4_eutectic_temp_std"].to_numpy(np.float32),
        miscibility_gap=scalar_df["m5_miscibility_gap"].to_numpy(bool),
        topology=topology,
        topology_classes=TOPOLOGY_CLASSES,
        weights=weights,
        n_sources=scalar_df["n_sources"].to_numpy(np.int32),
        high_disagreement=scalar_df["high_disagreement"].to_numpy(bool),
        train_idx=np.asarray(splits["train"], dtype=np.int64),
        val_idx=np.asarray(splits["val"], dtype=np.int64),
        test_idx=np.asarray(splits["test"], dtype=np.int64),
    )
    logger.info("Loaded %d systems × %d features; %d train / %d val / %d test",
                ds.n_systems, ds.n_features, len(ds.train_idx),
                len(ds.val_idx), len(ds.test_idx))
    return ds


# ---------------------------------------------------------------------------
# Specialist model interface
# ---------------------------------------------------------------------------

class SpecialistModel:
    """Interface shared by M1-M6. Subclasses implement fit / predict / evaluate;
    save / load use joblib unless overridden (the torch model does)."""

    name: str = "base"

    def fit(self, ds: PhaseDataset) -> "SpecialistModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        raise NotImplementedError

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SpecialistModel":
        import joblib
        return joblib.load(path)
