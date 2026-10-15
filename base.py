"""
Foundational utilities: dataset loader, GPU detection, common base classes.

The PhaseDataset class is the single entry point for all six specialist
models. It loads the preprocessing artefacts written by
`mpds_preprocess.pipeline` and exposes them as numpy arrays with the
correct alignment guaranteed (every row of every target array maps to the
same system as the same row of X).
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
# GPU detection (used by every model that supports it)
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return 'cuda' if a CUDA device is available, else 'cpu'.

    The string is what XGBoost ≥ 2.0 expects in `device=...` and is
    compatible with torch.device(...) construction.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def xgb_device_kwargs() -> dict:
    """Kwargs to pass to XGBoost constructors for GPU/CPU routing."""
    dev = detect_device()
    if dev == "cuda":
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist", "n_jobs": -1}


def torch_device():
    import torch
    return torch.device(detect_device())


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------

# Canonical topology class ordering — kept consistent with targets.py
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
    topology: np.ndarray                # (N,) int8 - index into topology_classes
    topology_classes: tuple[str, ...] = TOPOLOGY_CLASSES

    # Sample weights and provenance flags
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    n_sources: np.ndarray = field(default_factory=lambda: np.array([]))
    high_disagreement: np.ndarray = field(default_factory=lambda: np.array([]))

    # Splits
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # -- convenience properties -----------------------------------------

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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(processed_dir: str | Path) -> PhaseDataset:
    """Load every artefact produced by the preprocessing pipeline into one
    aligned PhaseDataset.

    All target arrays are realigned to the row order of features.parquet.
    """
    p = Path(processed_dir)

    # -- features --------------------------------------------------------
    X_df = pd.read_parquet(p / "features.parquet")
    pair_ids = X_df.index.tolist()
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
    # Replace NaN/Inf with 0; tree models tolerate NaN but it simplifies torch.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # -- scalar targets ---------------------------------------------------
    scalar_df = pd.read_parquet(p / "targets_scalar.parquet").reindex(pair_ids)
    topology_classes = TOPOLOGY_CLASSES
    cls_to_idx = {c: i for i, c in enumerate(topology_classes)}
    topology = np.asarray(
        [cls_to_idx.get(t, cls_to_idx["complex"]) for t in scalar_df["m6_topology"]],
        dtype=np.int8,
    )

    # -- liquidus / solidus / phases -------------------------------------
    liq = np.load(p / "targets_liquidus.npz", allow_pickle=False)
    sol = np.load(p / "targets_solidus.npz", allow_pickle=False)
    ph = np.load(p / "targets_phases.npz", allow_pickle=True)
    phase_vocab = list(ph["vocab"])

    # -- weights ---------------------------------------------------------
    weights = pd.read_parquet(p / "weights.parquet").reindex(pair_ids)["weight"].to_numpy(np.float32)

    # -- splits ----------------------------------------------------------
    with (p / "splits.json").open() as f:
        splits = json.load(f)

    ds = PhaseDataset(
        X=X,
        feature_names=feature_names,
        pair_ids=pair_ids,
        Y_phases=ph["Y"].astype(np.int8),
        phase_vocab=phase_vocab,
        T_liquidus=liq["T"].astype(np.float32),
        T_liquidus_mad=liq["MAD"].astype(np.float32),
        T_liquidus_mask=liq["mask"].astype(bool),
        comp_grid=liq["comp_grid"].astype(np.float32),
        T_solidus=sol["T"].astype(np.float32),
        T_solidus_mad=sol["MAD"].astype(np.float32),
        eutectic_exists=scalar_df["m4_eutectic_exists"].to_numpy(bool),
        eutectic_comp=scalar_df["m4_eutectic_comp"].to_numpy(np.float32),
        eutectic_temp=scalar_df["m4_eutectic_temp"].to_numpy(np.float32),
        eutectic_temp_std=scalar_df["m4_eutectic_temp_std"].to_numpy(np.float32),
        miscibility_gap=scalar_df["m5_miscibility_gap"].to_numpy(bool),
        topology=topology,
        topology_classes=topology_classes,
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
# Common base class for all specialist models
# ---------------------------------------------------------------------------

class SpecialistModel:
    """Minimal interface every M1-M6 model implements.

    Concrete subclasses must define `fit`, `predict`. `save` and `load`
    are joblib-based by default but can be overridden (e.g. torch models).
    """

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
