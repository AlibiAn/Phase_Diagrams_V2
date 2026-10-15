"""MPDS specialist model package.

Six specialist models + meta-learner + uncertainty sampling.

Quick start
-----------
    from mpds_models.train_all import train_all
    metrics = train_all("/data/processed/", "/data/models/")

    # load trained models for inference
    from mpds_models.base import load_dataset
    from mpds_models.meta_learner import FullDiagramReconstructor

    ds = load_dataset("/data/processed/")
    recon = FullDiagramReconstructor.from_dir("/data/models/", ds)
    diagrams = recon.reconstruct(ds.X[ds.test_idx], [ds.pair_ids[i] for i in ds.test_idx])

Models
------
  M1 (phase set)       : PhaseClassifier            XGBoost per phase
  M2/M3 (curves)       : LiquidusSolidusModel       Deep MLP + physics loss
  M4 (eutectic)        : EutecticDetector           XGBoost cls + 2× reg
  M5 (miscibility)     : MiscibilityGapDetector     XGBoost
  M6 (topology, solo)  : TopologyClassifier         XGBoost multiclass
  M6 (topology, stack) : StackedTopologyClassifier  XGBoost on M1-M5 outputs
  Reconstruction       : FullDiagramReconstructor   bundles all six
  Active learning      : uncertainty.rank_systems_for_acquisition,
                         uncertainty.rank_grid_points_within_system
"""
__all__ = ["train_all"]
__version__ = "0.1.0"


def __getattr__(name):
    # Lazy, so `python -m mpds_models.train_all` doesn't import the module twice
    if name == "train_all":
        from .train_all import train_all
        globals()["train_all"] = train_all   # the submodule import rebinds the name
        return train_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
