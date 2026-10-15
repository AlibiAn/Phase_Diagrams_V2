"""MPDS specialist model package.

Six specialist models + meta-learner + uncertainty sampling.

Quick start
-----------
    from mpds_models.train_all import train_all
    metrics = train_all("/data/processed/", "/data/models/")

    # or load trained models for inference
    from mpds_models.base import load_dataset
    from mpds_models.m1_phase import PhaseClassifier
    from mpds_models.deep_curve_net import LiquidusSolidusModel
    from mpds_models.meta_learner import FullDiagramReconstructor

    m1 = PhaseClassifier.load("/data/models/m1_phase.joblib")
    m23 = LiquidusSolidusModel.load("/data/models/m23_curves.pt")
    # ... etc.

Models
------
  M1 (phase set)       : PhaseClassifier            XGBoost per phase
  M2/M3 (curves)       : LiquidusSolidusModel       Deep MLP + physics loss
  M4 (eutectic)        : EutecticDetector           XGBoost cls + 2× reg
  M5 (miscibility)     : MiscibilityGapDetector     XGBoost
  M6 (topology, solo)  : TopologyClassifier         XGBoost multiclass
  M6 (topology, stack) : StackedTopologyClassifier  XGBoost on M1-M5 outputs
  Reconstruction       : FullDiagramReconstructor   bundles all six
  Active learning      : uncertainty.rank_systems_for_acquisition / ...
"""
from .train_all import train_all
__all__ = ["train_all"]
__version__ = "0.1.0"
