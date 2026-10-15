"""
SHAP analysis helpers.

Each XGBoost-based specialist model exposes `.clf` (single classifier),
`.cls` / `.reg_comp` / `.reg_temp` (M4), `.models[j]` (per-phase M1), or
`.meta` (StackedTopologyClassifier). This module wraps shap.TreeExplainer
in a uniform interface so any classifier or regressor can be inspected.

Two outputs are typical:
  - global feature importance plots (which features drive predictions
    across the whole dataset?)
  - per-prediction explanations (which features drove this one
    particular topology call?)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _booster_of(model: Any):
    """Return an underlying xgboost.Booster (or compatible tree model) from
    one of our specialist objects."""
    if hasattr(model, "get_booster"):
        return model
    if hasattr(model, "clf") and model.clf is not None:
        return model.clf
    if hasattr(model, "meta") and model.meta is not None:
        return model.meta
    if hasattr(model, "cls") and model.cls is not None:
        return model.cls
    raise ValueError(f"don't know how to extract a tree booster from {type(model).__name__}")


def explain(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_display: int = 20,
) -> dict:
    """Return a dict with SHAP outputs ready for downstream plotting or saving.

    For multi-class classifiers, SHAP returns either:
      - a list of (N, F) arrays (one per class), older API
      - a single (N, F, C) array, newer API
    Both shapes are handled.
    """
    booster = _booster_of(model)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # Normalise to a list-of-2D-arrays for multiclass
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # (N, F, C)  ->  list of (N, F)
        per_class = [shap_values[..., c] for c in range(shap_values.shape[-1])]
        is_multiclass = True
    elif isinstance(shap_values, list):
        per_class = shap_values
        is_multiclass = True
    else:
        per_class = None
        is_multiclass = False

    if is_multiclass:
        global_importance = {
            f"class_{i}": pd.Series(
                np.abs(arr).mean(axis=0),
                index=feature_names if feature_names is not None else None,
            ).sort_values(ascending=False).head(max_display)
            for i, arr in enumerate(per_class)
        }
    else:
        global_importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_names if feature_names is not None else None,
        ).sort_values(ascending=False).head(max_display)

    return {
        "shap_values": shap_values,
        "global_importance": global_importance,
        "expected_value": explainer.expected_value,
        "explainer": explainer,
    }


def top_features_for_prediction(
    model: Any,
    X_row: np.ndarray,
    feature_names: list[str],
    k: int = 10,
    class_index: int | None = None,
) -> pd.DataFrame:
    """Per-prediction explanation: top-K features by absolute SHAP value.

    For multiclass models pass class_index to pick which class to explain.
    """
    booster = _booster_of(model)
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_row.reshape(1, -1))
    # Normalise to a 1-D vector for the target class
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        if class_index is None:
            raise ValueError("class_index required for multi-class models")
        s = shap_vals[0, :, class_index]
    elif isinstance(shap_vals, list):
        if class_index is None:
            raise ValueError("class_index required for multi-class models")
        s = shap_vals[class_index].ravel()
    else:
        s = shap_vals.ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "value": X_row,
        "shap": s,
        "abs_shap": np.abs(s),
    }).sort_values("abs_shap", ascending=False).head(k).reset_index(drop=True)
    return df
