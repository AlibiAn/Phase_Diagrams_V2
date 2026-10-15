"""
SHAP analysis helpers.

Wraps shap.TreeExplainer behind one interface for every XGBoost-based model:
pass a raw XGBoost estimator or a specialist exposing `.clf` (M5, M6 solo),
`.meta` (stacked M6) or `.cls` (M4a). For M4b/M4c or an M1 phase, pass
`m4.reg_comp`, `m4.reg_temp` or `m1.models[j]` directly.

  explain()                      global feature importance over a dataset
  top_features_for_prediction()  which features drove one prediction
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def _booster_of(model: Any):
    """The XGBoost estimator inside a specialist (or `model` itself if it is one)."""
    if hasattr(model, "get_booster"):
        return model
    for attr in ("clf", "meta", "cls"):
        if getattr(model, attr, None) is not None:
            return getattr(model, attr)
    raise ValueError(f"don't know how to extract a tree booster from {type(model).__name__}")


def explain(
    model: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_display: int = 20,
) -> dict:
    """SHAP values, global importance (top `max_display` features by mean |SHAP|,
    per class for multiclass models), expected value and the explainer.

    Multiclass output is handled in both SHAP formats: a list of (N, F)
    arrays (older releases) or one (N, F, C) array (newer).
    """
    explainer = shap.TreeExplainer(_booster_of(model))
    shap_values = explainer.shap_values(X)

    def importance(values: np.ndarray) -> pd.Series:
        return (pd.Series(np.abs(values).mean(axis=0), index=feature_names)
                .sort_values(ascending=False).head(max_display))

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        per_class = [shap_values[..., c] for c in range(shap_values.shape[-1])]
    elif isinstance(shap_values, list):
        per_class = shap_values
    else:
        per_class = None

    if per_class is not None:
        global_importance = {f"class_{i}": importance(arr) for i, arr in enumerate(per_class)}
    else:
        global_importance = importance(shap_values)

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
    """Top-k features of one prediction by |SHAP|. Multiclass models need
    `class_index` to choose the class being explained."""
    explainer = shap.TreeExplainer(_booster_of(model))
    shap_vals = explainer.shap_values(X_row.reshape(1, -1))
    multiclass = isinstance(shap_vals, list) or shap_vals.ndim == 3
    if multiclass and class_index is None:
        raise ValueError("class_index required for multi-class models")
    if isinstance(shap_vals, list):
        s = shap_vals[class_index].ravel()
    elif multiclass:
        s = shap_vals[0, :, class_index]
    else:
        s = shap_vals.ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "value": X_row,
        "shap": s,
        "abs_shap": np.abs(s),
    }).sort_values("abs_shap", ascending=False).head(k).reset_index(drop=True)
    return df
