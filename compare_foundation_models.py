"""
compare_foundation_models.py — tabular foundation models against the current
tree models, where they are most likely to help, in the same 5-fold
cross-validation over all 1,380 systems.

Models
------
  xgboost   the current scalar models (m4_eutectic, m5_miscibility, m6_topology)
  tabpfn    TabPFN v2 (Hollmann et al., Nature 637 (2025) 319, doi:10.1038/s41586-024-08328-6;
            weights under the Prior Labs Licence, Apache 2.0 with an attribution
            requirement: "Built with TabPFN"). The newer gated versions are not used.
  tabicl    TabICL v2 (Qu et al., Inria Soda team; BSD-3, weights on Hugging Face
            jingang/TabICL)
The foundation models run on the GPU; XGBoost on the CPU (MPDS_DEVICE=cpu), as
the recorded models were.

Tasks
-----
  gap        liquid miscibility gap (binary)
  eutectic   eutectic exists (binary)
  type       diagram type (5 classes)
             Binary tasks: ROC-AUC and average precision (no threshold), and F1
             at a threshold chosen on the training folds by an inner 5-fold split
             (as the XGBoost models do). Type: accuracy and macro-F1 of the most
             probable class; for the foundation models also with the class
             probabilities divided by the class frequency of the training folds
             ("balanced"), the counterpart of XGBoost's balanced class weights.
  height     the mean height of the liquidus above the melting-point line (one
             number per system; a third of the liquidus error is this number).
             The reference is the height implied by the v16 curve model's own
             out-of-fold curve. If a model predicts it better, the v16 curve is
             shifted towards it (with a window that keeps both ends at Tm) and
             the liquidus error is scored.

Usage
-----
    MPDS_DEVICE=cpu python compare_foundation_models.py /home/alibi/lab/processed_v8 \\
        --curves results/v16/cv_v8_all/oof_v14+mp+extra.npz --out results/v17 2>&1 | tee results/v17/foundation_models.txt
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cross_validate import folds
from mpds_models.base import load_dataset, mask_clipped_tops, curve_metrics, precision_recall_f1
from mpds_models.tree_curves import _fill_curves

warnings.filterwarnings("ignore")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def make(name: str, task: str, seed: int = 0):
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if task == "regression" else TabPFNClassifier
        return cls.create_default_for_version(ModelVersion.V2, device="cuda", random_state=seed,
                                              ignore_pretraining_limits=True)
    if name == "tabicl":
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLRegressor if task == "regression" else TabICLClassifier
        return cls(device="cuda", random_state=seed)
    raise ValueError(name)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    scores = np.array([precision_recall_f1(y, p >= t)[2] for t in THRESHOLDS])
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return float(THRESHOLDS[best[np.argmin(np.abs(THRESHOLDS[best] - 0.5))]])


def binary_oof(name, X, y, train, test, inner=5):
    """Test-fold probabilities and the threshold chosen on the training folds."""
    m = make(name, "classification").fit(X[train], y[train])
    p_test = m.predict_proba(X[test])[:, 1]
    p_inner = np.zeros(len(train))
    for itr, ite in folds(len(train), inner, seed=1):
        mi = make(name, "classification").fit(X[train[itr]], y[train[itr]])
        p_inner[ite] = mi.predict_proba(X[train[ite]])[:, 1]
    return p_test, best_threshold(y[train], p_inner)


def scalar_tasks(ds, fl, models, out) -> pd.DataFrame:
    from mpds_models.m4_eutectic import EutecticDetector, EutecticDetectorConfig
    from mpds_models.m5_miscibility import MiscibilityGapDetector
    from mpds_models.m6_topology import TopologyClassifier
    from dataclasses import replace
    X = ds.X
    labels = {"gap": ds.miscibility_gap.astype(int), "eutectic": ds.eutectic_exists.astype(int)}
    y_type = ds.topology.astype(int)
    rows = []
    for f, (train, test) in enumerate(fl):
        d = replace(ds, train_idx=train, val_idx=np.array([], dtype=np.int64), test_idx=test)
        for name in models:
            t0 = time.time()
            res = {}
            for task, y in labels.items():
                if name == "xgboost":
                    m = (MiscibilityGapDetector() if task == "gap" else
                         EutecticDetector(EutecticDetectorConfig(temp_base_features_only=True))).fit(d)
                    p = m.predict_proba(X[test])
                    thr = (m.cfg.decision_threshold)
                else:
                    p, thr = binary_oof(name, X, y, train, test)
                res[task] = (p, thr)
                rows.append(dict(model=name, fold=f, task=task, auc=roc_auc_score(y[test], p),
                                 ap=average_precision_score(y[test], p),
                                 f1=precision_recall_f1(y[test], p >= thr)[2], threshold=thr))
            if name == "xgboost":
                pt = TopologyClassifier().fit(d).predict_proba(X[test])
                variants = {"": pt}
            else:
                pt = make(name, "classification").fit(X[train], y_type[train]).predict_proba(X[test])
                full = np.zeros((len(test), ds.n_topology_classes))
                seen = np.unique(y_type[train])
                full[:, seen] = pt
                prior = np.bincount(y_type[train], minlength=ds.n_topology_classes) / len(train)
                variants = {"": full, " (balanced)": full / np.where(prior > 0, prior, 1)}
            for tag, pr in variants.items():
                pred = pr.argmax(1)
                rows.append(dict(model=name + tag, fold=f, task="type", accuracy=float((pred == y_type[test]).mean()),
                                 macro_f1=f1_score(y_type[test], pred, average="macro",
                                                   labels=list(range(ds.n_topology_classes)), zero_division=0)))
            print(f"  fold {f} {name}: {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scalars_by_fold.csv", index=False)
    return df


def print_scalars(df: pd.DataFrame) -> None:
    ref = "xgboost"
    for task, metrics in (("gap", ("auc", "ap", "f1")), ("eutectic", ("auc", "ap", "f1")),
                          ("type", ("accuracy", "macro_f1"))):
        t = df[df.task == task]
        print(f"\n{task}: mean over 5 folds (folds better than XGBoost in brackets)")
        for m in t.model.unique():
            parts = []
            for k in metrics:
                a = t[t.model == m].set_index("fold")[k]
                b = t[t.model == ref].set_index("fold")[k]
                wins = "" if m == ref else f" ({int((a > b).sum())}/5)"
                parts.append(f"{k} {a.mean():.3f}{wins}")
            print(f"  {m:<22} " + "   ".join(parts))


def height_task(ds, fl, models, curves_file, out) -> None:
    """The mean height of the liquidus above the melting-point line."""
    names = list(ds.feature_names)
    tm_a = ds.X[:, names.index("phys__melting_point__A")].astype(float)
    tm_b = ds.X[:, names.index("phys__melting_point__B")].astype(float)
    frac = ds.comp_grid.astype(float) / 100.0
    line = (1 - frac)[None] * tm_a[:, None] + frac[None] * tm_b[:, None]
    seen = ds.T_liquidus_mask.copy()
    seen[:, [0, -1]] = False
    filled = _fill_curves(ds.T_liquidus, seen, frac, tm_a, tm_b, 3)
    target = (filled - line).mean(axis=1)
    ok = np.isfinite(target)
    o = np.load(curves_file)
    if list(o["pair_ids"]) != list(ds.pair_ids):
        raise SystemExit("the curve file does not belong to this dataset")
    liq, sol = o["liquidus"], o["solidus"]
    implied = (liq - line).mean(axis=1)
    window = 4 * frac * (1 - frac)
    window /= window.mean()

    def fold_scores(curve):
        return np.array([curve_metrics(ds, te, curve[te], sol[te])["liquidus_MAE_K"] for _, te in fl])

    ref = fold_scores(liq)
    print(f"\nheight of the liquidus above the melting-point line ({int(ok.sum())} systems)")
    print(f"  predicting 0:                    MAE {np.abs(target[ok]).mean():6.1f} K")
    print(f"  v16 curve model's own height:    MAE {np.abs(implied - target)[ok].mean():6.1f} K"
          f"   (its liquidus: {ref.mean():.1f} K)")
    rows = []
    for name in models:
        if name == "xgboost":
            continue
        t0 = time.time()
        p = np.zeros(ds.n_systems)
        for train, test in fl:
            tr = train[ok[train]]
            Xt = np.concatenate([ds.X[tr], ds.X_swap[tr]])        # height is the same for A-B and B-A
            m = make(name, "regression").fit(Xt, np.tile(target[tr], 2))
            p[test] = 0.5 * (m.predict(ds.X[test]) + m.predict(ds.X_swap[test]))
        err = np.abs(p - target)[ok].mean()
        both = 0.5 * (p + implied)
        print(f"  {name:<32} MAE {err:6.1f} K   (averaged with the curve model's height: "
              f"{np.abs(both - target)[ok].mean():.1f} K)   {time.time() - t0:.0f} s")
        for a in (0.5, 1.0):
            shifted = liq + a * (p - implied)[:, None] * window[None]
            s = fold_scores(shifted)
            print(f"    v16 curve shifted {a:.0%} towards it: liquidus {s.mean():6.1f} K, "
                  f"better in {int((s < ref).sum())}/5 folds")
            rows.append(dict(model=name, shift=a, liquidus_MAE_K=s.mean(), folds_won=int((s < ref).sum()),
                             height_MAE_K=err))
        np.save(out / f"height_oof_{name}.npy", p)
    pd.DataFrame(rows).to_csv(out / "height.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--curves", required=True, help="out-of-fold v16 curves (cross_validate.py oof_*.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="extended+mp")
    ap.add_argument("--models", nargs="+", default=["xgboost", "tabpfn", "tabicl"])
    ap.add_argument("--tasks", nargs="+", default=["scalars", "height"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.processed_dir, features=a.features)
    mask_clipped_tops(ds)
    fl = folds(ds.n_systems, 5)
    print(f"{a.processed_dir}: {ds.n_systems} systems x {ds.n_features} features ({a.features}); 5-fold CV")
    if "scalars" in a.tasks:
        print_scalars(scalar_tasks(ds, fl, a.models, out))
    if "height" in a.tasks:
        height_task(ds, fl, a.models, a.curves, out)


if __name__ == "__main__":
    main()
