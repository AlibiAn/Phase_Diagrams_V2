"""
cross_validate.py — decide between model versions with K-fold cross-validation
over all systems: the liquidus / solidus curve model and the scalar models
(miscibility gap, diagram type, eutectic).

Every system is predicted once, by models that never saw it. A change is
accepted if it beats the reference (the first --configs name) in at least 4 of
5 folds. Unlike compare_curve_models.py, every model here trains on ALL the
systems of the training folds (tree models need no validation split), so the
v13 numbers are about 2 K better than the ones recorded in results/v13.

Configurations (--configs)
--------------------------
  v13            ExtraTrees, median over trees, A/B swap, v6 features, solidus = liquidus - gap
  v13+direct     v13 with the solidus predicted as its own deviation from the Tm line
  v14-features   v13 with the extended element / Miedema / regular-solution features
  v14            extended features + direct solidus (the v14 model)

Usage
-----
    python cross_validate.py $D --out results/v14/cv_v6 --configs v13 v13+direct v14-features v14
    # same systems and folds, targets from the two-liquid dataset, data groups as in $D
    python cross_validate.py /home/alibi/lab/processed_v7 --groups-from $D --out results/v14/cv_v7 --configs v13 v14
    # scalar models with both feature sets
    python cross_validate.py $D --out results/v14/cv_v6 --scalars --feature-sets v6 extended

Outputs (in --out)
------------------
  curves_by_fold.csv     liquidus / solidus MAE per configuration and fold
  curves_by_group.csv    per data group (clean, incomplete, volatile element, monotectic)
  curves_by_topology.csv per true diagram type
  oof_<config>.npz       the out-of-fold curves (liquidus, solidus, fold)
  scalars.csv            scalar-model scores per feature set and fold
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mpds_models.base import (FEATURE_SETS, curve_metrics, group_rows, load_dataset, mask_clipped_tops,
                              precision_recall_f1, system_quality)
from mpds_models.tree_curves import TreeCurveConfig, TreeCurveModel

CONFIGS = {
    "v13": dict(features="v6", solidus="gap"),
    "v13+direct": dict(features="v6", solidus="direct"),
    "v14-features": dict(features="extended", solidus="gap"),
    "v14": dict(features="extended", solidus="direct"),
}


def folds(n: int, k: int, seed: int = 0):
    """The folds compare_curve_models.py uses (KFold, shuffled, seed 0)."""
    return [(np.sort(tr), te) for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))]


def with_split(ds, train, test):
    return replace(ds, train_idx=train, val_idx=np.array([], dtype=np.int64), test_idx=test)


def curve_cv(datasets: dict, config: str, k: int, seed: int):
    """Out-of-fold (liquidus, solidus, fold index) for one configuration."""
    c = CONFIGS[config]
    ds = datasets[c["features"]]
    liq = np.zeros((ds.n_systems, len(ds.comp_grid)), dtype=np.float32)
    sol = np.zeros_like(liq)
    fold_of = np.zeros(ds.n_systems, dtype=np.int64)
    for f, (train, test) in enumerate(folds(ds.n_systems, k)):
        cfg = TreeCurveConfig(backend="extra_trees", aggregate="median", swap_augment=True,
                              solidus=c["solidus"], seed=seed)
        model = TreeCurveModel(cfg).fit(with_split(ds, train, test))
        liq[test], sol[test] = model.predict(ds.X[test])
        fold_of[test] = f
    return liq, sol, fold_of


def curve_tables(ds, quality, results: dict, k: int):
    by_fold, by_group, by_topo = [], [], []
    topo = np.array([ds.topology_name(i) for i in range(ds.n_systems)])
    for name, (liq, sol, fold_of) in results.items():
        for f in range(k):
            idx = np.flatnonzero(fold_of == f)
            m = curve_metrics(ds, idx, liq[idx], sol[idx])
            by_fold.append(dict(config=name, fold=f, liquidus_MAE_K=m["liquidus_MAE_K"],
                                solidus_MAE_K=m["solidus_MAE_K"]))
        for g, idx in group_rows(quality, np.arange(ds.n_systems)).items():
            m = curve_metrics(ds, idx, liq[idx], sol[idx])
            by_group.append(dict(config=name, group=g, systems=len(idx),
                                 liquidus_MAE_K=m["liquidus_MAE_K"], solidus_MAE_K=m["solidus_MAE_K"]))
        for t in ds.topology_classes:
            idx = np.flatnonzero(topo == t)
            m = curve_metrics(ds, idx, liq[idx], sol[idx])
            by_topo.append(dict(config=name, topology=t, systems=len(idx),
                                liquidus_MAE_K=m["liquidus_MAE_K"], solidus_MAE_K=m["solidus_MAE_K"]))
    return pd.DataFrame(by_fold), pd.DataFrame(by_group), pd.DataFrame(by_topo)


def print_curves(by_fold, by_group, by_topo, reference: str) -> None:
    for metric in ("liquidus_MAE_K", "solidus_MAE_K"):
        t = by_fold.pivot(index="config", columns="fold", values=metric).loc[by_fold.config.unique()]
        wins = (t.lt(t.loc[reference], axis=1)).sum(axis=1)
        t.columns = [f"fold {c}" for c in t.columns]
        t["mean"] = t.mean(axis=1)
        t[f"folds won vs {reference}"] = wins
        print(f"\n{metric.replace('_', ' ')} by fold:")
        print(t.to_string(float_format=lambda v: f"{v:.1f}"))
    for table, key, title in ((by_group, "group", "data group (groups overlap; clean = none of the others)"),
                              (by_topo, "topology", "true diagram type")):
        print(f"\nLiquidus / solidus MAE (K) by {title}:")
        t = table.assign(score=table.liquidus_MAE_K.map("{:.0f}".format) + " / "
                         + table.solidus_MAE_K.map("{:.0f}".format))
        wide = t.pivot(index="config", columns=key, values="score").loc[table.config.unique(), table[key].unique()]
        n = table.drop_duplicates(key).set_index(key).systems
        wide.columns = [f"{c} ({n[c]})" for c in wide.columns]
        print(wide.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Scalar models
# ─────────────────────────────────────────────────────────────────────────────

def scalar_cv(ds, k: int, feature_set: str, all_features_for_m4: bool = False) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from mpds_models.m4_eutectic import EutecticDetector, EutecticDetectorConfig
    from mpds_models.m5_miscibility import MiscibilityGapDetector
    from mpds_models.m6_topology import TopologyClassifier
    rows = []
    for f, (train, test) in enumerate(folds(ds.n_systems, k)):
        d = with_split(ds, train, test)
        t0 = time.time()
        m5 = MiscibilityGapDetector().fit(d)
        p = m5.predict_proba(ds.X[test])
        y = ds.miscibility_gap[test]
        prec, rec, f1 = precision_recall_f1(y, p >= m5.cfg.decision_threshold)
        row = dict(features=feature_set, fold=f, gap_f1=f1, gap_precision=prec, gap_recall=rec,
                   gap_roc_auc=roc_auc_score(y, p), gap_avg_precision=average_precision_score(y, p))

        m6 = TopologyClassifier().fit(d).evaluate(d, "test")
        row.update(type_accuracy=m6["accuracy"], type_macro_f1=m6["macro_f1"],
                   **{f"type_f1_{c}": v["f1"] for c, v in m6["per_class"].items()})

        m4 = EutecticDetector(EutecticDetectorConfig(temp_base_features_only=not all_features_for_m4)).fit(d)
        e = m4.evaluate(d, "test")
        # Position errors two ways: where truth AND prediction have a eutectic
        # (as predict.py reports them; the systems then depend on the existence
        # classifier), and over every system with a true eutectic (the same
        # systems for every feature set, so the regressors compare like for like)
        has = test[ds.eutectic_exists[test] & np.isfinite(ds.eutectic_temp[test]) & np.isfinite(ds.eutectic_comp[test])]
        temp_all = np.abs(m4.reg_temp.predict(m4._temp_X(ds.X[has])) - ds.eutectic_temp[has]).mean()
        comp_all = np.abs(np.clip(m4.reg_comp.predict(ds.X[has]), 0, 100) - ds.eutectic_comp[has]).mean()
        row.update(eutectic_f1=e["exists_f1"],
                   eutectic_roc_auc=roc_auc_score(ds.eutectic_exists[test], m4.predict_proba(ds.X[test])),
                   eutectic_comp_MAE_at_pct=e["comp_MAE_at_pct"], eutectic_temp_MAE_K=e["temp_MAE_K"],
                   eutectic_comp_MAE_all_true_at_pct=float(comp_all), eutectic_temp_MAE_all_true_K=float(temp_all))
        rows.append(row)
        print(f"  {feature_set} fold {f} ({time.time() - t0:.0f} s): gap F1 {f1:.3f}, type accuracy "
              f"{m6['accuracy']:.3f}, eutectic temperature MAE {e['temp_MAE_K']:.0f} K", flush=True)
    return rows


def print_scalars(df: pd.DataFrame, reference: str) -> None:
    higher_is_better = [c for c in df.columns if c not in ("features", "fold") and "MAE" not in c]
    metrics = [c for c in df.columns if c not in ("features", "fold")]
    mean = df.groupby("features", sort=False)[metrics].mean().T
    for fs in df.features.unique():
        if fs == reference:
            continue
        a = df[df.features == fs].set_index("fold")[metrics]
        b = df[df.features == reference].set_index("fold")[metrics]
        better = pd.Series({m: int(((a[m] > b[m]) if m in higher_is_better else (a[m] < b[m])).sum())
                            for m in metrics})
        mean[f"folds {fs} beats {reference}"] = better
    print("\nScalar models, mean over folds (F1, accuracy, AUC: higher is better; MAE: lower):")
    print(mean.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:.1f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="K-fold cross-validation of curve and scalar models.")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="*", choices=list(CONFIGS), default=[],
                    help="curve configurations; the first is the reference for the fold wins")
    ap.add_argument("--scalars", action="store_true", help="cross-validate the M4 / M5 / M6 scalar models")
    ap.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=["v6"],
                    help="feature sets for --scalars; the first is the reference")
    ap.add_argument("--m4-temp-all-features", action="store_true",
                    help="--scalars: let the eutectic temperature regressor use the extended features too "
                         "(train_all keeps it on the v6 columns)")
    ap.add_argument("--groups-from", default=None,
                    help="dataset that defines the data groups (default: processed_dir itself)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    needed = {CONFIGS[c]["features"] for c in a.configs} | (set(a.feature_sets) if a.scalars else set())
    datasets = {}
    for fs in sorted(needed, reverse=True):
        datasets[fs] = load_dataset(a.processed_dir, features=fs)
    any_ds = next(iter(datasets.values()))
    quality = system_quality(load_dataset(a.groups_from) if a.groups_from else
                             load_dataset(a.processed_dir))           # before masking
    n_gap, n_seen = int(any_ds.miscibility_gap.sum()), 0
    for ds in datasets.values():
        mask_clipped_tops(ds)
        n_seen = int(ds.T_liquidus_mask.sum())
    from mpds_models.base import detect_device
    print(f"{a.processed_dir}: {any_ds.n_systems} systems, {n_seen:,} liquidus points scored, "
          f"{n_gap} systems with a miscibility gap; {a.cv}-fold cross-validation; "
          f"XGBoost device: {detect_device()}")

    if a.configs:
        results = {}
        for c in a.configs:
            t0 = time.time()
            results[c] = curve_cv(datasets, c, a.cv, a.seed)
            np.savez_compressed(out / f"oof_{c}.npz", liquidus=results[c][0], solidus=results[c][1],
                                fold=results[c][2], pair_ids=np.asarray(any_ds.pair_ids))
            print(f"  {c}: done in {time.time() - t0:.0f} s", flush=True)
        tables = curve_tables(any_ds, quality, results, a.cv)
        for t, name in zip(tables, ("curves_by_fold", "curves_by_group", "curves_by_topology")):
            t.to_csv(out / f"{name}.csv", index=False)
        print_curves(*tables, reference=a.configs[0])

    if a.scalars:
        rows = []
        for fs in a.feature_sets:
            rows += scalar_cv(datasets[fs], a.cv, fs, a.m4_temp_all_features)
        df = pd.DataFrame(rows)
        df.to_csv(out / "scalars.csv", index=False)
        print_scalars(df, reference=a.feature_sets[0])
    print(f"\nSaved in {out}/")


if __name__ == "__main__":
    main()
