"""
cross_validate.py — decide between versions with 5-fold cross-validation over
all 1,380 systems (KFold, shuffled, seed 0: the folds of every result since v13).

Curves: one run trains the trees, the height correction and the
foundation-model curve in each fold, so it scores v16 (trees), v17 (height)
and v18 (blend) in the same folds. A change is real if it beats the
reference in at least 4 of 5 folds. --against compares with the out-of-fold
curves of an earlier run (its oof_curves.npz).

    MPDS_DEVICE=cpu python cross_validate.py /home/alibi/lab/processed_v8 --out results/v18/cv \\
        --groups-from /home/alibi/lab/processed_v6
    ... --height 0                          trees only (v16), a few minutes
    ... --features extended --out ...       without the compound features
    ... --scalars                           also the eutectic / gap / type models

Outputs in --out: curves_by_fold.csv, curves_by_group.csv, oof_curves.npz,
scalars.csv (with --scalars), and the console tables.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mpds_models.curves import TreeCurves
from mpds_models.data import FEATURE_SETS, add_extra_systems, curve_scores, group_rows, load_dataset, mask_clipped_tops, \
    precision_recall_f1, system_quality
from mpds_models.train import EXTRA_DATA

STAGE_NAMES = {"trees": "v16 trees", "height": "v17 + height", "blend": "v18 + curve blend"}


def folds(n: int, k: int = 5, seed: int = 0):
    return [(np.sort(tr), te) for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))]


def curve_cv(ds, extra_dir, k, height, blend, n_components, seed=0) -> tuple[dict, np.ndarray]:
    """Out-of-fold (liquidus, solidus) per stage, and the fold of each system."""
    fit_ds = add_extra_systems(ds, extra_dir) if extra_dir else ds
    out, fold_of = {}, np.zeros(ds.n_systems, dtype=int)
    for f, (train, test) in enumerate(folds(ds.n_systems, k)):
        t0 = time.time()
        fold = fit_ds.with_split(np.concatenate([train, fit_ds.extra_idx]), test)
        trees = TreeCurves(seed=seed).fit(fold)
        if height:
            from mpds_models.foundation import FoundationCurves
            stages = FoundationCurves(trees, height, blend, n_components, seed=seed).fit(fold).predict_stages(ds.X[test])
        else:
            stages = {"trees": trees.predict(ds.X[test])}
        for name, (liq, sol) in stages.items():
            if name not in out:
                out[name] = (np.zeros((ds.n_systems, 101), np.float32), np.zeros((ds.n_systems, 101), np.float32))
            out[name][0][test], out[name][1][test] = liq, sol
        fold_of[test] = f
        print(f"  fold {f}: {time.time() - t0:.0f} s", flush=True)
    return out, fold_of


def curve_tables(ds, quality, curves: dict, fold_of: np.ndarray):
    by_fold, by_group = [], []
    for name, (liq, sol) in curves.items():
        for f in np.unique(fold_of):
            idx = np.flatnonzero(fold_of == f)
            s = curve_scores(ds, idx, liq[idx], sol[idx])
            by_fold.append(dict(model=name, fold=f, liquidus_MAE_K=s["liquidus_MAE_K"], solidus_MAE_K=s["solidus_MAE_K"]))
        for g, idx in group_rows(quality, np.arange(ds.n_systems)).items():
            s = curve_scores(ds, idx, liq[idx], sol[idx])
            by_group.append(dict(model=name, group=g, systems=len(idx), liquidus_MAE_K=s["liquidus_MAE_K"],
                                 solidus_MAE_K=s["solidus_MAE_K"]))
    return pd.DataFrame(by_fold), pd.DataFrame(by_group)


def print_curves(by_fold: pd.DataFrame, by_group: pd.DataFrame, reference: str) -> None:
    order = list(dict.fromkeys(by_fold.model))
    for metric in ("liquidus_MAE_K", "solidus_MAE_K"):
        t = by_fold.pivot(index="model", columns="fold", values=metric).loc[order]
        wins = t.lt(t.loc[reference], axis=1).sum(axis=1)
        t.columns = [f"fold {c}" for c in t.columns]
        t["mean"] = t.mean(axis=1)
        t[f"folds better than {reference}"] = wins
        print(f"\n{metric.replace('_', ' ')} by fold:\n" + t.to_string(float_format=lambda v: f"{v:.1f}"))
    t = by_group.assign(score=by_group.liquidus_MAE_K.map("{:.1f}".format) + " / " + by_group.solidus_MAE_K.map("{:.1f}".format))
    wide = t.pivot(index="model", columns="group", values="score").loc[order, list(dict.fromkeys(by_group.group))]
    n = by_group.drop_duplicates("group").set_index("group").systems
    wide.columns = [f"{c} ({n[c]})" for c in wide.columns]
    print("\nLiquidus / solidus MAE (K) by data group (groups overlap; clean = none of the others):\n" + wide.to_string())


def scalar_cv(ds, k: int) -> pd.DataFrame:
    """Eutectic, miscibility-gap and solo diagram-type models in the same folds.
    The eutectic position is scored over every system with a true eutectic
    ("_all_true", the same systems for every model) and where truth and
    prediction both have one (as predict.py reports it)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from mpds_models.classifiers import EutecticDetector, MiscibilityGapDetector, TopologyClassifier
    rows = []
    for f, (train, test) in enumerate(folds(ds.n_systems, k)):
        d = ds.with_split(train, test)
        gap = MiscibilityGapDetector().fit(d)
        p, y = gap.predict_proba(ds.X[test]), ds.miscibility_gap[test]
        prec, rec, f1 = precision_recall_f1(y, p >= gap.threshold)
        topo = TopologyClassifier().fit(d).predict_proba(ds.X[test]).argmax(1)
        eut = EutecticDetector().fit(d)
        e = eut.predict(ds.X[test])
        has = ds.eutectic_exists[test] & np.isfinite(ds.eutectic_temp[test]) & np.isfinite(ds.eutectic_comp[test])
        both = has & e["exists"]
        Xh = ds.X[test[has]]
        rows.append(dict(fold=f, gap_f1=f1, gap_precision=prec, gap_recall=rec, gap_roc_auc=roc_auc_score(y, p),
                         gap_avg_precision=average_precision_score(y, p),
                         type_accuracy=float((topo == ds.topology[test]).mean()),
                         eutectic_f1=precision_recall_f1(ds.eutectic_exists[test], e["exists"])[2],
                         eutectic_roc_auc=roc_auc_score(ds.eutectic_exists[test], e["exists_proba"]),
                         eutectic_comp_MAE_at_pct=float(np.abs(e["comp"] - ds.eutectic_comp[test])[both].mean()),
                         eutectic_temp_MAE_K=float(np.abs(e["temp"] - ds.eutectic_temp[test])[both].mean()),
                         eutectic_comp_MAE_all_true_at_pct=float(np.abs(np.clip(eut.reg["comp"].predict(Xh), 0, 100)
                                                                        - ds.eutectic_comp[test[has]]).mean()),
                         eutectic_temp_MAE_all_true_K=float(np.abs(eut.reg["temp"].predict(Xh[:, :eut.n_temp_features])
                                                                   - ds.eutectic_temp[test[has]]).mean())))
        print(f"  scalars fold {f}: gap F1 {f1:.3f}, type accuracy {rows[-1]['type_accuracy']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", choices=FEATURE_SETS, default="extended+mp")
    ap.add_argument("--extra-data", default=EXTRA_DATA)
    ap.add_argument("--no-extra-data", action="store_true")
    ap.add_argument("--height", type=float, default=0.5, help="0 = trees only")
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--components", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--groups-from", help="dataset defining the data groups (default: processed_dir)")
    ap.add_argument("--against", nargs="*", default=[], metavar="NAME=OOF_NPZ",
                    help="out-of-fold curves of earlier runs to compare with, e.g. v18=results/v18/cv/oof_curves.npz")
    ap.add_argument("--reference", help="model the fold wins are counted against (default: the first)")
    ap.add_argument("--scalars", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(a.processed_dir, a.features)
    quality = system_quality(load_dataset(a.groups_from, a.features) if a.groups_from else ds)     # before masking
    mask_clipped_tops(ds)
    print(f"{a.processed_dir}: {ds.n_systems} systems, {int(ds.T_liquidus_mask.sum()):,} liquidus points, "
          f"{a.folds}-fold CV, features {a.features}, extra systems: {None if a.no_extra_data else a.extra_data}")
    curves, fold_of = curve_cv(ds, None if a.no_extra_data else a.extra_data, a.folds, a.height,
                               a.blend if a.height else 0.0, a.components)
    curves = {STAGE_NAMES[k]: v for k, v in curves.items()}
    np.savez_compressed(out / "oof_curves.npz", pair_ids=np.asarray(ds.pair_ids), fold=fold_of,
                        **{f"{k.split()[0]}_{part}": v[j] for k, v in curves.items() for j, part in enumerate(("liquidus", "solidus"))})
    for spec in a.against:
        name, path = spec.split("=", 1)
        o = np.load(path)
        if list(o["pair_ids"]) != ds.pair_ids or not np.array_equal(o["fold"], fold_of):
            raise SystemExit(f"{path}: different systems or folds")
        stage = sorted(k[:-len("_liquidus")] for k in o.files if k.endswith("_liquidus"))[-1]
        curves = {name: (o[f"{stage}_liquidus"], o[f"{stage}_solidus"]), **curves}
    by_fold, by_group = curve_tables(ds, quality, curves, fold_of)
    by_fold.to_csv(out / "curves_by_fold.csv", index=False)
    by_group.to_csv(out / "curves_by_group.csv", index=False)
    print_curves(by_fold, by_group, a.reference or next(iter(curves)))
    if a.scalars:
        df = scalar_cv(ds, a.folds)
        df.to_csv(out / "scalars.csv", index=False)
        print("\nScalar models, mean over folds:\n" + df.drop(columns="fold").mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved in {out}/")


if __name__ == "__main__":
    main()
