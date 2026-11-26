"""
pointwise_liquidus.py — the professor's suggestion (26 November 2026): one row
per measured liquidus point, inputs = the system's features + the composition
x (at% B), output = the liquidus temperature there. A diagram that covers
only part of 0-100 at% then simply contributes fewer rows; nothing is filled
in between or beyond its data (the curve model fills gaps by interpolation
and anchors the ends at Tm). Curves come from predicting x = 0, 1, ..., 100.

Variants (LightGBM, one model per fold):
  absolute    output T(x) directly (the suggestion as stated)
  deviation   the same inputs; the model learns T(x) minus the straight line
              between the two melting points (a function of the inputs), and
              the line is added back, so the output is still T(x)
Every training system also enters with A and B exchanged (x -> 100 - x), and
the 404 extra systems of v16-v18 contribute their points, as for the curve model.
Scored in the 5 folds of cross_validate.py against v16 / v17 / v18 (rebuilt
from the explore_v18 caches, experiments/v18_oof.py), on the same points.

    python experiments/pointwise_liquidus.py --out results/v20
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np

from cross_validate import folds
from mpds_models.curves import melting_point_line
from mpds_models.data import add_extra_systems, curve_scores, group_rows, load_dataset, mask_clipped_tops, system_quality

DATA, EXTRA, V6 = "/home/alibi/lab/processed_v8", "/home/alibi/lab/extra_v8_all", "/home/alibi/lab/processed_v6"
GRID = np.arange(101, dtype=np.float32)


def point_rows(ds, systems: np.ndarray, deviation: bool):
    """(X, y, w) with one row per observed liquidus point of `systems`, in both A/B orders."""
    Xs, ys, ws = [], [], []
    tm_a, tm_b = ds.melting_points()
    for X_sys, flip in ((ds.X, False), (ds.X_swap, True)):
        for i in systems:
            j = np.flatnonzero(ds.T_liquidus_mask[i])
            if not len(j):
                continue
            x = 100.0 - GRID[j] if flip else GRID[j]
            T = ds.T_liquidus[i, j].astype(np.float64)
            if deviation:
                a, b = (tm_b[i], tm_a[i]) if flip else (tm_a[i], tm_b[i])
                T = T - ((1 - x / 100.0) * a + x / 100.0 * b)
            Xs.append(np.column_stack([np.repeat(X_sys[i][None], len(j), axis=0), x]))
            ys.append(T)
            ws.append(np.full(len(j), ds.weights[i]))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(ws)


def predict_curves(model, ds, systems: np.ndarray, deviation: bool) -> np.ndarray:
    X = np.column_stack([np.repeat(ds.X[systems], 101, axis=0), np.tile(GRID, len(systems))]).astype(np.float32)
    out = model.predict(X, num_threads=20).reshape(len(systems), 101)
    if deviation:
        tm_a, tm_b = ds.melting_points(systems)
        out = out + melting_point_line(tm_a, tm_b, GRID / 100.0)
    return out.astype(np.float32)


def main() -> None:
    import lightgbm as lgb
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["absolute", "deviation"])
    ap.add_argument("--rounds", type=int, default=1200)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATA, "extended+mp")
    quality = system_quality(load_dataset(V6, "extended+mp"))         # the data groups of the v14-v18 tables
    mask_clipped_tops(ds)
    full = add_extra_systems(ds, EXTRA)
    params = dict(objective="l1", learning_rate=0.08, num_leaves=127, min_child_samples=40, feature_fraction=0.3,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=20, seed=0)
    results = {}
    for variant in a.variants:
        dev = variant == "deviation"
        liq = np.zeros((ds.n_systems, 101), np.float32)
        for f, (train, test) in enumerate(folds(ds.n_systems)):
            done = out / "folds" / f"{variant}_{f}.npy"          # resumable: finished folds are kept
            if done.exists():
                liq[test] = np.load(done)
                continue
            t0 = time.time()
            X, y, w = point_rows(full, np.concatenate([train, full.extra_idx]), dev)
            model = lgb.train(params, lgb.Dataset(X, y, weight=w, free_raw_data=True), num_boost_round=a.rounds)
            liq[test] = predict_curves(model, ds, test, dev)
            done.parent.mkdir(exist_ok=True)
            np.save(done, liq[test])
            print(f"  {variant} fold {f}: {len(y):,} training points, {time.time() - t0:.0f} s", flush=True)
        results[variant] = liq
        np.save(out / f"pointwise_{variant}_oof.npy", liq)

    from experiments.v18_oof import v18_oof
    ref = v18_oof(ds)
    rows = {**{k: v[0] for k, v in ref.items()}, **{f"pointwise {k}": v for k, v in results.items()}}
    fl = folds(ds.n_systems)
    by_fold = {k: np.array([curve_scores(ds, te, l[te], l[te])["liquidus_MAE_K"] for _, te in fl]) for k, l in rows.items()}
    groups = group_rows(quality, np.arange(ds.n_systems))
    print("\nLiquidus MAE (K), 5-fold CV on processed_v8, same points for every model")
    print(f"{'model':<22}{'mean':>7}" + "".join(f"{'fold ' + str(f):>8}" for f in range(5)) + "   folds better than v18")
    for k, v in by_fold.items():
        print(f"{k:<22}{v.mean():7.1f}" + "".join(f"{x:8.1f}" for x in v) + f"   {(v < by_fold['v18']).sum()}/5")
    print("\nBy data group (groups from processed_v6, overlapping):")
    print(f"{'model':<22}" + "".join(f"{g + f' ({len(i)})':>24}" for g, i in groups.items()))
    for k, l in rows.items():
        print(f"{k:<22}" + "".join(f"{curve_scores(ds, i, l[i], l[i])['liquidus_MAE_K']:24.1f}" for i in groups.values()))
    np.savez_compressed(out / "pointwise_oof.npz", pair_ids=np.asarray(ds.pair_ids),
                        **{k.replace(" ", "_"): v for k, v in rows.items()})


if __name__ == "__main__":
    main()
