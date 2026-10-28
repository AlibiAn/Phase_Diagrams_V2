"""
compare_curve_models.py — liquidus / solidus curves from tree ensembles
(ExtraTrees, RandomForest, XGBoost) against the CurveNet MLP, on one split.

Every tree model is trained here on the training split (see
mpds_models/tree_curves.py for how). CurveNet is either loaded from files
train_all.py saved (--curvenet, one file per seed) or trained here
(--train-curvenet, with the train_all settings).

Usage
-----
    python compare_curve_models.py processed_v6 --out curve_cmp \\
        --curvenet models_v9_s0/m23_curves.pt models_v9_s1/m23_curves.pt
    python compare_curve_models.py processed_v6 --out curve_cmp --seeds 0 1 2 \\
        --targets deviation absolute --train-curvenet

Scores (all in K, on observed grid points of --split, default test)
------
  liquidus_MAE / solidus_MAE   as in predict.py
  liquidus_MAE_interior        without the two pure-element points
  liquidus_MAE_at_25/50/75     the points LazyPredict scored (it used val)
  end_vs_Tm                    |curve end - Tm| at x = 0 and 100 (0 = exact)
  order_violation_fraction     solidus above liquidus before clamping

Outputs (in --out)
------------------
  scores.csv            one row per model and seed
  summary.csv           mean and std over seeds per model
  by_composition.csv/png liquidus MAE in 5 at% bins
  by_topology.csv       liquidus MAE per topology class
  examples.png          predicted curves for one system per topology class
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mpds_models.base import curve_metrics, load_dataset
from mpds_models.tree_curves import CURVE_BACKENDS, TreeCurveConfig, TreeCurveModel

POINTS = (25, 50, 75)
BINS = np.arange(0, 101, 5)

# chart colours (reference palette, light surface); one fixed colour per model
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
NICE = {"extra_trees": "ExtraTrees", "random_forest": "RandomForest", "xgboost": "XGBoost"}


def scores(ds, idx, liq, sol) -> dict:
    grid = np.asarray(ds.comp_grid)
    true, mask = ds.T_liquidus[idx], ds.T_liquidus_mask[idx]
    err = np.abs(liq - true)
    ia = list(ds.feature_names).index("phys__melting_point__A")
    ib = list(ds.feature_names).index("phys__melting_point__B")
    tm = np.stack([ds.X[idx, ia], ds.X[idx, ib]], axis=1)
    ends = np.stack([liq[:, 0], liq[:, -1]], axis=1)
    out = curve_metrics(ds, idx, liq, sol)
    out["liquidus_MAE_interior_K"] = float(err[:, 1:-1][mask[:, 1:-1]].mean())
    for x in POINTS:
        g = int(np.argmin(np.abs(grid - x)))
        e = err[:, g][mask[:, g]]
        out[f"liquidus_MAE_at_{x}_K"] = float(e.mean()) if e.size else np.nan
    out["end_vs_Tm_K"] = float(np.abs(ends - tm)[tm > 0].mean())
    return out


def by_bin(ds, idx, liq) -> list[float]:
    grid = np.asarray(ds.comp_grid)
    err = np.abs(liq - ds.T_liquidus[idx])
    mask = ds.T_liquidus_mask[idx]
    res = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        cols = (grid >= lo) & ((grid < hi) | (hi == BINS[-1]))
        e = err[:, cols][mask[:, cols]]
        res.append(float(e.mean()) if e.size else np.nan)
    return res


def by_topology(ds, idx, liq) -> dict:
    err = np.abs(liq - ds.T_liquidus[idx])
    mask = ds.T_liquidus_mask[idx]
    names = np.array([ds.topology_name(i) for i in idx])
    out = {}
    for cls in ds.topology_classes:
        rows = names == cls
        e = err[rows][mask[rows]]
        out[cls] = float(e.mean()) if e.size else np.nan
    return out


def load_curvenet(path: Path):
    try:
        from mpds_models.curves_io import load_curves_model
    except ImportError:
        from mpds_models.deep_curve_net import LiquidusSolidusModel
        load_curves_model = LiquidusSolidusModel.load
    return load_curves_model(path)


def train_curvenet(ds, seed: int, epochs: int):
    import torch
    from mpds_models.deep_curve_net import CurveNetConfig, LiquidusSolidusModel
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = CurveNetConfig(epochs=epochs, batch_size=min(32, max(8, len(ds.train_idx) // 4)),
                         lambda_endpoints=1.0, lambda_smooth=0.1)     # train_all defaults
    return LiquidusSolidusModel(cfg).fit(ds)


def predict_raw(model, X):
    try:
        return model.predict(X, clamp=False)
    except TypeError:
        return model.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_bins(df: pd.DataFrame, colors: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    mid = (BINS[:-1] + BINS[1:]) / 2
    for model, row in df.iterrows():
        ax.plot(mid, row.values, color=colors[model], lw=2, marker="o", ms=4, label=model)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("composition (at% B), 5 at% bins", fontsize=9, color=MUTED)
    ax.set_ylabel("liquidus MAE (K)", fontsize=9, color=MUTED)
    ax.set_title("Liquidus error across composition (mean over seeds)", fontsize=10,
                 color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_examples(ds, idx, curves: dict, colors: dict, out: Path) -> None:
    """One panel per topology class (first system of that class in the split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [ds.topology_name(i) for i in idx]
    picks = [names.index(c) for c in ds.topology_classes if c in names]
    if not picks:
        return
    grid = np.asarray(ds.comp_grid)
    ncol = min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow), squeeze=False)
    for ax, k in zip(axes.flat, picks):
        i = idx[k]
        m = ds.T_liquidus_mask[i]
        ax.scatter(grid[m], ds.T_liquidus[i, m], s=8, color=INK, label="data liquidus", zorder=3)
        s = np.isfinite(ds.T_solidus[i])
        ax.scatter(grid[s], ds.T_solidus[i, s], s=8, facecolor="white", edgecolor=INK,
                   linewidth=0.8, label="data solidus", zorder=3)
        for model, (liq, sol) in curves.items():
            ax.plot(grid, liq[k], color=colors[model], lw=2, label=model)
            ax.plot(grid, np.minimum(sol[k], liq[k]), color=colors[model], lw=1.2, ls="--")
        ax.set_title(f"{ds.pair_ids[i]} · {names[k]}", fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 100)
        _style(ax)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False, fontsize=8)
    fig.suptitle("x: composition (at% B)   solid lines: liquidus, dashed: solidus",
                 fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Tree-ensemble curves vs CurveNet.")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="curve_cmp")
    ap.add_argument("--backends", nargs="+", choices=CURVE_BACKENDS, default=list(CURVE_BACKENDS))
    ap.add_argument("--targets", nargs="+", choices=("deviation", "absolute"), default=["deviation"],
                    help="tree targets to try (deviation = exact end points)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--curvenet", nargs="*", type=Path, default=[],
                    help="trained CurveNet files (m23_curves.pt), one per seed")
    ap.add_argument("--train-curvenet", action="store_true",
                    help="also train CurveNet here, once per --seeds value")
    ap.add_argument("--curve-epochs", type=int, default=200)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--save-models", action="store_true",
                    help="save each tree model (ExtraTrees files can be a few hundred MB)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.processed_dir)
    idx = ds.split(a.split)
    if a.split == "val" and (a.curvenet or a.train_curvenet):
        print("NOTE: CurveNet picks its epoch on val, so val flatters it; use --split test.")

    runs = []                      # (model label, seed, fitted model, fit seconds)
    for path in a.curvenet:
        runs.append(("CurveNet", str(path), load_curvenet(path), np.nan))
    for seed in a.seeds:
        if a.train_curvenet:
            t0 = time.time()
            runs.append(("CurveNet", seed, train_curvenet(ds, seed, a.curve_epochs), time.time() - t0))
        for target in a.targets:
            for backend in a.backends:
                label = NICE[backend] + ("" if target == "deviation" else " (absolute)")
                t0 = time.time()
                m = TreeCurveModel(TreeCurveConfig(backend=backend, target=target, seed=seed)).fit(ds)
                runs.append((label, seed, m, time.time() - t0))
                if a.save_models:
                    m.save(out / f"tree_curves_{backend}_{target}_s{seed}.joblib")

    labels = list(dict.fromkeys(r[0] for r in runs))
    colors = {lab: SERIES[k % len(SERIES)] for k, lab in enumerate(labels)}
    rows, bins, topo, example = [], {}, {}, {}
    for label, seed, model, secs in runs:
        liq, sol = predict_raw(model, ds.X[idx])
        rows.append({"model": label, "seed": seed, "fit_seconds": secs, **scores(ds, idx, liq, sol)})
        bins.setdefault(label, []).append(by_bin(ds, idx, liq))
        topo.setdefault(label, []).append(by_topology(ds, idx, liq))
        example.setdefault(label, (liq, sol))
        print(f"{label:<26} seed {seed}: liquidus MAE {rows[-1]['liquidus_MAE_K']:.1f} K, "
              f"ends vs Tm {rows[-1]['end_vs_Tm_K']:.1f} K")

    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    metric_cols = [c for c in df.columns if c not in ("model", "seed")]
    agg = df.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    agg.to_csv(out / "summary.csv")

    bins_df = pd.DataFrame({lab: np.nanmean(v, axis=0) for lab, v in bins.items()},
                           index=[f"{lo}-{hi}" for lo, hi in zip(BINS[:-1], BINS[1:])]).T
    bins_df.to_csv(out / "by_composition.csv")
    topo_df = pd.DataFrame({lab: pd.DataFrame(v).mean() for lab, v in topo.items()}).T
    topo_df.to_csv(out / "by_topology.csv")
    plot_bins(bins_df, colors, out / "by_composition.png")
    plot_examples(ds, idx, example, colors, out / "examples.png")

    show = ["liquidus_MAE_K", "liquidus_MAE_interior_K", *[f"liquidus_MAE_at_{x}_K" for x in POINTS],
            "solidus_MAE_K", "end_vs_Tm_K", "order_violation_fraction", "fit_seconds"]
    table = df.groupby("model", sort=False)[show].mean()
    n_seeds = df.groupby("model", sort=False).size()
    std = df.groupby("model", sort=False)["liquidus_MAE_K"].std()
    table.insert(1, "liq_MAE_std", std)
    table.insert(0, "runs", n_seeds)
    print(f"\n=== {a.split} split, {len(idx)} systems (mean over runs) ===")
    print(table.to_string(float_format=lambda v: f"{v:.1f}" if abs(v) >= 1 else f"{v:.3f}"))
    print("\nLiquidus MAE by topology (K):")
    print(topo_df.to_string(float_format=lambda v: f"{v:.0f}"))
    print(f"\nSaved scores, summaries and charts in {out}/")


if __name__ == "__main__":
    main()
