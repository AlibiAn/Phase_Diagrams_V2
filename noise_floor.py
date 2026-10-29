"""
noise_floor.py — how much do the sources themselves disagree, and how close
are the curve models to that limit?

Where several MPDS diagrams describe the same system, preprocessing merges
them into one consensus curve and stores, per grid point, how far the sources
spread around it (the MAD). No model can match the consensus much more
closely than the sources match each other, so this spread is a floor under
the curve error.

The script reports the spread for systems with two or more sources and,
with --model-dir, the model's liquidus error grouped by that spread. If the
error is about as large as the spread where the sources disagree a lot, and
much larger where they agree, the model, not the data, is what limits it.

Usage
-----
    python noise_floor.py processed_v6
    python noise_floor.py processed_v6 --model-dir models_v10_trees --mask-clipped
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mpds_models.base import load_dataset, mask_clipped_tops

BINS = (0, 10, 25, 50, 100, 200, np.inf)


def spread_table(ds, idx) -> pd.DataFrame:
    rows = []
    multi = ds.n_sources[idx] >= 2
    for name, T, mad, seen in (
        ("liquidus", ds.T_liquidus, ds.T_liquidus_mad, ds.T_liquidus_mask),
        ("solidus", ds.T_solidus, ds.T_solidus_mad, np.isfinite(ds.T_solidus)),
    ):
        m = mad[idx][multi][seen[idx][multi]]
        m = m[np.isfinite(m)]
        rows.append(dict(curve=name, systems=int(multi.sum()), points=int(m.size),
                         mean_K=m.mean() if m.size else np.nan,
                         median_K=np.median(m) if m.size else np.nan,
                         p90_K=np.quantile(m, 0.9) if m.size else np.nan))
    return pd.DataFrame(rows)


def error_by_spread(ds, idx, liq) -> pd.DataFrame:
    err = np.abs(liq - ds.T_liquidus[idx])
    seen = ds.T_liquidus_mask[idx]
    mad = ds.T_liquidus_mad[idx]
    multi = np.repeat((ds.n_sources[idx] >= 2)[:, None], seen.shape[1], axis=1)
    rows = [dict(group="single source (spread unknown)",
                 points=int((seen & ~multi).sum()),
                 model_MAE_K=float(err[seen & ~multi].mean()) if (seen & ~multi).any() else np.nan,
                 mean_spread_K=np.nan)]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = seen & multi & (mad >= lo) & (mad < hi)
        rows.append(dict(group=f"spread {lo:g}-{hi:g} K", points=int(sel.sum()),
                         model_MAE_K=float(err[sel].mean()) if sel.any() else np.nan,
                         mean_spread_K=float(mad[sel].mean()) if sel.any() else np.nan))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Source disagreement vs model error.")
    ap.add_argument("processed_dir")
    ap.add_argument("--model-dir", help="models from train_all to compare with the spread")
    ap.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    ap.add_argument("--mask-clipped", action="store_true",
                    help="leave liquidus tops cut off by the source chart out")
    ap.add_argument("--out", help="save the tables as CSV files with this prefix")
    a = ap.parse_args()

    ds = load_dataset(a.processed_dir)
    if a.mask_clipped:
        mask_clipped_tops(ds)
    idx = np.arange(ds.n_systems) if a.split == "all" else ds.split(a.split)
    n_multi = int((ds.n_sources[idx] >= 2).sum())
    print(f"{a.split}: {len(idx)} systems, {n_multi} with two or more sources")

    spread = spread_table(ds, idx)
    print("\n=== How far the sources spread around the consensus (MAD) ===")
    print(spread.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if a.out:
        spread.to_csv(f"{a.out}_spread.csv", index=False)

    if a.model_dir:
        from mpds_models.meta_learner import FullDiagramReconstructor
        m23 = FullDiagramReconstructor.from_dir(a.model_dir, ds).m23
        liq, _ = m23.predict(ds.X[idx])
        table = error_by_spread(ds, idx, liq)
        print(f"\n=== Liquidus error of {a.model_dir} by how much the sources disagree ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print("\nWhere the model error is far above the spread, the model has room to improve; "
              "where they are close, the data is the limit.")
        if a.out:
            table.to_csv(f"{a.out}_model_error.csv", index=False)


if __name__ == "__main__":
    main()
