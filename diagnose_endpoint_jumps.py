"""
diagnose_endpoint_jumps.py — find out WHY liquidus curves jump at x=0 and x=100.

Put this file next to predict.py / compare_diagrams.py and run:

    # check only the ground-truth data
    python diagnose_endpoint_jumps.py ~/data/processed

    # check the data AND the trained CurveNet's predictions
    python diagnose_endpoint_jumps.py ~/data/processed ~/data/models

It prints 4 short reports and saves endpoint_jumps.png with the 12 worst curves.
Each report answers one question; read them top to bottom.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mpds_models.base import load_dataset

JUMP_RATIO = 5.0   # an end step >5x steeper than the curve's typical step counts as a "jump"
TOL_K = 25.0       # how close (in K) two temperatures must be to count as "the same"


def end_step_ratio(curve: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """How much steeper the FIRST and LAST step are than a typical step.

    Uses only valid grid points, and divides by the grid distance so a gap
    in the data (e.g. points at x=0 and x=10 only) is not mistaken for a jump.
    """
    idx = np.flatnonzero(mask & np.isfinite(curve))
    if idx.size < 6:
        return np.nan, np.nan
    v = curve[idx]
    slope = np.abs(np.diff(v)) / np.diff(idx)      # K per grid step
    typical = np.median(slope[1:-1]) + 1.0          # +1 K so flat curves don't divide by ~0
    return slope[0] / typical, slope[-1] / typical


def jump_report(title: str, curves: np.ndarray, masks: np.ndarray) -> np.ndarray:
    ratios = np.array([end_step_ratio(c, m) for c, m in zip(curves, masks)])
    ok = np.isfinite(ratios).all(axis=1)
    left = ratios[ok, 0] > JUMP_RATIO
    right = ratios[ok, 1] > JUMP_RATIO
    print(f"\n{title}")
    print(f"  curves checked         : {ok.sum()}")
    print(f"  jump at x=0   (left)   : {left.sum()}  ({left.mean():.1%})")
    print(f"  jump at x=100 (right)  : {right.sum()}  ({right.mean():.1%})")
    print(f"  jump at BOTH ends      : {(left & right).sum()}")
    return np.nan_to_num(ratios.max(axis=1), nan=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("processed_dir")
    p.add_argument("model_dir", nargs="?", help="optional: also check predictions")
    p.add_argument("--out", default="endpoint_jumps.png")
    a = p.parse_args()

    ds = load_dataset(a.processed_dir)
    ia = ds.feature_names.index("phys__melting_point__A")
    ib = ds.feature_names.index("phys__melting_point__B")
    TmA, TmB = ds.X[:, ia], ds.X[:, ib]
    L, M = ds.T_liquidus, ds.T_liquidus_mask

    # ── [1] Are the melting-point features sane? ─────────────────────────────
    print("\n[1] Melting-point FEATURES (used by the endpoint penalty)")
    print(f"  Tm_A range: {TmA.min():.0f} .. {TmA.max():.0f}")
    print(f"  Tm_B range: {TmB.min():.0f} .. {TmB.max():.0f}")
    print("  (max near ~3700 -> Kelvin;  max near ~3400 -> Celsius)")
    n_zero = int(((TmA <= 0) | (TmB <= 0)).sum())
    print(f"  systems with Tm <= 0 (missing value turned into 0 by nan_to_num): {n_zero}")

    # ── [2] Do the TRUE curve endpoints agree with those features? ───────────
    has0, has1 = M[:, 0], M[:, -1]
    print("\n[2] TRUE liquidus endpoints vs melting-point features")
    print(f"  systems with data AT x=0  : {has0.sum()} / {len(M)}")
    print(f"  systems with data AT x=100: {has1.sum()} / {len(M)}")
    d = np.concatenate([L[has0, 0] - TmA[has0], L[has1, -1] - TmB[has1]])
    if d.size:
        print(f"  median (true endpoint - Tm feature): {np.median(d):+.0f} K")
        print(f"  agree within {TOL_K:.0f} K              : {(np.abs(d) < TOL_K).mean():.1%}")
        print(f"  off by about -273 K (data in C?)   : {(np.abs(d + 273.15) < TOL_K).mean():.1%}")
        print(f"  off by about +273 K (feature in C?): {(np.abs(d - 273.15) < TOL_K).mean():.1%}")
        swapped = has0 & (np.abs(L[:, 0] - TmB) < TOL_K) & (np.abs(L[:, 0] - TmA) > TOL_K)
        print(f"  x=0 matches Tm_B instead of Tm_A (A/B swapped?): {int(swapped.sum())}")

    # ── [3] Does the GROUND TRUTH itself jump at the ends? ───────────────────
    ratio_true = jump_report("[3] Jumps in the GROUND-TRUTH curves (all systems)", L, M)

    # ── [4] Does the MODEL's prediction jump at the ends? ────────────────────
    ratio_pred = None
    idx = ds.test_idx
    if a.model_dir:
        from mpds_models.curves_io import load_curves_model
        m23 = load_curves_model(Path(a.model_dir) / "m23_curves.pt")
        pred_liq, _ = m23.predict(ds.X[idx])
        full_mask = np.ones_like(pred_liq, dtype=bool)
        ratio_pred = jump_report("[4] Jumps in the PREDICTED curves (test split)",
                                 pred_liq, full_mask)
        pull0 = np.abs(pred_liq[:, 0] - TmA[idx])
        print(f"  median |pred(x=0) - Tm_A| : {np.median(pull0):.0f} K  "
              "(small = endpoint penalty is winning)")

    # ── plot the 12 worst examples ───────────────────────────────────────────
    if ratio_pred is not None:
        rows = idx[np.argsort(-ratio_pred)[:12]]
        pred_by_row = dict(zip(idx, pred_liq))
    else:
        rows = np.argsort(-ratio_true)[:12]
        pred_by_row = {}
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    x = ds.comp_grid
    for ax, r in zip(axes.ravel(), rows):
        m = M[r]
        # NaN where there is no data, so gaps are not bridged by straight lines
        ax.plot(x, np.where(m, L[r], np.nan), "o-", ms=2, color="#2980B9",
                label="true liquidus")
        if r in pred_by_row:
            ax.plot(x, pred_by_row[r], color="#C0392B", label="predicted")
        ax.scatter([0, 100], [TmA[r], TmB[r]], marker="*", s=150,
                   color="black", zorder=5, label="Tm features")
        ax.set_title(ds.pair_ids[r], fontsize=10)
        ax.set_xlim(-2, 102)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Worst end-jumps  (black stars = melting points the model is pulled toward)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
