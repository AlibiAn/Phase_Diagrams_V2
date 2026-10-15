"""
plot_diagrams.py — Visualise predicted phase diagrams against ground truth.

Usage
-----
    # Plot all test-split systems, save to ~/diagrams/
    python plot_diagrams.py ~/data/processed ~/data/models --out ~/diagrams

    # Plot specific pairs
    python plot_diagrams.py ~/data/processed ~/data/models --pairs Ag_Al Cu_Zr Fe_Ni --out ~/diagrams

    # Summary grid of best and worst predictions
    python plot_diagrams.py ~/data/processed ~/data/models --grid --out ~/diagrams
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from mpds_models.base import load_dataset, PhaseDataset, precision_recall_f1
from mpds_models.meta_learner import FullDiagramReconstructor, PredictedDiagram
from predict import liquidus_mae


# ─────────────────────────────────────────────────────────────────────────────
# Colours / style
# ─────────────────────────────────────────────────────────────────────────────

PRED_LIQ_COLOR  = "#C0392B"   # red
PRED_SOL_COLOR  = "#8E44AD"   # purple
TRUE_LIQ_COLOR  = "#2980B9"   # blue
TRUE_SOL_COLOR  = "#27AE60"   # green
EUT_TRUE_COLOR  = "#2980B9"
EUT_PRED_COLOR  = "#C0392B"
BAND_ALPHA      = 0.15
GRID_ALPHA      = 0.25

TOPOLOGY_LABELS = {
    "isomorphous":                  "Isomorphous (lens)",
    "simple_eutectic":              "Simple eutectic",
    "eutectic_with_intermetallics": "Eutectic + intermetallics",
    "monotectic":                   "Monotectic",
    "complex":                      "Complex",
}


# ─────────────────────────────────────────────────────────────────────────────
# Single diagram plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_single(
    d: PredictedDiagram,
    ds: PhaseDataset,
    ds_idx: int,
    ax: plt.Axes,
    show_legend: bool = True,
) -> float:
    """Draw one phase diagram on `ax`. Returns the liquidus MAE in K (0 if no data)."""
    comp = d.comp_grid                       # (101,) mol% B
    elem_a, elem_b = d.pair.split("_")

    # ── predicted curves ───────────────────────────────────────────────────
    ax.plot(comp, d.liquidus_T, color=PRED_LIQ_COLOR, lw=2.0,
            label="Predicted liquidus", zorder=4)
    ax.plot(comp, d.solidus_T,  color=PRED_SOL_COLOR, lw=2.0,
            linestyle="--", label="Predicted solidus", zorder=4)

    if d.liquidus_T_std is not None:
        ax.fill_between(comp,
                        d.liquidus_T - d.liquidus_T_std,
                        d.liquidus_T + d.liquidus_T_std,
                        color=PRED_LIQ_COLOR, alpha=BAND_ALPHA, zorder=2)
        ax.fill_between(comp,
                        d.solidus_T - d.solidus_T_std,
                        d.solidus_T + d.solidus_T_std,
                        color=PRED_SOL_COLOR, alpha=BAND_ALPHA, zorder=2)

    # ── ground truth ───────────────────────────────────────────────────────
    liq_true = ds.T_liquidus[ds_idx]
    sol_true = ds.T_solidus[ds_idx]
    mask_l = ds.T_liquidus_mask[ds_idx]
    mask_s = ~np.isnan(sol_true)

    liq_mae = None
    # NaN where there is no data, so gaps are not bridged by straight lines
    if mask_l.any():
        ax.plot(comp, np.where(mask_l, liq_true, np.nan),
                color=TRUE_LIQ_COLOR, lw=2.0, linestyle=":",
                label="True liquidus", zorder=5, alpha=0.85)
        liq_mae = float(np.abs(d.liquidus_T - liq_true)[mask_l].mean())

    if mask_s.any():
        ax.plot(comp, np.where(mask_s, sol_true, np.nan),
                color=TRUE_SOL_COLOR, lw=2.0, linestyle=":",
                label="True solidus", zorder=5, alpha=0.85)

    # ── predicted eutectic point ───────────────────────────────────────────
    if d.eutectic_exists and d.eutectic_comp is not None:
        ax.scatter([d.eutectic_comp], [d.eutectic_temp],
                   color=EUT_PRED_COLOR, s=80, zorder=6,
                   marker="v", label=f"Predicted eutectic ({d.eutectic_comp:.1f}%, {d.eutectic_temp:.0f} K)")

    # ── true eutectic point ────────────────────────────────────────────────
    if ds.eutectic_exists[ds_idx]:
        tc = float(ds.eutectic_comp[ds_idx])
        tt = float(ds.eutectic_temp[ds_idx])
        if np.isfinite(tc) and np.isfinite(tt):
            ax.scatter([tc], [tt],
                       color=EUT_TRUE_COLOR, s=80, zorder=6,
                       marker="^", label=f"True eutectic ({tc:.1f}%, {tt:.0f} K)")

    # ── miscibility gap marker ─────────────────────────────────────────────
    if d.miscibility_gap:
        lo, hi = ax.get_ylim()
        ax.axhspan(hi - 0.03 * (hi - lo), hi, color="#F39C12", alpha=0.25)

    # ── topology annotation ────────────────────────────────────────────────
    true_topo  = ds.topology_name(ds_idx)
    pred_topo  = d.topology
    correct    = pred_topo == true_topo
    conf       = d.topology_proba[pred_topo]
    topo_color = "#27AE60" if correct else "#E74C3C"
    tick       = "✓" if correct else "✗"

    ax.set_title(f"{elem_a}–{elem_b}", fontsize=13, fontweight="bold", pad=4)
    ax.text(0.01, 0.98,
            f"{tick} {TOPOLOGY_LABELS.get(pred_topo, pred_topo)}  ({conf:.0%})",
            transform=ax.transAxes, fontsize=8,
            color=topo_color, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=topo_color, alpha=0.8))

    if not correct:
        ax.text(0.01, 0.88,
                f"True: {TOPOLOGY_LABELS.get(true_topo, true_topo)}",
                transform=ax.transAxes, fontsize=7.5,
                color=TRUE_LIQ_COLOR, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=TRUE_LIQ_COLOR, alpha=0.7))

    if liq_mae is not None:
        ax.text(0.99, 0.98,
                f"MAE {liq_mae:.0f} K",
                transform=ax.transAxes, fontsize=7.5,
                color="#555", va="top", ha="right")

    if d.inconsistencies:
        ax.text(0.5, 0.02,
                f"⚠ {d.inconsistencies[0][:60]}",
                transform=ax.transAxes, fontsize=6.5,
                color="#E67E22", va="bottom", ha="center")

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xlabel(f"Composition (mol% {elem_b})", fontsize=9)
    ax.set_ylabel("Temperature (K)", fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=8)

    if show_legend:
        ax.legend(fontsize=7, loc="lower center", ncol=2, framealpha=0.85)

    return liq_mae or 0.0


def save_single_diagram(
    d: PredictedDiagram,
    ds: PhaseDataset,
    ds_idx: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_single(d, ds, ds_idx, ax, show_legend=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_grid(
    diagrams: list[PredictedDiagram],
    ds: PhaseDataset,
    idx_list: np.ndarray,
    out_path: Path,
    n_cols: int = 4,
    max_systems: int = 24,
) -> None:
    """Grid of up to `max_systems` diagrams, best liquidus MAE first."""
    maes = np.nan_to_num([liquidus_mae(d, ds, i) for d, i in zip(diagrams, idx_list)], nan=9999.0)
    order = np.argsort(maes)[:max_systems]
    n_show = len(order)
    n_rows = (n_show + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 4.5, n_rows * 3.8))
    fig.suptitle("Predicted Phase Diagrams — Test Split (sorted by liquidus MAE)",
                 fontsize=13, fontweight="bold", y=1.01)

    for plot_i, orig_i in enumerate(order):
        ax = fig.add_subplot(n_rows, n_cols, plot_i + 1)
        plot_single(diagrams[orig_i], ds, idx_list[orig_i], ax, show_legend=(plot_i == 0))

    legend_elements = [
        mpatches.Patch(color=PRED_LIQ_COLOR, label="Predicted liquidus"),
        mpatches.Patch(color=PRED_SOL_COLOR, label="Predicted solidus"),
        mpatches.Patch(color=TRUE_LIQ_COLOR, label="True liquidus"),
        mpatches.Patch(color=TRUE_SOL_COLOR, label="True solidus"),
        plt.Line2D([0], [0], marker="v", color=EUT_PRED_COLOR,
                   linestyle="None", markersize=8, label="Predicted eutectic"),
        plt.Line2D([0], [0], marker="^", color=EUT_TRUE_COLOR,
                   linestyle="None", markersize=8, label="True eutectic"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy summary bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_bars(
    diagrams: list[PredictedDiagram],
    ds: PhaseDataset,
    idx_list: np.ndarray,
    out_path: Path,
) -> None:
    """Per-class topology accuracy bar chart + overall metrics panel."""
    classes = list(ds.topology_classes)
    y_true  = np.array([ds.topology_name(i) for i in idx_list])
    y_pred  = np.array([d.topology for d in diagrams])

    precisions, recalls, f1s = zip(*(precision_recall_f1(y_true == c, y_pred == c) for c in classes))
    supports = [int((y_true == c).sum()) for c in classes]

    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── panel 1: per-class F1 ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    x   = np.arange(len(classes))
    w   = 0.25
    bars_f1 = ax1.bar(x - w, f1s,      width=w, color="#C0392B", label="F1")
    ax1.bar(x,     precisions, width=w, color="#2980B9", label="Precision")
    ax1.bar(x + w, recalls,    width=w, color="#27AE60", label="Recall")

    for bar, sup in zip(bars_f1, supports):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"n={sup}", ha="center", va="bottom", fontsize=7.5, color="#555")

    ax1.set_xticks(x)
    ax1.set_xticklabels([TOPOLOGY_LABELS.get(c, c) for c in classes],
                        rotation=18, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Topology Classification — Per-Class Performance", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.axhline(0.8, color="#999", linestyle="--", lw=0.8)
    ax1.grid(axis="y", alpha=0.3)

    # ── panel 2: liquidus MAE histogram ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    liq_maes = [m for m in (liquidus_mae(d, ds, i) for d, i in zip(diagrams, idx_list))
                if np.isfinite(m)]
    ax2.hist(liq_maes, bins=20, color=PRED_LIQ_COLOR, edgecolor="white", alpha=0.85)
    ax2.axvline(np.mean(liq_maes), color="#2C3E50", linestyle="--", lw=1.5,
                label=f"Mean = {np.mean(liq_maes):.0f} K")
    ax2.axvline(np.median(liq_maes), color="#7F8C8D", linestyle=":", lw=1.5,
                label=f"Median = {np.median(liq_maes):.0f} K")
    ax2.set_xlabel("Liquidus MAE (K)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Liquidus Error Distribution", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── panel 3: eutectic position scatter ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    pred_comps, true_comps = [], []
    for d, i in zip(diagrams, idx_list):
        if d.eutectic_exists and ds.eutectic_exists[i] and d.eutectic_comp is not None:
            tc = float(ds.eutectic_comp[i])
            if np.isfinite(tc):
                pred_comps.append(d.eutectic_comp)
                true_comps.append(tc)

    if pred_comps:
        ax3.scatter(true_comps, pred_comps, alpha=0.7, color=PRED_LIQ_COLOR, s=45, zorder=3)
        lim = [0, 100]
        ax3.plot(lim, lim, "k--", lw=1, alpha=0.5, label="Perfect prediction")
        mae_comp = np.mean(np.abs(np.array(pred_comps) - np.array(true_comps)))
        ax3.text(0.05, 0.92, f"MAE = {mae_comp:.1f} at%",
                 transform=ax3.transAxes, fontsize=9, color="#2C3E50",
                 bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
        ax3.set_xlabel("True eutectic composition (at%)", fontsize=10)
        ax3.set_ylabel("Predicted eutectic composition (at%)", fontsize=10)
        ax3.set_title("Eutectic Composition: Predicted vs True", fontsize=10, fontweight="bold")
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No matched eutectic pairs", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=11, color="#999")

    overall_acc = float((y_true == y_pred).mean())
    fig.suptitle(
        f"Model Accuracy Summary  —  {len(diagrams)} test systems  "
        f"(topology accuracy: {overall_acc:.1%})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Accuracy summary saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading dataset ...")
    ds = load_dataset(args.processed_dir)

    print("Loading models ...")
    recon = FullDiagramReconstructor.from_dir(args.model_dir, ds)

    idx_list = ds.indices_of(args.pairs) if args.pairs else ds.test_idx

    pairs = [ds.pair_ids[i] for i in idx_list]
    print(f"Predicting {len(pairs)} systems ...")

    diagrams = recon.reconstruct(ds.X[idx_list], pairs, with_uncertainty=True, n_mc_samples=20)
    print(f"Done. Saving plots to {out}/")

    for d, idx in zip(diagrams, idx_list):
        save_single_diagram(d, ds, idx, out / f"{d.pair}.png")

    if len(diagrams) >= 4:
        make_summary_grid(diagrams, ds, idx_list, out / "summary_grid.png",
                          n_cols=4, max_systems=24)
    plot_accuracy_bars(diagrams, ds, idx_list, out / "accuracy_summary.png")

    print(f"\nAll done. Outputs in {out}/")
    print(f"  • {len(diagrams)} individual diagram PNGs")
    print("  • summary_grid.png — all diagrams sorted by accuracy")
    print("  • accuracy_summary.png — classification metrics + error distributions")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Plot predicted vs true phase diagrams.")
    p.add_argument("processed_dir")
    p.add_argument("model_dir")
    p.add_argument("--pairs", nargs="+",
                   help="specific element pairs to plot, e.g. --pairs Ag_Al Cu_Zr")
    p.add_argument("--out",   default="./diagrams",
                   help="output directory for PNG files (default: ./diagrams)")
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
