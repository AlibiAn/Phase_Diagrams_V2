"""
compare_diagrams.py — Compare predicted phase diagrams against ground truth.

Three output types
------------------
  1. full_grid.png          All test systems on one page — compact overlay
                            (predicted vs true on the same axes). Colour-coded
                            border: green = topology correct, red = wrong.
                            Sorted by topology correctness then liquidus MAE.

  2. ranked_comparison.png  Three tiers (best / median / worst) × N systems,
                            each shown as a 3-panel figure (true | pred | error).
                            Use --n-each to control how many per tier.

  3. <Pair>_comparison.png  Per-system 3-panel figure for named pairs.

Usage
-----
    # Full grid of all test systems (default — the most informative view)
    python compare_diagrams.py ~/data/processed ~/data/models --out ~/comparisons

    # Fewer or more columns in the grid
    python compare_diagrams.py ~/data/processed ~/data/models --out ~/comp --cols 6

    # Ranked page only (fastest)
    python compare_diagrams.py ~/data/processed ~/data/models --out ~/comp --ranked-only

    # Individual 3-panel figures for named pairs
    python compare_diagrams.py ~/data/processed ~/data/models \\
        --pairs Ag_Al Cu_Zr Al_Sb Fe_Ni --out ~/comparisons
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from mpds_models.base import load_dataset, PhaseDataset
from mpds_models.meta_learner import FullDiagramReconstructor, PredictedDiagram
from predict import liquidus_mae

# ── colours ──────────────────────────────────────────────────────────────────
C_LIQ_TRUE  = "#1A6FA8"
C_SOL_TRUE  = "#1A8C4E"
C_LIQ_PRED  = "#C0392B"
C_SOL_PRED  = "#7D3C98"
C_EUT_TRUE  = "#1A6FA8"
C_EUT_PRED  = "#C0392B"
BAND_ALPHA  = 0.15
GRID_ALPHA  = 0.22

TOPOLOGY_SHORT = {
    "isomorphous":                  "Isomorphous",
    "simple_eutectic":              "Simple eut.",
    "eutectic_with_intermetallics": "Eut.+IMs",
    "monotectic":                   "Monotectic",
    "complex":                      "Complex",
}


def _topo_correct(d: PredictedDiagram, ds: PhaseDataset, ds_idx: int) -> bool:
    return d.topology == ds.topology_name(ds_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Compact overlay panel  (one axis: predicted + true on same plot)
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    ax: plt.Axes,
    d: PredictedDiagram,
    ds: PhaseDataset,
    ds_idx: int,
) -> float:
    """Overlay predicted and true curves on one axis. Returns liquidus MAE."""
    comp = d.comp_grid
    elem_a, elem_b = d.pair.split("_")

    # ── predicted ──────────────────────────────────────────────────────────
    ax.plot(comp, d.liquidus_T, color=C_LIQ_PRED, lw=1.6, zorder=4)
    ax.plot(comp, d.solidus_T,  color=C_SOL_PRED, lw=1.2,
            linestyle="--", zorder=4)

    if d.liquidus_T_std is not None:
        ax.fill_between(comp,
                        d.liquidus_T - d.liquidus_T_std,
                        d.liquidus_T + d.liquidus_T_std,
                        color=C_LIQ_PRED, alpha=BAND_ALPHA, zorder=2)

    if d.eutectic_exists and d.eutectic_comp is not None:
        ax.scatter([d.eutectic_comp], [d.eutectic_temp],
                   color=C_EUT_PRED, s=30, zorder=6, marker="v")

    # ── ground truth ───────────────────────────────────────────────────────
    liq_true = ds.T_liquidus[ds_idx]
    sol_true = ds.T_solidus[ds_idx]
    mask_l   = ds.T_liquidus_mask[ds_idx]
    mask_s   = ~np.isnan(sol_true)

    liq_mae = float("nan")
    # NaN where there is no data, so gaps are not bridged by straight lines
    if mask_l.any():
        ax.plot(comp, np.where(mask_l, liq_true, np.nan),
                color=C_LIQ_TRUE, lw=1.6, linestyle=":", zorder=5, alpha=0.9)
        liq_mae = float(np.abs(d.liquidus_T - liq_true)[mask_l].mean())

    if mask_s.any():
        ax.plot(comp, np.where(mask_s, sol_true, np.nan),
                color=C_SOL_TRUE, lw=1.2, linestyle=":", zorder=5, alpha=0.9)

    if ds.eutectic_exists[ds_idx]:
        tc = float(ds.eutectic_comp[ds_idx])
        tt = float(ds.eutectic_temp[ds_idx])
        if np.isfinite(tc) and np.isfinite(tt):
            ax.scatter([tc], [tt], color=C_EUT_TRUE, s=30, zorder=7, marker="^")

    # ── title & labels ─────────────────────────────────────────────────────
    correct   = _topo_correct(d, ds, ds_idx)
    true_topo = ds.topology_name(ds_idx)
    pred_topo = d.topology
    tick      = "✓" if correct else "✗"
    mae_str   = f"{liq_mae:.0f}K" if np.isfinite(liq_mae) else "—"

    title_col = "#27AE60" if correct else "#E74C3C"
    ax.set_title(
        f"{elem_a}–{elem_b}   {tick}  {mae_str}",
        fontsize=7.5, fontweight="bold", color=title_col, pad=2,
    )

    if not correct:
        ax.text(0.99, 0.97,
                f"pred: {TOPOLOGY_SHORT.get(pred_topo, pred_topo)}\n"
                f"true: {TOPOLOGY_SHORT.get(true_topo, true_topo)}",
                transform=ax.transAxes, fontsize=5.5,
                va="top", ha="right", color="#E74C3C",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#E74C3C", alpha=0.85))

    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=5.5, pad=1)
    ax.set_xlabel(f"mol% {elem_b}", fontsize=5.5, labelpad=1)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.4)

    # Colour the axes border by topology correctness
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_edgecolor("#27AE60" if correct else "#E74C3C")

    return liq_mae


# ─────────────────────────────────────────────────────────────────────────────
# Full grid  (all systems, compact overlay)
# ─────────────────────────────────────────────────────────────────────────────

def save_full_grid(
    diagrams: list[PredictedDiagram],
    ds: PhaseDataset,
    idx_list: np.ndarray,
    out_path: Path,
    n_cols: int = 8,
) -> None:
    """One panel per system, all on a single page.

    Sort order:
      1. Topology-correct systems first (green borders), sorted by MAE ascending
      2. Topology-wrong systems second (red borders), sorted by MAE ascending
    """
    n = len(diagrams)
    maes = [liquidus_mae(d, ds, i) for d, i in zip(diagrams, idx_list)]
    correct = [_topo_correct(d, ds, i) for d, i in zip(diagrams, idx_list)]

    order = sorted(
        range(n),
        key=lambda i: (0 if correct[i] else 1,
                       maes[i] if np.isfinite(maes[i]) else 9999),
    )

    n_rows = (n + n_cols - 1) // n_cols
    # 2.6 × 2.3 in per panel, plus a strip for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.3 + 0.9),
                             squeeze=False)

    n_correct = sum(correct)
    n_total   = len(diagrams)
    finite_maes = [m for m in maes if np.isfinite(m)]
    mean_mae  = np.mean(finite_maes) if finite_maes else float("nan")

    fig.suptitle(
        f"All {n_total} test systems — topology accuracy {n_correct}/{n_total} "
        f"({n_correct/n_total:.0%})   |   mean liquidus MAE {mean_mae:.0f} K\n"
        f"── Borders: green = topology correct, red = wrong  "
        f"── Curves: red solid = predicted, blue dotted = ground truth",
        fontsize=8.5, fontweight="bold", y=0.995,
    )

    flat_axes = axes.ravel()
    for ax, orig_i in zip(flat_axes, order):
        draw_overlay(ax, diagrams[orig_i], ds, idx_list[orig_i])
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=C_LIQ_PRED, lw=2,   label="Predicted liquidus"),
        Line2D([0], [0], color=C_SOL_PRED, lw=1.5, linestyle="--", label="Predicted solidus"),
        Line2D([0], [0], color=C_LIQ_TRUE, lw=2,   linestyle=":", label="True liquidus"),
        Line2D([0], [0], color=C_SOL_TRUE, lw=1.5, linestyle=":", label="True solidus"),
        Line2D([0], [0], marker="v", color=C_EUT_PRED, linestyle="None",
               markersize=6, label="Predicted eutectic"),
        Line2D([0], [0], marker="^", color=C_EUT_TRUE, linestyle="None",
               markersize=6, label="True eutectic"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0), framealpha=0.95)

    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_wrong = n_total - n_correct
    print(f"Full grid saved → {out_path}")
    print(f"  {n_correct} correct (green)  |  {n_wrong} wrong (red)  |  "
          f"mean MAE {mean_mae:.0f} K")


# ─────────────────────────────────────────────────────────────────────────────
# Ranked comparison page  (best / median / worst, 3-panel each)
# ─────────────────────────────────────────────────────────────────────────────

def draw_ground_truth(ax, d, ds, ds_idx):
    comp = d.comp_grid
    elem_b = d.pair.split("_")[1]
    liq_true, sol_true = ds.T_liquidus[ds_idx], ds.T_solidus[ds_idx]
    mask_l, mask_s = ds.T_liquidus_mask[ds_idx], ~np.isnan(sol_true)
    if mask_l.any():
        ax.plot(comp, np.where(mask_l, liq_true, np.nan), color=C_LIQ_TRUE, lw=2)
    if mask_s.any():
        ax.plot(comp, np.where(mask_s, sol_true, np.nan),
                color=C_SOL_TRUE, lw=2, linestyle="--")
    if ds.eutectic_exists[ds_idx]:
        tc, tt = float(ds.eutectic_comp[ds_idx]), float(ds.eutectic_temp[ds_idx])
        if np.isfinite(tc) and np.isfinite(tt):
            ax.scatter([tc], [tt], color=C_EUT_TRUE, s=70, zorder=5, marker="*")
    true_topo = ds.topology_name(ds_idx)
    ax.set_title("Ground Truth", fontsize=8.5, fontweight="bold", color="#2C3E50")
    ax.text(0.03, 0.97, TOPOLOGY_SHORT.get(true_topo, true_topo),
            transform=ax.transAxes, fontsize=7.5, va="top", color=C_LIQ_TRUE,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_LIQ_TRUE, alpha=0.85))
    ax.set_xlabel(f"mol% {elem_b}", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylabel("T (K)", fontsize=7.5)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=7)


def draw_prediction(ax, d, ds, ds_idx):
    comp = d.comp_grid
    elem_b = d.pair.split("_")[1]
    ax.plot(comp, d.liquidus_T, color=C_LIQ_PRED, lw=2)
    ax.plot(comp, d.solidus_T,  color=C_SOL_PRED, lw=2, linestyle="--")
    if d.liquidus_T_std is not None:
        ax.fill_between(comp, d.liquidus_T - d.liquidus_T_std,
                        d.liquidus_T + d.liquidus_T_std,
                        color=C_LIQ_PRED, alpha=BAND_ALPHA)
    if d.eutectic_exists and d.eutectic_comp is not None:
        ax.scatter([d.eutectic_comp], [d.eutectic_temp],
                   color=C_EUT_PRED, s=70, zorder=5, marker="*")
    pred_topo = d.topology
    conf = d.topology_proba[pred_topo]
    correct = _topo_correct(d, ds, ds_idx)
    col = "#27AE60" if correct else "#E74C3C"
    tick = "✓" if correct else "✗"
    ax.set_title("Prediction", fontsize=8.5, fontweight="bold", color="#2C3E50")
    ax.text(0.03, 0.97, f"{tick} {TOPOLOGY_SHORT.get(pred_topo, pred_topo)} ({conf:.0%})",
            transform=ax.transAxes, fontsize=7.5, va="top", color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))
    ax.set_xlabel(f"mol% {elem_b}", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=7)


def draw_error_profile(ax, d, ds, ds_idx):
    comp = d.comp_grid
    liq_true, mask_l = ds.T_liquidus[ds_idx], ds.T_liquidus_mask[ds_idx]
    sol_true = ds.T_solidus[ds_idx]
    mask_s = ~np.isnan(sol_true)
    liq_mae = float("nan")
    if mask_l.any():
        err = np.where(mask_l, np.abs(d.liquidus_T - liq_true), np.nan)
        ax.plot(comp, err, color=C_LIQ_PRED, lw=1.6, label="Liq error")
        ax.fill_between(comp, 0, np.nan_to_num(err), color=C_LIQ_PRED, alpha=0.18)
        liq_mae = float(np.nanmean(err))
    if mask_s.any():
        err_s = np.where(mask_s, np.abs(d.solidus_T - np.nan_to_num(sol_true)), np.nan)
        ax.plot(comp, err_s, color=C_SOL_PRED, lw=1.6, linestyle="--", label="Sol error")
    for ref, sty in [(50, ":"), (100, "--"), (200, "-.")]:
        ax.axhline(ref, color="#AAB", lw=0.8, linestyle=sty, alpha=0.6)
        ax.text(101, ref, f"{ref}K", fontsize=5.5, va="center", color="#999")
    if np.isfinite(liq_mae):
        ax.axhline(liq_mae, color=C_LIQ_PRED, lw=1.2, linestyle="--", alpha=0.6)
        ax.text(3, liq_mae + 6, f"mean {liq_mae:.0f} K", fontsize=6.5, color=C_LIQ_PRED)
    ax.set_title("|Pred − True|", fontsize=8.5, fontweight="bold", color="#2C3E50")
    ax.set_xlabel("mol%", fontsize=7.5)
    ax.set_ylabel("Error (K)", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")
    return liq_mae


def save_single_comparison(d, ds, ds_idx, out_path):
    elem_a, elem_b = d.pair.split("_")
    correct = _topo_correct(d, ds, ds_idx)
    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)
    ax_true = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1], sharey=ax_true)
    ax_err  = fig.add_subplot(gs[2])
    draw_ground_truth(ax_true, d, ds, ds_idx)
    draw_prediction(ax_pred, d, ds, ds_idx)
    liq_mae = draw_error_profile(ax_err, d, ds, ds_idx)
    tick = "✓" if correct else "✗"
    col  = "#27AE60" if correct else "#E74C3C"
    fig.suptitle(
        f"{elem_a}–{elem_b}  {tick} topology {'correct' if correct else 'wrong'}"
        f"   │  Liquidus MAE = {liq_mae:.0f} K",
        fontsize=11, fontweight="bold", color=col, y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return liq_mae


def save_ranked_page(diagrams, ds, idx_list, maes, out_path, n_each=5):
    finite = [(i, m) for i, m in enumerate(maes) if np.isfinite(m)]
    if len(finite) < 3:
        print("Not enough finite-MAE systems for ranked page.")
        return
    finite.sort(key=lambda x: x[1])
    n = len(finite)
    k = min(n_each, n)
    best   = finite[:k]
    worst  = finite[::-1][:k]
    start  = n // 2 - n_each // 2
    median = [finite[max(0, start + i)] for i in range(k)]

    tiers = [
        ("Best predictions  (lowest liquidus MAE)",   best,   "#27AE60"),
        ("Median predictions",                         median, "#2980B9"),
        ("Worst predictions  (highest liquidus MAE)", worst,  "#E74C3C"),
    ]
    n_rows = len(tiers)
    n_cols = n_each * 3
    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 4.0))
    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55,
                              left=0.03, right=0.97)
    for row_i, (label, items, col) in enumerate(tiers):
        fig.text(0.005, 1 - (row_i + 0.06) / n_rows, label,
                 fontsize=9, fontweight="bold", color=col,
                 transform=fig.transFigure)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(items) * 3, subplot_spec=outer[row_i], wspace=0.42)
        for col_i, (orig_i, mae) in enumerate(items):
            d, ds_i = diagrams[orig_i], idx_list[orig_i]
            base = col_i * 3
            ax_t = fig.add_subplot(inner[base])
            ax_p = fig.add_subplot(inner[base + 1], sharey=ax_t)
            ax_e = fig.add_subplot(inner[base + 2])
            draw_ground_truth(ax_t, d, ds, ds_i)
            draw_prediction(ax_p, d, ds, ds_i)
            draw_error_profile(ax_e, d, ds, ds_i)
            elem_a, elem_b = d.pair.split("_")
            correct = _topo_correct(d, ds, ds_i)
            tick = "✓" if correct else "✗"
            ax_t.set_title(f"{elem_a}–{elem_b}  {tick}  {mae:.0f} K",
                           fontsize=8, fontweight="bold",
                           color="#27AE60" if correct else "#E74C3C")
            if col_i > 0:
                ax_t.set_ylabel("")
                ax_p.set_ylabel("")

    legend_elements = [
        Line2D([0], [0], color=C_LIQ_TRUE, lw=2, label="True liquidus"),
        Line2D([0], [0], color=C_SOL_TRUE, lw=2, linestyle="--", label="True solidus"),
        Line2D([0], [0], color=C_LIQ_PRED, lw=2, label="Predicted liquidus"),
        Line2D([0], [0], color=C_SOL_PRED, lw=2, linestyle="--", label="Predicted solidus"),
        Line2D([0], [0], marker="*", color=C_EUT_TRUE, linestyle="None",
               markersize=8, label="True eutectic"),
        Line2D([0], [0], marker="*", color=C_EUT_PRED, linestyle="None",
               markersize=8, label="Predicted eutectic"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
    fig.suptitle("Ranked comparison — Liquidus MAE",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Ranked comparison saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading dataset ...")
    ds = load_dataset(args.processed_dir)

    print("Loading models ...")
    recon = FullDiagramReconstructor.from_dir(args.model_dir, ds)

    idx_list = ds.indices_of(args.pairs) if args.pairs else ds.test_idx

    pairs = [ds.pair_ids[i] for i in idx_list]
    print(f"Predicting {len(pairs)} systems ...")

    diagrams = recon.reconstruct(
        ds.X[idx_list], pairs,
        with_uncertainty=True, n_mc_samples=20,
    )
    maes = [liquidus_mae(d, ds, i) for d, i in zip(diagrams, idx_list)]

    if not args.ranked_only:
        print(f"Saving full grid ({len(diagrams)} panels) ...")
        save_full_grid(diagrams, ds, idx_list,
                       out / "full_grid.png", n_cols=args.cols)

    if len(diagrams) >= 3:
        save_ranked_page(diagrams, ds, idx_list, maes,
                         out / "ranked_comparison.png",
                         n_each=args.n_each)

    if args.pairs:
        print("Saving individual 3-panel comparisons ...")
        for d, i in zip(diagrams, idx_list):
            mae = save_single_comparison(d, ds, i, out / f"{d.pair}_comparison.png")
            correct = _topo_correct(d, ds, i)
            print(f"  {d.pair:15}  {'✓' if correct else '✗'}  "
                  f"liq MAE = {mae:.0f} K")

    n_correct = sum(_topo_correct(d, ds, i) for d, i in zip(diagrams, idx_list))
    finite_maes = [m for m in maes if np.isfinite(m)]
    print(f"\n  Topology accuracy : {n_correct}/{len(diagrams)} "
          f"({n_correct/len(diagrams):.1%})")
    if finite_maes:
        print(f"  Liquidus MAE      : {np.mean(finite_maes):.0f} K  "
              f"(median {np.median(finite_maes):.0f} K, "
              f"best {np.min(finite_maes):.0f} K, "
              f"worst {np.max(finite_maes):.0f} K)")
    print(f"\n  Outputs in {out}/")
    if not args.ranked_only:
        print(f"  • full_grid.png          — all {len(diagrams)} systems, overlay")
    print(f"  • ranked_comparison.png  — best/median/worst ({args.n_each} each)")
    if args.pairs:
        print("  • <pair>_comparison.png  — 3-panel per named pair")


def _cli():
    p = argparse.ArgumentParser(
        description="Compare predicted vs true phase diagrams."
    )
    p.add_argument("processed_dir")
    p.add_argument("model_dir")
    p.add_argument("--pairs",    nargs="+",
                   help="specific pairs for 3-panel detail plots")
    p.add_argument("--out",      default="./comparisons")
    p.add_argument("--cols",     type=int, default=8,
                   help="columns in the full grid (default: 8)")
    p.add_argument("--n-each",   type=int, default=5,
                   help="systems per tier in ranked page (default: 5)")
    p.add_argument("--ranked-only", action="store_true",
                   help="skip full grid, only save ranked page")
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
