"""
make_review_figures.py — the figures of docs/research_review.md.

  two_liquid_correction.png   six two-liquid systems: the "liquidus" of
                              processed_v6 (the dome) and the liquidus of
                              processed_v7 (the monotectic line)
  learning_curve.png          v14 cross-validated error against the number of
                              training systems (read from the explore log)
  examples_v13_v14.png        test-split diagrams: data, v13 and v14 curves

Usage
-----
    python make_review_figures.py --v6 /home/alibi/lab/processed_v6 --v7 /home/alibi/lab/processed_v7 \\
        --v13 /home/alibi/lab/models_v13 --v14 /home/alibi/lab/models_v14 \\
        --explore-log results/v14/explore_v6.txt --out docs/figures
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpds_models.base import load_dataset, load_dataset_for_models, mask_clipped_tops
from mpds_models.meta_learner import FullDiagramReconstructor

# reference palette, light surface: categorical slots 1-3 in fixed order
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def two_liquid(v6: str, v7: str, out: Path, pairs: list[str]) -> None:
    d6, d7 = load_dataset(v6), load_dataset(v7)
    grid = d6.comp_grid
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE)
    for ax, pair in zip(axes.flat, pairs):
        i = d6.pair_ids.index(pair)
        a, b = pair.split("_")
        old = np.where(d6.T_liquidus_mask[i], d6.T_liquidus[i], np.nan)
        new = np.where(d7.T_liquidus_mask[i], d7.T_liquidus[i], np.nan)
        ax.plot(grid, old, color=ORANGE, lw=2, label="liquidus target before (processed_v6): top of the two-liquid dome")
        ax.plot(grid, new, color=BLUE, lw=2, label="liquidus target after (processed_v7): monotectic line")
        ax.set_title(f"{a}–{b}   {d6.topology_name(i).replace('_', ' ')} → {d7.topology_name(i).replace('_', ' ')}",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 100)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("composition (at% of the second element)", fontsize=8.5, color=SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("temperature (K)", fontsize=8.5, color=SECONDARY)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Two-liquid systems: the old target follows the dome, the corrected one the monotectic line",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "two_liquid_correction.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def learning_curve(log: str, out: Path) -> None:
    rows = re.findall(r"\d+% of the training systems \((\d+)\)\s+liquidus\s+([\d.]+)\s+solidus\s+([\d.]+)",
                      Path(log).read_text())
    if not rows:
        print(f"no learning-curve lines in {log}; skipped")
        return
    n, liq, sol = (np.array([float(r[k]) for r in rows]) for k in range(3))
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    for y, color, name in ((sol, ORANGE, "solidus"), (liq, BLUE, "liquidus")):
        ax.plot(n, y, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=name)
        ax.annotate(f"{name}  {y[-1]:.0f} K", (n[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [f"{int(v)}" for v in n])
    ax.minorticks_off()
    ax.set_xlim(n[0] * 0.9, n[-1] * 1.45)
    ax.set_ylim(0, max(sol.max(), liq.max()) * 1.1)
    ax.set_xlabel("systems the model was trained on (log scale)", fontsize=8.5, color=SECONDARY)
    ax.set_ylabel("cross-validated MAE (K)", fontsize=8.5, color=SECONDARY)
    ax.set_title("The error is still falling with more systems (v14, 5-fold CV)", fontsize=10, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def examples(v6: str, v13: str, v14: str, out: Path, pairs: list[str]) -> None:
    curves = {}
    for name, model_dir in (("v13", v13), ("v14", v14)):
        ds = load_dataset_for_models(v6, model_dir)
        mask_clipped_tops(ds)
        idx = ds.indices_of(pairs)
        liq, sol = FullDiagramReconstructor.from_dir(model_dir, ds).m23.predict(ds.X[idx])
        curves[name] = (liq, np.minimum(sol, liq))
    grid = ds.comp_grid
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor=SURFACE)
    for k, (ax, i) in enumerate(zip(axes.flat, idx)):
        seen = ds.T_liquidus_mask[i]
        ax.scatter(grid[seen], ds.T_liquidus[i, seen], s=9, color=INK, zorder=3, label="data: liquidus")
        s_seen = np.isfinite(ds.T_solidus[i])
        ax.scatter(grid[s_seen], ds.T_solidus[i, s_seen], s=9, facecolor=SURFACE, edgecolor=INK, linewidth=0.7,
                   zorder=3, label="data: solidus")
        notes = []
        for name, color in (("v13", BLUE), ("v14", ORANGE)):
            liq, sol = curves[name]
            err_l = np.abs(liq[k] - ds.T_liquidus[i])[seen].mean()
            err_s = np.abs(sol[k] - ds.T_solidus[i])[s_seen].mean() if s_seen.any() else float("nan")
            ax.plot(grid, liq[k], color=color, lw=2, label=f"{name}: liquidus (solid), solidus (dashed)")
            ax.plot(grid, sol[k], color=color, lw=1.3, ls="--")
            notes.append(f"{name} {err_l:.0f} / {err_s:.0f} K")
        a, b = ds.pair_ids[i].split("_")
        ax.set_title(f"{a}–{b}   {ds.topology_name(i).replace('_', ' ')}\n" + "     ".join(notes), fontsize=9,
                     color=INK, loc="left", linespacing=1.5)
        ax.set_xlim(0, 100)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("composition (at% of the second element)", fontsize=8.5, color=SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("temperature (K)", fontsize=8.5, color=SECONDARY)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Six test-split systems the models never saw, from well to badly predicted "
                 "(numbers: liquidus / solidus MAE)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "examples_v13_v14.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures for docs/research_review.md.")
    ap.add_argument("--v6", required=True)
    ap.add_argument("--v7", required=True)
    ap.add_argument("--v13", required=True)
    ap.add_argument("--v14", required=True)
    ap.add_argument("--explore-log", required=True)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--two-liquid-pairs", nargs=6,
                    default=["Ag_V", "Cu_Pb", "Al_In", "Ag_Ni", "Bi_Zn", "Ag_Rh"])
    ap.add_argument("--example-pairs", nargs=6, default=None,
                    help="six test-split pairs (default: every 23rd test system by v14 error rank)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    two_liquid(a.v6, a.v7, out, a.two_liquid_pairs)
    learning_curve(a.explore_log, out)
    pairs = a.example_pairs
    if pairs is None:
        ds = load_dataset_for_models(a.v6, a.v14)
        mask_clipped_tops(ds)
        liq, _ = FullDiagramReconstructor.from_dir(a.v14, ds).m23.predict(ds.X[ds.test_idx])
        err = np.array([np.abs(liq[k] - ds.T_liquidus[i])[ds.T_liquidus_mask[i]].mean()
                        if ds.T_liquidus_mask[i].any() else np.nan for k, i in enumerate(ds.test_idx)])
        order = np.argsort(np.nan_to_num(err, nan=1e9))
        pairs = [ds.pair_ids[ds.test_idx[j]] for j in order[11::23][:6]]     # spread from good to bad
    examples(a.v6, a.v13, a.v14, out, pairs)
    print(f"Saved figures in {out}/ (examples: {', '.join(pairs)})")


if __name__ == "__main__":
    main()
