"""
plot_pointwise.py — parity plots and example curves for pointwise_liquidus.py.

    python experiments/plot_pointwise.py --out results/v20

parity_liquidus.png   predicted vs measured liquidus temperature at every
                      measured point (out of fold), one panel per model
parity_by_group.png   the same for the per-point model and v18, clean vs incomplete diagrams
example_curves.png    12 systems drawn at random (seed 0): 6 incomplete, 6 others
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from mpds_models.data import load_dataset, mask_clipped_tops, system_quality

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
NAMES = {"v18": "v18 (current model)", "pointwise_absolute": "per point: features + x -> T",
         "pointwise_deviation": "per point, T relative to the Tm line"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def parity(ax, true, pred, title, lim):
    err = pred - true
    mae, rmse = np.abs(err).mean(), np.sqrt((err ** 2).mean())
    r2 = 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum()
    style(ax)
    hb = ax.hexbin(true, pred, gridsize=70, extent=(*lim, *lim), cmap=SEQ, norm=LogNorm(), mincnt=1, linewidths=0)
    ax.plot(lim, lim, color=INK, lw=1)
    for d in (-100, 100):
        ax.plot(lim, np.add(lim, d), color=MUTED, lw=0.8, ls="--")
    within = (np.abs(err) <= 100).mean()
    ax.text(0.03, 0.97, f"MAE {mae:.1f} K\nRMSE {rmse:.1f} K\nR² {r2:.3f}\nwithin ±100 K: {within:.0%}\n{len(true):,} points",
            transform=ax.transAxes, va="top", fontsize=8, color=INK, linespacing=1.4)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("measured liquidus temperature (K)", fontsize=8, color=INK2)
    return hb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = Path(a.out)
    o = np.load(out / "pointwise_oof.npz")
    ds = load_dataset("/home/alibi/lab/processed_v8", "extended+mp")
    quality = system_quality(load_dataset("/home/alibi/lab/processed_v6", "extended+mp"))
    mask_clipped_tops(ds)
    assert list(o["pair_ids"]) == ds.pair_ids
    m = ds.T_liquidus_mask
    true = ds.T_liquidus[m].astype(float)
    models = [k for k in NAMES if k in o.files]
    lim = (150.0, float(np.percentile(true, 99.9)) + 200)

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.8), facecolor="white")
    for ax, k in zip(axes, models):
        hb = parity(ax, true, o[k][m].astype(float), NAMES[k], lim)
    axes[0].set_ylabel("predicted liquidus temperature (K)", fontsize=8, color=INK2)
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("measured points per hexagon", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle("Liquidus: predicted vs measured at every measured point (5-fold cross-validation, "
                 "1,380 systems of processed_v8; dashed: ±100 K)", fontsize=10, color=INK, x=0.02, ha="left")
    fig.savefig(out / "parity_liquidus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    groups = {"clean diagrams": quality["group"].to_numpy() == "clean",
              "incomplete diagrams": quality["incomplete"].to_numpy()}
    best = min((k for k in models if k != "v18"), key=lambda k: np.abs(o[k][m] - ds.T_liquidus[m]).mean())
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 9.0), facecolor="white")
    for r, (g, rows) in enumerate(groups.items()):
        mg = m & rows[:, None]
        for c, k in enumerate(("v18", best)):
            hb = parity(axes[r, c], ds.T_liquidus[mg].astype(float), o[k][mg].astype(float),
                        f"{NAMES[k]}\n{g} ({int(rows.sum())} systems)", lim)
    fig.suptitle("Parity by data group (groups as in the v14-v18 tables)", fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out / "parity_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    inc = np.flatnonzero(groups["incomplete diagrams"] & m.any(1))
    rest = np.flatnonzero(~groups["incomplete diagrams"] & m.any(1))
    pick = np.r_[rng.choice(inc, 6, replace=False), rng.choice(rest, 6, replace=False)]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6), facecolor="white")
    x = np.arange(101)
    for ax, i in zip(axes.flat, pick):
        style(ax)
        seen = m[i]
        ax.scatter(x[seen], ds.T_liquidus[i, seen], s=9, color=MUTED, lw=0, zorder=3)
        for k, color in (("v18", BLUE), (best, ORANGE)):
            e = np.abs(o[k][i] - ds.T_liquidus[i])[seen].mean()
            ax.plot(x, o[k][i], color=color, lw=2, zorder=4, label=f"{'v18' if k == 'v18' else 'per point'}: {e:.0f} K")
        a_, b_ = ds.pair_ids[i].split("_")
        tag = "incomplete" if groups["incomplete diagrams"][i] else quality["group"].iloc[i]
        ax.set_title(f"{a_}–{b_}  ({tag})", fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"at% {b_}", fontsize=7.5, color=INK2)
        ax.legend(fontsize=7, frameon=False, loc="best", labelcolor=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("liquidus (K)", fontsize=8, color=INK2)
    fig.suptitle("Out-of-fold liquidus of 12 systems drawn at random (top row and first two of the middle: "
                 "incomplete diagrams). Grey dots: data; blue: v18; orange: per-point model; legend: MAE on the dots",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "example_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved parity_liquidus.png, parity_by_group.png, example_curves.png in {out}/ (per-point model shown: {best})")


if __name__ == "__main__":
    main()
