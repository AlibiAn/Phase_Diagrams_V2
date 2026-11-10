"""
plot_results.py — charts of how the models improve from version to version,
drawn from the JSON reports predict.py writes.

Pass the reports oldest first, scored the same way (same split and the same
--mask-clipped / --fix-end-points flags), so the numbers are comparable.

Charts (in --out)
-----------------
  progress.png        liquidus and solidus error per version
  data_groups.png     liquidus error per data group (clean, incomplete,
                      volatile element, monotectic) per version
  topology_f1.png     F1 per diagram type per version
  per_system.png      every system's liquidus error, first vs last version
  distribution.png    share of systems below each liquidus error, per version
  summary.csv         the numbers behind the first three charts

Usage
-----
    python plot_results.py results/v11/report_v10_trees.json results/v12/report_v11.json \\
        results/v13/report_v13.json --names v10 v11 v13 --out results/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (light surface). Versions are ordered, so they take one hue,
# light to dark (ordinal steps 250-650); the two curves take categorical slots 1-2.
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
C_LIQ, C_SOL = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GROUPS = ("all", "clean", "incomplete", "volatile element", "monotectic")
SHORT = {"isomorphous": "isomorphous", "simple_eutectic": "simple\neutectic",
         "eutectic_with_intermetallics": "eutectic +\ncompounds", "monotectic": "monotectic",
         "complex": "complex"}


def ramp(n: int) -> list[str]:
    """n colours from the ordinal ramp, light (oldest) to dark (newest)."""
    if n == 1:
        return [RAMP[2]]
    return [RAMP[int(round(k))] for k in np.linspace(0, len(RAMP) - 1, n)]


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h) and h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom",
                    fontsize=7, color=INK)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_progress(reports, names, out):
    liq = [r["evaluation"]["curves"]["liquidus_MAE_K"] for r in reports]
    sol = [r["evaluation"]["curves"]["solidus_MAE_K"] for r in reports]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(names) + 2), 3.8))
    bar_labels(ax, ax.bar(x - w / 2 - 0.01, liq, w, color=C_LIQ, label="liquidus"))
    bar_labels(ax, ax.bar(x + w / 2 + 0.01, sol, w, color=C_SOL, label="solidus"))
    ax.set_xticks(x, names, fontsize=9, color=INK)
    ax.set_ylabel("mean absolute error (K)", fontsize=9, color=MUTED)
    first, last = liq[0], liq[-1]
    ax.set_title(f"Curve error per version: liquidus {first:.0f} → {last:.0f} K "
                 f"({100 * (last - first) / first:+.0f} %)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.set_ylim(0, max(max(liq), max(sol)) * 1.18)
    style(ax)
    save(fig, out)


def grouped_bars(ax, table: pd.DataFrame, colors, fmt="{:.0f}"):
    """Columns of `table` = versions, rows = categories."""
    n = table.shape[1]
    w = 0.8 / n
    x = np.arange(len(table))
    for k, (name, col) in enumerate(table.items()):
        bars = ax.bar(x - 0.4 + w * (k + 0.5), col.to_numpy(dtype=float), w * 0.92,
                      color=colors[k], label=name)
        if k == n - 1:                      # label the newest version only
            bar_labels(ax, bars, fmt)
    return x


def plot_groups(reports, names, out):
    rows = {}
    for r, name in zip(reports, names):
        g = r["evaluation"].get("curves_by_group") or {}
        rows[name] = {k: (g[k]["liquidus_MAE_K"] if k in g else np.nan) for k in GROUPS}
    table = pd.DataFrame(rows).reindex(list(GROUPS))
    missing = [c for c in table if table[c].isna().all()]
    shown = table.drop(columns=missing)
    if shown.empty:
        print("No report has curves_by_group (predict.py before the data groups); skipping data_groups.png")
        return table
    if missing:
        print(f"data_groups.png leaves out {', '.join(missing)}: report made before the data groups")
    n_sys = next((r["evaluation"]["curves_by_group"] for r in reversed(reports)
                  if r["evaluation"].get("curves_by_group")), {})
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [c for c, n in zip(ramp(len(names)), names) if n not in missing]
    x = grouped_bars(ax, shown, colors)
    ax.set_xticks(x, [f"{g}\n({n_sys[g]['n_systems']} systems)" if g in n_sys else g for g in shown.index],
                  fontsize=8, color=INK)
    ax.set_ylabel("liquidus MAE (K)", fontsize=9, color=MUTED)
    ax.set_title("Liquidus error by data group (groups overlap; clean = none of the others)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=len(shown.columns), loc="upper left")
    ax.set_ylim(0, np.nanmax(shown.to_numpy()) * 1.18)
    style(ax)
    save(fig, out)
    return table


def plot_topology(reports, names, out):
    table = pd.DataFrame({name: {c: v["f1"] for c, v in r["evaluation"]["topology"]["per_class"].items()}
                          for r, name in zip(reports, names)})
    support = reports[-1]["evaluation"]["topology"]["per_class"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = grouped_bars(ax, table, ramp(len(names)), fmt="{:.2f}")
    ax.set_xticks(x, [f"{SHORT.get(c, c)}\n(n={support[c]['support']})" for c in table.index],
                  fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (1 = every system of this type found, no false alarms)", fontsize=8.5, color=MUTED)
    acc = [r["evaluation"]["topology"]["accuracy"] for r in reports]
    ax.set_title("Diagram type: F1 per class   (accuracy " +
                 ", ".join(f"{n} {100 * a:.0f} %" for n, a in zip(names, acc)) + ")",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper right")
    style(ax)
    save(fig, out)
    return table


def per_system(report) -> pd.Series:
    return pd.Series({p["pair"]: p["liquidus_MAE_K"] for p in report["predictions"]}, dtype=float)


def plot_per_system(reports, names, out, n_labels: int = 4):
    a, b = per_system(reports[0]), per_system(reports[-1])
    common = a.index.intersection(b.index)
    a, b = a[common].dropna(), b[common].dropna()
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    better = b < a
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    lo, hi = max(1.0, min(a.min(), b.min()) * 0.8), max(a.max(), b.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax.scatter(a[better], b[better], s=16, color=C_LIQ, lw=0, alpha=0.8,
               label=f"better in {names[-1]} ({int(better.sum())})")
    ax.scatter(a[~better], b[~better], s=16, color=C_SOL, lw=0, alpha=0.8,
               label=f"worse or same ({int((~better).sum())})")
    change = (b - a).sort_values()
    # Label the biggest gains and losses, alternating sides so neighbours do not collide
    picks = list(change.index[:n_labels // 2]) + list(change.index[-(n_labels // 2):])
    for k, pair in enumerate(picks):
        ax.annotate(f"{pair.replace('_', '–')} ({change[pair]:+.0f} K)", (a[pair], b[pair]),
                    fontsize=7, color=MUTED, textcoords="offset points",
                    xytext=(6, -9) if k % 2 else (-6, 5), ha="left" if k % 2 else "right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"liquidus MAE in {names[0]} (K, log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel(f"liquidus MAE in {names[-1]} (K, log scale)", fontsize=9, color=MUTED)
    ax.set_title(f"Every system, {names[0]} vs {names[-1]}: below the line = improved",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax, "both")
    save(fig, out)


def plot_distribution(reports, names, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, r, name in zip(ramp(len(names)), reports, names):
        v = np.sort(per_system(r).dropna().to_numpy())
        share = np.arange(1, len(v) + 1) / len(v) * 100
        ax.step(v, share, where="post", color=color, lw=2,
                label=f"{name}: median {np.median(v):.0f} K")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("per-system liquidus MAE (K, log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("systems with at most this error (%)", fontsize=9, color=MUTED)
    ax.set_title("How the per-system error is spread (further left = better)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "both")
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Charts of model results across versions.")
    ap.add_argument("reports", nargs="+", help="predict.py JSON reports, oldest first")
    ap.add_argument("--names", nargs="+", help="a label per report (default: file names)")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()

    names = a.names or [Path(r).stem.replace("report_", "") for r in a.reports]
    if len(names) != len(a.reports):
        raise SystemExit("give one --names label per report")
    reports = [json.loads(Path(r).read_text()) for r in a.reports]
    for r, path in zip(reports, a.reports):
        if not r.get("evaluation"):
            raise SystemExit(f"{path} has no evaluation (was predict.py run with --no-eval?)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    plot_progress(reports, names, out / "progress.png")
    groups = plot_groups(reports, names, out / "data_groups.png")
    topo = plot_topology(reports, names, out / "topology_f1.png")
    if len(reports) >= 2:
        plot_per_system(reports, names, out / "per_system.png")
    plot_distribution(reports, names, out / "distribution.png")

    curves = pd.DataFrame({n: {"liquidus_MAE_K": r["evaluation"]["curves"]["liquidus_MAE_K"],
                               "solidus_MAE_K": r["evaluation"]["curves"]["solidus_MAE_K"],
                               "topology_accuracy": r["evaluation"]["topology"]["accuracy"],
                               "topology_macro_F1": r["evaluation"]["topology"]["macro_f1"]}
                           for r, n in zip(reports, names)})
    summary = pd.concat({"overall": curves, "liquidus MAE by group": groups,
                         "topology F1": topo})
    summary.to_csv(out / "summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
