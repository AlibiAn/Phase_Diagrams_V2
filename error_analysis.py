"""
error_analysis.py — where do the predicted phase diagrams go wrong, and why?

Draws every predicted diagram over its true curves, worst first, and breaks
the liquidus / solidus error down so the causes show:

  gallery_NN.png            every system, worst first: liquidus in blue, solidus in
                            orange (dots = data, line = prediction, shaded = the
                            scored error), both errors in the title, each tagged
                            with its main kind of error
  error_by_composition.png  error and bias (over- or under-prediction) across
                            composition, for the liquidus and the solidus
  error_near_features.png   error by distance from the true eutectic point and
                            by how steep the true curve is
  error_by_system.png       per-system error by topology class, and how much of
                            all the error the worst systems carry
  offset_vs_shape.png       per system: is the curve shifted up/down as a whole
                            (offset) or wrongly shaped?
  data_quality.png          error by how much of the range has data, by the
                            largest gap, and by how each part of the true curve
                            was drawn (smooth, straight-line runs, flat plateau,
                            next to a gap)
  error_drivers.png         system properties that go with high error
                            (Spearman correlation with per-system error)
  systems.csv               one row per system with every number above
  summary.md                the main numbers and findings in words

Kinds of error (one per system, the first that applies)
-------------------------------------------------------
  eutectic   the error within 10 at% of the true eutectic point is more than
             twice the error elsewhere (a V that is too shallow, too deep or
             in the wrong place)
  edges      the error within 10 at% of a pure element is more than twice the
             error in the middle
  offset     the whole curve sits too high or too low: removing the mean
             signed error leaves less than half the error
  shape      none of these: the curve is wrongly shaped across the diagram

Usage
-----
    python error_analysis.py processed_v6 models_v10_trees --out error_analysis \\
        --mask-clipped --fix-end-points
    python error_analysis.py processed_v6 models_v10_trees --split val --per-page 20

Score the same points predict.py does: pass the same --mask-clipped and
--fix-end-points flags.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from mpds_models.base import DATA_GROUPS, fix_end_points, load_dataset, load_dataset_for_models, mask_clipped_tops, system_quality
from mpds_models.meta_learner import FullDiagramReconstructor
from compare_diagrams import TOPOLOGY_SHORT

KINDS = ("eutectic", "edges", "offset", "shape")
NEAR_AT_PCT = 10.0

# chart colours (reference palette, light surface)
C_MAE, C_BIAS = "#2a78d6", "#eb6834"
C_LIQ, C_SOL, C_WRONG = "#2a78d6", "#eb6834", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DERIVED_LABELS = {
    "nn_distance": "distance to the nearest training systems",
    "n_sources": "number of source diagrams",
    "coverage": "share of the composition range with data",
    "tm_max": "higher melting point",
    "tm_gap": "melting-point difference",
    "topology_wrong": "diagram type predicted wrongly",
    "eutectic_wrong": "eutectic yes/no predicted wrongly",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-system numbers
# ─────────────────────────────────────────────────────────────────────────────

def nearest_train_distance(ds, idx, k: int = 5) -> np.ndarray:
    """Mean distance from each system to its k nearest training systems, on
    features standardised with the training statistics. A large value means
    the system is unlike anything the model learned from."""
    from sklearn.neighbors import NearestNeighbors
    tr = ds.train_idx
    mu, sd = ds.X[tr].mean(0), ds.X[tr].std(0) + 1e-6
    Z = lambda rows: (ds.X[rows] - mu) / sd
    nn = NearestNeighbors(n_neighbors=k).fit(Z(tr))
    dist, _ = nn.kneighbors(Z(idx))
    return dist.mean(axis=1)


def system_table(ds, idx, diagrams) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per system, plus the signed liquidus / solidus errors (N, G),
    NaN where there is no ground truth."""
    grid = np.asarray(ds.comp_grid, dtype=np.float64)
    liq = np.stack([d.liquidus_T for d in diagrams])
    sol = np.stack([d.solidus_T for d in diagrams])
    e_liq = np.where(ds.T_liquidus_mask[idx], liq - ds.T_liquidus[idx], np.nan)
    e_sol = sol - ds.T_solidus[idx]                       # NaN where unobserved
    names = {n: j for j, n in enumerate(ds.feature_names)}
    tm = np.stack([ds.X[idx, names["phys__melting_point__A"]],
                   ds.X[idx, names["phys__melting_point__B"]]], axis=1)
    near_edge = (grid <= NEAR_AT_PCT) | (grid >= 100 - NEAR_AT_PCT)

    rows = []
    for r, (i, d) in enumerate(zip(idx, diagrams)):
        e = e_liq[r]
        seen = np.isfinite(e)
        mae = float(np.abs(e[seen]).mean()) if seen.any() else np.nan
        bias = float(e[seen].mean()) if seen.any() else np.nan
        shape = float(np.abs(e[seen] - bias).mean()) if seen.any() else np.nan
        comp = float(ds.eutectic_comp[i])
        near_eut = (np.abs(grid - comp) <= NEAR_AT_PCT) if ds.eutectic_exists[i] and np.isfinite(comp) \
            else np.zeros_like(grid, dtype=bool)

        def ratio(near):
            a, b = seen & near, seen & ~near
            if a.sum() < 3 or b.sum() < 3:
                return np.nan
            return float(np.abs(e[a]).mean() / max(np.abs(e[b]).mean(), 1e-9))

        eut_ratio, edge_ratio = ratio(near_eut), ratio(near_edge)
        if not seen.any():
            kind = "no data"
        elif np.isfinite(eut_ratio) and eut_ratio > 2:
            kind = "eutectic"
        elif np.isfinite(edge_ratio) and edge_ratio > 2:
            kind = "edges"
        elif shape < 0.5 * mae:
            kind = "offset"
        else:
            kind = "shape"
        s = e_sol[r][np.isfinite(e_sol[r])]
        rows.append(dict(
            pair=ds.pair_ids[i], topology=ds.topology_name(i), predicted_topology=d.topology,
            liquidus_MAE_K=mae, liquidus_bias_K=bias, offset_K=abs(bias), shape_K=shape,
            solidus_MAE_K=float(np.abs(s).mean()) if s.size else np.nan,
            kind=kind, eutectic_error_ratio=eut_ratio, edge_error_ratio=edge_ratio,
            n_sources=int(ds.n_sources[i]), coverage=float(ds.T_liquidus_mask[i].mean()),
            tm_max=float(tm[r].max()), tm_gap=float(abs(tm[r, 0] - tm[r, 1])),
            topology_wrong=int(d.topology != ds.topology_name(i)),
            eutectic_wrong=int(d.eutectic_exists != bool(ds.eutectic_exists[i])),
        ))
    df = pd.DataFrame(rows)
    df["nn_distance"] = nearest_train_distance(ds, idx)
    df["liquidus_MAE_pct_of_Tm"] = 100 * df["liquidus_MAE_K"] / df["tm_max"].where(df["tm_max"] > 0)
    return df, e_liq, e_sol



# ─────────────────────────────────────────────────────────────────────────────
# Data quality: coverage, gaps and how each data point was drawn
# ─────────────────────────────────────────────────────────────────────────────

POINT_CLASSES = ("smooth curve", "straight 5-15 at%", "straight 15-30 at%", "straight > 30 at%",
                 "flat plateau", "next to a gap")
BIG_GAP_AT_PCT, NEAR_GAP_AT_PCT = 5.0, 3.0


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of each run of True values."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def point_classes(ds, idx, tol_K: float = 0.05, flat_K_per_pct: float = 0.05):
    """Label every observed liquidus point by how its stretch of curve was drawn.

    Data digitised from a few points and joined by straight lines lie exactly
    on a line (second difference below `tol_K`); the longer such a straight
    run, the sparser the original data. A straight run with no slope is a flat
    plateau. Points within NEAR_GAP_AT_PCT of a stretch of at least
    BIG_GAP_AT_PCT with no data are "next to a gap". Returns the (N, G) labels
    ("" where there is no data) and per-system largest gap / straight / flat
    shares.
    """
    grid = np.asarray(ds.comp_grid, dtype=np.float64)
    step = float(np.diff(grid).mean())
    T = np.where(ds.T_liquidus_mask[idx], ds.T_liquidus[idx], np.nan).astype(np.float64)
    labels = np.full(T.shape, "", dtype=object)
    stats = []
    for r, t in enumerate(T):
        seen = np.isfinite(t)
        gaps = [(a, b) for a, b in _runs(~seen)]
        largest_gap = max(((b - a) * step for a, b in gaps), default=0.0)
        near_gap = np.zeros_like(seen)
        reach = int(round(NEAR_GAP_AT_PCT / step))
        for a, b in gaps:
            if (b - a) * step >= BIG_GAP_AT_PCT:
                near_gap[max(0, a - reach):min(len(t), b + reach)] = True
        d2 = np.full(len(t), np.nan)
        d2[1:-1] = t[:-2] - 2 * t[1:-1] + t[2:]
        straight = np.abs(np.nan_to_num(d2, nan=np.inf)) < tol_K
        run_len = np.zeros(len(t))
        run_flat = np.zeros(len(t), dtype=bool)
        for a, b in _runs(straight):
            lo, hi = a - 1, b                           # the run's own end points (stop is exclusive)
            length = (hi - lo) * step
            flat = abs(t[hi] - t[lo]) / length < flat_K_per_pct
            run_len[lo:hi + 1] = np.maximum(run_len[lo:hi + 1], length)
            run_flat[lo:hi + 1] |= flat
        lab = np.where(run_len > 30, POINT_CLASSES[3],
                       np.where(run_len > 15, POINT_CLASSES[2],
                                np.where(run_len >= 5, POINT_CLASSES[1], POINT_CLASSES[0]))).astype(object)
        lab[near_gap & seen] = POINT_CLASSES[5]
        lab[run_flat & (run_len >= 5) & seen] = POINT_CLASSES[4]
        lab[~seen] = ""
        labels[r] = lab
        n = max(int(seen.sum()), 1)
        stats.append(dict(largest_gap_pct=largest_gap,
                          straight_share=float(np.isin(lab, POINT_CLASSES[2:4]).sum() / n),
                          flat_share=float((lab == POINT_CLASSES[4]).sum() / n)))
    return labels, pd.DataFrame(stats)


def quality_tables(df, labels, e_liq):
    """Per-system error by coverage and by largest gap; per-point error by class."""
    v = df.dropna(subset=["liquidus_MAE_K"])
    total = v["liquidus_MAE_K"].sum()

    def by(col, edges, names):
        b = pd.cut(v[col], edges, labels=names, right=False)
        t = v.groupby(b, observed=False)["liquidus_MAE_K"].agg(systems="size", mean_MAE_K="mean",
                                                               median_MAE_K="median", total="sum")
        t["share_of_error_pct"] = 100 * t.pop("total") / total
        return t

    coverage = by("coverage", [0, 0.5, 0.8, 0.95, 1.01], ["< 50 %", "50-80 %", "80-95 %", "95-100 %"])
    gaps = by("largest_gap_pct", [0, 2, 10, 30, 101], ["< 2 at%", "2-10 at%", "10-30 at%", "> 30 at%"])
    err = np.abs(e_liq)
    rows = []
    for c in POINT_CLASSES:
        sel = (labels == c) & np.isfinite(err)
        rows.append(dict(point_class=c, points=int(sel.sum()),
                         MAE_K=float(err[sel].mean()) if sel.any() else np.nan,
                         error_sum=float(err[sel].sum())))
    points = pd.DataFrame(rows).set_index("point_class")
    points["share_of_points_pct"] = 100 * points["points"] / points["points"].sum()
    points["share_of_error_pct"] = 100 * points.pop("error_sum") / points["MAE_K"].mul(points["points"]).sum()
    return coverage, gaps, points


def group_table(df) -> pd.DataFrame:
    """Liquidus / solidus error per data group (base.system_quality)."""
    v = df.dropna(subset=["liquidus_MAE_K"])
    total = v["liquidus_MAE_K"].sum()
    sel = {"all": np.ones(len(v), dtype=bool), "clean": (v["group"] == "clean").to_numpy(),
           "incomplete": v["incomplete"].to_numpy(bool), "volatile element": v["volatile"].to_numpy(bool),
           "monotectic": v["monotectic"].to_numpy(bool)}
    rows = []
    for g in DATA_GROUPS:
        t = v[sel[g]]
        rows.append(dict(group=g, systems=len(t), liquidus_mean_K=t["liquidus_MAE_K"].mean(),
                         liquidus_median_K=t["liquidus_MAE_K"].median(),
                         solidus_mean_K=t["solidus_MAE_K"].mean(),
                         share_of_error_pct=100 * t["liquidus_MAE_K"].sum() / total))
    return pd.DataFrame(rows).set_index("group")


def plot_data_quality(coverage, gaps, points, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    panels = (
        (axes[0], coverage, "mean_MAE_K", "systems", "Per system: share of the range with data"),
        (axes[1], gaps, "mean_MAE_K", "systems", "Per system: largest stretch with no data"),
        (axes[2], points, "MAE_K", "points", "Per point: how that part of the curve was drawn"),
    )
    for ax, t, col, count, title in panels:
        vals = t[col].to_numpy(dtype=float)
        bars = ax.bar(range(len(t)), np.nan_to_num(vals), color=C_MAE, width=0.62)
        for b, v, n, sh in zip(bars, vals, t[count], t["share_of_error_pct"]):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} K\n{n:,} {count}\n{sh:.0f} % of error",
                        ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(t)), [str(i).replace(" at%", "\nat%") for i in t.index], fontsize=7.5)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax, "y")
    axes[0].set_ylabel("liquidus MAE (K)", fontsize=9, color=MUTED)
    top = np.nanmax([np.nanmax(t[c].to_numpy(dtype=float)) for _, t, c, _, _ in panels])
    axes[0].set_ylim(0, top * 1.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_by_composition(grid, e_liq, e_sol, out):
    bins = np.arange(0, 101, 5)
    mid = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, e, name in zip(axes, (e_liq, e_sol), ("Liquidus", "Solidus")):
        mae, bias = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            cols = (grid >= lo) & ((grid < hi) | (hi == 100))
            v = e[:, cols][np.isfinite(e[:, cols])]
            mae.append(np.abs(v).mean() if v.size else np.nan)
            bias.append(v.mean() if v.size else np.nan)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot(mid, mae, color=C_MAE, lw=2, marker="o", ms=4, label="mean absolute error")
        ax.plot(mid, bias, color=C_BIAS, lw=2, marker="o", ms=4,
                label="mean signed error (> 0: predicted too high)")
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("composition (at% B), 5 at% bins", fontsize=9, color=MUTED)
        ax.set_xlim(0, 100)
        _style(ax)
    axes[0].set_ylabel("K", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _binned(values, errors, edges, labels):
    out = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (values >= lo) & (values < hi) & np.isfinite(errors)
        out.append((lab, float(np.abs(errors[sel]).mean()) if sel.any() else np.nan, int(sel.sum())))
    return out


def near_feature_tables(ds, idx, e_liq):
    grid = np.asarray(ds.comp_grid, dtype=np.float64)
    comp = ds.eutectic_comp[idx].astype(np.float64)
    has = ds.eutectic_exists[idx] & np.isfinite(comp)
    dist = np.abs(grid[None, :] - comp[:, None])
    eut = _binned(dist[has].ravel(), e_liq[has].ravel(), [0, 2, 5, 10, 20, 40, 101],
                  ["0-2", "2-5", "5-10", "10-20", "20-40", "> 40"])
    # Steepness of the true liquidus: K per at% between the two neighbours
    T = np.where(ds.T_liquidus_mask[idx], ds.T_liquidus[idx], np.nan).astype(np.float64)
    slope = np.full(T.shape, np.nan)
    step = np.diff(grid).mean()
    slope[:, 1:-1] = np.abs(T[:, 2:] - T[:, :-2]) / (2 * step)
    steep = _binned(slope.ravel(), e_liq.ravel(), [0, 5, 10, 20, 40, 80, np.inf],
                    ["0-5", "5-10", "10-20", "20-40", "40-80", "> 80"])
    return eut, steep


def plot_near_features(eut, steep, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, rows, title, xlabel in (
        (axes[0], eut, "Liquidus error by distance from the true eutectic (eutectic systems)",
         "distance from the eutectic composition (at%)"),
        (axes[1], steep, "Liquidus error by steepness of the true curve",
         "slope of the true liquidus (K per at%)"),
    ):
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=C_MAE, width=0.62)
        for b, (_, v, n) in zip(bars, rows):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}\n{n:,} pts",
                        ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax, "y")
    axes[0].set_ylabel("liquidus MAE (K)", fontsize=9, color=MUTED)
    top = np.nanmax([r[1] for r in eut + steep])
    axes[0].set_ylim(0, top * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_system(df, classes, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    present = [c for c in classes if (df["topology"] == c).any()]
    for k, cls in enumerate(present):
        v = df.loc[df["topology"] == cls, "liquidus_MAE_K"].dropna().to_numpy()
        y = len(present) - 1 - k
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=C_MAE, alpha=0.6, lw=0)
        ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=2, zorder=3)
        ax.text(np.median(v), y + 0.34, f" median {np.median(v):.0f} K, n={len(v)}",
                fontsize=7, color=MUTED, va="bottom", ha="left")
    ax.set_yticks(range(len(present))[::-1], [c.replace("_", " ") for c in present], fontsize=8, color=INK)
    ax.set_xlabel("liquidus MAE per system (K); bar = median", fontsize=9, color=MUTED)
    ax.set_title("Per-system error by true diagram type", fontsize=10, color=INK, loc="left")
    _style(ax, "x")

    ax = axes[1]
    err = np.sort(df["liquidus_MAE_K"].dropna().to_numpy())[::-1]
    share_sys = np.arange(1, len(err) + 1) / len(err) * 100
    share_err = np.cumsum(err) / err.sum() * 100
    ax.plot(share_sys, share_err, color=C_MAE, lw=2)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
    for p in (10, 25):
        j = int(np.ceil(p / 100 * len(err))) - 1
        ax.scatter([share_sys[j]], [share_err[j]], color=C_MAE, s=30, zorder=3)
        ax.text(share_sys[j] + 2, share_err[j] - 4, f"worst {p} % of systems: {share_err[j]:.0f} % of the error",
                fontsize=8, color=INK, va="top")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("share of systems, worst first (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("share of all liquidus error (%)", fontsize=9, color=MUTED)
    ax.set_title("How concentrated the error is", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_offset_vs_shape(df, out):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    d = df.dropna(subset=["offset_K", "shape_K"])
    lim = max(d["offset_K"].max(), d["shape_K"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    ax.scatter(d["shape_K"], d["offset_K"], s=16, color=C_MAE, alpha=0.7, lw=0)
    n_off = int((d["offset_K"] > d["shape_K"]).sum())
    ax.text(0.03, 0.97, f"above the line: mostly shifted ({n_off} of {len(d)})",
            transform=ax.transAxes, fontsize=8, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.text(0.97, 0.03, "below: mostly wrongly shaped", transform=ax.transAxes,
            fontsize=8, color=INK, ha="right")
    for _, r in d.nlargest(6, "liquidus_MAE_K").iterrows():
        ax.annotate(r["pair"].replace("_", "–"), (r["shape_K"], r["offset_K"]), fontsize=7,
                    color=MUTED, xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("shape error: MAE after removing the shift (K)", fontsize=9, color=MUTED)
    ax.set_ylabel("offset: |mean signed error| (K)", fontsize=9, color=MUTED)
    ax.set_title("Is each curve shifted or misshapen?", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def driver_table(ds, idx, df, top: int = 15) -> pd.DataFrame:
    y = df["liquidus_MAE_K"].to_numpy()
    ok = np.isfinite(y)
    rows = []
    cand = {f"feature: {n}": ds.X[idx, j] for j, n in enumerate(ds.feature_names)}
    cand.update({DERIVED_LABELS[c]: df[c].to_numpy(dtype=float) for c in DERIVED_LABELS})
    for name, x in cand.items():
        x = np.asarray(x, dtype=float)
        if np.nanstd(x[ok]) == 0:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        if np.isfinite(rho):
            rows.append(dict(property=name, spearman=float(rho), p_value=float(p)))
    t = pd.DataFrame(rows)
    return t.reindex(t["spearman"].abs().sort_values(ascending=False).index).head(top)


def plot_drivers(t, out):
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(t) + 1.4))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["spearman"], color=[C_BIAS if v > 0 else C_MAE for v in t["spearman"]], height=0.62)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [p.replace("feature: ", "") for p in t["property"]], fontsize=8, color=INK)
    ax.set_xlabel("Spearman correlation with per-system liquidus MAE "
                  "(> 0: more of it, more error)", fontsize=9, color=MUTED)
    ax.set_title("What goes with high error", fontsize=10, color=INK, loc="left")
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_panel(ax, d, ds, i: int, row, rank: int) -> None:
    """One system: each curve in one colour (dots = data, line = prediction),
    the scored gap between them shaded, both errors in the title."""
    grid = np.asarray(d.comp_grid, dtype=np.float64)
    liq_seen = ds.T_liquidus_mask[i]
    sol_true = ds.T_solidus[i]
    sol_seen = np.isfinite(sol_true)
    liq_true = np.where(liq_seen, ds.T_liquidus[i], np.nan)
    for pred, true, seen, color in ((d.liquidus_T, liq_true, liq_seen, C_LIQ),
                                    (d.solidus_T, sol_true, sol_seen, C_SOL)):
        if seen.sum() >= 2:
            # Shade the scored gap, bridging single missing grid points so it reads as a band
            band = seen | (np.r_[seen[1:], False] & np.r_[False, seen[:-1]])
            filled = np.interp(grid, grid[seen], true[seen])
            ax.fill_between(grid, pred, filled, where=band, color=color, alpha=0.16, lw=0)
        ax.plot(grid, pred, color=color, lw=1.7, zorder=3)
        ax.scatter(grid[seen], true[seen], s=5, color=color, lw=0, zorder=4)
    if ds.eutectic_exists[i] and np.isfinite(ds.eutectic_comp[i]) and np.isfinite(ds.eutectic_temp[i]):
        ax.scatter([ds.eutectic_comp[i]], [ds.eutectic_temp[i]], marker="^", s=34, color=INK, zorder=5)
    if d.eutectic_exists and d.eutectic_comp is not None and d.eutectic_temp is not None:
        ax.scatter([d.eutectic_comp], [d.eutectic_temp], marker="v", s=34, facecolor="white",
                   edgecolor=INK, lw=1.1, zorder=5)

    a, b = d.pair.split("_")[:2]
    sol = f"{row['solidus_MAE_K']:.0f} K" if np.isfinite(row["solidus_MAE_K"]) else "no data"
    ax.set_title(f"{a}–{b}\nliquidus {row['liquidus_MAE_K']:.0f} K · solidus {sol}",
                 fontsize=7.5, color=INK, loc="left", pad=3)
    true_t, pred_t = ds.topology_name(i), d.topology
    short = lambda t: TOPOLOGY_SHORT.get(t, t)
    right = true_t == pred_t
    ax.text(0.98, 0.97, f"✓ {short(true_t)}" if right else f"✗ predicted {short(pred_t)}\n   true {short(true_t)}",
            transform=ax.transAxes, fontsize=6, ha="right", va="top", color=MUTED if right else C_WRONG,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.text(0.02, 0.03, f"#{rank} {row['kind']}", transform=ax.transAxes, fontsize=6, color=INK,
            va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=GRID, alpha=0.9))
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"at% {b}", fontsize=6, color=MUTED, labelpad=1)
    ax.tick_params(colors=MUTED, labelsize=6, pad=1)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)


def save_gallery(diagrams, ds, idx, df, out_dir: Path, per_page: int, cols: int = 6) -> list[Path]:
    order = df["liquidus_MAE_K"].fillna(-1).sort_values(ascending=False).index.to_list()
    rank = {r: k + 1 for k, r in enumerate(order)}
    paths = []
    legend = [Line2D([0], [0], color=C_LIQ, lw=2, label="liquidus: predicted (line)"),
              Line2D([0], [0], marker="o", color=C_LIQ, ls="None", ms=4, label="liquidus: data (dots)"),
              Line2D([0], [0], color=C_SOL, lw=2, label="solidus: predicted (line)"),
              Line2D([0], [0], marker="o", color=C_SOL, ls="None", ms=4, label="solidus: data (dots)"),
              Patch(facecolor=C_LIQ, alpha=0.25, label="shaded: the error that is scored"),
              Line2D([0], [0], marker="^", color=INK, ls="None", ms=6, label="true eutectic"),
              Line2D([0], [0], marker="v", markerfacecolor="white", markeredgecolor=INK, ls="None", ms=6,
                     label="predicted eutectic")]
    for page, start in enumerate(range(0, len(order), per_page), 1):
        chunk = order[start:start + per_page]
        rows = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.75 + 1.1), squeeze=False)
        for ax, r in zip(axes.flat, chunk):
            draw_panel(ax, diagrams[r], ds, idx[r], df.loc[r], rank[r])
        for ax in list(axes.flat)[len(chunk):]:
            ax.set_visible(False)
        fig.suptitle(f"Systems ranked by liquidus error, worst first: {start + 1}-{start + len(chunk)} of "
                     f"{len(order)}.  Each panel has its own temperature scale (K): compare the error "
                     f"numbers, not how far apart the curves look. Errors count only where there are "
                     f"data dots.", fontsize=9, y=0.995)
        fig.legend(handles=legend, loc="lower center", ncol=7, fontsize=7.5, frameon=False)
        fig.tight_layout(rect=(0, 0.03, 1, 0.975))
        path = out_dir / f"gallery_{page:02d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Where and why the predicted diagrams go wrong.")
    ap.add_argument("processed_dir")
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="error_analysis")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--per-page", type=int, default=24, help="diagrams per gallery page")
    ap.add_argument("--mask-clipped", action="store_true",
                    help="leave liquidus tops cut off by the source chart out (as predict.py)")
    ap.add_argument("--fix-end-points", action="store_true",
                    help="leave end points that disagree with Tm out (as predict.py)")
    ap.add_argument("--uncertainty", action="store_true", help="draw the uncertainty bands (slower)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset_for_models(a.processed_dir, a.model_dir)
    quality = system_quality(ds)               # before any masking
    if a.mask_clipped:
        mask_clipped_tops(ds)
    if a.fix_end_points:
        fix_end_points(ds, inject=False)
    idx = ds.split(a.split)
    recon = FullDiagramReconstructor.from_dir(a.model_dir, ds)
    diagrams = recon.reconstruct(ds.X[idx], [ds.pair_ids[i] for i in idx],
                                 with_uncertainty=a.uncertainty)

    df, e_liq, e_sol = system_table(ds, idx, diagrams)
    grid = np.asarray(ds.comp_grid, dtype=np.float64)

    plot_by_composition(grid, e_liq, e_sol, out / "error_by_composition.png")
    eut, steep = near_feature_tables(ds, idx, e_liq)
    plot_near_features(eut, steep, out / "error_near_features.png")
    plot_by_system(df, ds.topology_classes, out / "error_by_system.png")
    plot_offset_vs_shape(df, out / "offset_vs_shape.png")
    labels, qstats = point_classes(ds, idx)
    groups = quality.iloc[idx].reset_index(drop=True)[["group", "incomplete", "volatile", "volatile_elements",
                                                       "monotectic", "above_boiling_share"]]
    df = pd.concat([df, qstats, groups], axis=1)
    by_group = group_table(df)
    df.to_csv(out / "systems.csv", index=False)
    q_cov, q_gap, q_pts = quality_tables(df, labels, e_liq)
    plot_data_quality(q_cov, q_gap, q_pts, out / "data_quality.png")
    drivers = driver_table(ds, idx, df)
    drivers.to_csv(out / "error_drivers.csv", index=False)
    plot_drivers(drivers, out / "error_drivers.png")
    pages = save_gallery(diagrams, ds, idx, df, out, a.per_page)

    # ---- numbers in words -------------------------------------------------
    v = df.dropna(subset=["liquidus_MAE_K"])
    err = np.sort(v["liquidus_MAE_K"].to_numpy())[::-1]
    worst10 = err[: max(1, int(np.ceil(0.1 * len(err))))].sum() / err.sum()
    all_e = e_liq[np.isfinite(e_liq)]
    kinds = (v.groupby("kind")["liquidus_MAE_K"].agg(systems="size", mean_MAE_K="mean", total="sum")
             .reindex([k for k in KINDS if k in set(v["kind"])]))
    kinds["share_of_error_pct"] = 100 * kinds.pop("total") / v["liquidus_MAE_K"].sum()
    by_topo = (v.groupby("topology")[["liquidus_MAE_K", "liquidus_bias_K", "solidus_MAE_K"]]
               .agg(["median", "mean"]).round(0))
    q = v["nn_distance"].quantile([0.25, 0.75])
    near, far = v[v["nn_distance"] <= q.iloc[0]], v[v["nn_distance"] >= q.iloc[1]]
    elems = (pd.DataFrame([(e, r["liquidus_MAE_K"]) for _, r in v.iterrows()
                           for e in r["pair"].split("_")], columns=["element", "MAE"])
             .groupby("element")["MAE"].agg(systems="size", mean_MAE_K="mean"))
    elems = elems[elems["systems"] >= 3].sort_values("mean_MAE_K", ascending=False)

    lines = [
        f"# Error analysis: {a.model_dir}, {a.split} split ({len(idx)} systems)", "",
        f"- Liquidus MAE {np.abs(all_e).mean():.1f} K over {all_e.size:,} points; mean signed error "
        f"{all_e.mean():+.1f} K ({'predicted too high' if all_e.mean() > 0 else 'predicted too low'} on average).",
        f"- The worst 10 % of systems carry {100 * worst10:.0f} % of all liquidus error; "
        f"median system {np.median(err):.0f} K, worst {err[0]:.0f} K ({v.loc[v['liquidus_MAE_K'].idxmax(), 'pair']}).",
        f"- Systems least like the training data (top quarter by distance) have liquidus MAE "
        f"{far['liquidus_MAE_K'].mean():.0f} K, the most similar quarter {near['liquidus_MAE_K'].mean():.0f} K.",
        f"- Relative to the higher melting point the error is {v['liquidus_MAE_pct_of_Tm'].median():.1f} % "
        f"(median); its correlation with that melting point is "
        f"{spearmanr(v['tm_max'], v['liquidus_MAE_pct_of_Tm'], nan_policy='omit')[0]:+.2f}, against "
        f"{spearmanr(v['tm_max'], v['liquidus_MAE_K'])[0]:+.2f} in kelvin "
        f"(near 0 = high-melting systems are only hard because their temperatures are bigger).",
        f"- Clean systems (complete diagram, no volatile element, not monotectic; "
        f"{int(by_group.loc['clean', 'systems'])} of {len(v)}): liquidus MAE "
        f"{by_group.loc['clean', 'liquidus_mean_K']:.0f} K mean, {by_group.loc['clean', 'liquidus_median_K']:.0f} K "
        f"median. Volatile-element systems ({int(by_group.loc['volatile element', 'systems'])}): "
        f"{by_group.loc['volatile element', 'liquidus_mean_K']:.0f} K mean.",
        f"- Diagram type wrong: MAE {v.loc[v['topology_wrong'] == 1, 'liquidus_MAE_K'].mean():.0f} K "
        f"vs {v.loc[v['topology_wrong'] == 0, 'liquidus_MAE_K'].mean():.0f} K when right.",
        "", "## By data group (groups overlap; clean = none of the others)",
        "incomplete = under half the composition range with liquidus data, or a gap of 10 at% or "
        "more; volatile element = an element boiling below 1300 K",
        by_group.round(1).to_string(),
        "", "## Data quality: coverage, gaps and straight-line data",
        "Per system, by the share of the composition range with liquidus data:",
        q_cov.round(1).to_string(),
        "", "Per system, by the largest stretch of composition with no liquidus data:",
        q_gap.round(1).to_string(),
        "", "Per data point, by how that part of the curve was drawn (straight = the source points "
        "were joined by a straight line over that many at%; such stretches are interpolated, not measured):",
        q_pts.round(1).to_string(),
        "", "## Main kind of error per system", kinds.round(1).to_string(),
        "", "## By true diagram type (K)", by_topo.to_string(),
        "", "## Error near the eutectic (eutectic systems)",
        "\n".join(f"- {lab} at% from it: {m:.0f} K ({n:,} points)" for lab, m, n in eut),
        "", "## Error by steepness of the true liquidus",
        "\n".join(f"- {lab} K/at%: {m:.0f} K ({n:,} points)" for lab, m, n in steep),
        "", "## Elements in the worst systems (at least 3 systems)",
        elems.head(12).round(0).to_string(),
        "", "## Properties that go with high error (Spearman)",
        drivers.round(3).to_string(index=False),
        "", "## Worst 15 systems",
        v.nlargest(15, "liquidus_MAE_K")[["pair", "topology", "predicted_topology", "liquidus_MAE_K",
                                           "liquidus_bias_K", "kind", "nn_distance"]].round(1)
        .to_string(index=False),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    print(f"Saved {len(pages)} gallery pages, 6 charts, systems.csv and summary.md in {out}/")


if __name__ == "__main__":
    main()
