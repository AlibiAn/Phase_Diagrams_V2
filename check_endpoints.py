"""
check_endpoints.py — audit the pure-element end points of the phase diagrams.

At x = 0 and x = 100 at% the liquidus and solidus must both equal the
melting point of the pure element. The curve models lean on this (the
CurveNet endpoint penalty, the hard / deviation modes), so this script checks
it three ways:

1. Features vs reference: phys__melting_point__A/B against standard elemental
   melting points (CRC Handbook values, built in below), using the element
   symbols in the pair id ("Ag_Al" -> A = Ag, B = Al). Catches missing
   values, wrong elements, A/B swapped and °C stored as K.
2. Data vs features: the observed liquidus / solidus at each end against
   Tm(A) / Tm(B). Catches curves in °C, curves stored the wrong way round
   (x = 0 is B), solidus ends detached from the liquidus, and steep jumps
   at the first / last grid point (mis-digitised end points).
2b. Clipped tops: liquidus curves that run flat at their maximum, above both
   melting points, are the source chart's top edge rather than data
   (train_all / predict.py --mask-clipped leave those points out).
3. Models vs both (optional, --curve-model): how far each trained curve
   model's ends are from the melting points and from the data, how large the
   end jump is, and its liquidus error in 5 at% bins (are the ends worse than
   the interior?). Accepts CurveNet .pt files and TreeCurveModel .joblib files.

Usage
-----
    python check_endpoints.py processed_v6 --out endpoint_audit
    python check_endpoints.py processed_v6 --out endpoint_audit \\
        --curve-model models_v8/m23_curves.pt models_trees/tree_curves.joblib

Outputs (in --out)
------------------
  features_vs_reference.csv  one row per system and end with a reference mismatch
  data_endpoints.csv         one row per system and end: values, differences, flags
  clipped_tops.csv           systems whose liquidus is flat at the source chart's top edge
  model_endpoints.csv        per model: end-point errors (with --curve-model)
  model_error_by_bin.csv     per model: liquidus MAE per 5 at% bin
  endpoints_scatter.png      data end points against Tm, coloured by flag
  model_error_by_bin.png     liquidus MAE across composition, one line per model
  summary.json               all counts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpds_models.base import clipped_top_mask, load_dataset

C_TO_K = 273.15

# Elemental melting points in K (CRC Handbook of Chemistry and Physics).
# Elements in UNCERTAIN sublime, have several allotropes or are poorly
# measured, so a mismatch there is reported but may be the reference's fault.
REFERENCE_TM_K = {
    "H": 13.99, "Li": 453.65, "Be": 1560, "B": 2350, "C": 3823, "N": 63.15,
    "O": 54.36, "F": 53.53, "Ne": 24.56, "Na": 370.94, "Mg": 923, "Al": 933.47,
    "Si": 1687, "P": 317.3, "S": 388.36, "Cl": 171.6, "Ar": 83.81, "K": 336.7,
    "Ca": 1115, "Sc": 1814, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519,
    "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1357.77, "Zn": 692.68, "Ga": 302.91,
    "Ge": 1211.4, "As": 1090, "Se": 494, "Br": 265.8, "Kr": 115.78, "Rb": 312.45,
    "Sr": 1050, "Y": 1799, "Zr": 2128, "Nb": 2750, "Mo": 2896, "Tc": 2430,
    "Ru": 2607, "Rh": 2237, "Pd": 1828.05, "Ag": 1234.93, "Cd": 594.22, "In": 429.75,
    "Sn": 505.08, "Sb": 903.78, "Te": 722.66, "I": 386.85, "Xe": 161.4, "Cs": 301.7,
    "Ba": 1000, "La": 1193, "Ce": 1068, "Pr": 1208, "Nd": 1297, "Pm": 1315,
    "Sm": 1345, "Eu": 1099, "Gd": 1585, "Tb": 1629, "Dy": 1680, "Ho": 1734,
    "Er": 1802, "Tm": 1818, "Yb": 1097, "Lu": 1925, "Hf": 2506, "Ta": 3290,
    "W": 3695, "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041.4, "Au": 1337.33,
    "Hg": 234.32, "Tl": 577, "Pb": 600.61, "Bi": 544.55, "Po": 527, "Rn": 202,
    "Ra": 973, "Ac": 1323, "Th": 2023, "Pa": 1841, "U": 1405.3, "Np": 917,
    "Pu": 912.5, "Am": 1449, "Cm": 1613,
}
UNCERTAIN = {"B", "C", "P", "S", "As", "Se", "Tc", "Pm", "Po", "Pa", "Np", "Am", "Cm"}

# Data flags, most specific first (a row gets the first that applies)
FLAG_ORDER = ("missing_Tm", "celsius", "swapped", "mismatch", "ok")

# chart colours (reference palette, light surface); fixed order, never cycled
C_OK, C_UNIT, C_OTHER = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def melting_point_columns(ds) -> tuple[int, int]:
    names = list(ds.feature_names)
    return names.index("phys__melting_point__A"), names.index("phys__melting_point__B")


def split_names(ds) -> np.ndarray:
    out = np.full(ds.n_systems, "", dtype=object)
    for name in ("train", "val", "test"):
        out[ds.split(name)] = name
    return out


def elements(pair: str) -> tuple[str | None, str | None]:
    parts = pair.split("_")
    if len(parts) == 2 and all(p in REFERENCE_TM_K for p in parts):
        return parts[0], parts[1]
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Features vs reference
# ─────────────────────────────────────────────────────────────────────────────

def check_features(ds, tol: float) -> tuple[pd.DataFrame, dict]:
    ia, ib = melting_point_columns(ds)
    rows, n_checked, n_unknown = [], 0, 0
    for i, pair in enumerate(ds.pair_ids):
        ea, eb = elements(pair)
        if ea is None:
            n_unknown += 1
            continue
        n_checked += 1
        fa, fb = float(ds.X[i, ia]), float(ds.X[i, ib])
        ra, rb = REFERENCE_TM_K[ea], REFERENCE_TM_K[eb]
        swapped = abs(fa - rb) <= tol and abs(fb - ra) <= tol and abs(ra - rb) > tol
        for end, el, f, r in (("A", ea, fa, ra), ("B", eb, fb, rb)):
            if f <= 0:
                flag = "missing"
            elif abs(f - r) <= tol:
                continue
            elif swapped:
                flag = "A_B_swapped"
            elif abs(f + C_TO_K - r) <= tol:
                flag = "celsius"
            else:
                flag = "mismatch"
            rows.append(dict(pair=pair, end=end, element=el, Tm_feature=f, Tm_reference=r,
                             difference=f - r, flag=flag, reference_uncertain=el in UNCERTAIN))
    df = pd.DataFrame(rows, columns=["pair", "end", "element", "Tm_feature", "Tm_reference",
                                     "difference", "flag", "reference_uncertain"])
    summary = {"systems_checked": n_checked, "systems_without_element_symbols": n_unknown,
               "ends_flagged": df["flag"].value_counts().to_dict()}
    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data end points vs the melting-point features
# ─────────────────────────────────────────────────────────────────────────────

def check_data(ds, tol: float, jump_K: float) -> tuple[pd.DataFrame, dict]:
    ia, ib = melting_point_columns(ds)
    L, S, Lm = ds.T_liquidus, ds.T_solidus, ds.T_liquidus_mask
    splits = split_names(ds)
    rows = []
    for i, pair in enumerate(ds.pair_ids):
        tm = (float(ds.X[i, ia]), float(ds.X[i, ib]))
        ends = {"A": 0, "B": -1}
        liq = {e: float(L[i, g]) if Lm[i, g] else np.nan for e, g in ends.items()}
        sol = {e: float(S[i, g]) for e, g in ends.items()}
        # System level: curves the wrong way round fit (Tm_B, Tm_A) clearly better
        swapped = False
        if all(np.isfinite(v) for v in liq.values()) and min(tm) > 0 and abs(tm[0] - tm[1]) > 2 * tol:
            straight = abs(liq["A"] - tm[0]) + abs(liq["B"] - tm[1])
            crossed = abs(liq["A"] - tm[1]) + abs(liq["B"] - tm[0])
            swapped = crossed + 2 * tol < straight and crossed <= 2 * tol
        for k, (end, g) in enumerate(ends.items()):
            t, lv, sv = tm[k], liq[end], sol[end]
            nxt = 1 if g == 0 else -2
            jump = (abs(float(L[i, g]) - float(L[i, nxt]))
                    if Lm[i, g] and Lm[i, nxt] else np.nan)
            row = dict(pair=pair, split=splits[i], topology=ds.topology_name(i), end=end,
                       Tm_feature=t, liquidus_end=lv, solidus_end=sv,
                       liquidus_minus_Tm=lv - t if t > 0 else np.nan,
                       solidus_minus_liquidus=sv - lv,
                       end_jump_K=jump)
            if not np.isfinite(lv):
                row["flag"] = "not_observed"
            elif t <= 0:
                row["flag"] = "missing_Tm"
            elif abs(lv - t) <= tol:
                row["flag"] = "ok"
            elif abs(lv + C_TO_K - t) <= tol:
                row["flag"] = "celsius"
            elif swapped:
                row["flag"] = "swapped"
            else:
                row["flag"] = "mismatch"
            row["solidus_detached"] = bool(np.isfinite(sv) and np.isfinite(lv) and abs(sv - lv) > tol)
            row["steep_end_jump"] = bool(np.isfinite(jump) and jump > jump_K)
            rows.append(row)
    df = pd.DataFrame(rows)
    obs = df[df["flag"] != "not_observed"]
    d = obs["liquidus_minus_Tm"].dropna()
    summary = {
        "ends_total": int(len(df)),
        "ends_observed": int(len(obs)),
        "flags": df["flag"].value_counts().to_dict(),
        "solidus_detached": int(df["solidus_detached"].sum()),
        "steep_end_jump": int(df["steep_end_jump"].sum()),
        "systems_with_any_problem": int(df.loc[df["flag"].isin(FLAG_ORDER[:-1]) |
                                               df["solidus_detached"] |
                                               df["steep_end_jump"], "pair"].nunique()),
        "liquidus_minus_Tm_K": {
            "median_abs": float(d.abs().median()) if len(d) else None,
            "p90_abs": float(d.abs().quantile(0.9)) if len(d) else None,
            "mean": float(d.mean()) if len(d) else None,
        },
    }
    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trained curve models
# ─────────────────────────────────────────────────────────────────────────────

def load_curve_model(path: Path):
    if path.suffix == ".joblib":
        import joblib
        return joblib.load(path)
    try:
        from mpds_models.curves_io import load_curves_model
    except ImportError:
        from mpds_models.deep_curve_net import LiquidusSolidusModel
        load_curves_model = LiquidusSolidusModel.load
    return load_curves_model(path)


def check_model(model, label: str, ds, idx: np.ndarray, bins: np.ndarray) -> tuple[dict, list]:
    ia, ib = melting_point_columns(ds)
    try:
        liq, sol = model.predict(ds.X[idx], clamp=False)
    except TypeError:
        liq, sol = model.predict(ds.X[idx])
    tm = np.stack([ds.X[idx, ia], ds.X[idx, ib]], axis=1)
    pred_end = np.stack([liq[:, 0], liq[:, -1]], axis=1)
    sol_end = np.stack([sol[:, 0], sol[:, -1]], axis=1)
    has_tm = tm > 0
    true = ds.T_liquidus[idx]
    mask = ds.T_liquidus_mask[idx]
    data_end = np.stack([true[:, 0], true[:, -1]], axis=1)
    data_obs = np.stack([mask[:, 0], mask[:, -1]], axis=1)
    # End jump: distance of the end point from the straight extension of its
    # two neighbours (0 for a curve that runs smoothly into the end)
    jump = np.stack([np.abs(liq[:, 0] - (2 * liq[:, 1] - liq[:, 2])),
                     np.abs(liq[:, -1] - (2 * liq[:, -2] - liq[:, -3]))], axis=1)
    err = np.abs(liq - true)
    grid = np.asarray(ds.comp_grid)
    res = {
        "model": label, "n_systems": int(len(idx)),
        "liquidus_end_vs_Tm_MAE_K": float(np.abs(pred_end - tm)[has_tm].mean()),
        "solidus_end_vs_Tm_MAE_K": float(np.abs(sol_end - tm)[has_tm].mean()),
        "liquidus_end_vs_data_MAE_K": float(np.abs(pred_end - data_end)[data_obs].mean())
        if data_obs.any() else None,
        "end_jump_median_K": float(np.median(jump)),
        "end_jump_p90_K": float(np.quantile(jump, 0.9)),
        "liquidus_MAE_ends_K": float(err[:, [0, -1]][mask[:, [0, -1]]].mean())
        if mask[:, [0, -1]].any() else None,
        "liquidus_MAE_interior_K": float(err[:, 1:-1][mask[:, 1:-1]].mean()),
    }
    by_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        cols = (grid >= lo) & ((grid < hi) | ((hi == bins[-1]) & (grid <= hi)))
        e = err[:, cols][mask[:, cols]]
        by_bin.append(dict(model=label, bin_lo=lo, bin_hi=hi,
                           liquidus_MAE_K=float(e.mean()) if e.size else np.nan, n_points=int(e.size)))
    return res, by_bin


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


def plot_scatter(df: pd.DataFrame, tol: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df["flag"] != "not_observed") & (df["Tm_feature"] > 0)]
    groups = (("within ±%g K" % tol, d["flag"] == "ok", C_OK),
              ("°C or swapped", d["flag"].isin(["celsius", "swapped"]), C_UNIT),
              ("other mismatch", d["flag"] == "mismatch", C_OTHER))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = [0, max(d["Tm_feature"].max(), d["liquidus_end"].max()) * 1.05] if len(d) else [0, 1]
    ax.plot(lim, lim, color=MUTED, lw=1)
    ax.fill_between(lim, [lim[0] - tol, lim[1] - tol], [lim[0] + tol, lim[1] + tol],
                    color=GRID, alpha=0.6, lw=0)
    ax.plot(lim, [lim[0] - C_TO_K, lim[1] - C_TO_K], color=MUTED, lw=1, ls="--")
    xt = 0.8 * lim[1]
    ax.text(xt + 40, xt - C_TO_K - 40, "Tm − 273 K (curve in °C)", color=MUTED, fontsize=8,
            va="top", ha="left")
    for label, sel, color in groups:
        sub = d[sel]
        ax.scatter(sub["Tm_feature"], sub["liquidus_end"], s=18, color=color,
                   edgecolor="white", linewidth=0.6, label=f"{label} ({len(sub)})", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Tm feature (K)", fontsize=9, color=MUTED)
    ax.set_ylabel("observed liquidus at the pure-element end (K)", fontsize=9, color=MUTED)
    ax.set_title("Data end points vs melting-point features", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_bins(by_bin: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for color, (label, g) in zip(SERIES, by_bin.groupby("model", sort=False)):
        mid = (g["bin_lo"] + g["bin_hi"]) / 2
        ax.plot(mid, g["liquidus_MAE_K"], color=color, lw=2, marker="o", ms=4, label=label)
    ax.set_xlabel("composition (at% B), 5 at% bins", fontsize=9, color=MUTED)
    ax.set_ylabel("liquidus MAE (K)", fontsize=9, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_title("Liquidus error across composition", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the pure-element end points.")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="endpoint_audit")
    ap.add_argument("--tol", type=float, default=50.0,
                    help="largest |difference| in K still counted as agreeing (default 50)")
    ap.add_argument("--jump", type=float, default=100.0,
                    help="change in K between the end point and its neighbour that counts "
                         "as a steep jump (default 100)")
    ap.add_argument("--curve-model", nargs="+", type=Path, default=[],
                    help="trained curve models to check (.pt CurveNet or .joblib tree model)")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test",
                    help="split the models are checked on (default test)")
    ap.add_argument("--list", type=int, default=15, help="worst systems printed per check")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.processed_dir)
    summary = {"tol_K": a.tol, "jump_K": a.jump}

    print("\n=== 1. Melting-point features vs reference values ===")
    feat, summary["features"] = check_features(ds, a.tol)
    feat.to_csv(out / "features_vs_reference.csv", index=False)
    s = summary["features"]
    print(f"Checked {s['systems_checked']} systems "
          f"({s['systems_without_element_symbols']} pair ids are not two element symbols).")
    if len(feat):
        print(f"Ends that disagree with the reference by more than {a.tol:g} K:")
        print(feat["flag"].value_counts().to_string())
        worst = feat.reindex(feat["difference"].abs().sort_values(ascending=False).index)
        print(worst.head(a.list).to_string(index=False))
    else:
        print(f"All features agree with the reference within {a.tol:g} K.")

    print("\n=== 2. Data end points vs melting-point features ===")
    data, summary["data"] = check_data(ds, a.tol, a.jump)
    data.to_csv(out / "data_endpoints.csv", index=False)
    s = summary["data"]
    print(f"{s['ends_observed']} of {s['ends_total']} ends have an observed liquidus point.")
    print(data["flag"].value_counts().to_string())
    print(f"Solidus end > {a.tol:g} K from the liquidus end: {s['solidus_detached']}")
    print(f"Steep jump (> {a.jump:g} K) between end point and neighbour: {s['steep_end_jump']}")
    q = s["liquidus_minus_Tm_K"]
    if q["median_abs"] is not None:
        print(f"|liquidus end − Tm|: median {q['median_abs']:.1f} K, 90th pct {q['p90_abs']:.1f} K, "
              f"mean signed {q['mean']:+.1f} K")
    bad = data[data["flag"].isin(FLAG_ORDER[:-1])]
    if len(bad):
        bad = bad.reindex(bad["liquidus_minus_Tm"].abs().sort_values(ascending=False).index)
        print("Worst ends:")
        print(bad[["pair", "split", "end", "Tm_feature", "liquidus_end", "liquidus_minus_Tm", "flag"]]
              .head(a.list).to_string(index=False))
    plot_scatter(data, a.tol, out / "endpoints_scatter.png")

    print("\n=== 2b. Liquidus tops cut off by the source chart ===")
    clipped = clipped_top_mask(ds)
    rows = [dict(pair=ds.pair_ids[i], split=split_names(ds)[i], topology=ds.topology_name(i),
                 plateau_K=float(ds.T_liquidus[i, clipped[i]][0]),
                 from_at_pct=float(ds.comp_grid[clipped[i]].min()),
                 to_at_pct=float(ds.comp_grid[clipped[i]].max()),
                 n_points=int(clipped[i].sum()))
            for i in np.flatnonzero(clipped.any(axis=1))]
    clip_df = pd.DataFrame(rows, columns=["pair", "split", "topology", "plateau_K",
                                          "from_at_pct", "to_at_pct", "n_points"])
    clip_df.to_csv(out / "clipped_tops.csv", index=False)
    summary["clipped_tops"] = {"systems": len(clip_df), "points": int(clipped.sum()),
                               "by_topology": clip_df["topology"].value_counts().to_dict()}
    print(f"{len(clip_df)} systems have a flat liquidus top above both melting points "
          f"({int(clipped.sum())} grid points); train_all / predict.py --mask-clipped leave them out.")
    if len(clip_df):
        print(clip_df["topology"].value_counts().to_string())
        print(clip_df.sort_values("n_points", ascending=False).head(a.list).to_string(index=False))

    if a.curve_model:
        print(f"\n=== 3. Trained curve models on the {a.split} split ===")
        idx = ds.split(a.split)
        bins = np.arange(0, 101, 5)
        res, by_bin = [], []
        for path in a.curve_model:
            label = f"{path.parent.name}/{path.stem}"
            r, b = check_model(load_curve_model(path), label, ds, idx, bins)
            res.append(r)
            by_bin += b
        res_df = pd.DataFrame(res)
        res_df.to_csv(out / "model_endpoints.csv", index=False)
        by_bin_df = pd.DataFrame(by_bin)
        by_bin_df.to_csv(out / "model_error_by_bin.csv", index=False)
        print(res_df.set_index("model").T.to_string(float_format=lambda v: f"{v:.1f}"))
        plot_bins(by_bin_df, out / "model_error_by_bin.png")
        summary["models"] = res

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved the tables, charts and summary.json in {out}/")


if __name__ == "__main__":
    main()
