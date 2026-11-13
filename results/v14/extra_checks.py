"""Four small checks quoted in docs/research_review.md, printed to results/v14/extra_checks.txt.
Run from the repository root after cross_validate.py:
    PYTHONPATH=. python results/v14/extra_checks.py /home/alibi/lab

1. Is the solidus gain of the direct solidus real, given that the test split does not show it?
2. The eutectic point read off the predicted liquidus, against the eutectic model (M4).
3. Which systems are worst for v14 on the two-liquid targets (out-of-fold, all 1,380 systems);
   also saves the thirty worst as results/v14/worst_30_two_liquid.png.
4. Systems in the raw MPDS files that the preprocessing rejected.
"""
import copy
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cross_validate import folds
from mpds_models.base import fix_end_points, load_dataset, mask_clipped_tops, system_quality

V6, V7 = "/home/alibi/lab/processed_v6", "/home/alibi/lab/processed_v7"
ds = load_dataset(V6)
quality = system_quality(ds)
mask_clipped_tops(ds)

print("1. Direct solidus vs gap solidus (both with the extended features), out-of-fold, scored as predict.py")
scored = copy.deepcopy(ds)
fix_end_points(scored, inject=False)
gap, direct = np.load("results/v14/cv_v6/oof_v14-features.npz"), np.load("results/v14/cv_v6/oof_v14.npz")
err = {n: np.abs(np.minimum(o["solidus"], o["liquidus"]) - scored.T_solidus) for n, o in (("gap", gap), ("direct", direct))}
for label, idx in (("all 1,380 systems", np.arange(ds.n_systems)), ("the 138 test-split systems", ds.test_idx)):
    print(f"   {label:<28} gap {np.nanmean(err['gap'][idx]):6.1f} K   direct {np.nanmean(err['direct'][idx]):6.1f} K")
per = {n: np.array([np.nanmean(e[i]) if np.isfinite(e[i]).any() else np.nan for i in ds.test_idx]) for n, e in err.items()}
npts = np.isfinite(scored.T_solidus[ds.test_idx]).sum(axis=1)
worst = np.argsort(-np.nan_to_num((per["direct"] - per["gap"]) * npts))[:3]
print("   test systems that cost the direct solidus most: " + "; ".join(
    f"{ds.pair_ids[ds.test_idx[k]]} {per['gap'][k]:.0f} -> {per['direct'][k]:.0f} K" for k in worst))
rng = np.random.default_rng(0)
diffs = np.array([np.nanmean(err["direct"][s]) - np.nanmean(err["gap"][s])
                  for s in (rng.choice(ds.n_systems, 138, replace=False) for _ in range(2000))])
print(f"   2,000 random subsets of 138 systems: direct minus gap = {diffs.mean():.1f} K on average, "
      f"5th to 95th percentile {np.percentile(diffs, 5):.1f} to {np.percentile(diffs, 95):.1f} K; "
      f"direct is better in {100 * (diffs < 0).mean():.1f} % of the subsets")

print("\n2. Eutectic point: lowest point of the out-of-fold v14 liquidus vs the eutectic model (M4), 5-fold CV")
has = ds.eutectic_exists & np.isfinite(ds.eutectic_temp) & np.isfinite(ds.eutectic_comp)
liq = direct["liquidus"]
print(f"   lowest point of the v14 liquidus: temperature MAE {np.abs(liq.min(1) - ds.eutectic_temp)[has].mean():.1f} K, "
      f"composition MAE {np.abs(liq.argmin(1) - ds.eutectic_comp)[has].mean():.1f} at%  ({int(has.sum())} systems with a eutectic)")
sc = pd.read_csv("results/v14/cv_v6/scalars.csv")
for fs in ("v6", "extended"):
    r = sc[sc.features == fs]
    print(f"   eutectic model, {fs} features: temperature MAE {r.eutectic_temp_MAE_all_true_K.mean():.1f} K, "
          f"composition MAE {r.eutectic_comp_MAE_all_true_at_pct.mean():.1f} at%")

print("\n3. Worst systems for v14 on the two-liquid targets (out-of-fold)")
d7 = load_dataset(V7)
q7 = system_quality(d7)
mask_clipped_tops(d7)
o = np.load("results/v14/cv_v7/oof_v14.npz")
l7, s7 = o["liquidus"], np.minimum(o["solidus"], o["liquidus"])
seen = d7.T_liquidus_mask
e7 = np.abs(l7 - d7.T_liquidus)
total = np.array([e7[i][seen[i]].sum() for i in range(d7.n_systems)])
mean = np.array([e7[i][seen[i]].mean() if seen[i].any() else np.nan for i in range(d7.n_systems)])
order = np.argsort(-total)
n10 = d7.n_systems // 10
print(f"   the worst 10 % of systems ({n10}) carry {100 * total[order[:n10]].sum() / total.sum():.0f} % of the liquidus error; "
      f"per-system MAE: median {np.nanmedian(mean):.0f} K, mean {np.nanmean(mean):.0f} K")
elements = pd.DataFrame({"element": [e for p in d7.pair_ids for e in p.split("_")], "mae": np.repeat(mean, 2)})
by_el = elements.groupby("element").mae.agg(["mean", "count"]).sort_values("mean", ascending=False)
print("   elements with the highest mean system error (K, systems): " + ", ".join(
    f"{e} {r['mean']:.0f} ({int(r['count'])})" for e, r in by_el.head(10).iterrows()))
print("   thirty worst: " + ", ".join(d7.pair_ids[i] for i in order[:30]))
names = list(d7.feature_names)
tm = d7.X[:, [names.index("phys__melting_point__A"), names.index("phys__melting_point__B")]]
fig, axes = plt.subplots(5, 6, figsize=(24, 17))
for ax, i in zip(axes.flat, order[:30]):
    g = d7.comp_grid
    ax.plot(g[seen[i]], d7.T_liquidus[i][seen[i]], "k.", ms=4)
    s = np.isfinite(d7.T_solidus[i])
    ax.plot(g[s], d7.T_solidus[i][s], ".", color="gray", ms=3)
    ax.plot(g, l7[i], "-", color="#eb6834")
    ax.plot(g, s7[i], "--", color="#eb6834", lw=0.8)
    ax.plot([0, 100], tm[i], ":", color="#2a78d6", lw=0.8)
    ax.set_title(f"{d7.pair_ids[i]}  {d7.topology_name(i)[:16]}  MAE {mean[i]:.0f} K  {q7.group[i][:24]}", fontsize=9)
fig.suptitle("Thirty worst systems, v14 out-of-fold on the two-liquid targets: data (black liquidus, grey solidus), "
             "prediction (orange), melting-point line (blue)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("results/v14/worst_30_two_liquid.png", dpi=60)

print("\n4. Systems in the raw MPDS files that are not in the dataset")
sys.path.insert(0, sys.argv[1])
from mpds_preprocess.loader import load_entry, quality_filter
from mpds_preprocess.units import decide_units
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/mpds_data/*.json")):
    e = load_entry(f)
    if e is None or len(e.get("elements") or []) != 2:
        continue
    rep = quality_filter(e)
    rows.append(dict(pair="_".join(sorted(e["elements"])), passed=rep.passed,
                     unit_known=decide_units(e).unit != "?" if rep.passed else False))
raw = pd.DataFrame(rows)
missing = raw[~raw.pair.isin(set(ds.pair_ids))].groupby("pair").agg(passed=("passed", "max"), entries=("passed", "size"))
print(f"   raw files: {len(raw)} entries, {raw.pair.nunique()} systems; in the dataset: {ds.n_systems}; "
      f"rejected: {len(missing)} systems ({int(missing.entries.sum())} entries)")
print(f"   of the rejected systems, {int(missing.passed.sum())} have an entry that passes the quality filter "
      f"(dropped because its temperature unit could not be determined); the other {int((~missing.passed).sum())} "
      f"have only partial-range entries or no usable liquidus")
