"""Checks of the height shift (compare_foundation_models.py) printed to results/v17/height_shift_checks.txt:
1. cross-validation by data group and diagram type, and with TabPFN and TabICL heights averaged;
2. the test split: height models trained on train + validation (as v16 was), applied to the v16 test curves.
Run from the repository root:  MPDS_DEVICE=cpu PYTHONPATH=. python results/v17/height_shift_checks.py"""
import numpy as np
import pandas as pd
from compare_foundation_models import make
from cross_validate import folds
from mpds_models.base import (curve_metrics, group_rows, load_dataset, load_dataset_for_models, mask_clipped_tops,
                              fix_end_points, system_quality)
from mpds_models.meta_learner import FullDiagramReconstructor
from mpds_models.tree_curves import _fill_curves

V8, V6 = "/home/alibi/lab/processed_v8", "/home/alibi/lab/processed_v6"
ds = load_dataset(V8, features="extended+mp")
quality = system_quality(load_dataset(V6))
mask_clipped_tops(ds)
names = list(ds.feature_names)
tm_a, tm_b = (ds.X[:, names.index(f"phys__melting_point__{k}")].astype(float) for k in "AB")
frac = ds.comp_grid.astype(float) / 100.0
line = (1 - frac)[None] * tm_a[:, None] + frac[None] * tm_b[:, None]
seen = ds.T_liquidus_mask.copy(); seen[:, [0, -1]] = False
target = (_fill_curves(ds.T_liquidus, seen, frac, tm_a, tm_b, 3) - line).mean(axis=1)
ok = np.isfinite(target)
window = 4 * frac * (1 - frac); window /= window.mean()
o = np.load("results/v16/cv_v8_all/oof_v14+mp+extra.npz"); liq, sol = o["liquidus"], o["solidus"]
implied = (liq - line).mean(axis=1)
p = {m: np.load(f"results/v17/height_oof_{m}.npy") for m in ("tabpfn", "tabicl")}
p["average of both"] = 0.5 * (p["tabpfn"] + p["tabicl"])
fl = folds(ds.n_systems, 5)
shift = lambda base, h, imp, a=0.5: base + a * (h - imp)[:, None] * window[None]

print("1. Cross-validation (processed_v8, 1,380 systems), liquidus MAE in K; v16 curve vs shifted 50 % towards the height model")
ref = np.array([curve_metrics(ds, te, liq[te], sol[te])["liquidus_MAE_K"] for _, te in fl])
for m, h in p.items():
    s = shift(liq, h, implied)
    f = np.array([curve_metrics(ds, te, s[te], sol[te])["liquidus_MAE_K"] for _, te in fl])
    print(f"   {m:<18} height MAE {np.abs(h - target)[ok].mean():5.1f} K | liquidus {ref.mean():.1f} -> {f.mean():.1f} K, better in {(f < ref).sum()}/5 folds")
s = shift(liq, p["average of both"], implied)
print("   by data group (v16 -> shifted, average of both height models):")
for g, idx in group_rows(quality, np.arange(ds.n_systems)).items():
    a, b = curve_metrics(ds, idx, liq[idx], sol[idx])["liquidus_MAE_K"], curve_metrics(ds, idx, s[idx], sol[idx])["liquidus_MAE_K"]
    print(f"     {g:<18} n={len(idx):5d}  {a:6.1f} -> {b:6.1f} K")
topo = np.array([ds.topology_name(i) for i in range(ds.n_systems)])
print("   by diagram type:")
for t in ds.topology_classes:
    idx = np.flatnonzero(topo == t)
    a, b = curve_metrics(ds, idx, liq[idx], sol[idx])["liquidus_MAE_K"], curve_metrics(ds, idx, s[idx], sol[idx])["liquidus_MAE_K"]
    print(f"     {t:<30} n={len(idx):4d}  {a:6.1f} -> {b:6.1f} K")

print("\n2. Test split (138 systems), scored as predict.py (--mask-clipped --fix-end-points)")
scored = load_dataset(V8); mask_clipped_tops(scored); fix_end_points(scored, inject=False)
d16 = load_dataset_for_models(V8, "/home/alibi/lab/models_v16"); mask_clipped_tops(d16)
test = ds.test_idx
train = np.sort(np.concatenate([ds.train_idx, ds.val_idx]))
lt, st = FullDiagramReconstructor.from_dir("/home/alibi/lab/models_v16", d16).m23.predict(d16.X[test])
imp_t = (lt - line[test]).mean(axis=1)
tr = train[ok[train]]
hs = {}
for m in ("tabpfn", "tabicl"):
    model = make(m, "regression").fit(np.concatenate([ds.X[tr], ds.X_swap[tr]]), np.tile(target[tr], 2))
    hs[m] = 0.5 * (model.predict(ds.X[test]) + model.predict(ds.X_swap[test]))
hs["average of both"] = 0.5 * (hs["tabpfn"] + hs["tabicl"])
base = curve_metrics(scored, test, lt, st)
print(f"   v16: liquidus {base['liquidus_MAE_K']:.1f} K")
for m, h in hs.items():
    s = shift(lt, h, imp_t)
    r = curve_metrics(scored, test, s, st)
    q = group_rows(quality, test)
    pos = {int(i): k for k, i in enumerate(test)}
    clean = q["clean"]; rows = [pos[int(i)] for i in clean]
    c0 = curve_metrics(scored, clean, lt[rows], st[rows])["liquidus_MAE_K"]; c1 = curve_metrics(scored, clean, s[rows], st[rows])["liquidus_MAE_K"]
    print(f"   v16 shifted towards {m:<16}: liquidus {r['liquidus_MAE_K']:.1f} K   (clean systems {c0:.1f} -> {c1:.1f} K)")
