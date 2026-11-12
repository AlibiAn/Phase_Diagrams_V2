"""Two checks behind the two-liquid dataset (processed_v7), printed to results/v14/two_liquid_checks.txt:
1. how much of the v13 cross-validated liquidus error sits in two-liquid systems;
2. the monotectic temperature the corrected liquidus gives for well-known systems, against
   handbook values QUOTED FROM MEMORY (to be checked against a reference before they are cited).
Run from the repository root after cross_validate.py:  python results/v14/two_liquid_checks.py"""
import numpy as np
import pandas as pd
from mpds_models.base import load_dataset, mask_clipped_tops

V6, V7 = "/home/alibi/lab/processed_v6", "/home/alibi/lab/processed_v7"
ds = load_dataset(V6)
mask_clipped_tops(ds)
ch = pd.read_csv(f"{V7}/changes_vs_reference.csv", index_col=0).reindex(ds.pair_ids)
seen = ds.T_liquidus_mask
print("1. Share of the cross-validated liquidus error (processed_v6 targets) by kind of system")
for name in ("v13", "v14"):
    o = np.load(f"results/v14/cv_v6/oof_{name}.npz")
    err = np.abs(o["liquidus"] - ds.T_liquidus)
    total = err[seen].sum()
    print(f"   {name}:")
    for label, rows in (("two liquids in most sources (gap flag of processed_v7)", ch.gap_new.to_numpy()),
                        ("   flagged as a miscibility gap in processed_v6", (ch.gap_new & ch.gap_reference).to_numpy()),
                        ("   NOT flagged in processed_v6", (ch.gap_new & ~ch.gap_reference).to_numpy()),
                        ("all other systems", ~ch.gap_new.to_numpy())):
        e = err[rows][seen[rows]]
        print(f"     {label:<56} {int(rows.sum()):5d} systems   liquidus MAE {e.mean():6.1f} K   "
              f"{100 * e.sum() / total:5.1f} % of the error")

print("\n2. Monotectic temperature (K): corrected liquidus vs handbook value quoted from memory")
known = {"Cu_Pb": 1228, "Al_In": 912, "Bi_Zn": 689, "Pb_Zn": 691, "Ag_Ni": 1708, "Al_Pb": 932, "Ga_Pb": 586,
         "Al_Bi": 930, "Cu_Tl": 1241, "Bi_Ga": 495, "Ag_Co": 1762, "Fe_Sn": 1403, "Al_Cd": 922, "Ga_Tl": 558,
         "Cd_Ga": 555, "Li_Na": 444}
d7 = load_dataset(V7)
raw6 = load_dataset(V6)
diffs = []
for pair, ref in known.items():
    i = ds.pair_ids.index(pair)
    cut = (d7.T_liquidus[i] < raw6.T_liquidus[i] - 1) & d7.T_liquidus_mask[i]
    line = float(np.median(d7.T_liquidus[i][cut])) if cut.any() else float("nan")
    top = float(np.nanmax(np.where(raw6.T_liquidus_mask[i], raw6.T_liquidus[i], np.nan)))
    diffs.append(abs(line - ref))
    print(f"   {pair:<6} handbook {ref:5d}   processed_v7 line {line:6.0f}   processed_v6 highest liquidus {top:6.0f}")
print(f"   median |difference| {np.nanmedian(diffs):.0f} K, largest {np.nanmax(diffs):.0f} K over {len(diffs)} systems")
