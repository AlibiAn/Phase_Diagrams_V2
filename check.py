import sys, json, glob, collections
import numpy as np
sys.path.insert(0, ".")
from mpds_models.base import load_dataset

ds = load_dataset("processed_v3")
both = ds.T_liquidus_mask & np.isfinite(ds.T_solidus)
d = (ds.T_liquidus - ds.T_solidus)[both]
print("solidus within 5 K of liquidus: %.0f%% of grid points" % (100 * (np.abs(d) < 5).mean()))
print("median liquidus - solidus: %.0f K" % np.median(d))

c = collections.Counter()
for f in sorted(glob.glob("mpds_data/*.json"))[:500]:
    for p in json.load(open(f)).get("phases", []):
        if p.get("phases") == 2:
            c[(repr(p.get("label"))[:20], p.get("is_solid"))] += 1
print("two-phase regions (label, is_solid), most common:")
for k, n in c.most_common(15):
    print("  ", k, n)