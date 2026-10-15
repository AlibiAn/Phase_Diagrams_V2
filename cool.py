import sys; import numpy as np
sys.path.insert(0, ".")
from mpds_models.base import load_dataset
ds = load_dataset("processed_v4")

has = ds.eutectic_exists & np.isfinite(ds.eutectic_temp)
sol_min = np.nanmin(np.where(np.isfinite(ds.T_solidus), ds.T_solidus, np.nan), axis=1)
d = (sol_min - ds.eutectic_temp)[has & np.isfinite(sol_min)]
print("eutectic systems checked:", d.size)
print("lowest solidus within 30 K of eutectic temp: %.0f%%" % (100 * (np.abs(d) < 30).mean()))
print("lowest solidus > 100 K BELOW eutectic temp: %.0f%%  (too low = misread region)" % (100 * (d < -100).mean()))
print("lowest solidus > 100 K ABOVE eutectic temp: %.0f%%" % (100 * (d > 100).mean()))