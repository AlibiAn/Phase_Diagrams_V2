import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/alibi/lab_new/Phase_Diagrams_V2")
import numpy as np
from multiprocessing import Pool
from explore_v18 import Lab
from engine_learn import curves_from, NAMES
lab = Lab("/home/alibi/lab/processed_v8", "/home/alibi/lab/cache_v18")
ids = list(lab.score_ds.pair_ids)
fits = json.loads(open("results/v19/engine_fit/fits.json").read())
P = np.array([fits[p]["params"] if p in fits else [np.nan]*4 for p in ids])
have = np.isfinite(P).all(1)
pred = np.load("results/v19/thermo_params_oof.npy")
v17 = lab.v17()
ref = lab.report("v17", *v17)
with Pool(16) as pool:
    for keep in [(), (0,), (1,), (2,), (3,), (0, 1), (2, 3), (0, 1, 2), (0, 1, 2, 3)]:
        th = pred.copy()
        for j in keep:
            th[have, j] = P[have, j]
        eng = curves_from(pool, ids, th, v17[0])
        lab.report("fitted: " + (",".join(NAMES[j] for j in keep) or "none") + "; rest predicted", eng, np.minimum(v17[1], eng), ref)
