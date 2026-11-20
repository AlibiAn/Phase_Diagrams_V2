"""
fit_thermo_engine.py — how well can the small CALPHAD-style engine
(mpds_models/thermo_engine.py) reproduce each system's liquidus if its four
parameters (L0, L1, Ls, c) are FITTED to that system? This is an upper bound
for the approach "predict a few thermodynamic parameters, compute the diagram":
if the best fit is not far better than the direct curve model, the approach
cannot win. Also scores the engine with untuned parameters (L0 from Miedema,
L1 = 0, no solid solubility, OQMD compound energies as computed).

    python fit_thermo_engine.py /home/alibi/lab/processed_v8 --out results/v19/engine_fit --workers 14
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
BOUNDS = [(-200.0, 200.0), (-150.0, 150.0), (-40.0, 200.0), (0.0, 2.0)]


def fit_one(args):
    pid, T, mask, use_compounds = args
    from mpds_models.element_features import interaction_parameter
    from mpds_models.thermo_engine import system_for
    a, b = pid.split("_")
    s = system_for(a, b, use_compounds=use_compounds, t_step=10.0)
    if s is None or mask.sum() < 10:
        return pid, None
    om = interaction_parameter(a, b)
    om = float(om) if np.isfinite(om) else 0.0

    def loss(p):
        return float(np.abs(s.liquidus(*p) - T)[mask].mean())

    default = np.array([om, 0.0, 60.0, 1.0])
    best = (loss(default), default)
    starts = [default, np.array([om, 0.0, 10.0, 1.0]), np.array([om - 30, 0.0, 60.0, 0.7]),
              np.array([om + 30, 0.0, 60.0, 1.3]), np.array([om, 20.0, 60.0, 1.0]), np.array([om, -20.0, 60.0, 1.0])]
    for start in starts:                 # the compiled engine takes < 1 ms per diagram, so search widely
        r = minimize(loss, start, method="Powell", bounds=BOUNDS, options={"maxfev": 600, "xtol": 0.2, "ftol": 0.2})
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    return pid, dict(default_mae=loss(default), fit_mae=best[0], params=[float(v) for v in best[1]],
                     n_compounds=len(s.comp_x), n_points=int(mask.sum()),
                     fitted_liquidus=s.liquidus(*best[1]).tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--no-compounds", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from mpds_models.base import load_dataset, mask_clipped_tops
    ds = load_dataset(a.processed_dir)
    mask_clipped_tops(ds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, ds.T_liquidus[i].astype(float), ds.T_liquidus_mask[i].copy(), not a.no_compounds)
            for i, p in enumerate(ds.pair_ids)]
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time()
    res = {}
    with Pool(a.workers) as pool:
        for k, (pid, r) in enumerate(pool.imap_unordered(fit_one, jobs, chunksize=4)):
            if r is not None:
                res[pid] = r
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} systems, {(time.time() - t0) / 60:.1f} min", flush=True)
    (out / "fits.json").write_text(json.dumps(res))
    pts = np.array([r["n_points"] for r in res.values()])
    d = np.array([r["default_mae"] for r in res.values()])
    f = np.array([r["fit_mae"] for r in res.values()])
    print(f"\n{len(res)} systems fitted ({'without' if a.no_compounds else 'with'} OQMD compounds), {(time.time() - t0) / 60:.1f} min")
    print(f"  untuned engine (Miedema liquid, OQMD compounds):  liquidus MAE {np.sum(d * pts) / pts.sum():6.1f} K  (median system {np.median(d):.0f} K)")
    print(f"  engine with the 4 parameters fitted per system:   liquidus MAE {np.sum(f * pts) / pts.sum():6.1f} K  (median system {np.median(f):.0f} K)")
    topo = {p: ds.topology_name(i) for i, p in enumerate(ds.pair_ids)}
    for t in ds.topology_classes:
        ks = [p for p in res if topo[p] == t]
        if ks:
            w = np.array([res[p]["n_points"] for p in ks]); e = np.array([res[p]["fit_mae"] for p in ks])
            print(f"    {t:<30} {len(ks):5d} systems  fitted MAE {np.sum(e * w) / w.sum():6.1f} K")


if __name__ == "__main__":
    main()
