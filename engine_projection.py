"""
engine_projection.py — redraw a predicted liquidus as a thermodynamically
consistent diagram: fit the parameters of the engine (mpds_models/thermo_engine.py)
to the PREDICTED curve of v18 (not to the data), so the engine draws flat
monotectic plateaus, sharp eutectics and compound peaks where v18 places them.

Optionally the parameters are also pulled towards the values TabPFN + TabICL
predict for them (predict_thermo_params.py):

    loss(theta) = mean_x |engine(theta, x) - v18(x)| + lam * sum_k |theta_k - theta_pred_k| / scale_k

Everything is out of fold: the v18 curves and the predicted parameters come from
the same 5 folds, and no step uses the system's own data. Scored like
explore_v18.py: alone, and blended with v18.

    python engine_projection.py /home/alibi/lab/processed_v8 --cache /home/alibi/lab/cache_v18 \\
        --params results/v19/thermo_params_oof.npy --out results/v19
"""
from __future__ import annotations

import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
BOUNDS = [(-200.0, 200.0), (-150.0, 150.0), (-40.0, 200.0), (0.0, 2.0)]
SCALE = np.array([76.0, 37.0, 73.0, 0.52])          # spread of the fitted parameters (results/v19)


def project(args):
    pid, target, theta_pred, lam = args
    from mpds_models.thermo_engine import system_for
    a, b = pid.split("_")
    s = system_for(a, b, t_step=10.0)
    if s is None or not np.isfinite(theta_pred).all():
        return None
    theta_pred = np.clip(theta_pred, [lo for lo, _ in BOUNDS], [hi for _, hi in BOUNDS])

    def loss(p):
        fit = float(np.abs(s.liquidus(*p) - target).mean())
        return fit + lam * float(np.sum(np.abs(np.asarray(p) - theta_pred) / SCALE))

    best = None
    for start in (theta_pred, np.array([theta_pred[0], 0.0, 60.0, 1.0]), np.array([theta_pred[0], 0.0, 10.0, 1.0]),
                  np.array([theta_pred[0] - 30, 0.0, 60.0, theta_pred[3]]),
                  np.array([theta_pred[0] + 30, 0.0, 60.0, theta_pred[3]])):
        r = minimize(loss, start, method="Powell", bounds=BOUNDS, options={"maxfev": 600, "xtol": 0.2, "ftol": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return s.liquidus(*best.x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--params", required=True, help="out-of-fold predicted parameters (predict_thermo_params.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0, 5.0, 20.0])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from explore_v18 import Lab
    lab = Lab(a.processed_dir, a.cache)
    v17 = lab.v17()
    fm = np.load(lab.cache / f"pca_fm_10_{lab._key(lab.data, 'extended+mp')}.npy")
    v18_liq = 0.5 * v17[0] + 0.5 * (lab.line + fm)
    v18 = (v18_liq, np.minimum(v17[1], v18_liq))
    ref = lab.report("v17", *v17)
    lab.report("v18", *v18, ref, groups=True)
    theta = np.load(a.params)
    ids = lab.score_ds.pair_ids
    for lam in a.lams:
        with Pool(a.workers) as pool:
            res = pool.map(project, [(p, v18_liq[i].astype(float), theta[i], lam) for i, p in enumerate(ids)], chunksize=8)
        proj = np.array([r if r is not None else v18_liq[i] for i, r in enumerate(res)], dtype=np.float32)
        np.save(Path(a.out) / f"engine_projection_lam{lam:g}.npy", proj)
        lab.report(f"engine fitted to the v18 curve, prior weight {lam:g}", proj, np.minimum(v18[1], proj), ref, groups=True)
        for w in (0.25, 0.5):
            liq = (1 - w) * v18_liq + w * proj
            lab.report(f"  v18 blended {w:.0%} with it", liq, np.minimum(v18[1], liq), ref)


if __name__ == "__main__":
    main()
