"""
predict_thermo_params.py — the "predict the thermodynamics, compute the diagram"
route, scored in the same 5 folds as v17 / v18.

1. Targets: the four parameters (L0, L1, Ls, c) fitted to each system's
   liquidus by fit_thermo_engine.py.
2. TabPFN v2 + TabICL v2 predict each parameter from the element properties,
   out of fold, trained on both A/B orders (swapping A and B maps L1 -> -L1;
   L0, Ls and c are unchanged).
3. The engine (mpds_models/thermo_engine.py) computes each liquidus from the
   predicted parameters.
Scored alone, and blended with the v18 curve (v17 + 50 % of the 10-component
foundation-model curve; explore_v18.py caches).

    MPDS_DEVICE=cpu python predict_thermo_params.py /home/alibi/lab/processed_v8 \\
        --fits results/v19/engine_fit/fits.json --cache /home/alibi/lab/cache_v18 --out results/v19
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
NAMES = ("L0", "L1", "Ls", "c")


def _engine_curve(args):
    pid, params = args
    from mpds_models.thermo_engine import system_for
    a, b = pid.split("_")
    s = system_for(a, b, t_step=10.0)
    return None if s is None else s.liquidus(*params)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--fits", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from explore_v18 import FMS, Lab
    from mpds_models.height_correction import make_regressor
    lab = Lab(a.processed_dir, a.cache)
    out = Path(a.out)
    fits = json.loads(Path(a.fits).read_text())
    ids = lab.score_ds.pair_ids
    P = np.full((lab.n, 4), np.nan)
    fitted_curves = np.full((lab.n, 101), np.nan)
    for i, p in enumerate(ids):
        if p in fits:
            P[i] = fits[p]["params"]
            fitted_curves[i] = fits[p]["fitted_liquidus"]
    have = np.isfinite(P).all(axis=1)
    print(f"{int(have.sum())} of {lab.n} systems have fitted parameters")

    # reference curves
    v17 = lab.v17()
    fm = np.load(lab.cache / f"pca_fm_10_{lab._key(lab.data, 'extended+mp')}.npy")
    v18_liq = 0.5 * v17[0] + 0.5 * (lab.line + fm)
    v18 = (v18_liq, np.minimum(v17[1], v18_liq))
    ref = lab.report("v17", *v17)
    lab.report("v18 (v17 + 50 % foundation-model curve, 10 components)", *v18, ref)
    fc = np.where(np.isfinite(fitted_curves), fitted_curves, v18[0])
    lab.report("UPPER BOUND: engine with parameters fitted to each system's own data", fc, np.minimum(v18[1], fc), ref)

    # predict the parameters out of fold
    ds = lab.ds("extended+mp")
    pred = np.zeros((lab.n, 4))
    for train, test in lab.folds:
        tr = train[have[train]]
        X = np.concatenate([ds.X[tr], ds.X_swap[tr]]).astype(np.float32)
        for j, name in enumerate(NAMES):
            y = P[tr, j]
            y_sw = -y if name == "L1" else y
            preds = [make_regressor(m).fit(X, np.r_[y, y_sw].astype(np.float32)).predict(ds.X[test].astype(np.float32))
                     for m in FMS]
            pred[test, j] = np.mean(preds, axis=0)
    for j, name in enumerate(NAMES):
        e = np.abs(pred[have, j] - P[have, j])
        print(f"  {name}: predicted out of fold with mean error {e.mean():.2f} (spread of the fitted values {P[have, j].std():.2f})")
    np.save(out / "thermo_params_oof.npy", pred)

    with Pool(a.workers) as pool:
        curves = pool.map(_engine_curve, [(p, pred[i]) for i, p in enumerate(ids)], chunksize=8)
    eng = np.array([c if c is not None else v18[0][i] for i, c in enumerate(curves)], dtype=np.float32)
    np.save(out / "thermo_engine_oof.npy", eng)
    lab.report("engine with PREDICTED parameters, alone", eng, np.minimum(v18[1], eng), ref, groups=True)
    for w in (0.25, 0.5):
        liq = (1 - w) * v18[0] + w * eng
        lab.report(f"v18 blended {w:.0%} with the engine curve", liq, np.minimum(v18[1], liq), ref, groups=True)


if __name__ == "__main__":
    main()
