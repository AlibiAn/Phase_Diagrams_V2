"""
engine_learn.py — two ways to make the "predict the thermodynamic parameters,
compute the diagram" route robust to the engine's sensitivity (results/v19:
parameters predicted to ~20 % of their spread still give a 150 K liquidus).

A. Average curves, not parameters. For each system, compute the engine curve
   for the predicted parameters plus K residual vectors (prediction minus fit)
   drawn from systems in the OTHER folds, and take the median curve. This is
   the expected diagram under the predictor's own uncertainty, which removes
   the jumps a single slightly-wrong parameter set produces.

B. Regularised refit ("EM"). Fitted parameters are partly interchangeable, so
   the fit picks one of many equally good sets, some of them unpredictable.
   Refit each system with
       loss = mean |engine(theta) - liquidus| + lam * sum_k |theta_k - theta_oof_k| / scale_k
   (theta_oof = the out-of-fold prediction from the previous round), so that
   among nearly equal fits the predictable one is chosen; retrain the
   predictor on the new targets; repeat. Screening caveat: theta_oof of a
   training system was itself predicted by models that saw the test fold's
   fitted parameters (a second-order leak); a win must be re-run with the
   refit nested inside each fold.

    MPDS_DEVICE=cpu python engine_learn.py /home/alibi/lab/processed_v8 --fits results/v19/engine_fit/fits.json \\
        --params results/v19/thermo_params_oof.npy --cache /home/alibi/lab/cache_v18 --out results/v19
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
NAMES = ("L0", "L1", "Ls", "c")
BOUNDS = [(-200.0, 200.0), (-150.0, 150.0), (-40.0, 200.0), (0.0, 2.0)]
LO, HI = np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS])
SCALE = np.array([76.0, 37.0, 73.0, 0.52])


def _system(pid):
    from mpds_models.thermo_engine import system_for
    a, b = pid.split("_")
    return system_for(a, b, t_step=10.0)


def ensemble_curve(args):
    """Median engine curve over parameter sets theta_pred + residuals."""
    pid, thetas = args
    s = _system(pid)
    if s is None:
        return None
    curves = np.array([s.liquidus(*np.clip(t, LO, HI)) for t in thetas])
    return np.median(curves, axis=0)


def refit(args):
    pid, T, mask, start, prior, lam = args
    s = _system(pid)
    if s is None or mask.sum() < 10:
        return None
    prior = np.clip(prior, LO, HI)

    def loss(p):
        return float(np.abs(s.liquidus(*p) - T)[mask].mean()) + lam * float(np.sum(np.abs(np.asarray(p) - prior) / SCALE))

    best = None
    for x0 in (np.clip(start, LO, HI), prior, 0.5 * (np.clip(start, LO, HI) + prior)):
        r = minimize(loss, x0, method="Powell", bounds=BOUNDS, options={"maxfev": 600, "xtol": 0.2, "ftol": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    fit = float(np.abs(s.liquidus(*best.x) - T)[mask].mean())
    return best.x, fit


def curves_from(pool, ids, params, fallback):
    res = pool.map(ensemble_curve, [(p, params[i][None]) for i, p in enumerate(ids)], chunksize=8)
    return np.array([r if r is not None else fallback[i] for i, r in enumerate(res)], dtype=np.float32)


def predict_oof(lab, P, have):
    from explore_v18 import FMS
    from mpds_models.height_correction import make_regressor
    ds = lab.ds("extended+mp")
    pred = np.zeros((lab.n, 4))
    for train, test in lab.folds:
        tr = train[have[train]]
        X = np.concatenate([ds.X[tr], ds.X_swap[tr]]).astype(np.float32)
        for j, name in enumerate(NAMES):
            y = P[tr, j]
            y_sw = -y if name == "L1" else y
            pred[test, j] = np.mean([make_regressor(m).fit(X, np.r_[y, y_sw].astype(np.float32))
                                     .predict(ds.X[test].astype(np.float32)) for m in FMS], axis=0)
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--fits", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from explore_v18 import Lab
    from mpds_models.base import mask_clipped_tops
    lab = Lab(a.processed_dir, a.cache)
    out = Path(a.out)
    ids = list(lab.score_ds.pair_ids)
    fits = json.loads(Path(a.fits).read_text())
    P = np.full((lab.n, 4), np.nan)
    for i, p in enumerate(ids):
        if p in fits:
            P[i] = fits[p]["params"]
    have = np.isfinite(P).all(axis=1)
    pred = np.load(a.params)

    v17 = lab.v17()
    fm = np.load(lab.cache / f"pca_fm_10_{lab._key(lab.data, 'extended+mp')}.npy")
    v18_liq = 0.5 * v17[0] + 0.5 * (lab.line + fm)
    v18 = (v18_liq, np.minimum(v17[1], v18_liq))
    ref = lab.report("v17", *v17)
    lab.report("v18", *v18, ref, groups=True)

    def score(name, eng, groups=False):
        lab.report(name, eng, np.minimum(v18[1], eng), ref, groups=groups)
        for w in (0.25, 0.5):
            liq = (1 - w) * v18_liq + w * eng
            lab.report(f"  v18 blended {w:.0%} with it", liq, np.minimum(v18[1], liq), ref)

    fold_of = np.zeros(lab.n, int)
    for f, (_, test) in enumerate(lab.folds):
        fold_of[test] = f
    rng = np.random.default_rng(0)
    with Pool(a.workers) as pool:
        # A: median curve under the predictor's uncertainty
        print("\nA. median engine curve over predicted parameters + out-of-fold residuals")
        resid = pred - P
        for shrink in (1.0, 0.5):
            jobs = []
            for i, p in enumerate(ids):
                pool_i = np.flatnonzero(have & (fold_of != fold_of[i]))
                draw = resid[rng.choice(pool_i, a.samples - 1, replace=False)]
                jobs.append((p, np.vstack([pred[i], pred[i] + shrink * draw])))
            res = pool.map(ensemble_curve, jobs, chunksize=4)
            eng = np.array([r if r is not None else v18_liq[i] for i, r in enumerate(res)], dtype=np.float32)
            np.save(out / f"engine_ensemble_shrink{shrink:g}.npy", eng)
            score(f"engine median over {a.samples} parameter sets (residuals x {shrink:g})", eng, groups=True)

        if a.skip_b:
            return
        # B: regularised refit, then retrain the predictor
        ds = lab.score_ds
        mask_clipped_tops(ds)
        cur, prior = P.copy(), pred.copy()
        for rnd in range(1, a.rounds + 1):
            jobs = [(p, ds.T_liquidus[i].astype(float), ds.T_liquidus_mask[i].copy(), cur[i], prior[i], a.lam)
                    for i, p in enumerate(ids) if have[i]]
            res = pool.map(refit, jobs, chunksize=4)
            hi = np.flatnonzero(have)
            pts = ds.T_liquidus_mask[hi].sum(axis=1)
            fit_err = np.full(len(hi), np.nan)
            for k, (i, r) in enumerate(zip(hi, res)):
                if r is not None:
                    cur[i], fit_err[k] = r
            ok = np.isfinite(fit_err)
            print(f"\nB round {rnd}: refit with prior weight {a.lam:g}; fit MAE {np.sum(fit_err[ok] * pts[ok]) / pts[ok].sum():.1f} K "
                  f"(unregularised 57.8 K); parameters moved by "
                  + ", ".join(f"{n} {np.abs(cur[have, j] - P[have, j]).mean():.1f}" for j, n in enumerate(NAMES)))
            prior = predict_oof(lab, cur, have)
            print("  predicted out of fold with mean error "
                  + ", ".join(f"{n} {np.abs(prior[have, j] - cur[have, j]).mean():.2f} (spread {cur[have, j].std():.2f})"
                              for j, n in enumerate(NAMES)))
            np.save(out / f"thermo_params_em{rnd}.npy", prior)
            eng = curves_from(pool, ids, prior, v18_liq)
            np.save(out / f"thermo_engine_em{rnd}.npy", eng)
            score(f"engine with parameters predicted after refit round {rnd}", eng, groups=True)


if __name__ == "__main__":
    main()
