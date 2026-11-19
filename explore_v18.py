"""
explore_v18.py — the experiments after v17, each scored the way v17 is: the
ExtraTrees curve model (v16 settings, trained with the extra systems), then
the liquidus height correction by TabPFN v2 + TabICL v2 (strength 0.5),
in the same 5 folds over the 1,380 systems of processed_v8.

Every experiment prints the liquidus (and solidus) MAE and the folds in
which it beats v17 (the reference row). Out-of-fold curves and heights are
cached in --cache, so a rerun only computes what is new.

Experiments (--run)
-------------------
  features    compound-stability inputs: the 2018 Materials Project snapshot
              (mp, v16/v17), ORCHID's OQMD hulls (oqmd), ORCHID's Materials
              Project 2026 hulls (mp26), alone and combined
  rs          the equilibrium liquidus of a regular solution with pure solids
              (miscibility gap and monotectic line included) as inputs
  plateau-fix lift likely two-liquid systems to a monotectic level predicted by the
              foundation models, over the regular-solution model's gap range
  tta         average each prediction with the one for the swapped pair (B-A), mirrored
  scale       curve deviations learnt in units of |Tm(A) - Tm(B)| + 300 K
  pca-fm      the foundation models predict the leading principal components
              of the whole deviation curve, not only its mean height; the tree
              curve is blended with their curve
  height      variants of the height correction: strength, extra systems in
              the height training, a solidus correction
  plateau     upper bound: monotectic systems drawn as a flat line at their
              TRUE monotectic temperature over their TRUE two-liquid range
  mpds-open   extra training systems from the keyless MPDS download

    MPDS_DEVICE=cpu python explore_v18.py /home/alibi/lab/processed_v8 --run features scale \\
        --cache /home/alibi/lab/cache_v18 2>&1 | tee results/v18/explore.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from cross_validate import folds
from mpds_models.base import (add_extra_curve_systems, curve_metrics, group_rows, load_dataset, mask_clipped_tops,
                              system_quality)
from mpds_models.height_correction import make_regressor
from mpds_models.tree_curves import TreeCurveConfig, TreeCurveModel, _fill_curves

warnings.filterwarnings("ignore")
V6 = "/home/alibi/lab/processed_v6"
EXTRA = "/home/alibi/lab/extra_v8_all"
FMS = ("tabpfn", "tabicl")


class Lab:
    def __init__(self, data: str, cache: str):
        self.data, self.cache = data, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._ds: dict = {}
        self.quality = system_quality(load_dataset(V6))
        ref = self.ds("extended+mp")
        self.n = ref.n_systems
        self.folds = folds(self.n, 5)
        self.fold_of = np.zeros(self.n, dtype=int)
        for f, (_, te) in enumerate(self.folds):
            self.fold_of[te] = f
        names = list(ref.feature_names)
        self.tm_a = ref.X[:, names.index("phys__melting_point__A")].astype(float)
        self.tm_b = ref.X[:, names.index("phys__melting_point__B")].astype(float)
        self.frac = ref.comp_grid.astype(float) / 100.0
        self.line = (1 - self.frac)[None] * self.tm_a[:, None] + self.frac[None] * self.tm_b[:, None]
        seen = ref.T_liquidus_mask.copy()
        seen[:, [0, -1]] = False
        self.filled = _fill_curves(ref.T_liquidus, seen, self.frac, self.tm_a, self.tm_b, 3)
        self.height = (self.filled - self.line).mean(axis=1)
        self.ok = np.isfinite(self.height)
        w = self.frac * (1 - self.frac)
        self.window = w / w.mean()
        self.score_ds = ref

    def ds(self, features: str):
        if features not in self._ds:
            d = load_dataset(self.data, features=features)
            mask_clipped_tops(d)
            self._ds[features] = d
        return self._ds[features]

    def _key(self, *parts) -> str:
        return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ---- the curve model ------------------------------------------------
    def curves(self, features="extended+mp", extra=EXTRA, **cfg):
        """Out-of-fold (liquidus, solidus) of the v16 curve model with these options."""
        f = self.cache / f"curves_{self._key(self.data, features, extra, cfg)}.npz"
        if f.exists():
            o = np.load(f)
            return o["liq"], o["sol"]
        ds = self.ds(features)
        fit = ds
        if extra:
            fit = add_extra_curve_systems(ds, extra)
            mask_clipped_tops(fit)
        liq = np.zeros((self.n, 101), np.float32)
        sol = np.zeros_like(liq)
        for train, test in self.folds:
            c = TreeCurveConfig(backend="extra_trees", aggregate="median", swap_augment=True, solidus="direct", **cfg)
            m = TreeCurveModel(c).fit(replace(fit, train_idx=np.concatenate([train, fit.extra_idx]), test_idx=test))
            liq[test], sol[test] = m.predict(ds.X[test])
        np.savez(f, liq=liq, sol=sol)
        return liq, sol

    # ---- foundation-model targets ----------------------------------------
    def fm_oof(self, target: np.ndarray, target_swap: np.ndarray, features="extended+mp", model="tabpfn",
               rows: np.ndarray | None = None, tag="height", extra_X=None):
        """Out-of-fold predictions of a per-system target by a foundation model, trained on
        both A/B orders. `rows`: systems allowed in training (default: target finite).
        `extra_X`: (X, X_swap, y, y_swap) of extra training rows."""
        f = self.cache / f"fm_{tag}_{model}_{self._key(self.data, features, tag, rows is None, extra_X is not None)}.npy"
        if f.exists():
            return np.load(f)
        ds = self.ds(features)
        use = np.isfinite(target) if rows is None else rows & np.isfinite(target)
        p = np.zeros(self.n)
        for train, test in self.folds:
            tr = train[use[train]]
            X = np.concatenate([ds.X[tr], ds.X_swap[tr]])
            y = np.concatenate([target[tr], target_swap[tr]])
            if extra_X is not None:
                X, y = np.concatenate([X, extra_X[0], extra_X[1]]), np.concatenate([y, extra_X[2], extra_X[3]])
            m = make_regressor(model).fit(X.astype(np.float32), y.astype(np.float32))
            p[test] = m.predict(ds.X[test].astype(np.float32))
        np.save(f, p)
        return p

    def height_fm(self, features="extended+mp", extra_heights=False):
        extra_X = None
        if extra_heights:
            ex = add_extra_curve_systems(self.ds(features), EXTRA)
            idx = ex.extra_idx
            ta, tb = ex.X[idx, list(ex.feature_names).index("phys__melting_point__A")], \
                ex.X[idx, list(ex.feature_names).index("phys__melting_point__B")]
            line = (1 - self.frac)[None] * ta[:, None] + self.frac[None] * tb[:, None]
            seen = ex.T_liquidus_mask[idx].copy()
            seen[:, [0, -1]] = False
            h = (_fill_curves(ex.T_liquidus[idx], seen, self.frac, ta, tb, 3) - line).mean(axis=1)
            good = np.isfinite(h) & (ta > 0) & (tb > 0)
            extra_X = (ex.X[idx][good], ex.X_swap[idx][good], h[good], h[good])
        return np.mean([self.fm_oof(self.height, self.height, features, m, tag="height" + ("+x" if extra_heights else ""),
                                    extra_X=extra_X) for m in FMS], axis=0)

    def shift(self, liq, sol, h, strength=0.5):
        implied = (liq - self.line).mean(axis=1)
        liq2 = liq + strength * (h - implied)[:, None] * self.window[None]
        return liq2.astype(np.float32), np.minimum(sol, liq2).astype(np.float32)

    # ---- scoring ------------------------------------------------------------
    def by_fold(self, liq, sol):
        d = self.score_ds
        return np.array([[curve_metrics(d, te, liq[te], sol[te])[k] for k in ("liquidus_MAE_K", "solidus_MAE_K")]
                         for _, te in self.folds])

    def report(self, label, liq, sol, ref=None, groups=False):
        f = self.by_fold(liq, sol)
        s = f"  {label:<60} liquidus {f[:, 0].mean():6.1f}  solidus {f[:, 1].mean():6.1f}"
        if ref is not None:
            s += f"   folds won: {(f[:, 0] < ref[:, 0]).sum()}/5, {(f[:, 1] < ref[:, 1]).sum()}/5"
        print(s, flush=True)
        if groups:
            d = self.score_ds
            g = group_rows(self.quality, np.arange(self.n))
            print("      " + "  ".join(f"{k}: {curve_metrics(d, i, liq[i], sol[i])['liquidus_MAE_K']:.1f}"
                                      for k, i in g.items()))
        return f

    def v17(self, features="extended+mp", **cfg):
        liq, sol = self.curves(features, **cfg)
        return self.shift(liq, sol, self.height_fm(features))


# ─────────────────────────────────────────────────────────────────────────────

def exp_features(lab, ref):
    for fs in ("extended", "extended+oqmd", "extended+mp26", "extended+mp+oqmd", "extended+oqmd+mp26",
               "extended+mp+oqmd+mp26"):
        t0 = time.time()
        liq, sol = lab.curves(fs)
        lab.report(f"curve model only, {fs}", liq, sol)
        lab.report(f"  + height correction ({time.time() - t0:.0f} s)", *lab.v17(fs), ref, groups=True)


def exp_rs(lab, ref):
    """The equilibrium regular-solution liquidus (regular_solution.py) as inputs."""
    for fs in ("extended+mp+rs", "extended+oqmd+rs", "extended+mp+oqmd+mp26+rs", "extended+rs"):
        liq, sol = lab.curves(fs)
        lab.report(f"curve model only, {fs}", liq, sol)
        lab.report("  + height correction", *lab.v17(fs), ref, groups=True)


def exp_tta(lab, ref, features="extended+mp"):
    """Average each curve with the curve predicted for B-A, mirrored back."""
    f = lab.cache / f"curves_tta_{lab._key(lab.data, features, EXTRA)}.npz"
    ds = lab.ds(features)
    if f.exists():
        o = np.load(f); liq, sol, liq_s, sol_s = o["liq"], o["sol"], o["liq_s"], o["sol_s"]
    else:
        fit = add_extra_curve_systems(ds, EXTRA)
        mask_clipped_tops(fit)
        liq = np.zeros((lab.n, 101), np.float32); sol = np.zeros_like(liq)
        liq_s = np.zeros_like(liq); sol_s = np.zeros_like(liq)
        for train, test in lab.folds:
            c = TreeCurveConfig(backend="extra_trees", aggregate="median", swap_augment=True, solidus="direct")
            m = TreeCurveModel(c).fit(replace(fit, train_idx=np.concatenate([train, fit.extra_idx]), test_idx=test))
            liq[test], sol[test] = m.predict(ds.X[test])
            a_, b_ = m.predict(ds.X_swap[test])
            liq_s[test], sol_s[test] = a_[:, ::-1], b_[:, ::-1]
        np.savez(f, liq=liq, sol=sol, liq_s=liq_s, sol_s=sol_s)
    lab.report("curve model, one A/B order (as v16)", liq, sol)
    la, sa = 0.5 * (liq + liq_s), np.minimum(0.5 * (sol + sol_s), 0.5 * (liq + liq_s))
    lab.report("curve model, both orders averaged", la, sa)
    lab.report("  + height correction", *lab.shift(la, sa, lab.height_fm(features)), ref, groups=True)


def _fm_classifier_oof(lab, y, features, model, tag):
    from compare_foundation_models import make
    f = lab.cache / f"fmclf_{tag}_{model}_{lab._key(lab.data, features, tag)}.npy"
    if f.exists():
        return np.load(f)
    ds = lab.ds(features)
    p = np.zeros(lab.n)
    for train, test in lab.folds:
        m = make(model, "classification").fit(ds.X[train], y[train])
        p[test] = m.predict_proba(ds.X[test])[:, 1]
    np.save(f, p)
    return p


def _plateaus(lab):
    """(level, mask) of the flat liquidus stretch (>= 10 at%) of each two-liquid system; NaN / False if none."""
    d = lab.score_ds
    T = np.where(d.T_liquidus_mask, d.T_liquidus, np.nan)
    level = np.full(lab.n, np.nan)
    mask = np.zeros(T.shape, dtype=bool)
    for i in np.flatnonzero(d.miscibility_gap):
        t = T[i]
        if np.isfinite(t).sum() < 20:
            continue
        run = np.r_[False, np.abs(np.diff(t)) < 1.0] & np.isfinite(t)
        if run.sum() >= 10:
            mask[i], level[i] = run, np.nanmedian(t[run])
    return level, mask


def _rs_gap(lab, features):
    """Two-liquid composition mask of the regular-solution model, per system (cached)."""
    from mpds_models.element_features import element_table, interaction_parameter
    from mpds_models.regular_solution import equilibrium_state
    f = lab.cache / "rs_gap_masks.npy"
    if f.exists():
        return np.load(f)
    t = element_table()
    out = np.zeros((lab.n, 101), dtype=bool)
    for i, pid in enumerate(lab.score_ds.pair_ids):
        a, b = pid.split("_")
        om = interaction_parameter(a, b)
        vals = [t.at[a, "MeltingT"], t.at[b, "MeltingT"], t.at[a, "HeatFusion"], t.at[b, "HeatFusion"]]
        if om > 0 and all(np.isfinite(v) and v > 0 for v in vals):
            out[i] = equilibrium_state(*vals, om, t_step=5.0)[1]
    np.save(f, out)
    return out


def exp_plateau_fix(lab, ref, features="extended+mp"):
    """Lift the liquidus of likely two-liquid systems to a predicted monotectic level."""
    d = lab.score_ds
    liq, sol = lab.v17(features)
    p_gap = np.mean([_fm_classifier_oof(lab, d.miscibility_gap.astype(int), features, m, "gap") for m in FMS], axis=0)
    level, true_mask = _plateaus(lab)
    tmax = np.maximum(lab.tm_a, lab.tm_b)
    has = np.isfinite(level)
    target = level - tmax
    lhat = tmax + np.mean([lab.fm_oof(target, target, features, m, rows=has, tag="plateau_level") for m in FMS], axis=0)
    err = np.abs(lhat - level)[has]
    implied = np.array([np.median(liq[i, true_mask[i]]) if has[i] else np.nan for i in range(lab.n)])
    print(f"    plateau level: foundation models {np.mean(err):.0f} K mean error (median {np.median(err):.0f}), "
          f"v17 curve's own level {np.nanmean(np.abs(implied - level)[has]):.0f} K, on {int(has.sum())} systems")
    rs = _rs_gap(lab, features)
    print(f"    regular-solution model predicts a two-liquid range in {int(rs.any(axis=1).sum())} systems")
    for thr in (0.5, 0.7):
        for rng in ("rs", "true"):
            for beta in (0.5, 1.0):
                liq2 = liq.copy()
                for i in np.flatnonzero(p_gap >= thr):
                    m = rs[i] if rng == "rs" else true_mask[i]
                    if m.sum() < 3:
                        continue
                    lifted = np.where(m, np.maximum(liq[i], lhat[i]) if beta else liq[i], liq[i])
                    flat = np.where(m, lhat[i], liq[i])
                    liq2[i] = liq[i] + beta * (flat - liq[i]) if beta == 1.0 else liq[i] + beta * (lifted - liq[i])
                tag = "TRUE range (upper bound)" if rng == "true" else "regular-solution range"
                lab.report(f"P(gap) >= {thr}, {tag}, {'flat at' if beta == 1.0 else 'half-way up to'} predicted level",
                           liq2, np.minimum(sol, liq2), ref)


def exp_scale(lab, ref):
    for off in (300.0, 100.0, 1000.0):
        liq, sol = lab.curves("extended+mp", target_scale="tm_gap", scale_offset_K=off)
        lab.report(f"scaled by |dTm| + {off:.0f} K: curve model only", liq, sol)
        lab.report("  + height correction", *lab.shift(liq, sol, lab.height_fm("extended+mp")), ref)


def exp_pca_fm(lab, ref, features="extended+mp", k=6):
    """Foundation models predict the first k principal components of the deviation curve."""
    from sklearn.decomposition import PCA
    dev = lab.filled - lab.line
    ok = np.isfinite(dev).all(axis=1)
    knots = np.arange(0, 101, 2)
    liq0, sol0 = lab.curves(features)
    cached = lab.cache / f"pca_fm_{k}_{lab._key(lab.data, features)}.npy"
    fm = np.load(cached) if cached.exists() else np.zeros((lab.n, 101))
    for f, (train, test) in enumerate([] if cached.exists() else lab.folds):
        tr = train[ok[train]]
        Y = np.concatenate([dev[tr][:, knots], dev[tr][:, ::-1][:, knots]])
        pca = PCA(n_components=k).fit(Y)
        S, S_swap = pca.transform(dev[tr][:, knots]), pca.transform(dev[tr][:, ::-1][:, knots])
        scores = np.zeros((len(test), k))
        ds = lab.ds(features)
        for c in range(k):
            preds = []
            for model in FMS:
                m = make_regressor(model).fit(np.concatenate([ds.X[tr], ds.X_swap[tr]]).astype(np.float32),
                                              np.concatenate([S[:, c], S_swap[:, c]]).astype(np.float32))
                preds.append(m.predict(ds.X[test].astype(np.float32)))
            scores[:, c] = np.mean(preds, axis=0)
        at_knots = pca.inverse_transform(scores)
        fm[test] = np.stack([np.interp(np.arange(101), knots, r) for r in at_knots])
        print(f"    fold {f}: explained variance of {k} components {pca.explained_variance_ratio_.sum():.2f}", flush=True)
    np.save(cached, fm)
    v17 = lab.v17(features)
    for a in (0.25, 0.5, 0.75, 1.0):
        liq = (1 - a) * liq0 + a * (lab.line + fm)
        lab.report(f"tree curve blended {a:.0%} with the {k}-component foundation-model curve", liq,
                   np.minimum(sol0, liq), ref)
    v17 = lab.v17(features)
    for a in (0.25, 0.5):
        liq = (1 - a) * v17[0] + a * (lab.line + fm)
        lab.report(f"v17 curve blended {a:.0%} with it", liq, np.minimum(v17[1], liq), ref)


def exp_pca_k(lab, ref):
    for k in (3, 10, 16):
        print(f"  -- {k} components")
        exp_pca_fm(lab, ref, k=k)


def exp_pca_hulls(lab, ref):
    """The v18 recipe (10 components, v17 curve blended 50 %) with all compound-stability inputs."""
    exp_pca_fm(lab, ref, features="extended+mp+oqmd+mp26", k=10)


def exp_height(lab, ref, features="extended+mp"):
    liq, sol = lab.curves(features)
    h = lab.height_fm(features)
    for s in (0.3, 0.7, 1.0):
        lab.report(f"height correction strength {s}", *lab.shift(liq, sol, h, s), ref)
    hx = lab.height_fm(features, extra_heights=True)
    lab.report("height models also trained on the 404 extra systems", *lab.shift(liq, sol, hx), ref)
    # solidus: same idea on the solidus height
    seen = lab.score_ds.T_solidus.copy()
    sfill = _fill_curves(seen, np.isfinite(seen), lab.frac, lab.tm_a, lab.tm_b, 1)
    sh = (sfill - lab.line).mean(axis=1)
    hs = np.mean([lab.fm_oof(sh, sh, features, m, tag="solidus_height") for m in FMS], axis=0)
    l2, s2 = lab.shift(liq, sol, h)
    implied = (s2 - lab.line).mean(axis=1)
    for s in (0.5, 1.0):
        sol3 = np.minimum(s2 + s * (hs - implied)[:, None] * lab.window[None], l2)
        lab.report(f"v17 + solidus height correction, strength {s}", l2, sol3, ref)


def exp_plateau(lab, ref, features="extended+mp"):
    """Upper bound: flatten monotectic systems at their true monotectic level."""
    d = lab.score_ds
    liq, sol = lab.v17(features)
    T = np.where(d.T_liquidus_mask, d.T_liquidus, np.nan)
    flat = np.zeros_like(T, dtype=bool)
    for i in np.flatnonzero(d.miscibility_gap):
        t = T[i]
        if np.isfinite(t).sum() < 20:
            continue
        dt = np.abs(np.diff(t))
        run = np.r_[False, dt < 1.0] & np.isfinite(t)
        if run.sum() >= 10:
            flat[i] = run
    level = np.array([np.nanmedian(T[i][flat[i]]) if flat[i].any() else np.nan for i in range(lab.n)])
    liq2 = liq.copy()
    for i in np.flatnonzero(np.isfinite(level)):
        liq2[i, flat[i]] = level[i]
    print(f"    {int(np.isfinite(level).sum())} monotectic systems with a flat plateau of >= 10 at%")
    lab.report("ORACLE: plateau drawn at the true level over the true range", liq2, np.minimum(sol, liq2), ref)
    liq3 = liq.copy()
    for i in np.flatnonzero(np.isfinite(level)):
        liq3[i, flat[i]] = np.median(liq[i, flat[i]])
    lab.report("ORACLE range only: flat at the predicted median over the true range", liq3, np.minimum(sol, liq3), ref)


def exp_mpds_open(lab, ref, features="extended+mp", extra="/home/alibi/lab/extra_v8_open"):
    liq, sol = lab.curves(features, extra=extra)
    lab.report("curve model with extras + keyless MPDS systems", liq, sol)
    lab.report("  + height correction", *lab.shift(liq, sol, lab.height_fm(features)), ref, groups=True)


EXPERIMENTS = {"pca-hulls": exp_pca_hulls, "pca-k": exp_pca_k, "plateau-fix": exp_plateau_fix, "tta": exp_tta, "rs": exp_rs, "features": exp_features, "scale": exp_scale, "pca-fm": exp_pca_fm, "height": exp_height,
               "plateau": exp_plateau, "mpds-open": exp_mpds_open}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--run", nargs="+", choices=list(EXPERIMENTS), required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    lab = Lab(a.processed_dir, a.cache)
    print(f"{a.processed_dir}: {lab.n} systems, 5-fold CV; MAE in K; folds won = liquidus, solidus against v17")
    print("Reference:")
    ref = lab.report("v17 (v16 curve model + height correction 0.5)", *lab.v17(), groups=True)
    for name in a.run:
        t0 = time.time()
        print(f"\n[{name}]", flush=True)
        EXPERIMENTS[name](lab, ref)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
