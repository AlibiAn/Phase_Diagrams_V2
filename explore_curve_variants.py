"""
explore_curve_variants.py — the experiments behind the v14 decisions, each in
5-fold cross-validation against the v14 curve model: what was tried and lost
(so nobody repeats it without a new reason), and what limits the error.

Every experiment prints liquidus / solidus MAE (mean over the folds) and in how
many folds it beats the reference. All models train on all systems of the
training folds, with A/B swap augmentation and the median over the trees.

Experiments (--run, any number)
-------------------------------
  settings         tree settings and knot spacing around the v14 values
  feature-groups   v6 features plus each group of new features on its own
  learning-curve   v14 trained on 25 / 50 / 75 / 100 % of the training systems
  relative-target  liquidus deviation divided by the mean melting point
  physics-base     ideal / regular-solution liquidus as the base curve instead
                   of the straight line between the melting points
  analogue         curves borrowed from systems that share one element,
                   weighted by how alike the other two elements behave
  per-point        LightGBM on one row per (system, composition) with
                   composition-specific features (about 15 minutes)
  gap-stacking     out-of-fold miscibility-gap probability as an extra feature
  oracle           upper bounds: features that give away the true diagram
                   type, gap flag, eutectic point or the curve's mean height
  height-model     a separate model for the mean height of the liquidus above
                   the melting-point line, used to shift the predicted curve
  compound-energies  computed formation energies of the A-B compounds
                   (Materials Project; needs --mp-file, see the caveat in
                   mpds_models/compound_features.py)

Usage
-----
    python explore_curve_variants.py $D --run settings feature-groups 2>&1 | tee results/v14/explore_v6.txt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from cross_validate import folds
from mpds_models.base import curve_metrics, load_dataset, mask_clipped_tops
from mpds_models.element_features import interaction_parameter, regular_solution_liquidus
from mpds_models.tree_curves import _fill_curves

KNOTS = np.arange(0, 101, 2)


class Harness:
    """Filled training targets (as tree_curves builds them) and helpers to
    fit ExtraTrees on any feature matrix and score out-of-fold curves."""

    def __init__(self, processed_dir: str, k: int = 5):
        ds = load_dataset(processed_dir, features="extended")
        mask_clipped_tops(ds)
        self.ds, self.n, self.k = ds, ds.n_systems, k
        names = list(ds.feature_names)
        self.n_v6 = names.index("el__MendeleevNumber__A")          # first extended column
        self.names = names
        self.A = np.array([p.split("_")[0] for p in ds.pair_ids])
        self.B = np.array([p.split("_")[1] for p in ds.pair_ids])
        self.tm_a = ds.X[:, names.index("phys__melting_point__A")].astype(float)
        self.tm_b = ds.X[:, names.index("phys__melting_point__B")].astype(float)
        self.frac = ds.comp_grid.astype(float) / 100.0
        self.line = (1 - self.frac)[None] * self.tm_a[:, None] + self.frac[None] * self.tm_b[:, None]
        seen = ds.T_liquidus_mask.copy()
        seen[:, [0, -1]] = False
        self.liq = _fill_curves(ds.T_liquidus, seen, self.frac, self.tm_a, self.tm_b, 3)
        self.dev = self.liq - self.line
        s_seen = np.isfinite(ds.T_solidus)
        s_seen[:, [0, -1]] = False
        self.sdev = _fill_curves(ds.T_solidus, s_seen, self.frac, self.tm_a, self.tm_b, 1) - self.line
        self.ok_liq, self.ok_sol = np.isfinite(self.dev).all(1), np.isfinite(self.sdev).all(1)
        self.folds = folds(self.n, k)
        self.fold_of = np.zeros(self.n, dtype=int)
        for f, (_, te) in enumerate(self.folds):
            self.fold_of[te] = f
        self.X, self.Xs = ds.X, ds.X_swap

    def columns(self, pred) -> np.ndarray:
        return np.array([j for j, n in enumerate(self.names) if pred(j, n)])

    @staticmethod
    def forest(n=300, leaf=2, mf=1.0, seed=0):
        return ExtraTreesRegressor(n_estimators=n, min_samples_leaf=leaf, max_features=mf,
                                   n_jobs=-1, random_state=seed)

    @staticmethod
    def to_grid(at_knots: np.ndarray, knots=KNOTS) -> np.ndarray:
        return np.stack([np.interp(np.arange(101), knots, r) for r in at_knots])

    def fit_predict(self, X, Xs, train, test, Y, ok, make=None, agg="median", knots=KNOTS):
        rows = train[ok[train]]
        w = self.ds.weights
        model = (make or self.forest)().fit(
            np.concatenate([X[rows], Xs[rows]]),
            np.concatenate([Y[rows][:, knots], Y[rows][:, ::-1][:, knots]]),
            sample_weight=np.concatenate([w[rows], w[rows]]))
        Xt = np.asarray(X[test], dtype=np.float32)
        if agg == "median":
            p = np.median([t.predict(Xt).reshape(len(Xt), -1) for t in model.estimators_], axis=0)
        else:
            p = model.predict(Xt)
        return self.to_grid(p, knots)

    def oof(self, X=None, Xs=None, *, liquidus=True, solidus=True, fold_list=None, **kw):
        """Out-of-fold (liquidus, solidus); features default to the extended set."""
        X = self.X if X is None else X
        Xs = self.Xs if Xs is None else Xs
        liq = np.zeros((self.n, 101), dtype=np.float32)
        sol = np.zeros_like(liq)
        for train, test in (fold_list or self.folds):
            if liquidus:
                liq[test] = self.line[test] + self.fit_predict(X, Xs, train, test, self.dev, self.ok_liq, **kw)
            if solidus:
                sol[test] = self.line[test] + self.fit_predict(X, Xs, train, test, self.sdev, self.ok_sol, **kw)
        return liq, sol

    def by_fold(self, liq, sol) -> np.ndarray:
        sol = np.minimum(sol, liq)
        return np.array([[curve_metrics(self.ds, np.flatnonzero(self.fold_of == f), liq[self.fold_of == f],
                                        sol[self.fold_of == f])[m]
                          for m in ("liquidus_MAE_K", "solidus_MAE_K")] for f in range(self.k)])

    def report(self, label, liq, sol, ref=None, *, liquidus=True, solidus=True) -> np.ndarray:
        f = self.by_fold(liq, sol)
        s = f"  {label:<52}"
        if liquidus:
            s += f" liquidus {f[:, 0].mean():6.1f}"
        if solidus:
            s += f"  solidus {f[:, 1].mean():6.1f}"
        if ref is not None:
            s += "   folds won:"
            if liquidus:
                s += f" liquidus {(f[:, 0] < ref[:, 0]).sum()}/{self.k}"
            if solidus:
                s += f" solidus {(f[:, 1] < ref[:, 1]).sum()}/{self.k}"
        print(s, flush=True)
        return f


# ─────────────────────────────────────────────────────────────────────────────

def settings(h: Harness, base, ref) -> None:
    for label, kw in (("min_samples_leaf 1", dict(make=lambda: h.forest(leaf=1))),
                      ("min_samples_leaf 3", dict(make=lambda: h.forest(leaf=3))),
                      ("max_features 0.5", dict(make=lambda: h.forest(mf=0.5))),
                      ("max_features 0.33", dict(make=lambda: h.forest(mf=0.33))),
                      ("600 trees", dict(make=lambda: h.forest(n=600))),
                      ("mean over the trees", dict(agg="mean")),
                      ("knots every 1 at%", dict(knots=np.arange(101)))):
        h.report(label, *h.oof(**kw), ref)


def feature_groups(h: Harness, base, ref) -> None:
    v6 = lambda j, n: j < h.n_v6
    for label, pred in (("v6 features only (v13 features)", v6),
                        ("v6 + element properties (el__)", lambda j, n: v6(j, n) or n.startswith("el__")),
                        ("v6 + Miedema enthalpies (mied__)", lambda j, n: v6(j, n) or n.startswith("mied__")),
                        ("v6 + regular-solution liquidus (phys__ new)",
                         lambda j, n: v6(j, n) or (j >= h.n_v6 and n.startswith("phys__"))),
                        ("new features only", lambda j, n: j >= h.n_v6)):
        c = h.columns(pred)
        h.report(f"{label} [{len(c)}]", *h.oof(h.X[:, c], h.Xs[:, c]), ref)


def learning_curve(h: Harness, base, ref) -> None:
    rng = np.random.default_rng(1)
    for share in (0.25, 0.5, 0.75):
        fl = [(np.sort(rng.choice(tr, int(len(tr) * share), replace=False)), te) for tr, te in h.folds]
        h.report(f"{share:.0%} of the training systems ({len(fl[0][0])})", *h.oof(fold_list=fl))
    h.report(f"100% of the training systems ({len(h.folds[0][0])})", *base)


def relative_target(h: Harness, base, ref) -> None:
    tmean = ((h.tm_a + h.tm_b) / 2)[:, None]
    liq = np.zeros((h.n, 101), dtype=np.float32)
    for train, test in h.folds:
        liq[test] = h.line[test] + tmean[test] * h.fit_predict(h.X, h.Xs, train, test, h.dev / tmean, h.ok_liq)
    h.report("target = deviation / mean melting point", liq, base[1], ref, solidus=False)


def physics_base(h: Harness, base, ref) -> None:
    m = h.ds.T_liquidus_mask
    print(f"  (straight line between the melting points alone: liquidus MAE "
          f"{np.abs(h.line - h.ds.T_liquidus)[m].mean():.1f} K)")
    for kind in ("ideal", "regular"):
        curve = np.zeros((h.n, 101))
        for i in range(h.n):
            om = interaction_parameter(h.A[i], h.B[i]) if kind == "regular" else 0.0
            curve[i] = regular_solution_liquidus(h.A[i], h.B[i], om if np.isfinite(om) else 0.0, h.frac)
        print(f"  ({kind}-solution curve alone: liquidus MAE {np.abs(curve - h.ds.T_liquidus)[m].mean():.1f} K)")
        liq = np.zeros((h.n, 101), dtype=np.float32)
        for train, test in h.folds:
            liq[test] = curve[test] + h.fit_predict(h.X, h.Xs, train, test, h.liq - curve, h.ok_liq)
        h.report(f"base = {kind}-solution liquidus + trees", liq, base[1], ref, solidus=False)
        h.report(f"  averaged with the v14 liquidus", (liq + base[0]) / 2, base[1], ref, solidus=False)


def analogue(h: Harness, base, ref) -> None:
    """A-B from the curves of A-B' and A'-B, weighted by exp(-d / tau), where
    d(B, B') is the mean distance between the curves of B and B' with the
    partners they share in the training folds (never using A-B itself)."""
    els = sorted(set(h.A) | set(h.B))
    ei = {e: i for i, e in enumerate(els)}
    ne, nk = len(els), len(KNOTS)
    C = np.full((ne, ne, nk), np.nan, dtype=np.float32)       # C[e, x]: curve of e-x with e on the left
    for i in range(h.n):
        if h.ok_liq[i]:
            a, b = ei[h.A[i]], ei[h.B[i]]
            C[a, b], C[b, a] = h.dev[i][KNOTS], h.dev[i][::-1][KNOTS]
    for tau in (20.0, 40.0):
        liq = np.zeros((h.n, 101), dtype=np.float32)
        for train, test in h.folds:
            usable = np.zeros((ne, ne), dtype=bool)
            for i in train:
                if h.ok_liq[i]:
                    usable[ei[h.A[i]], ei[h.B[i]]] = usable[ei[h.B[i]], ei[h.A[i]]] = True
            Cm = np.where(usable[:, :, None], C, np.nan)
            with np.errstate(all="ignore"):
                D = np.nanmean(np.abs(Cm[:, None] - Cm[None]), axis=3)       # D[e, e', partner]
            for i in test:
                a, b = ei[h.A[i]], ei[h.B[i]]
                curves, weights = [], []
                for p, q in ((a, b), (b, a)):              # p is shared, q is replaced by q2
                    Dq = D[q].copy()
                    Dq[:, p] = np.nan
                    with np.errstate(all="ignore"):
                        d = np.where(np.isfinite(Dq).sum(1) >= 3, np.nanmean(Dq, axis=1), np.nan)
                    for q2 in np.flatnonzero(usable[p] & np.isfinite(d)):
                        if q2 != q:
                            curves.append(Cm[p, q2] if p == a else Cm[p, q2][::-1])
                            weights.append(np.exp(-d[q2] / tau))
                if curves:
                    w = np.array(weights)
                    liq[i] = h.line[i] + h.to_grid(((np.array(curves) * w[:, None]).sum(0) / w.sum())[None])[0]
                else:
                    liq[i] = h.line[i]
        h.report(f"analogue curves, tau = {tau:g} K", liq, base[1], ref, solidus=False)
        h.report("  averaged with the v14 liquidus", (liq + base[0]) / 2, base[1], ref, solidus=False)


def per_point(h: Harness, base, ref) -> None:
    import lightgbm as lgb
    dev_phys = {}
    for kind in ("ideal", "regular"):
        out = np.zeros((h.n, 101), dtype=np.float32)
        for i in range(h.n):
            om = interaction_parameter(h.A[i], h.B[i]) if kind == "regular" else 0.0
            out[i] = regular_solution_liquidus(h.A[i], h.B[i], om if np.isfinite(om) else 0.0, h.frac) - h.line[i]
        dev_phys[kind] = out
    xk, nk = h.frac[KNOTS], len(KNOTS)

    def rows(idx, swapped):
        X = (h.Xs if swapped else h.X)[idx]
        flip = (lambda a: a[:, ::-1]) if swapped else (lambda a: a)
        ta, tb = (h.tm_b, h.tm_a) if swapped else (h.tm_a, h.tm_b)
        ln = (1 - xk)[None] * ta[idx][:, None] + xk[None] * tb[idx][:, None]
        pi, pr = flip(dev_phys["ideal"][idx])[:, KNOTS], flip(dev_phys["regular"][idx])[:, KNOTS]
        extra = np.stack([np.broadcast_to(xk, ln.shape), np.broadcast_to(np.minimum(xk, 1 - xk), ln.shape),
                          ln, pi, pr, ln + pr], axis=-1).astype(np.float32)
        d = h.ds.T_liquidus[idx] - h.line[idx]
        seen = h.ds.T_liquidus_mask[idx].copy()
        d[:, [0, -1]], seen[:, [0, -1]] = 0.0, True               # ends pinned to Tm
        return (np.concatenate([np.repeat(X, nk, 0), extra.reshape(-1, extra.shape[-1])], axis=1),
                flip(d)[:, KNOTS].reshape(-1), flip(seen)[:, KNOTS].reshape(-1), np.repeat(idx, nk))

    liq = np.zeros((h.n, 101), dtype=np.float32)
    for train, test in h.folds:
        parts = [rows(train, s) for s in (False, True)]
        Xt = np.concatenate([p[0][p[2]] for p in parts])
        yt = np.concatenate([p[1][p[2]] for p in parts])
        wt = np.concatenate([h.ds.weights[p[3][p[2]]] for p in parts])
        model = lgb.LGBMRegressor(objective="l1", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
                                  feature_fraction=0.3, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                                  n_estimators=1500, n_jobs=-1, verbose=-1).fit(Xt, yt, sample_weight=wt)
        p = (model.predict(rows(test, False)[0]).reshape(len(test), -1)
             + model.predict(rows(test, True)[0]).reshape(len(test), -1)[:, ::-1]) / 2
        p[:, [0, -1]] = 0.0
        liq[test] = h.line[test] + h.to_grid(p)
    h.report("LightGBM (L1) on one row per composition point", liq, base[1], ref, solidus=False)
    h.report("  0.3 x LightGBM + 0.7 x v14 liquidus", 0.3 * liq + 0.7 * base[0], base[1], ref, solidus=False)


def gap_stacking(h: Harness, base, ref) -> None:
    """Add P(miscibility gap) from a classifier as a feature. Training rows get
    out-of-fold probabilities from an inner 5-fold split of the training folds."""
    y = h.ds.miscibility_gap.astype(int)
    make = lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                        n_jobs=-1, random_state=0)
    liq = np.zeros((h.n, 101), dtype=np.float32)
    sol = np.zeros_like(liq)
    for train, test in h.folds:
        p = np.zeros(h.n)
        for itr, ite in folds(len(train), 5, seed=1):
            p[train[ite]] = make().fit(np.concatenate([h.X[train[itr]], h.Xs[train[itr]]]),
                                       np.tile(y[train[itr]], 2)).predict_proba(h.X[train[ite]])[:, 1]
        p[test] = make().fit(np.concatenate([h.X[train], h.Xs[train]]),
                             np.tile(y[train], 2)).predict_proba(h.X[test])[:, 1]
        X2, X2s = np.column_stack([h.X, p]), np.column_stack([h.Xs, p])
        liq[test] = h.line[test] + h.fit_predict(X2, X2s, train, test, h.dev, h.ok_liq)
        sol[test] = h.line[test] + h.fit_predict(X2, X2s, train, test, h.sdev, h.ok_sol)
    h.report("+ out-of-fold P(miscibility gap) as a feature", liq, sol, ref)


def oracle(h: Harness, base, ref) -> None:
    """Upper bounds: give the trees the truth as extra features."""
    ds = h.ds
    seen = ds.T_liquidus_mask
    err = base[0] - ds.T_liquidus
    shift = np.array([err[i][seen[i]].mean() if seen[i].any() else 0.0 for i in range(h.n)])
    h.report("v14 liquidus shifted by each system's true mean error", base[0] - shift[:, None], base[1], ref,
             solidus=False)
    topo = np.eye(ds.n_topology_classes, dtype=np.float32)[ds.topology]
    gap = ds.miscibility_gap.astype(np.float32)[:, None]
    eut = np.column_stack([ds.eutectic_exists, np.nan_to_num(ds.eutectic_comp, nan=-1.0),
                           np.nan_to_num(ds.eutectic_temp, nan=-1.0)]).astype(np.float32)
    eut_s = eut.copy()
    eut_s[:, 1] = np.where(eut[:, 1] >= 0, 100.0 - eut[:, 1], -1.0)
    for label, add, add_s in (("+ TRUE diagram type", topo, topo), ("+ TRUE miscibility-gap flag", gap, gap),
                              ("+ TRUE eutectic point (deepest)", eut, eut_s),
                              ("+ all three", np.column_stack([topo, gap, eut]),
                               np.column_stack([topo, gap, eut_s]))):
        h.report(label, *h.oof(np.column_stack([h.X, add]), np.column_stack([h.Xs, add_s])), ref)


def height_model(h: Harness, base, ref) -> None:
    """A dedicated model for one number per system, the mean height of the
    liquidus above the melting-point line, and the curve shifted to match it."""
    target, ok = h.dev.mean(axis=1), h.ok_liq
    implied = (base[0] - h.line).mean(axis=1)
    print(f"  mean height of the liquidus above the line: MAE of the v14 curve's own value "
          f"{np.abs(implied - target)[ok].mean():.1f} K (predicting 0: {np.abs(target[ok]).mean():.1f} K)")
    window = 4 * h.frac * (1 - h.frac)
    window = window / window.mean()                       # shifts the middle, keeps the ends at Tm

    def lgbm():
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=15, min_data_in_leaf=10,
                                 feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, n_jobs=-1,
                                 verbose=-1)
    for name, make in (("ExtraTrees", lambda: ExtraTreesRegressor(500, min_samples_leaf=2, n_jobs=-1,
                                                                 random_state=0)),
                       ("LightGBM", lgbm)):
        p = np.zeros(h.n)
        for train, test in h.folds:
            rows = train[ok[train]]
            model = make().fit(np.concatenate([h.X[rows], h.Xs[rows]]), np.tile(target[rows], 2))
            p[test] = (model.predict(h.X[test]) + model.predict(h.Xs[test])) / 2
        print(f"  {name} trained on that number alone: MAE {np.abs(p - target)[ok].mean():.1f} K")
        liq = base[0] + 0.5 * (p - implied)[:, None] * window[None]
        h.report(f"  v14 liquidus shifted half-way to the {name} height", liq, base[1], ref, solidus=False)


MP_FILE = None


def compound_energies(h: Harness, base, ref) -> None:
    from mpds_models.compound_features import build_features
    if MP_FILE is None:
        raise SystemExit("compound-energies needs --mp-file")
    X, Xs, names = build_features(h.ds.pair_ids, MP_FILE)
    print(f"  ({int((X[:, names.index('cmp__n_entries')] > 0).sum())} of {h.n} systems have computed "
          f"compounds; {int((X[:, names.index('cmp__n_stable')] > 0).sum())} have a stable one)")
    h.report(f"+ compound formation energies [{len(names)} features]",
             *h.oof(np.column_stack([h.X, X]), np.column_stack([h.Xs, Xs])), ref)
    keep = [j for j, n in enumerate(names) if n.startswith("cmp__hull") or n in ("cmp__min_eform", "cmp__x_at_min")]
    h.report("+ energies only (no counts of known compounds)",
             *h.oof(np.column_stack([h.X, X[:, keep]]), np.column_stack([h.Xs, Xs[:, keep]])), ref)


EXPERIMENTS = {"compound-energies": compound_energies, "height-model": height_model, "settings": settings, "feature-groups": feature_groups, "learning-curve": learning_curve,
               "relative-target": relative_target, "physics-base": physics_base, "analogue": analogue,
               "per-point": per_point, "gap-stacking": gap_stacking, "oracle": oracle}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed_dir")
    ap.add_argument("--run", nargs="+", choices=list(EXPERIMENTS), required=True)
    ap.add_argument("--mp-file", default=None,
                    help="mp_nostruct_20181018.json.gz for the compound-energies experiment")
    a = ap.parse_args()
    global MP_FILE
    MP_FILE = a.mp_file
    h = Harness(a.processed_dir)
    print(f"{a.processed_dir}: {h.n} systems, {int(h.ds.T_liquidus_mask.sum()):,} liquidus points, "
          f"{h.k}-fold cross-validation; MAE in K")
    base = h.oof()
    print("Reference:")
    ref = h.report("v14 curve model (extended features, direct solidus)", *base)
    for name in a.run:
        t0 = time.time()
        print(f"\n[{name}]")
        EXPERIMENTS[name](h, base, ref)
        print(f"  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
