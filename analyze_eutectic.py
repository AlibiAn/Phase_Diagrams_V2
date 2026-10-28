"""
analyze_eutectic.py — why do bagged tree ensembles (RandomForest, ExtraTrees)
do so well on eutectics, and is their lead over XGBoost real?

LazyPredict scored each model once, on the 138-system validation split,
where one system is worth ~0.7 % accuracy. This script replaces that single
score with repeated cross-validation and adds the diagnostics that explain
*why* one family wins:

1. Repeated stratified k-fold CV on train + val (test stays untouched), every
   model on the same folds: F1, ROC-AUC, PR-AUC (average precision) and
   Brier score for "does a eutectic exist", plus MAE for its composition
   and temperature. Each model is compared with XGBoost (the current M4
   settings) using the Nadeau-Bengio corrected paired t-test, which accounts
   for the overlap between CV training sets. F1 at 0.5 depends on where a
   model's probabilities sit (class-balanced forests sit higher), so
   --tune-threshold adds F1 at the threshold M4 would tune; ROC-AUC and
   PR-AUC need no threshold at all.
2. Overfitting: training-fold score next to held-out score. Boosting can
   fit the training rows almost perfectly; a large gap with a lower
   held-out score is variance, which bagging reduces.
3. Learning curves: held-out ROC-AUC with 20-100 % of each training fold.
   Bagging's advantage, if it is a small-data / noisy-label effect, is
   largest at small sizes and shrinks as data grows.
4. Label noise: held-out accuracy on high-disagreement systems (sources
   disagree) vs the rest.
5. What the models use: permutation importance on held-out folds for
   XGBoost and RandomForest, per feature and grouped by property (the
   __A / __B suffix removed).
6. Baselines for the position regressors: the training mean, and for the
   temperature a fixed fraction of the lower melting point. A model that
   barely beats them is not learning much.

Usage
-----
    python analyze_eutectic.py processed_v6 --out eutectic_study
    python analyze_eutectic.py processed_v6 --out eutectic_study --repeats 3 --quick

Outputs (in --out): cv_folds.csv, summary.csv, regression.csv,
learning_curve.csv, importance.csv, importance_groups.csv, report.md and
charts cv_scores.png, learning_curve.png, importance.png.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold

from mpds_models.base import load_dataset, make_tree_classifier, make_tree_regressor, tune_threshold
from mpds_models.m4_eutectic import EutecticDetectorConfig

# chart colours (reference palette, light surface); one fixed colour per model
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
REFERENCE = "XGBoost (M4)"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def classifiers(scale_pos: float, seed: int, quick: bool) -> dict:
    """name -> factory. XGBoost (M4) uses the M4 settings; the others are what
    the pipeline's backend switch would train, plus LazyPredict-style defaults."""
    c = EutecticDetectorConfig()
    n_forest = 300 if quick else 1000
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    from mpds_models.base import xgb_device_kwargs
    return {
        REFERENCE: lambda: make_tree_classifier(
            "xgboost", n_estimators=c.cls_n_estimators, max_depth=c.cls_max_depth,
            learning_rate=c.cls_learning_rate, scale_pos_weight=scale_pos, seed=seed),
        "XGBoost (defaults)": lambda: xgb.XGBClassifier(
            verbosity=0, random_state=seed, **xgb_device_kwargs()),
        "RandomForest": lambda: make_tree_classifier(
            "random_forest", n_estimators=n_forest, max_depth=None, learning_rate=0, seed=seed),
        "ExtraTrees": lambda: make_tree_classifier(
            "extra_trees", n_estimators=n_forest, max_depth=None, learning_rate=0, seed=seed),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
        "Logistic (linear)": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
    }


def regressors(seed: int, quick: bool) -> dict:
    c = EutecticDetectorConfig()
    n_forest = 300 if quick else 1000
    return {
        REFERENCE: lambda: make_tree_regressor(
            "xgboost", n_estimators=c.reg_n_estimators, max_depth=c.reg_max_depth,
            learning_rate=c.reg_learning_rate, seed=seed),
        "RandomForest": lambda: make_tree_regressor(
            "random_forest", n_estimators=n_forest, max_depth=None, learning_rate=0, seed=seed),
        "ExtraTrees": lambda: make_tree_regressor(
            "extra_trees", n_estimators=n_forest, max_depth=None, learning_rate=0, seed=seed),
    }


def fit(model, X, y, w):
    if hasattr(model, "steps"):             # sklearn Pipeline: route the weights to the last step
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def corrected_ttest(diff: np.ndarray, n_train: int, n_test: int) -> float:
    """Two-sided p-value of the Nadeau-Bengio corrected resampled t-test."""
    k = len(diff)
    var = diff.var(ddof=1)
    if k < 2 or var == 0:
        return float("nan")
    t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def cv_classification(X, y, w, noisy, folds, models_for, tune: bool, seed: int) -> pd.DataFrame:
    """Per-fold scores for every model. With `tune`, also F1 at the threshold
    M4 would pick (tune_threshold on the training fold, 3 inner folds)."""
    rows = []
    for k, (tr, te) in enumerate(folds):
        scale_pos = max((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), 1.0)
        for name, make in models_for(scale_pos).items():
            t0 = time.time()
            m = fit(make(), X[tr], y[tr], w[tr])
            p = m.predict_proba(X[te])[:, 1]
            p_tr = m.predict_proba(X[tr])[:, 1]
            pred = p >= 0.5
            f1_tuned = np.nan
            if tune and not hasattr(m, "steps"):
                t = tune_threshold(make, X[tr], y[tr], w[tr], n_folds=3, seed=seed)
                f1_tuned = f1_score(y[te], p >= (0.5 if t is None else t))
            rows.append(dict(
                fold=k, model=name, seconds=time.time() - t0,
                f1=f1_score(y[te], pred), f1_tuned=f1_tuned, roc_auc=roc_auc_score(y[te], p),
                pr_auc=average_precision_score(y[te], p), brier=brier_score_loss(y[te], p),
                accuracy=float((pred == y[te]).mean()),
                train_f1=f1_score(y[tr], p_tr >= 0.5), train_roc_auc=roc_auc_score(y[tr], p_tr),
                acc_high_disagreement=float((pred == y[te])[noisy[te]].mean()) if noisy[te].any() else np.nan,
                acc_rest=float((pred == y[te])[~noisy[te]].mean()),
            ))
        print(f"  fold {k + 1}/{len(folds)} done")
    return pd.DataFrame(rows)


def summarise(folds_df: pd.DataFrame, n_train: int, n_test: int) -> pd.DataFrame:
    metrics = ["f1", "f1_tuned", "roc_auc", "pr_auc", "brier", "accuracy", "train_f1", "train_roc_auc",
               "acc_high_disagreement", "acc_rest", "seconds"]
    out = folds_df.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    ref = folds_df[folds_df["model"] == REFERENCE].set_index("fold")
    for metric in ("f1", "f1_tuned", "roc_auc", "pr_auc"):
        diffs, ps, wins = [], [], []
        for name in out.index:
            d = (folds_df[folds_df["model"] == name].set_index("fold")[metric] - ref[metric]).to_numpy()
            diffs.append(d.mean())
            ps.append(corrected_ttest(d, n_train, n_test) if name != REFERENCE else np.nan)
            wins.append(float((d > 0).mean()))
        out[f"{metric}_vs_xgb"] = diffs
        out[f"{metric}_p_value"] = ps
        out[f"{metric}_folds_won"] = wins
    out["overfit_gap_f1"] = out["train_f1_mean"] - out["f1_mean"]
    return out


def cv_regression(X, ds_rows, w, Tmin, seed, quick, n_splits) -> pd.DataFrame:
    """Eutectic composition / temperature MAE against simple baselines."""
    rows = []
    for target, y_all in ds_rows.items():
        ok = np.isfinite(y_all)
        Xr, yr, wr, tm = X[ok], y_all[ok], w[ok], Tmin[ok]
        for k, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(Xr)):
            rows.append(dict(target=target, fold=k, model="mean of training rows",
                             mae=float(np.abs(yr[te] - np.average(yr[tr], weights=wr[tr])).mean())))
            if target == "temperature_K":
                known = tr[tm[tr] > 0]
                frac = np.median(yr[known] / tm[known])
                guess = np.where(tm[te] > 0, frac * tm[te], np.average(yr[tr], weights=wr[tr]))
                rows.append(dict(target=target, fold=k, model="fixed fraction of lower Tm",
                                 mae=float(np.abs(yr[te] - guess).mean())))
            for name, make in regressors(seed, quick).items():
                m = make().fit(Xr[tr], yr[tr], sample_weight=wr[tr])
                rows.append(dict(target=target, fold=k, model=name,
                                 mae=float(np.abs(m.predict(Xr[te]) - yr[te]).mean())))
    df = pd.DataFrame(rows)
    return df.groupby(["target", "model"], sort=False)["mae"].agg(["mean", "std"]).reset_index()


def learning_curve(X, y, w, folds, models_for, fractions, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k, (tr, te) in enumerate(folds):
        order = rng.permutation(tr)
        for f in fractions:
            sub = order[: max(int(f * len(tr)), 20)]
            if len(np.unique(y[sub])) < 2:
                continue
            scale_pos = max((len(sub) - y[sub].sum()) / max(y[sub].sum(), 1), 1.0)
            for name, make in models_for(scale_pos).items():
                m = fit(make(), X[sub], y[sub], w[sub])
                rows.append(dict(fold=k, fraction=f, n_train=len(sub), model=name,
                                 roc_auc=roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return pd.DataFrame(rows)


def permutation_importance(X, y, w, folds, models_for, names, seed, n_repeats) -> pd.DataFrame:
    """Drop in held-out ROC-AUC when one feature is shuffled, averaged over folds."""
    from sklearn.inspection import permutation_importance as perm_imp
    rows = []
    for k, (tr, te) in enumerate(folds):
        scale_pos = max((len(tr) - y[tr].sum()) / max(y[tr].sum(), 1), 1.0)
        for name, make in models_for(scale_pos).items():
            m = fit(make(), X[tr], y[tr], w[tr])
            r = perm_imp(m, X[te], y[te], scoring="roc_auc", n_repeats=n_repeats,
                         random_state=seed, n_jobs=-1)
            rows += [dict(fold=k, model=name, feature=f, importance=v)
                     for f, v in zip(names, r.importances_mean)]
    df = pd.DataFrame(rows)
    return df.groupby(["model", "feature"])["importance"].mean().reset_index()


def markdown(df: pd.DataFrame, **kw) -> str:
    try:
        return df.to_markdown(**kw)
    except ImportError:                     # to_markdown needs the tabulate package
        return "```\n" + df.to_string(index=kw.get("index", True)) + "\n```"


def feature_group(name: str) -> str:
    for suffix in ("__A", "__B"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_cv(folds_df, colors, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(folds_df["model"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.42 * len(models) + 1.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, title in zip(axes, ("f1", "roc_auc", "pr_auc"), ("F1 at 0.5", "ROC-AUC", "PR-AUC")):
        for j, m in enumerate(models):
            v = folds_df.loc[folds_df["model"] == m, metric].to_numpy()
            yy = len(models) - 1 - j
            ax.scatter(v, yy + rng.uniform(-0.15, 0.15, len(v)), s=10, color=colors[m], alpha=0.5, lw=0)
            ax.plot([v.mean()] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=2)
        ax.set_title(f"{title}, one dot per fold, bar = mean", fontsize=9, color=INK, loc="left")
        _style(ax, "x")
    axes[0].set_yticks(range(len(models))[::-1], models, fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_learning(lc, colors, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    g = lc.groupby(["model", "fraction"], sort=False).agg(n=("n_train", "mean"), auc=("roc_auc", "mean"),
                                                          sd=("roc_auc", "std")).reset_index()
    for m, sub in g.groupby("model", sort=False):
        ax.plot(sub["n"], sub["auc"], color=colors[m], lw=2, marker="o", ms=5, label=m)
        ax.fill_between(sub["n"], sub["auc"] - sub["sd"], sub["auc"] + sub["sd"],
                        color=colors[m], alpha=0.12, lw=0)
    ax.set_xlabel("training systems", fontsize=9, color=MUTED)
    ax.set_ylabel("held-out ROC-AUC (eutectic exists)", fontsize=9, color=MUTED)
    ax.set_title("Learning curves (band = ±1 std over folds)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(groups, colors, out, top=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pivot = groups.pivot(index="group", columns="model", values="importance").fillna(0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index[:top]]
    models = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pivot) + 1.5))
    y = np.arange(len(pivot))[::-1]
    h = 0.8 / len(models)
    for j, m in enumerate(models):
        ax.barh(y + (len(models) / 2 - j - 0.5) * h, pivot[m], height=h * 0.9, color=colors[m], label=m)
    ax.set_yticks(y, pivot.index, fontsize=8, color=INK)
    ax.set_xlabel("drop in held-out ROC-AUC when shuffled (A and B summed)", fontsize=9, color=MUTED)
    ax.set_title("What decides 'eutectic exists'", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Why do tree ensembles win on eutectics?")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="eutectic_study")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="300 instead of 1000 forest trees, fewer importance repeats")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="also score F1 at the tuned threshold M4 uses (about 4x slower)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.processed_dir)
    rows = np.concatenate([ds.train_idx, ds.val_idx])     # test stays untouched
    X, w = ds.X[rows], ds.weights[rows]
    y = ds.eutectic_exists[rows].astype(int)
    noisy = ds.high_disagreement[rows].astype(bool)
    names = list(ds.feature_names)
    ia, ib = names.index("phys__melting_point__A"), names.index("phys__melting_point__B")
    print(f"{len(rows)} systems (train + val), {y.mean():.0%} with a eutectic, "
          f"{noisy.sum()} high-disagreement")

    folds = list(RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats,
                                         random_state=a.seed).split(X, y))
    n_test = len(folds[0][1])
    n_train = len(rows) - n_test

    def all_models(scale_pos):
        return classifiers(scale_pos, a.seed, a.quick)

    def main_models(scale_pos):
        m = all_models(scale_pos)
        return {k: m[k] for k in (REFERENCE, "RandomForest", "ExtraTrees")}

    colors = {m: SERIES[k] for k, m in enumerate(all_models(1.0))}

    print(f"\n1-2. {a.repeats}x{a.folds}-fold CV, {len(all_models(1.0))} classifiers")
    folds_df = cv_classification(X, y, w, noisy, folds, all_models, a.tune_threshold, a.seed)
    folds_df.to_csv(out / "cv_folds.csv", index=False)
    summary = summarise(folds_df, n_train, n_test)
    summary.to_csv(out / "summary.csv")
    plot_cv(folds_df, colors, out / "cv_scores.png")

    print("\n3. Learning curves (first CV repeat)")
    lc = learning_curve(X, y, w, folds[: a.folds], main_models, (0.2, 0.4, 0.6, 0.8, 1.0), a.seed)
    lc.to_csv(out / "learning_curve.csv", index=False)
    plot_learning(lc, colors, out / "learning_curve.png")

    print("\n5. Permutation importance (first CV repeat)")
    imp = permutation_importance(X, y, w, folds[: a.folds],
                                 lambda s: {k: v for k, v in main_models(s).items() if k != "ExtraTrees"},
                                 names, a.seed, 3 if a.quick else 10)
    imp.to_csv(out / "importance.csv", index=False)
    imp["group"] = imp["feature"].map(feature_group)
    groups = imp.groupby(["model", "group"])["importance"].sum().reset_index()
    groups.to_csv(out / "importance_groups.csv", index=False)
    plot_importance(groups, colors, out / "importance.png")

    print("\n6. Position regressors vs baselines")
    tmin = np.minimum(X[:, ia], X[:, ib])
    eut = ds.eutectic_exists[rows].astype(bool)
    reg = cv_regression(X[eut], {"composition_at_pct": ds.eutectic_comp[rows][eut],
                                 "temperature_K": ds.eutectic_temp[rows][eut]},
                        w[eut], tmin[eut], a.seed, a.quick, a.folds)
    reg.to_csv(out / "regression.csv", index=False)

    # ---- report ----------------------------------------------------------
    cols = ["f1_mean", "f1_std", "f1_vs_xgb", "f1_p_value",
            *(["f1_tuned_mean", "f1_tuned_vs_xgb", "f1_tuned_p_value"] if a.tune_threshold else []),
            "roc_auc_mean", "roc_auc_vs_xgb",
            "roc_auc_p_value", "pr_auc_mean", "brier_mean", "train_f1_mean", "overfit_gap_f1",
            "acc_high_disagreement_mean", "acc_rest_mean", "seconds_mean"]
    fmt = lambda v: f"{v:.3f}"
    lc_tab = lc.groupby(["model", "n_train"], sort=False)["roc_auc"].mean().unstack()
    top = (groups.sort_values("importance", ascending=False).groupby("model").head(8)
           .sort_values(["model", "importance"], ascending=[True, False]))
    report = [
        "# Eutectic study",
        f"{len(rows)} systems (train + val), {a.repeats}×{a.folds}-fold CV; "
        f"p-values: Nadeau–Bengio corrected paired t-test against {REFERENCE}.",
        "", "## Classification (eutectic exists)", markdown(summary[cols], floatfmt=".3f"),
        "", "## Learning curve (held-out ROC-AUC by training size)", markdown(lc_tab, floatfmt=".3f"),
        "", "## Position regressors (MAE)", markdown(reg, index=False, floatfmt=".1f"),
        "", "## Most important property groups", markdown(top, index=False, floatfmt=".4f"),
    ]
    (out / "report.md").write_text("\n".join(report) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n=== Classification, mean over folds ===")
        print(summary[cols].to_string(float_format=fmt))
        print("\n=== Learning curve: held-out ROC-AUC by training size ===")
        print(lc_tab.to_string(float_format=fmt))
        print("\n=== Position regressors, MAE ===")
        print(reg.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print("\n=== Top property groups (permutation importance, ROC-AUC drop) ===")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved report.md, tables and charts in {out}/")


if __name__ == "__main__":
    main()
