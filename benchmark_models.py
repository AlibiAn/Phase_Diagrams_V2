"""
benchmark_models.py — try ~30-40 scikit-learn style models on every prediction
task in the pipeline with LazyPredict, and rank them.

This is a SCREENING tool. It answers "which model families look promising for
our data?" It does not change train_all.py. Once a winner is clear for a task,
swap it into that task's model file (m4_eutectic.py, m6_topology.py, ...).

What it does
------------
1. Loads the processed dataset exactly like train_all.py (same features,
   same pair-disjoint train / val / test split).
2. Builds one table of (X, y) per task:
      M6_topology            5-class classifier
      M5_miscibility_gap     yes / no
      M4_eutectic_exists     yes / no
      M4_eutectic_comp_pct   regression (eutectic systems only)
      M4_eutectic_temp_K     regression (eutectic systems only)
      M1_phase_<name>        yes / no for the --m1-phases most balanced phases
      M2_liquidus_at_<x>     regression: liquidus temperature at x at%
      M3_solidus_at_50       regression: solidus temperature at 50 at%
   LazyPredict predicts one number per system, so the 101-point curves are
   represented by their value at a few compositions.
3. Fits every LazyPredict model on TRAIN and scores it on VAL (not test: the
   test split stays untouched for the final comparison).
4. Optionally (--model-dir) also scores YOUR trained M4/M5/M6 models on the
   same rows, so you can see where they rank.

Outputs (in --out)
------------------
  <task>.csv          full ranking for that task
  <task>.png          top models as a bar chart (XGBoost / your model marked)
  summary.csv         one row per task: best model, its score, XGBoost's rank

Usage
-----
    pip install lazypredict
    python benchmark_models.py processed_v6 --out benchmark_v6
    python benchmark_models.py processed_v6 --out benchmark_v6 --model-dir models_v6
    python benchmark_models.py processed_v6 --out quick --tasks M6_topology M5_miscibility_gap
    python benchmark_models.py processed_v6 --out full --models all     # every model

Speed
-----
LazyPredict trains its models one after another on one core. Here each model
runs through LazyPredict in its own worker process, so all CPU cores are busy
at once. XGBoost / LightGBM / CatBoost go to the GPU when CUDA is found
(--gpu auto). By default (--models fast) models that duplicate another model,
don't suit this data, or are slow and beaten by a faster equivalent are
skipped; the reasons are listed in SKIP_CLASSIFIERS / SKIP_REGRESSORS below.

Ranking metric: macro F1 for classifiers (every class counts equally, which
matters for rare topology classes), MAE in the target's units for regressors.
"""
from __future__ import annotations

import argparse
import inspect
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_absolute_error

from mpds_models.base import load_dataset

warnings.filterwarnings("ignore")

MIN_ROWS = 20          # skip a task with fewer training rows than this
LIQ_POINTS = (25, 50, 75)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# chart colours (reference palette, light surface)
C_OTHER, C_XGB, C_OURS = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


# ─────────────────────────────────────────────────────────────────────────────
# Metrics passed to LazyPredict as its custom metric
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(ds, n_phases: int) -> dict:
    """task name -> dict(kind='clf'|'reg', y=array over all systems, rows=bool mask)."""
    n = ds.n_systems
    all_rows = np.ones(n, dtype=bool)
    tasks = {}

    topo_names = np.asarray(ds.topology_classes, dtype=object)[ds.topology.astype(int)]
    tasks["M6_topology"] = dict(kind="clf", y=topo_names, rows=all_rows)
    tasks["M5_miscibility_gap"] = dict(kind="clf", y=ds.miscibility_gap.astype(int), rows=all_rows)
    tasks["M4_eutectic_exists"] = dict(kind="clf", y=ds.eutectic_exists.astype(int), rows=all_rows)

    eut = ds.eutectic_exists.astype(bool)
    comp = np.asarray(ds.eutectic_comp, dtype=float)
    temp = np.asarray(ds.eutectic_temp, dtype=float)
    tasks["M4_eutectic_comp_pct"] = dict(kind="reg", y=comp, rows=eut & np.isfinite(comp))
    tasks["M4_eutectic_temp_K"] = dict(kind="reg", y=temp, rows=eut & np.isfinite(temp))

    # M1: the most balanced phases (a phase present in 2% or 98% of systems
    # teaches little); skip the always-present Liquid and the <rare> bucket.
    Y = ds.Y_phases[ds.train_idx]
    rate = Y.mean(axis=0)
    order = np.argsort(np.abs(rate - 0.5))
    picked = 0
    for j in order:
        name = str(ds.phase_vocab[j])
        if name in ("Liquid", "<rare>") or not (0.05 <= rate[j] <= 0.95):
            continue
        safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        tasks[f"M1_phase_{safe}"] = dict(kind="clf", y=ds.Y_phases[:, j].astype(int), rows=all_rows)
        picked += 1
        if picked >= n_phases:
            break

    grid = np.asarray(ds.comp_grid)
    for x in LIQ_POINTS:
        g = int(np.argmin(np.abs(grid - x)))
        y = ds.T_liquidus[:, g]
        tasks[f"M2_liquidus_at_{x}"] = dict(kind="reg", y=y, rows=ds.T_liquidus_mask[:, g] & np.isfinite(y))
    g = int(np.argmin(np.abs(grid - 50)))
    y = ds.T_solidus[:, g]
    tasks["M3_solidus_at_50"] = dict(kind="reg", y=y, rows=np.isfinite(y))
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Your own trained models, scored on the same rows (optional)
# ─────────────────────────────────────────────────────────────────────────────

def score_own_models(model_dir: Path, ds, X_eval: np.ndarray, eval_idx: np.ndarray, tasks: dict) -> dict:
    """task name -> (label, score) for the models train_all.py saved."""
    import joblib
    out = {}

    def rows_of(task):
        return tasks[task]["rows"][eval_idx]

    try:
        m6 = joblib.load(model_dir / "m6_topology_solo.joblib")
        pred = np.asarray(ds.topology_classes, dtype=object)[m6.predict(X_eval).astype(int)]
        r = rows_of("M6_topology")
        out["M6_topology"] = ("Our M6 (XGBoost)", macro_f1(tasks["M6_topology"]["y"][eval_idx][r], pred[r]))
    except Exception as e:
        print(f"  (could not score M6: {e})")
    try:
        m5 = joblib.load(model_dir / "m5_miscibility.joblib")
        thr = float(getattr(getattr(m5, "cfg", None), "decision_threshold", 0.5))
        pred = (m5.predict_proba(X_eval) >= thr).astype(int)
        r = rows_of("M5_miscibility_gap")
        out["M5_miscibility_gap"] = ("Our M5 (XGBoost)", macro_f1(tasks["M5_miscibility_gap"]["y"][eval_idx][r], pred[r]))
    except Exception as e:
        print(f"  (could not score M5: {e})")
    try:
        m4 = joblib.load(model_dir / "m4_eutectic.joblib")
        p = m4.predict(X_eval)
        r = rows_of("M4_eutectic_exists")
        out["M4_eutectic_exists"] = ("Our M4 (XGBoost)",
                                     macro_f1(tasks["M4_eutectic_exists"]["y"][eval_idx][r], np.asarray(p["exists"]).astype(int)[r]))
        for task, key in (("M4_eutectic_comp_pct", "comp"), ("M4_eutectic_temp_K", "temp")):
            r = rows_of(task)
            pv = np.asarray(p[key], dtype=float)[r]
            yt = tasks[task]["y"][eval_idx][r]
            ok = np.isfinite(pv)          # M4 gives NaN where it predicts "no eutectic"
            if ok.any():
                out[task] = (f"Our M4 (XGBoost, {ok.sum()}/{ok.size} rows)", mae(yt[ok], pv[ok]))
    except Exception as e:
        print(f"  (could not score M4: {e})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Running LazyPredict
# ─────────────────────────────────────────────────────────────────────────────

# Models we skip by default, and why. "--models all" runs everything anyway.
SKIP_CLASSIFIERS = {
    # duplicates of a model we keep
    "RidgeClassifier": "same as RidgeClassifierCV",
    "LinearSVC": "linear like LogisticRegression; often fails to converge",
    "CalibratedClassifierCV": "LinearSVC fitted 5 times",
    "NuSVC": "same as SVC; often infeasible on imbalanced classes",
    "ExtraTreeClassifier": "a single tree; ExtraTreesClassifier is the real model",
    # wrong assumptions for our data
    "BernoulliNB": "turns every continuous feature into 0/1",
    "CategoricalNB": "needs integer categorical features",
    "QuadraticDiscriminantAnalysis": "unstable with ~200 correlated features",
    "LabelPropagation": "semi-supervised; all our rows are labelled",
    "LabelSpreading": "semi-supervised; all our rows are labelled",
    "SelfTrainingClassifier": "semi-supervised; all our rows are labelled",
    # weak and noisy
    "Perceptron": "weak linear model, results jump between runs",
    "PassiveAggressiveClassifier": "weak linear model, results jump between runs",
    "SGDClassifier": "weak linear model, results jump between runs",
    # slow wrappers with no meaning at default settings
    "StackingClassifier": "slow; a default stack of default models",
    "TunedThresholdClassifierCV": "slow cross-validated wrapper",
    "FixedThresholdClassifier": "wrapper, not a model",
}
SKIP_REGRESSORS = {
    "LinearRegression": "unstable with ~200 correlated features; RidgeCV covers it",
    "TransformedTargetRegressor": "plain LinearRegression",
    "Ridge": "same as RidgeCV", "Lasso": "same as LassoCV", "ElasticNet": "same as ElasticNetCV",
    "Lars": "same family as LassoCV, numerically fragile", "LarsCV": "same family as LassoCV",
    "LassoLars": "same family as LassoCV", "LassoLarsCV": "same family as LassoCV",
    "LassoLarsIC": "same family as LassoCV",
    "OrthogonalMatchingPursuit": "sparse linear, covered by LassoCV",
    "OrthogonalMatchingPursuitCV": "sparse linear, covered by LassoCV",
    "KernelRidge": "default linear kernel = Ridge",
    "LinearSVR": "linear, covered by RidgeCV and SVR", "NuSVR": "same as SVR",
    "ExtraTreeRegressor": "a single tree; ExtraTreesRegressor is the real model",
    "GammaRegressor": "assumes positive skewed targets; fails on 0 at%",
    "PoissonRegressor": "assumes count targets", "TweedieRegressor": "assumes count-like targets",
    "RANSACRegressor": "random outlier rejection, results jump between runs",
    "PassiveAggressiveRegressor": "weak, results jump between runs",
    "SGDRegressor": "weak, results jump between runs",
    "HuberRegressor": "slow solver, robust linear covered by others",
    "QuantileRegressor": "the slowest model in the list; a linear median fit",
    "GaussianProcessRegressor": "cubic cost in rows; default kernel suits 200 features poorly",
    "GradientBoostingRegressor": "single-core and slow; HistGradientBoosting/XGBoost/LightGBM do the same faster",
}
# Good models LazyPredict leaves out of its classifier list; we add them back.
EXTRA = {
    "clf": ["sklearn.ensemble.HistGradientBoostingClassifier", "sklearn.neural_network.MLPClassifier"],
    "reg": [],
}
GPU_LIBS = ("xgboost", "lightgbm", "catboost")


def model_list(kind: str, which: str) -> list:
    """List of (name, class) to run for this task kind."""
    import importlib
    from lazypredict.Supervised import CLASSIFIERS, REGRESSORS
    base = list(CLASSIFIERS if kind == "clf" else REGRESSORS)
    names = {n for n, _ in base}
    for path in EXTRA[kind]:
        mod, cls = path.rsplit(".", 1)
        if cls not in names:
            base.append((cls, getattr(importlib.import_module(mod), cls)))
    if which == "all":
        return base
    skip = SKIP_CLASSIFIERS if kind == "clf" else SKIP_REGRESSORS
    return [(n, c) for n, c in base if n not in skip]


def is_gpu_model(cls) -> bool:
    return any(lib in (getattr(cls, "__module__", "") or "") for lib in GPU_LIBS)


def _fit_one(kind, model_cls, X_tr, X_ev, y_tr, y_ev, use_gpu, seed):
    """Fit ONE model through LazyPredict. Runs inside a worker process."""
    import warnings as _w
    _w.filterwarnings("ignore")
    from lazypredict.Supervised import LazyClassifier, LazyRegressor
    cls = LazyClassifier if kind == "clf" else LazyRegressor
    key = "classifiers" if kind == "clf" else "regressors"
    kw = dict(verbose=0, ignore_warnings=True, predictions=False, random_state=seed,
              custom_metric=macro_f1 if kind == "clf" else mae, use_gpu=use_gpu)
    kw[key] = [model_cls]
    ok = set(inspect.signature(cls.__init__).parameters)
    lazy = cls(**{k: v for k, v in kw.items() if k in ok})
    try:
        res = lazy.fit(X_tr, X_ev, y_tr, y_ev)
    except Exception:
        return None
    df = res[0] if isinstance(res, tuple) else res
    return df if df is not None and len(df) else None


def run_task(name, task, Xdf, train_idx, eval_idx, models, n_jobs, gpu):
    tr = train_idx[task["rows"][train_idx]]
    ev = eval_idx[task["rows"][eval_idx]]
    y = task["y"]
    if len(tr) < MIN_ROWS or len(ev) < 5:
        print(f"  skip {name}: only {len(tr)} train / {len(ev)} eval rows")
        return None
    if task["kind"] == "clf" and len(np.unique(y[tr])) < 2:
        print(f"  skip {name}: training rows contain a single class")
        return None
    if task["kind"] == "clf":
        # numeric labels 0..k-1: some models (XGBoost, LightGBM) refuse text labels
        classes = {c: i for i, c in enumerate(sorted(set(y), key=str))}
        y = np.array([classes[v] for v in y])

    X_tr, X_ev, y_tr, y_ev = Xdf.iloc[tr], Xdf.iloc[ev], y[tr], y[ev]
    gpu_models = [(n, c) for n, c in models if gpu and is_gpu_model(c)]
    cpu_models = [(n, c) for n, c in models if (n, c) not in gpu_models]

    t0 = time.time()
    frames = []
    # CPU models: one model per worker, all cores busy, 1 thread per worker
    # (so 24 workers don't each start 24 threads).
    from joblib import Parallel, delayed
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        frames += Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_one)(task["kind"], c, X_tr, X_ev, y_tr, y_ev, False, 0)
            for _, c in cpu_models)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # GPU models: one at a time on the GPU
    for _, c in gpu_models:
        frames.append(_fit_one(task["kind"], c, X_tr, X_ev, y_tr, y_ev, True, 0))

    frames = [f for f in frames if f is not None]
    if not frames:
        print(f"  {name}: every model failed")
        return None
    scores = pd.concat(frames)
    failed = len(models) - len(scores)

    metric_col = macro_f1.__name__ if task["kind"] == "clf" else mae.__name__
    if metric_col not in scores.columns:                     # older/newer naming
        cand = [c for c in scores.columns if c.lower().replace(" ", "_") == metric_col]
        metric_col = cand[0] if cand else ("F1 Score" if task["kind"] == "clf" else "RMSE")
    asc = task["kind"] == "reg"
    scores = scores.sort_values(metric_col, ascending=asc)
    scores.attrs.update(metric=metric_col, ascending=asc, n_train=len(tr), n_eval=len(ev),
                        seconds=time.time() - t0, failed=failed)
    return scores


def plot_task(name, scores, own, out_path, top=15):
    metric, asc = scores.attrs["metric"], scores.attrs["ascending"]
    s = scores[metric].dropna().head(top)
    labels, values = list(s.index), list(s.values)
    colors = [C_XGB if "XGB" in lab else C_OTHER for lab in labels]
    if own is not None:
        labels.append(own[0]); values.append(own[1]); colors.append(C_OURS)
        order = np.argsort(values) if asc else np.argsort(values)[::-1]
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]
        colors = [colors[i] for i in order]
    labels = [f"{l}  (current)" if "XGB" in l and not l.startswith("Our") else l for l in labels]

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {v:.3g}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8, color=INK)
    unit = "macro F1 (higher is better)" if not asc else "MAE (lower is better)"
    ax.set_xlabel(unit, fontsize=9, color=MUTED)
    ax.set_title(f"{name}  ·  {scores.attrs['n_train']} train / {scores.attrs['n_eval']} val systems",
                 fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlim(0, max(values) * 1.15 if values else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Screen many models per task with LazyPredict.")
    ap.add_argument("processed_dir")
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--model-dir", help="also score your trained M4/M5/M6 on the same rows")
    ap.add_argument("--tasks", nargs="+", help="only these tasks (default: all)")
    ap.add_argument("--m1-phases", type=int, default=5, help="how many phases to test for M1")
    ap.add_argument("--eval-split", choices=("val", "test"), default="val",
                    help="score on val (default) so test stays untouched")
    ap.add_argument("--models", choices=("fast", "all"), default="fast",
                    help="'fast' skips duplicate, unsuitable and slow models (default)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="CPU worker processes (-1 = all cores)")
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                    help="run XGBoost / LightGBM / CatBoost on the GPU (auto = if CUDA is found)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(a.processed_dir)
    Xdf = pd.DataFrame(ds.X, columns=list(ds.feature_names))
    train_idx = np.asarray(ds.train_idx)
    eval_idx = np.asarray(ds.val_idx if a.eval_split == "val" else ds.test_idx)
    if a.eval_split == "test":
        print("NOTE: choosing models on the test split leaks it; use val for model selection.")

    tasks = all_tasks = build_tasks(ds, a.m1_phases)
    if a.tasks:
        unknown = set(a.tasks) - set(tasks)
        if unknown:
            raise SystemExit(f"unknown task(s) {sorted(unknown)}; available: {sorted(tasks)}")
        tasks = {k: v for k, v in tasks.items() if k in a.tasks}

    own = {}
    if a.model_dir:
        own = score_own_models(Path(a.model_dir), ds, ds.X[eval_idx], eval_idx, all_tasks)

    from lazypredict.config import is_gpu_available
    gpu = a.gpu == "on" or (a.gpu == "auto" and is_gpu_available())
    n_cores = os.cpu_count() or 1
    print(f"CPU workers: {n_cores if a.n_jobs == -1 else a.n_jobs} | GPU for boosters: {'yes' if gpu else 'no'}")
    lists = {k: model_list(k, a.models) for k in ("clf", "reg")}
    print(f"Models per task: {len(lists['clf'])} classifiers, {len(lists['reg'])} regressors ({a.models})")

    summary = []
    t_all = time.time()
    for name, task in tasks.items():
        print(f"\n=== {name} ({'classification' if task['kind'] == 'clf' else 'regression'}) ===")
        scores = run_task(name, task, Xdf, train_idx, eval_idx, lists[task['kind']], a.n_jobs, gpu)
        if scores is None:
            continue
        scores.to_csv(out / f"{name}.csv")
        plot_task(name, scores, own.get(name), out / f"{name}.png")

        metric = scores.attrs["metric"]
        best_name = scores.index[0]
        xgb = [i for i, m in enumerate(scores.index) if "XGB" in m]
        row = {
            "task": name, "metric": metric,
            "n_train": scores.attrs["n_train"], "n_eval": scores.attrs["n_eval"],
            "best_model": best_name, "best_score": round(float(scores[metric].iloc[0]), 4),
            "xgboost_rank": (xgb[0] + 1) if xgb else None,
            "xgboost_score": round(float(scores[metric].iloc[xgb[0]]), 4) if xgb else None,
            "n_models": len(scores),
            "n_failed": scores.attrs["failed"],
            "seconds": round(scores.attrs["seconds"], 1),
        }
        if name in own:
            row["our_model_score"] = round(float(own[name][1]), 4)
        summary.append(row)
        print(scores[[metric]].head(5).to_string())
        print(f"  {len(scores)} models in {scores.attrs['seconds']:.0f} s")
        if xgb:
            print(f"  XGBoost ranks {xgb[0] + 1} of {len(scores)}")

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(out / "summary.csv", index=False)
        print("\n================ SUMMARY ================")
        print(df.drop(columns=["seconds"]).to_string(index=False))
        print(f"\nTotal time {time.time() - t_all:.0f} s. Per-task rankings and charts saved in {out}/")


if __name__ == "__main__":
    main()
