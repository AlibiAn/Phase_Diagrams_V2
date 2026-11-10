"""
Shared foundations: dataset loader, device detection, metrics helpers and the
SpecialistModel base class.

`load_dataset` reads the artefacts written by `mpds_preprocess.pipeline` into a
single PhaseDataset whose target arrays are all row-aligned with X.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """'cuda' if a CUDA device is available, else 'cpu' (valid for XGBoost ≥ 2.0 and torch)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def xgb_device_kwargs() -> dict:
    """Device kwargs for XGBoost constructors."""
    if detect_device() == "cuda":
        return {"device": "cuda", "tree_method": "hist"}
    return {"device": "cpu", "tree_method": "hist", "n_jobs": -1}


def torch_device():
    import torch
    return torch.device(detect_device())


# ---------------------------------------------------------------------------
# Tree-ensemble backends
# ---------------------------------------------------------------------------

TREE_BACKENDS = ("xgboost", "random_forest", "extra_trees")


def make_tree_classifier(backend: str, *, n_estimators: int, max_depth: int | None,
                         learning_rate: float, scale_pos_weight: float = 1.0, seed: int = 0):
    """Binary classifier for `backend`.

    xgboost: boosted trees of depth `max_depth`; positives up-weighted by
    `scale_pos_weight`. random_forest / extra_trees: bagged, fully grown trees
    (`max_depth` and `learning_rate` are ignored) with balanced class weights,
    the bagging counterpart of scale_pos_weight.
    """
    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight, objective="binary:logistic",
            eval_metric="aucpr", verbosity=0, random_state=seed, **xgb_device_kwargs())
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    forest = {"random_forest": RandomForestClassifier,
              "extra_trees": ExtraTreesClassifier}.get(backend)
    if forest is None:
        raise ValueError(f"backend must be one of {TREE_BACKENDS}")
    return forest(n_estimators=n_estimators, class_weight="balanced",
                  n_jobs=-1, random_state=seed)


def make_tree_regressor(backend: str, *, n_estimators: int, max_depth: int | None,
                        learning_rate: float, seed: int = 0):
    """Regressor for `backend` (see make_tree_classifier for what each uses)."""
    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            objective="reg:squarederror", eval_metric="mae", verbosity=0,
            random_state=seed, **xgb_device_kwargs())
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    forest = {"random_forest": RandomForestRegressor,
              "extra_trees": ExtraTreesRegressor}.get(backend)
    if forest is None:
        raise ValueError(f"backend must be one of {TREE_BACKENDS}")
    return forest(n_estimators=n_estimators, n_jobs=-1, random_state=seed)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Binary precision / recall / F1 from boolean (or 0/1) arrays of any shape."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights giving every class the same total weight (mean 1),
    so rare classes count as much as common ones in training."""
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return (len(y) / (len(classes) * counts))[inverse].astype(np.float32)


def tune_threshold(make_clf, X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
                   n_folds: int = 5, seed: int = 0) -> float | None:
    """Decision threshold maximising F1 on out-of-fold predictions.

    `make_clf()` returns a fresh, unfitted binary classifier. The training
    rows are split into `n_folds` stratified folds; each fold is predicted by
    a model fitted on the others, so the threshold is chosen on predictions
    for unseen rows. Returns None when there are too few positives to split.
    """
    from sklearn.model_selection import StratifiedKFold

    if min(int(y.sum()), int(len(y) - y.sum())) < n_folds:
        return None
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_clf()
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = np.array([precision_recall_f1(y, oof >= t)[2] for t in grid])
    # Among (near-)ties for the best F1, prefer the threshold closest to 0.5
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])


def curve_metrics(ds: "PhaseDataset", idx: np.ndarray, raw_liq: np.ndarray,
                  raw_sol: np.ndarray) -> dict:
    """Liquidus / solidus errors on the observed grid points of rows `idx`.

    `raw_liq` / `raw_sol` are unclamped predictions; the solidus is capped at
    the liquidus before scoring, and the order violations are counted before
    the cap (the capped output never violates).
    """
    sol_pred = np.minimum(raw_sol, raw_liq)
    sol_true = ds.T_solidus[idx]
    liq_err = np.abs(raw_liq - ds.T_liquidus[idx])[ds.T_liquidus_mask[idx]]
    sol_err = np.abs(sol_pred - sol_true)[~np.isnan(sol_true)]

    def mae(e):
        return float(e.mean()) if e.size else float("nan")

    def rmse(e):
        return float(np.sqrt((e ** 2).mean())) if e.size else float("nan")

    return {
        "liquidus_MAE_K": mae(liq_err), "liquidus_RMSE_K": rmse(liq_err),
        "solidus_MAE_K": mae(sol_err), "solidus_RMSE_K": rmse(sol_err),
        "order_violation_fraction": float((raw_sol > raw_liq + 1.0).mean()),
    }


def pad_class_proba(raw: np.ndarray, seen_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand XGBoost probabilities over the classes seen in training to all
    `n_classes` columns (zero for classes absent from the training set)."""
    if raw.shape[1] == n_classes:
        return raw
    full = np.zeros((raw.shape[0], n_classes), dtype=raw.dtype)
    full[:, seen_classes] = raw
    return full


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------

# Canonical topology class order (must match mpds_preprocess/targets.py)
TOPOLOGY_CLASSES = (
    "isomorphous",
    "simple_eutectic",
    "eutectic_with_intermetallics",
    "monotectic",
    "complex",
)


@dataclass
class PhaseDataset:
    """All preprocessing outputs, row-aligned and ready for training."""

    # Features
    X: np.ndarray                       # (N, F) float32
    feature_names: list[str]
    pair_ids: list[str]                 # e.g. 'Ag_Al'

    # M1: phase multi-hot
    Y_phases: np.ndarray                # (N, V) int8
    phase_vocab: list[str]

    # M2: liquidus
    T_liquidus: np.ndarray              # (N, 101) float32
    T_liquidus_mad: np.ndarray
    T_liquidus_mask: np.ndarray         # which grid points have data
    comp_grid: np.ndarray               # (101,) float32

    # M3: solidus
    T_solidus: np.ndarray
    T_solidus_mad: np.ndarray

    # M4: eutectic
    eutectic_exists: np.ndarray         # (N,) bool
    eutectic_comp: np.ndarray           # (N,) float, NaN where exists=False
    eutectic_temp: np.ndarray
    eutectic_temp_std: np.ndarray

    # M5: miscibility gap
    miscibility_gap: np.ndarray         # (N,) bool

    # M6: topology (as integer class index)
    topology: np.ndarray                # (N,) int8 index into topology_classes
    topology_classes: tuple[str, ...] = TOPOLOGY_CLASSES

    # Sample weights and provenance
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    n_sources: np.ndarray = field(default_factory=lambda: np.array([]))
    high_disagreement: np.ndarray = field(default_factory=lambda: np.array([]))

    # Splits
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


    @property
    def n_systems(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_phases(self) -> int:
        return len(self.phase_vocab)

    @property
    def n_topology_classes(self) -> int:
        return len(self.topology_classes)

    def split(self, name: str) -> np.ndarray:
        """Return the row indices for 'train' / 'val' / 'test'."""
        return {"train": self.train_idx, "val": self.val_idx,
                "test": self.test_idx}[name]

    def topology_name(self, i: int) -> str:
        return self.topology_classes[int(self.topology[i])]

    def indices_of(self, pairs: list[str]) -> np.ndarray:
        """Row indices of the named pairs; unknown pairs are skipped with a warning."""
        row = {p: i for i, p in enumerate(self.pair_ids)}
        missing = [p for p in pairs if p not in row]
        if missing:
            logger.warning("pairs not in dataset, skipping: %s", ", ".join(missing))
        return np.array([row[p] for p in pairs if p in row], dtype=np.int64)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(processed_dir: str | Path) -> PhaseDataset:
    """Load the preprocessing artefacts into one PhaseDataset, with every
    target realigned to the row order of features.parquet."""
    p = Path(processed_dir)

    X_df = pd.read_parquet(p / "features.parquet")
    pair_ids = X_df.index.tolist()
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=np.float32, na_value=np.nan)
    # Trees tolerate NaN but the torch model does not, so zero-fill everywhere.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scalar_df = pd.read_parquet(p / "targets_scalar.parquet").reindex(pair_ids)
    cls_to_idx = {c: i for i, c in enumerate(TOPOLOGY_CLASSES)}
    topology = np.asarray(
        [cls_to_idx.get(t, cls_to_idx["complex"]) for t in scalar_df["m6_topology"]],
        dtype=np.int8,
    )

    liq = np.load(p / "targets_liquidus.npz", allow_pickle=False)
    sol = np.load(p / "targets_solidus.npz", allow_pickle=False)
    ph = np.load(p / "targets_phases.npz", allow_pickle=True)
    for name, arrs in (("targets_liquidus.npz", liq), ("targets_solidus.npz", sol),
                       ("targets_phases.npz", ph)):
        if "pair_ids" in arrs.files and list(arrs["pair_ids"]) != pair_ids:
            raise ValueError(f"{name} rows are not aligned with features.parquet")

    weights = pd.read_parquet(p / "weights.parquet").reindex(pair_ids)["weight"].to_numpy(np.float32)
    with (p / "splits.json").open() as f:
        splits = json.load(f)

    ds = PhaseDataset(
        X=X,
        feature_names=feature_names,
        pair_ids=pair_ids,
        Y_phases=ph["Y"].astype(np.int8),
        phase_vocab=list(ph["vocab"]),
        T_liquidus=liq["T"].astype(np.float32),
        T_liquidus_mad=liq["MAD"].astype(np.float32),
        T_liquidus_mask=liq["mask"].astype(bool) & np.isfinite(liq["T"]),
        comp_grid=liq["comp_grid"].astype(np.float32),
        T_solidus=sol["T"].astype(np.float32),
        T_solidus_mad=sol["MAD"].astype(np.float32),
        eutectic_exists=scalar_df["m4_eutectic_exists"].to_numpy(bool),
        eutectic_comp=scalar_df["m4_eutectic_comp"].to_numpy(np.float32),
        eutectic_temp=scalar_df["m4_eutectic_temp"].to_numpy(np.float32),
        eutectic_temp_std=scalar_df["m4_eutectic_temp_std"].to_numpy(np.float32),
        miscibility_gap=scalar_df["m5_miscibility_gap"].to_numpy(bool),
        topology=topology,
        topology_classes=TOPOLOGY_CLASSES,
        weights=weights,
        n_sources=scalar_df["n_sources"].to_numpy(np.int32),
        high_disagreement=scalar_df["high_disagreement"].to_numpy(bool),
        train_idx=np.asarray(splits["train"], dtype=np.int64),
        val_idx=np.asarray(splits["val"], dtype=np.int64),
        test_idx=np.asarray(splits["test"], dtype=np.int64),
    )
    logger.info("Loaded %d systems × %d features; %d train / %d val / %d test",
                ds.n_systems, ds.n_features, len(ds.train_idx),
                len(ds.val_idx), len(ds.test_idx))
    return ds


def clipped_top_mask(ds: PhaseDataset, *, min_points: int = 5, tol_K: float = 0.5,
                     margin_K: float = 50.0) -> np.ndarray:
    """(N, G) mask of liquidus points that sit on a flat top cut off by the
    source chart's upper edge.

    A real liquidus maximum above both melting points is a rounded peak (a
    congruent compound or a miscibility-gap dome). A run of at least
    `min_points` grid points level (within `tol_K`) with the curve's maximum,
    more than `margin_K` above both Tm, is the chart's border instead: the true
    curve lies somewhere above it (Ba-Sm is flat at 2,000 K over 20-80 at%).
    """
    names = {n: i for i, n in enumerate(ds.feature_names)}
    tm_max = np.maximum(ds.X[:, names["phys__melting_point__A"]],
                        ds.X[:, names["phys__melting_point__B"]])
    T = np.where(ds.T_liquidus_mask, ds.T_liquidus, np.nan)
    out = np.zeros(T.shape, dtype=bool)
    for i in np.flatnonzero(np.isfinite(T).any(axis=1)):
        top = np.nanmax(T[i])
        if top <= tm_max[i] + margin_K:
            continue
        level = np.abs(T[i] - top) <= tol_K          # False at unobserved points
        # Longest runs of consecutive level points
        edges = np.flatnonzero(np.diff(np.r_[0, level.astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= min_points:
                out[i, start:stop] = True
    return out


def mask_clipped_tops(ds: PhaseDataset, **kwargs) -> list[str]:
    """Mark clipped liquidus tops (see clipped_top_mask) as unobserved, in
    place, so they are neither trained on nor scored. Returns the pairs hit."""
    clipped = clipped_top_mask(ds, **kwargs)
    ds.T_liquidus_mask &= ~clipped
    pairs = [ds.pair_ids[i] for i in np.flatnonzero(clipped.any(axis=1))]
    logger.info("Masked clipped liquidus tops: %d points in %d systems",
                int(clipped.sum()), len(pairs))
    return pairs


def _melting_points(ds: PhaseDataset) -> tuple[np.ndarray, np.ndarray]:
    names = {n: i for i, n in enumerate(ds.feature_names)}
    return (ds.X[:, names["phys__melting_point__A"]].astype(np.float64),
            ds.X[:, names["phys__melting_point__B"]].astype(np.float64))


def fix_end_points(ds: PhaseDataset, *, inject: bool, tol_K: float = 50.0,
                   edge_pct: float = 5.0) -> dict:
    """Correct the pure-element end points of the curve targets, in place.

    At x = 0 and 100 at% both curves must equal Tm of that element. An
    observed end more than `tol_K` off is handled by what went wrong
    (check_endpoints.py found both kinds in the MPDS data):

    - liquidus end too HIGH, or solidus end off: the curve changes too steeply
      for the 1 at% grid (a near-vertical liquidus, a solid solution narrower
      than one step), so the end point is wrong but the rest of the curve is
      fine;
    - liquidus end too LOW: the source diagram stops short of the pure
      element and the end values are extrapolated, so the outer `edge_pct`
      at% of both curves at that end is unreliable too.

    inject=True (for training) sets the bad end points to Tm and treats the
    extrapolated edge stretches as unobserved. inject=False (for scoring)
    treats all of those points as unobserved, so no model is scored against
    values that were either wrong or filled in from Tm. Returns the counts.
    """
    tm = np.stack(_melting_points(ds), axis=1)                    # (N, 2)
    grid = np.asarray(ds.comp_grid, dtype=np.float64)
    ends = (0, len(grid) - 1)
    stretches = (np.flatnonzero((grid > 0) & (grid <= edge_pct)),
                 np.flatnonzero((grid < 100) & (grid >= 100 - edge_pct)))
    counts = {"liquidus_ends": 0, "solidus_ends": 0, "short_edges": 0}
    for k, g in enumerate(ends):
        has_tm = tm[:, k] > 0
        liq_seen = ds.T_liquidus_mask[:, g]
        liq_diff = ds.T_liquidus[:, g] - tm[:, k]
        sol_diff = ds.T_solidus[:, g] - tm[:, k]                   # NaN where unobserved
        short = has_tm & liq_seen & (liq_diff < -tol_K)
        liq_bad = has_tm & liq_seen & (liq_diff > tol_K) | short
        sol_bad = has_tm & np.isfinite(sol_diff) & (np.abs(sol_diff) > tol_K) | short & np.isfinite(sol_diff)
        counts["liquidus_ends"] += int(liq_bad.sum())
        counts["solidus_ends"] += int(sol_bad.sum())
        counts["short_edges"] += int(short.sum())

        rows = np.flatnonzero(short)
        ds.T_liquidus_mask[np.ix_(rows, stretches[k])] = False
        ds.T_solidus[np.ix_(rows, stretches[k])] = np.nan
        if inject:
            ds.T_liquidus[liq_bad, g] = tm[liq_bad, k]
            ds.T_solidus[sol_bad, g] = tm[sol_bad, k]
        else:
            ds.T_liquidus_mask[liq_bad, g] = False
            ds.T_solidus[sol_bad, g] = np.nan
    logger.info("End points %s: %d liquidus and %d solidus ends, %d diagrams short of "
                "a pure element (outer %g at%% unobserved)",
                "set to Tm" if inject else "left out of scoring", counts["liquidus_ends"],
                counts["solidus_ends"], counts["short_edges"], edge_pct)
    return counts


# ---------------------------------------------------------------------------
# A/B swap augmentation
# ---------------------------------------------------------------------------

# How a feature changes when A and B trade places, if it is not an __A / __B column
_KEEP, _NEGATE, _RECIPROCAL = "keep", "negate", "reciprocal"


def suspect_antisymmetric(ds: PhaseDataset, rules: dict[str, str]) -> dict[str, tuple[str, float]]:
    """"keep?" features that look like they change sign when A and B swap:
    they take both signs and follow some A - B difference (|correlation| > 0.5).
    Returns name -> (the difference it follows, correlation)."""
    names = list(ds.feature_names)
    col = {n: j for j, n in enumerate(names)}
    diffs = {n[:-3]: ds.X[:, col[n]] - ds.X[:, col[n[:-3] + "__B"]]
             for n in names if n.endswith("__A") and n[:-3] + "__B" in col}
    out = {}
    for n, rule in rules.items():
        f = ds.X[:, col[n]].astype(np.float64)
        if rule != "keep?" or min((f > 0).mean(), (f < 0).mean()) < 0.1 or f.std() == 0:
            continue
        best = max(((abs(np.corrcoef(f, d)[0, 1]), base) for base, d in diffs.items() if d.std() > 0),
                   default=(0.0, ""))
        if best[0] > 0.5:
            out[n] = (best[1], float(best[0]))
    return out


def swap_rules(ds: PhaseDataset) -> dict[str, str]:
    """feature name -> what happens to it when A and B swap.

    __A / __B columns trade values ("swap"). Every other column built from a
    pair X__A, X__B under the name X__<anything> is matched against the usual
    formulas on the data itself: a/b or b/a -> reciprocal, a-b or b-a ->
    negate, |a-b|, a+b, mean, product, max, min, max/min -> keep. Of the
    columns matching no formula, those that take both signs and follow an
    A - B difference are negated ("negate?": which element is A is arbitrary,
    so a symmetric quantity cannot follow A - B); the rest are assumed
    symmetric ("keep?"), as composition-averaged and equiatomic quantities
    are. Check both lists with python -m mpds_models.base PROCESSED_DIR.
    """
    names = list(ds.feature_names)
    col = {n: j for j, n in enumerate(names)}
    rules = {}
    for n in names:
        if n.endswith("__A") or n.endswith("__B"):
            rules[n] = "swap"
            continue
        base = n.rsplit("__", 1)[0]
        a_name, b_name = base + "__A", base + "__B"
        rules[n] = "keep?"
        if a_name not in col or b_name not in col:
            continue
        f = ds.X[:, col[n]].astype(np.float64)
        a = ds.X[:, col[a_name]].astype(np.float64)
        b = ds.X[:, col[b_name]].astype(np.float64)
        ok = (a != 0) & (b != 0)
        if ok.sum() < 10:
            continue
        f, a, b = f[ok], a[ok], b[ok]
        hi, lo = np.maximum(a, b), np.minimum(a, b)
        candidates = ((_RECIPROCAL, a / b), (_RECIPROCAL, b / a), (_NEGATE, a - b), (_NEGATE, b - a),
                      (_KEEP, np.abs(a - b)), (_KEEP, a + b), (_KEEP, (a + b) / 2), (_KEEP, a * b),
                      (_KEEP, hi), (_KEEP, lo), (_KEEP, hi / lo), (_KEEP, lo / hi))
        for rule, c in candidates:
            if np.mean(np.isclose(f, c, rtol=1e-3, atol=1e-6)) >= 0.99:
                rules[n] = rule
                break
    for n in suspect_antisymmetric(ds, rules):
        rules[n] = "negate?"
    return rules


def swap_augment(ds: PhaseDataset, rules: dict[str, str] | None = None) -> PhaseDataset:
    """A copy of `ds` whose training split also holds every training system
    with A and B swapped: features swapped by `swap_rules`, curves mirrored,
    the eutectic composition c -> 100 - c. Which element is "A" is only the
    order in the pair id, so each system teaches the model twice. The val and
    test splits are unchanged."""
    from dataclasses import replace
    rules = rules or swap_rules(ds)
    names = list(ds.feature_names)
    col = {n: j for j, n in enumerate(names)}
    tr = ds.train_idx
    X = ds.X[tr].copy()
    for n, rule in rules.items():
        j = col[n]
        if rule == "swap" and n.endswith("__A"):
            k = col.get(n[:-3] + "__B")
            if k is not None:
                X[:, [j, k]] = X[:, [k, j]]
        elif rule in (_NEGATE, "negate?"):
            X[:, j] = -X[:, j]
        elif rule == _RECIPROCAL:
            v = X[:, j]
            X[:, j] = np.divide(1.0, v, out=np.zeros_like(v), where=v != 0)

    def both(arr: np.ndarray, new: np.ndarray | None = None) -> np.ndarray:
        return np.concatenate([arr, arr[tr] if new is None else new])

    flip = lambda a: a[tr][:, ::-1]                                # mirror along composition
    n = ds.n_systems
    out = replace(
        ds,
        X=both(ds.X, X),
        pair_ids=list(ds.pair_ids) + ["_".join(ds.pair_ids[i].split("_")[::-1]) for i in tr],
        Y_phases=both(ds.Y_phases),
        T_liquidus=both(ds.T_liquidus, flip(ds.T_liquidus)),
        T_liquidus_mad=both(ds.T_liquidus_mad, flip(ds.T_liquidus_mad)),
        T_liquidus_mask=both(ds.T_liquidus_mask, flip(ds.T_liquidus_mask)),
        T_solidus=both(ds.T_solidus, flip(ds.T_solidus)),
        T_solidus_mad=both(ds.T_solidus_mad, flip(ds.T_solidus_mad)),
        eutectic_exists=both(ds.eutectic_exists),
        eutectic_comp=both(ds.eutectic_comp, 100.0 - ds.eutectic_comp[tr]),
        eutectic_temp=both(ds.eutectic_temp),
        eutectic_temp_std=both(ds.eutectic_temp_std),
        miscibility_gap=both(ds.miscibility_gap),
        topology=both(ds.topology),
        weights=both(ds.weights),
        n_sources=both(ds.n_sources),
        high_disagreement=both(ds.high_disagreement),
        train_idx=np.concatenate([tr, n + np.arange(len(tr))]).astype(np.int64),
    )
    return out


# ---------------------------------------------------------------------------
# Specialist model interface
# ---------------------------------------------------------------------------

class SpecialistModel:
    """Interface shared by M1-M6. Subclasses implement fit / predict / evaluate;
    save / load use joblib unless overridden (the torch model does)."""

    name: str = "base"

    def fit(self, ds: PhaseDataset) -> "SpecialistModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        raise NotImplementedError

    def evaluate(self, ds: PhaseDataset, split: str = "test") -> dict:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SpecialistModel":
        import joblib
        return joblib.load(path)


if __name__ == "__main__":
    import sys
    from collections import Counter
    ds_ = load_dataset(sys.argv[1])
    rules_ = swap_rules(ds_)
    print("How each feature changes when A and B swap:", dict(Counter(rules_.values())))
    for name, rule in rules_.items():
        if rule != "swap":
            print(f"  {rule:<11} {name}")
    print('"keep?" = matched no formula and assumed symmetric; check that none of these '
          "changes when A and B trade places.")
    print('"negate?" = matched no formula but takes both signs and follows an A - B difference, '
          "so it is negated; check that each really changes sign.")
