"""
The dataset: features and targets of every binary system, row-aligned.

load_dataset() reads a processed folder (features.parquet, targets_*.npz,
targets_scalar.parquet, weights.parquet, splits.json; written by
mpds_preprocess or data_build/build_two_liquid_dataset.py) and appends the
element and compound features of features.py. add_extra_systems() appends
systems that have curves but no labels, for training the curve model only.

Also here: the A/B swap of the features, the data fixes (clipped chart tops,
end points that disagree with the melting point) and the data groups used in
every report (clean, incomplete, volatile element, monotectic).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOPOLOGY_CLASSES = ("isomorphous", "simple_eutectic", "eutectic_with_intermetallics", "monotectic", "complex")
FEATURE_SETS = ("extended", "extended+mp")
# Materials Project 2018 snapshot for the "+mp" compound features; MPDS_MP_FILE overrides it
MP_FILE = "/home/alibi/lab/external/mp_nostruct_20181018.json.gz"
_EMPTY_IDX = lambda: np.array([], dtype=np.int64)


@dataclass
class PhaseDataset:
    X: np.ndarray                   # (N, F) float32, NaN -> 0
    X_swap: np.ndarray              # the same systems with A and B exchanged
    feature_names: list[str]
    pair_ids: list[str]             # 'Ag_Al'
    comp_grid: np.ndarray           # (101,) at% B
    T_liquidus: np.ndarray          # (N, 101) K
    T_liquidus_mask: np.ndarray     # where the liquidus has data
    T_solidus: np.ndarray           # (N, 101) K, NaN = no data
    Y_phases: np.ndarray            # (N, V) phases present
    phase_vocab: list[str]
    eutectic_exists: np.ndarray
    eutectic_comp: np.ndarray       # at% B, NaN without a eutectic
    eutectic_temp: np.ndarray
    miscibility_gap: np.ndarray
    topology: np.ndarray            # index into TOPOLOGY_CLASSES
    weights: np.ndarray
    n_sources: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    features: str = "extended+mp"
    n_base_features: int = 0        # leading columns that are the processed (v6) features
    extra_idx: np.ndarray = field(default_factory=_EMPTY_IDX)   # rows from add_extra_systems
    swap_rules: dict = field(default_factory=dict)              # how each v6 column changes under A <-> B
    topology_classes: tuple = TOPOLOGY_CLASSES

    @property
    def n_systems(self) -> int:
        return self.X.shape[0]

    def split(self, name: str) -> np.ndarray:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[name]

    def topology_name(self, i: int) -> str:
        return self.topology_classes[int(self.topology[i])]

    def column(self, name: str) -> int:
        return self.feature_names.index(name)

    def melting_points(self, rows=slice(None)) -> tuple[np.ndarray, np.ndarray]:
        return (self.X[rows, self.column("phys__melting_point__A")].astype(np.float64),
                self.X[rows, self.column("phys__melting_point__B")].astype(np.float64))

    def with_split(self, train: np.ndarray, test: np.ndarray = _EMPTY_IDX()) -> "PhaseDataset":
        return replace(self, train_idx=np.asarray(train, np.int64), val_idx=_EMPTY_IDX(),
                       test_idx=np.asarray(test, np.int64))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _pair_feature_blocks(pair_ids: list[str], features: str):
    """(X, X_swap, names) of the features computed from the element symbols."""
    from .features import compound_features, element_features
    blocks = [element_features(pair_ids)]
    if features == "extended+mp":
        blocks.append(compound_features(pair_ids, os.environ.get("MPDS_MP_FILE", MP_FILE)))
    return (np.concatenate([b[0] for b in blocks], axis=1), np.concatenate([b[1] for b in blocks], axis=1),
            [n for b in blocks for n in b[2]])


def load_dataset(processed_dir: str | Path, features: str = "extended+mp") -> PhaseDataset:
    """The processed folder's 207 features plus the element features
    ("extended", 147 columns) and optionally the Materials Project compound
    features ("extended+mp", 15 more; read the caveat in features.py)."""
    if features not in FEATURE_SETS:
        raise ValueError(f"features must be one of {FEATURE_SETS}")
    p = Path(processed_dir)
    X_df = pd.read_parquet(p / "features.parquet")
    pair_ids, base_names = X_df.index.tolist(), X_df.columns.tolist()
    X = _clean(X_df.to_numpy(dtype=np.float32, na_value=np.nan))
    scalar = pd.read_parquet(p / "targets_scalar.parquet").reindex(pair_ids)
    liq = np.load(p / "targets_liquidus.npz", allow_pickle=False)
    sol = np.load(p / "targets_solidus.npz", allow_pickle=False)
    ph = np.load(p / "targets_phases.npz", allow_pickle=True)
    for name, arrs in (("liquidus", liq), ("solidus", sol), ("phases", ph)):
        if "pair_ids" in arrs.files and list(arrs["pair_ids"]) != pair_ids:
            raise ValueError(f"targets_{name}.npz rows are not aligned with features.parquet")
    splits = json.loads((p / "splits.json").read_text())
    rules = swap_rules(X, base_names)
    extra, extra_swap, extra_names = _pair_feature_blocks(pair_ids, features)
    ds = PhaseDataset(
        X=np.concatenate([X, extra], axis=1),
        X_swap=np.concatenate([swap_features(X, base_names, rules), extra_swap], axis=1),
        feature_names=base_names + extra_names, pair_ids=pair_ids,
        comp_grid=liq["comp_grid"].astype(np.float32),
        T_liquidus=liq["T"].astype(np.float32), T_liquidus_mask=liq["mask"].astype(bool) & np.isfinite(liq["T"]),
        T_solidus=sol["T"].astype(np.float32),
        Y_phases=ph["Y"].astype(np.int8), phase_vocab=list(ph["vocab"]),
        eutectic_exists=scalar["m4_eutectic_exists"].to_numpy(bool),
        eutectic_comp=scalar["m4_eutectic_comp"].to_numpy(np.float32),
        eutectic_temp=scalar["m4_eutectic_temp"].to_numpy(np.float32),
        miscibility_gap=scalar["m5_miscibility_gap"].to_numpy(bool),
        topology=np.array([TOPOLOGY_CLASSES.index(t) if t in TOPOLOGY_CLASSES else 4 for t in scalar["m6_topology"]],
                          dtype=np.int8),
        weights=pd.read_parquet(p / "weights.parquet").reindex(pair_ids)["weight"].to_numpy(np.float32),
        n_sources=scalar["n_sources"].to_numpy(np.int32),
        train_idx=np.asarray(splits["train"], np.int64), val_idx=np.asarray(splits["val"], np.int64),
        test_idx=np.asarray(splits["test"], np.int64),
        features=features, n_base_features=len(base_names), swap_rules=rules)
    logger.info("Loaded %d systems x %d features (%s); %d train / %d val / %d test", ds.n_systems, ds.X.shape[1],
                features, len(ds.train_idx), len(ds.val_idx), len(ds.test_idx))
    return ds


def add_extra_systems(ds: PhaseDataset, extra_dir: str | Path) -> PhaseDataset:
    """A copy of `ds` with the systems of `extra_dir` (data_build/build_extra_systems.py)
    appended as TRAINING rows for the curve model. They have a liquidus (and
    sometimes a solidus) but no labels, which get placeholders: never fit
    the other models on the result. `extra_idx` lists the new rows."""
    p = Path(extra_dir)
    X_df = pd.read_parquet(p / "features.parquet")
    base_names = ds.feature_names[:ds.n_base_features]
    if list(X_df.columns) != base_names:
        raise ValueError(f"{extra_dir}: features do not match the dataset's")
    pair_ids = X_df.index.tolist()
    if set(pair_ids) & set(ds.pair_ids):
        raise ValueError(f"{extra_dir} contains systems of the dataset itself")
    X = _clean(X_df.to_numpy(dtype=np.float32, na_value=np.nan))
    extra, extra_swap, _ = _pair_feature_blocks(pair_ids, ds.features)
    liq = np.load(p / "targets_liquidus.npz", allow_pickle=False)
    sol = np.load(p / "targets_solidus.npz", allow_pickle=False)
    if list(liq["pair_ids"]) != pair_ids or list(sol["pair_ids"]) != pair_ids:
        raise ValueError(f"{extra_dir}: targets are not aligned with features.parquet")
    k, n = len(pair_ids), ds.n_systems
    cat = lambda a, b: np.concatenate([a, np.asarray(b, dtype=a.dtype)])
    extra_idx = n + np.arange(k, dtype=np.int64)
    out = replace(
        ds, X=cat(ds.X, np.concatenate([X, extra], axis=1)),
        X_swap=cat(ds.X_swap, np.concatenate([swap_features(X, base_names, ds.swap_rules), extra_swap], axis=1)),
        pair_ids=ds.pair_ids + pair_ids,
        T_liquidus=cat(ds.T_liquidus, liq["T"]),
        T_liquidus_mask=cat(ds.T_liquidus_mask, liq["mask"].astype(bool) & np.isfinite(liq["T"])),
        T_solidus=cat(ds.T_solidus, sol["T"]),
        Y_phases=cat(ds.Y_phases, np.zeros((k, ds.Y_phases.shape[1]))),
        eutectic_exists=cat(ds.eutectic_exists, np.zeros(k, bool)),
        eutectic_comp=cat(ds.eutectic_comp, np.full(k, np.nan)),
        eutectic_temp=cat(ds.eutectic_temp, np.full(k, np.nan)),
        miscibility_gap=cat(ds.miscibility_gap, np.zeros(k, bool)),
        topology=cat(ds.topology, np.full(k, TOPOLOGY_CLASSES.index("complex"))),
        weights=cat(ds.weights, pd.read_parquet(p / "weights.parquet").reindex(pair_ids)["weight"]),
        n_sources=cat(ds.n_sources, np.ones(k)),
        train_idx=np.concatenate([ds.train_idx, extra_idx]), extra_idx=extra_idx)
    mask_clipped_tops(out)                # on a copy of the masks: `ds` itself is not changed
    logger.info("Added %d extra systems for curve training from %s", k, extra_dir)
    return out


# ---------------------------------------------------------------------------
# A <-> B swap
# ---------------------------------------------------------------------------

def _suspect_antisymmetric(X: np.ndarray, names: list[str], rules: dict[str, str]) -> list[str]:
    """"keep?" columns that take both signs and follow an A - B difference
    (|correlation| > 0.5): which element is A is arbitrary, so a symmetric
    quantity cannot follow A - B; these are negated instead."""
    col = {n: j for j, n in enumerate(names)}
    diffs = [X[:, col[n]] - X[:, col[n[:-3] + "__B"]] for n in names if n.endswith("__A") and n[:-3] + "__B" in col]
    out = []
    for n, rule in rules.items():
        f = X[:, col[n]].astype(np.float64)
        if rule != "keep?" or min((f > 0).mean(), (f < 0).mean()) < 0.1 or f.std() == 0:
            continue
        if max((abs(np.corrcoef(f, d)[0, 1]) for d in diffs if d.std() > 0), default=0.0) > 0.5:
            out.append(n)
    return out


def swap_rules(X: np.ndarray, names: list[str]) -> dict[str, str]:
    """What happens to each processed feature when A and B trade places:
    __A / __B columns swap; other columns built from X__A, X__B are matched
    against a/b, b/a (reciprocal), a-b, b-a (negate), |a-b|, a+b, mean,
    product, max, min, max/min (keep) on the data itself; the rest are
    symmetric ("keep?") unless they follow an A - B difference ("negate?")."""
    col = {n: j for j, n in enumerate(names)}
    rules = {}
    for n in names:
        if n.endswith("__A") or n.endswith("__B"):
            rules[n] = "swap"
            continue
        base = n.rsplit("__", 1)[0]
        rules[n] = "keep?"
        if base + "__A" not in col or base + "__B" not in col:
            continue
        f, a, b = (X[:, col[c]].astype(np.float64) for c in (n, base + "__A", base + "__B"))
        ok = (a != 0) & (b != 0)
        if ok.sum() < 10:
            continue
        f, a, b = f[ok], a[ok], b[ok]
        hi, lo = np.maximum(a, b), np.minimum(a, b)
        for rule, c in (("reciprocal", a / b), ("reciprocal", b / a), ("negate", a - b), ("negate", b - a),
                        ("keep", np.abs(a - b)), ("keep", a + b), ("keep", (a + b) / 2), ("keep", a * b),
                        ("keep", hi), ("keep", lo), ("keep", hi / lo), ("keep", lo / hi)):
            if np.mean(np.isclose(f, c, rtol=1e-3, atol=1e-6)) >= 0.99:
                rules[n] = rule
                break
    for n in _suspect_antisymmetric(X, names, rules):
        rules[n] = "negate?"
    return rules


def swap_features(X: np.ndarray, names: list[str], rules: dict[str, str]) -> np.ndarray:
    col = {n: j for j, n in enumerate(names)}
    X = X.copy()
    for n, rule in rules.items():
        j = col[n]
        if rule == "swap" and n.endswith("__A") and n[:-3] + "__B" in col:
            k = col[n[:-3] + "__B"]
            X[:, [j, k]] = X[:, [k, j]]
        elif rule in ("negate", "negate?"):
            X[:, j] = -X[:, j]
        elif rule == "reciprocal":
            v = X[:, j]
            X[:, j] = np.divide(1.0, v, out=np.zeros_like(v), where=v != 0)
    return X


# ---------------------------------------------------------------------------
# Data fixes
# ---------------------------------------------------------------------------

def mask_clipped_tops(ds: PhaseDataset, min_points: int = 5, tol_K: float = 0.5, margin_K: float = 50.0) -> list[str]:
    """Mark liquidus tops cut off by the source chart's upper edge as unobserved,
    in place: a run of >= `min_points` grid points level with the curve's
    maximum, more than `margin_K` above both melting points (a real maximum
    is a rounded peak). Returns the pairs hit."""
    tm_max = np.maximum(ds.X[:, ds.column("phys__melting_point__A")], ds.X[:, ds.column("phys__melting_point__B")])
    T = np.where(ds.T_liquidus_mask, ds.T_liquidus, np.nan)
    clipped = np.zeros(T.shape, dtype=bool)
    for i in np.flatnonzero(np.isfinite(T).any(axis=1)):
        top = np.nanmax(T[i])
        if top > tm_max[i] + margin_K:
            for start, stop in _runs(np.abs(T[i] - top) <= tol_K):
                if stop - start >= min_points:
                    clipped[i, start:stop] = True
    ds.T_liquidus_mask &= ~clipped
    pairs = [ds.pair_ids[i] for i in np.flatnonzero(clipped.any(axis=1))]
    logger.info("Masked clipped liquidus tops: %d points in %d systems", int(clipped.sum()), len(pairs))
    return pairs


def fix_end_points(ds: PhaseDataset, tol_K: float = 50.0, edge_pct: float = 5.0) -> dict:
    """Leave pure-element end points that disagree with Tm by more than
    `tol_K` out of scoring, in place. A liquidus end too LOW means the source
    diagram stops short of the element, so the outer `edge_pct` at% of both
    curves at that end is left out too. Returns the counts."""
    tm = np.stack(ds.melting_points(), axis=1)
    grid = np.asarray(ds.comp_grid, dtype=np.float64)
    stretches = (np.flatnonzero((grid > 0) & (grid <= edge_pct)), np.flatnonzero((grid < 100) & (grid >= 100 - edge_pct)))
    counts = {"liquidus_ends": 0, "solidus_ends": 0, "short_edges": 0}
    for k, g in enumerate((0, len(grid) - 1)):
        has_tm, liq_seen = tm[:, k] > 0, ds.T_liquidus_mask[:, g]
        liq_diff, sol_diff = ds.T_liquidus[:, g] - tm[:, k], ds.T_solidus[:, g] - tm[:, k]
        short = has_tm & liq_seen & (liq_diff < -tol_K)
        liq_bad = has_tm & liq_seen & (liq_diff > tol_K) | short
        sol_bad = has_tm & np.isfinite(sol_diff) & (np.abs(sol_diff) > tol_K) | short & np.isfinite(sol_diff)
        counts["liquidus_ends"] += int(liq_bad.sum())
        counts["solidus_ends"] += int(sol_bad.sum())
        counts["short_edges"] += int(short.sum())
        rows = np.flatnonzero(short)
        ds.T_liquidus_mask[np.ix_(rows, stretches[k])] = False
        ds.T_solidus[np.ix_(rows, stretches[k])] = np.nan
        ds.T_liquidus_mask[liq_bad, g] = False
        ds.T_solidus[sol_bad, g] = np.nan
    return counts


# ---------------------------------------------------------------------------
# Data groups
# ---------------------------------------------------------------------------

VOLATILE_BOILING_K = 1300.0
DATA_GROUPS = ("all", "clean", "incomplete", "volatile element", "monotectic")


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def system_quality(ds: PhaseDataset, min_coverage: float = 0.5, max_gap_pct: float = 10.0) -> pd.DataFrame:
    """One row per system: coverage, largest_gap_pct, incomplete (coverage <
    min_coverage or a gap >= max_gap_pct), volatile_elements / volatile (an
    element boiling below 1,300 K), above_boiling_share, monotectic and the
    group ("clean" or the flags joined by " + "). Call it before masking."""
    step = float(np.diff(np.asarray(ds.comp_grid, dtype=np.float64)).mean())
    bp = np.stack([ds.X[:, ds.column("phys__boiling_point__A")],
                   ds.X[:, ds.column("phys__boiling_point__B")]], axis=1).astype(np.float64)
    rows = []
    for i, pair in enumerate(ds.pair_ids):
        seen = ds.T_liquidus_mask[i]
        gap = max(((b - a) * step for a, b in _runs(~seen)), default=0.0)
        coverage = float(seen.mean())
        vol = [e for e, b in zip(pair.split("_")[:2], bp[i]) if 0 < b < VOLATILE_BOILING_K]
        known = bp[i][bp[i] > 0]
        low_bp = known.min() if known.size else np.inf
        T = ds.T_liquidus[i][seen]
        flags = {"incomplete": coverage < min_coverage or gap >= max_gap_pct, "volatile element": bool(vol),
                 "monotectic": ds.topology_name(i) == "monotectic"}
        rows.append(dict(coverage=coverage, largest_gap_pct=gap, incomplete=flags["incomplete"],
                         volatile_elements=" ".join(vol), volatile=flags["volatile element"],
                         above_boiling_share=float((T > low_bp).mean()) if T.size and np.isfinite(low_bp) else 0.0,
                         monotectic=flags["monotectic"],
                         group=" + ".join(k for k, v in flags.items() if v) or "clean"))
    return pd.DataFrame(rows)


def group_rows(quality: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Data group -> the members of `idx` in it (groups overlap except "clean")."""
    q = quality.iloc[idx]
    return {"all": idx, "clean": idx[(q["group"] == "clean").to_numpy()],
            "incomplete": idx[q["incomplete"].to_numpy()], "volatile element": idx[q["volatile"].to_numpy()],
            "monotectic": idx[q["monotectic"].to_numpy()]}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    y_true, y_pred = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    tp, fp, fn = int((y_pred & y_true).sum()), int((y_pred & ~y_true).sum()), int((~y_pred & y_true).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def curve_errors(ds: PhaseDataset, idx, liq: np.ndarray, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute liquidus / solidus errors at the points of rows `idx` that have data."""
    sol_true = ds.T_solidus[idx]
    return (np.abs(liq - ds.T_liquidus[idx])[ds.T_liquidus_mask[idx]],
            np.abs(np.minimum(sol, liq) - sol_true)[np.isfinite(sol_true)])


def curve_scores(ds: PhaseDataset, idx, liq: np.ndarray, sol: np.ndarray) -> dict:
    e_liq, e_sol = (e.astype(np.float64) for e in curve_errors(ds, idx, liq, sol))
    mae = lambda e: float(e.mean()) if e.size else float("nan")
    rmse = lambda e: float(np.sqrt((e ** 2).mean())) if e.size else float("nan")
    return {"liquidus_MAE_K": mae(e_liq), "liquidus_RMSE_K": rmse(e_liq),
            "solidus_MAE_K": mae(e_sol), "solidus_RMSE_K": rmse(e_sol), "n_liquidus_points": int(e_liq.size)}
