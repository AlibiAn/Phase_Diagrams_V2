"""
build_two_liquid_dataset.py — derive a dataset in which two-liquid regions are
treated the same way in every diagram (processed_v7 from the raw MPDS files).

Why
---
MPDS files carry a `labels` dictionary with the name and position of every
region of the chart, including "L1 + L2" for the region where two liquids
coexist. The preprocessing that built processed_v6 does not read it, and so:

  * `m5_miscibility_gap` is true only when the liquid is drawn as two separate
    single-phase regions. When one liquid region wraps over the two-liquid
    dome (Ag-V, Cu-V, Ag-Co ...) the gap is missed and the system is labelled
    "simple_eutectic" or "eutectic_with_intermetallics".
  * the "liquidus" inside the gap is whatever bounds the single-phase liquid
    from below: the top of the dome (the binodal) in some diagrams, the flat
    monotectic line in others, the chart's upper edge in the rest.

What this script changes
------------------------
Liquid-containing regions are L, L + solid AND L1 + L2. With that one rule:

  liquidus   lowest temperature with no solid. Under a two-liquid dome this
             is the monotectic line, not the top of the dome: the hump of
             the source's liquidus that contains the "L1 + L2" label is cut
             down to the level of its higher foot (see flatten_dome).
  solidus    lowest temperature with any liquid; regions under a two-liquid
             region now count as liquid + solid.
  binodal    NEW target: the part of the dome that was cut off (where it is
             drawn below the chart's upper edge), in targets_binodal.npz.
  gap flag   true if the source has two liquid regions OR an "L1 + L2" label
             (weighted vote over the sources, as before).
  topology   re-derived by the unchanged rules from the new gap flag and liquidus.

Features, weights, splits, phases and the eutectic targets are copied from the
reference dataset unchanged, so the same systems sit in the same rows and
folds and models can be compared across the two datasets.

Nothing in the preprocessing package or in the reference dataset is modified:
the package's functions are wrapped at run time.

Usage
-----
    python build_two_liquid_dataset.py /home/alibi/lab/mpds_data /home/alibi/lab/processed_v7 \\
        --reference /home/alibi/lab/processed_v6 --preprocess-dir /home/alibi/lab
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("two_liquid")

EDGE_TOL_K = 5.0          # a point this far above the monotectic line is on the dome, not the line
TOP_TOL_K = 10.0          # a dome this close to the chart's upper edge is cut off, not measured


def is_two_liquid_label(label: str) -> bool:
    """True for region labels naming two (or more) coexisting liquids:
    'L$_{1}$ + L$_{2}$', "L' + L''", 'L1 + L2', 'L2 + L3'."""
    s = re.sub(r"[\$_{}\\\s]", "", label or "")
    parts = s.split("+")
    liquid = [p for p in parts if re.fullmatch(r"L(\d|'{1,3}|I{1,3})?|Liquid\d?", p)]
    return len(parts) >= 2 and len(liquid) >= 2


def _polygon(p: dict):
    try:
        x = np.asarray(p.get("x", []), dtype=float)
        y = np.asarray(p.get("y", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if x.size != y.size or x.size < 3:
        return None
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def two_liquid_regions(entry: dict) -> list[int]:
    """Indices (into entry['phases']) of the regions where two liquids coexist:
    for each 'L1 + L2' label, the smallest two-phase region containing the
    label's position."""
    from matplotlib.path import Path as MplPath

    points = []
    for name, pos in (entry.get("labels") or {}).items():
        if is_two_liquid_label(name):
            try:
                points.append((float(pos[0]), float(pos[1])))
            except (TypeError, ValueError, IndexError):
                pass
    if not points:
        return []
    found: list[int] = []
    for pt in points:
        best, best_area = None, np.inf
        for i, p in enumerate(entry.get("phases") or []):
            if p.get("phases") != 2:
                continue
            xy = _polygon(p)
            if xy is None:
                continue
            if MplPath(np.column_stack(xy)).contains_point(pt):
                a = _area(*xy)
                if a < best_area:
                    best, best_area = i, a
        if best is not None and best not in found:
            found.append(best)
    return found


def has_two_liquid_label(entry: dict) -> bool:
    return any(is_two_liquid_label(k) for k in (entry.get("labels") or {}))


def flatten_dome(T: np.ndarray, grid: np.ndarray, x_label: float, t_label: float, *,
                 tm_left: float = np.nan, tm_right: float = np.nan,
                 tol_K: float = EDGE_TOL_K, rise_K: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Cut the two-liquid dome off a liquidus curve.

    `T` is one source's liquidus on `grid` (NaN outside its range) and
    (x_label, t_label) the position of its "L1 + L2" label, which lies inside
    the two-liquid region: above the monotectic line and under the dome. The
    dome is the hump of the curve around x_label that rises above t_label.
    Walking down each side of the hump, the curve reaches a foot: a kink
    where it turns up again (the monotectic point on one side, often a
    eutectic further down on the other), or a pure element (melting points
    tm_left, tm_right). The monotectic temperature is the HIGHER foot, and
    everything of the hump above it is set to it.

    Nothing is changed if the curve at x_label is not above the label (the
    source already draws the monotectic line, or the label is elsewhere).
    Returns (new curve, mask of the points that were lowered).
    """
    T = np.asarray(T, dtype=float)
    out, cut = T.copy(), np.zeros(T.shape, dtype=bool)
    seen = np.flatnonzero(np.isfinite(T))
    if seen.size < 5 or not (grid[seen[0]] <= x_label <= grid[seen[-1]]):
        return out, cut
    i = int(seen[np.argmin(np.abs(grid[seen] - x_label))])
    if T[i] < t_label - tol_K:
        return out, cut

    def foot(step: int, tm_end: float) -> float:
        """Lowest value reached going from i in direction `step` before the
        curve rises again by more than rise_K (a kink). If the curve instead
        falls all the way to a pure element, the foot is that element's
        melting point at most: the dome's flank is too steep for the grid
        there, and the monotectic line cannot lie above Tm."""
        j, low = i, T[i]
        while 0 <= j + step < T.size and np.isfinite(T[j + step]):
            j += step
            if T[j] < low:
                low = T[j]
            elif T[j] > low + rise_K and low < t_label + tol_K:
                return float(low)
        at_pure_element = grid[j] <= 2.0 or grid[j] >= 98.0
        if at_pure_element and np.isfinite(tm_end):
            low = min(low, tm_end)
        return float(low)

    level = max(foot(-1, tm_left), foot(+1, tm_right))
    if level > t_label + tol_K:               # the label would lie below the line: not a dome
        return out, cut
    lo = hi = i                               # the hump: connected points above the level
    while lo - 1 >= 0 and np.isfinite(T[lo - 1]) and T[lo - 1] > level:
        lo -= 1
    while hi + 1 < T.size and np.isfinite(T[hi + 1]) and T[hi + 1] > level:
        hi += 1
    span = np.zeros(T.shape, dtype=bool)
    span[lo:hi + 1] = True
    cut = span & (T > level + tol_K)
    out[span & (T > level)] = level
    return out, cut


def _label_positions(entry: dict, pair_key) -> list[tuple[float, float]]:
    """Positions of the entry's two-liquid labels with x as at% of pair_key[1]."""
    flip = entry["elements"] != list(pair_key)
    out = []
    for name, pos in (entry.get("labels") or {}).items():
        if is_two_liquid_label(name):
            try:
                x, t = float(pos[0]), float(pos[1])
            except (TypeError, ValueError, IndexError):
                continue
            out.append((100.0 - x if flip else x, t))
    return out


# per-entry record of what flatten_dome removed: id(entry) -> (dome values, was cut)
_DOMES: dict[int, np.ndarray] = {}
STATS = {"labels": 0, "domes_cut": 0, "solidus_regions": 0, "solidus_regions_rejected": 0}


def install(pre) -> dict:
    """Wrap the preprocessing functions (module objects in `pre`) so that
    two-liquid regions count as liquid. Returns the original functions."""
    consensus, targets = pre["consensus"], pre["targets"]
    original = {"resample_liquidus": consensus.resample_liquidus,
                "detect_miscibility_gap": consensus.detect_miscibility_gap,
                "extract_solidus": targets.extract_solidus}

    def resample_liquidus(entry: dict, elements_canonical, comp_grid=consensus.DEFAULT_COMP_GRID):
        T = original["resample_liquidus"](entry, elements_canonical, comp_grid)
        dome = np.full(T.shape, np.nan)
        try:
            chart_top = float(entry["temp_range"][1])
        except (TypeError, ValueError, IndexError, KeyError):
            chart_top = np.inf
        for x_label, t_label in _label_positions(entry, elements_canonical):
            STATS["labels"] += 1
            new, cut = flatten_dome(T, comp_grid, x_label, t_label,
                                    tm_left=consensus.element_melting_point_K(elements_canonical[0]),
                                    tm_right=consensus.element_melting_point_K(elements_canonical[1]))
            if cut.any():
                STATS["domes_cut"] += 1
                measured = cut & (T < chart_top - TOP_TOL_K)      # not the chart's upper edge
                dome[measured] = T[measured]
                T = new
        _DOMES[id(entry)] = dome
        return T

    def detect_miscibility_gap(entry: dict) -> bool:
        return bool(original["detect_miscibility_gap"](entry)) or has_two_liquid_label(entry)

    def extract_solidus(entry: dict, elements_canonical, comp_grid=consensus.DEFAULT_COMP_GRID):
        regions = two_liquid_regions(entry)
        if regions:
            # A region can only hold liquid down to the lowest point of the
            # single-phase liquid; a "region" reaching below that is the chart
            # frame or a merged outline, not a two-liquid field: leave it alone.
            floors = [float(np.min(p["y"])) for p in entry.get("phases") or []
                      if targets._region_contains_liquid(p) and p.get("phases") == 1 and len(p.get("y") or [])]
            floor = min(floors) if floors else -np.inf
            phases = [dict(p) for p in entry.get("phases") or []]
            for i in regions:
                if float(np.min(phases[i]["y"])) >= floor - 15.0:
                    phases[i].update(phases=1, is_solid=False, label="Liquid")
                    STATS["solidus_regions"] += 1
                else:
                    STATS["solidus_regions_rejected"] += 1
            entry = dict(entry)
            entry["phases"] = phases
        return original["extract_solidus"](entry, elements_canonical, comp_grid)

    consensus.resample_liquidus = resample_liquidus
    consensus.detect_miscibility_gap = detect_miscibility_gap
    targets.extract_solidus = extract_solidus
    return original


def binodal_curve(c) -> tuple[np.ndarray, np.ndarray]:
    """Median over the sources of the dome that flatten_dome cut off, and how
    many sources drew it at each composition."""
    stack = np.vstack([_DOMES.get(id(e), np.full(c.comp_grid.shape, np.nan)) for e in c.raw_entries])
    n = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        med = np.nanmedian(stack, axis=0)
    return med, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json_dir", help="raw MPDS diagram files")
    ap.add_argument("out_dir", help="new folder for the derived dataset")
    ap.add_argument("--reference", required=True, help="dataset to copy features / splits from (processed_v6)")
    ap.add_argument("--preprocess-dir", required=True,
                    help="folder that contains the mpds_preprocess package")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out, ref = Path(a.out_dir), Path(a.reference)
    if out.resolve() == ref.resolve():
        raise SystemExit("out_dir must be a new folder, not the reference dataset")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(a.preprocess_dir).resolve()))
    from mpds_preprocess import consensus, pipeline, targets

    entries, reports, _, _ = pipeline.phase1_load_and_filter(Path(a.json_dir))
    n_label = sum(has_two_liquid_label(e) for e in entries)
    logger.info("%d entries kept; %d carry an 'L1 + L2' label", len(entries), n_label)

    cons_old = pipeline.phase2_consensus(copy.copy(entries), reports)
    install({"consensus": consensus, "targets": targets})
    cons_new = pipeline.phase2_consensus(entries, reports)

    pair_ids = [f"{c.pair_key[0]}_{c.pair_key[1]}" for c in cons_new]
    ref_features = pd.read_parquet(ref / "features.parquet")
    if list(ref_features.index) != pair_ids:
        raise SystemExit("systems differ from the reference dataset; refusing to mix them")

    vocab = targets.build_phase_vocabulary(cons_new, min_occurrences=5)
    recs = [targets.extract_targets(c, vocab) for c in cons_new]
    grid = cons_new[0].comp_grid
    ids = np.asarray(pair_ids, dtype=str)

    np.savez_compressed(out / "targets_liquidus.npz",
                        T=np.vstack([t.m2_liquidus_T for t in recs]).astype(np.float32),
                        MAD=np.vstack([t.m2_liquidus_T_mad for t in recs]).astype(np.float32),
                        mask=np.vstack([t.m2_liquidus_mask for t in recs]).astype(bool),
                        comp_grid=grid.astype(np.float32), pair_ids=ids)
    np.savez_compressed(out / "targets_solidus.npz",
                        T=np.vstack([t.m3_solidus_T for t in recs]).astype(np.float32),
                        MAD=np.vstack([t.m3_solidus_T_mad for t in recs]).astype(np.float32),
                        comp_grid=grid.astype(np.float32), pair_ids=ids)
    bino = [binodal_curve(c) for c in cons_new]
    np.savez_compressed(out / "targets_binodal.npz",
                        T=np.vstack([b[0] for b in bino]).astype(np.float32),
                        n_sources=np.vstack([b[1] for b in bino]).astype(np.int16),
                        comp_grid=grid.astype(np.float32), pair_ids=ids)

    ref_scalar = pd.read_parquet(ref / "targets_scalar.parquet").reindex(pair_ids)
    scalar = ref_scalar.copy()
    scalar["m5_miscibility_gap"] = [t.m5_miscibility_gap for t in recs]
    scalar["m6_topology"] = [t.m6_topology for t in recs]
    scalar.to_parquet(out / "targets_scalar.parquet")

    for name in ("features.parquet", "weights.parquet", "splits.json", "targets_phases.npz"):
        shutil.copy2(ref / name, out / name)

    # What changed, per system
    ref_liq = np.load(ref / "targets_liquidus.npz")
    new_T = np.vstack([t.m2_liquidus_T for t in recs])
    new_mask = np.vstack([t.m2_liquidus_mask for t in recs]).astype(bool)
    both = ref_liq["mask"].astype(bool) & new_mask
    diff = np.where(both, new_T - ref_liq["T"], 0.0)
    changes = pd.DataFrame({
        "pair": pair_ids,
        "gap_reference": ref_scalar["m5_miscibility_gap"].to_numpy(bool),
        "gap_new": scalar["m5_miscibility_gap"].to_numpy(bool),
        "gap_vote_fraction": [c.miscibility_vote_fraction for c in cons_new],
        "topology_reference": ref_scalar["m6_topology"].to_numpy(),
        "topology_new": scalar["m6_topology"].to_numpy(),
        "liquidus_points_changed": (np.abs(diff) > 1.0).sum(axis=1),
        "liquidus_mean_change_K": diff.sum(axis=1) / np.maximum((np.abs(diff) > 1.0).sum(axis=1), 1),
        "binodal_points": [int(np.isfinite(b[0]).sum()) for b in bino],
    }).set_index("pair")
    changes.to_csv(out / "changes_vs_reference.csv")

    manifest = {
        "derived_from": str(ref), "raw": str(a.json_dir), "rule": "L1 + L2 regions count as liquid",
        "n_systems": len(pair_ids),
        "entries_with_two_liquid_label": int(n_label), "two_liquid_labels_seen": STATS["labels"],
        "domes_cut_from_source_liquidus": STATS["domes_cut"],
        "two_liquid_regions_used_for_solidus": STATS["solidus_regions"],
        "two_liquid_regions_rejected_for_solidus": STATS["solidus_regions_rejected"],
        "gap_reference": int(changes.gap_reference.sum()), "gap_new": int(changes.gap_new.sum()),
        "systems_with_liquidus_changed": int((changes.liquidus_points_changed > 0).sum()),
        "liquidus_points_changed": int(changes.liquidus_points_changed.sum()),
        "systems_with_binodal": int((changes.binodal_points > 0).sum()),
        "topology_reference": ref_scalar["m6_topology"].value_counts().to_dict(),
        "topology_new": scalar["m6_topology"].value_counts().to_dict(),
        "consensus_old_matches_reference_gap": bool(
            [c.miscibility_gap for c in cons_old] == list(ref_scalar["m5_miscibility_gap"])),
    }
    with (out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))
    print("\nTopology, reference (rows) vs new (columns):")
    print(pd.crosstab(changes.topology_reference, changes.topology_new).to_string())


if __name__ == "__main__":
    main()
