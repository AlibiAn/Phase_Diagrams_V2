"""
check_solidus.py — find out WHY the lowest solidus sits below the listed
eutectic temperature, and which region of which source file causes it.

Usage
-----
    python check_solidus.py processed_v4 mpds_data

Report [1] sorts every eutectic system into one cause:
  ok              lowest solidus within 30 K of the listed eutectic
  element end     the solidus bottoms out at a pure element's melting point
                  (a eutectic right next to a low-melting element that the
                  file's eutectic list doesn't include). Physically fine.
  -273 K          off by about 273 K: a unit mismatch inside some files
  diagram floor   solidus reaches the bottom of the drawn diagram: a fully
                  solid region was taken as containing liquid (my bug)
  other           anything else

Report [2] (needs the raw JSON folder) traces the worst "diagram floor" and
"other" systems back to the source file and region that set the minimum.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mpds_models.base import load_dataset

TOL_K = 30.0


def classify(ds):
    ia = ds.feature_names.index("phys__melting_point__A")
    ib = ds.feature_names.index("phys__melting_point__B")
    TmA, TmB = ds.X[:, ia], ds.X[:, ib]
    rows = []
    for i in range(ds.n_systems):
        if not ds.eutectic_exists[i] or not np.isfinite(ds.eutectic_temp[i]):
            continue
        s = ds.T_solidus[i]
        if not np.isfinite(s).any():
            continue
        j = int(np.nanargmin(s))
        smin, x = float(s[j]), float(ds.comp_grid[j])
        d = smin - float(ds.eutectic_temp[i])
        tm_low = min(TmA[i], TmB[i])
        if abs(d) < TOL_K:
            kind = "ok"
        elif d > 0:
            kind = "above eutectic"
        elif abs(d + 273.15) < TOL_K:
            kind = "-273 K"
        elif abs(smin - tm_low) < TOL_K or (x <= 3 or x >= 97) and smin >= tm_low - TOL_K:
            kind = "element end"
        elif smin < 0.6 * tm_low and smin < 0.6 * float(ds.eutectic_temp[i]):
            kind = "diagram floor"
        else:
            kind = "other"
        rows.append(dict(i=i, pair=ds.pair_ids[i], kind=kind, smin=smin, x=x,
                         eut=float(ds.eutectic_temp[i]), d=d, tm_low=float(tm_low)))
    return rows


def trace(json_dir: Path, pairs: list[str], grid: np.ndarray) -> None:
    """For each pair, find which source + region sets the lowest solidus."""
    from mpds_preprocess.loader import load_directory, quality_filter
    from mpds_preprocess.units import normalise_entry, liquidus_points, polygon_lower_edge_at
    from mpds_preprocess.targets import (_polygon_xy, _region_contains_liquid,
                                         _two_phase_touches_liquid)
    wanted = {tuple(p.split("_")) for p in pairs}
    by_pair: dict = {}
    for e in load_directory(json_dir):
        key = tuple(sorted(e.get("elements") or []))
        if key not in wanted or not quality_filter(e).passed:
            continue
        if normalise_entry(e).unit == "?":
            continue
        by_pair.setdefault(key, []).append(e)

    for p in pairs:
        key = tuple(p.split("_"))
        print(f"\n  {p}")
        for e in by_pair.get(key, []):
            flip = e["elements"] != list(key)
            polys = [(q, *xy) for q in e.get("phases", []) for xy in [_polygon_xy(q)]
                     if xy is not None and xy[0].size >= 2]
            ref = [k for k, (q, _, _) in enumerate(polys)
                   if _region_contains_liquid(q) and q.get("phases") == 1]
            lv = (np.concatenate([np.column_stack([polys[k][1], polys[k][2]]) for k in ref])
                  if ref else np.empty((0, 2)))
            liq = liquidus_points(e)
            liq = liq[np.argsort(liq[:, 0])] if liq.size else np.empty((0, 2))
            best = None
            for k, (q, x, y) in enumerate(polys):
                if _region_contains_liquid(q):
                    why = "liquid by label"
                elif q.get("phases") == 2 and _two_phase_touches_liquid(x, y, lv, liq):
                    why = "touches liquid (geometry)"
                else:
                    continue
                xs = 100.0 - x if flip else x
                low = polygon_lower_edge_at(xs, y, grid)
                if np.isfinite(low).any():
                    m = float(np.nanmin(low))
                    if best is None or m < best[0]:
                        best = (m, k, q, x, y, why, float(grid[np.nanargmin(low)]))
            eut = e.get("eutectics") or []
            if best is None:
                print(f"    {e['entry']}: no liquid-containing region found | eutectics {eut}")
                continue
            m, k, q, x, y, why, xm = best
            print(f"    {e['entry']}: lowest solidus {m:.0f} K at x={xm:.0f} from region #{k} "
                  f"'{q.get('label')}' phases={q.get('phases')} [{why}] "
                  f"x {x.min():.1f}..{x.max():.1f}, T {y.min():.0f}..{y.max():.0f} K, "
                  f"{x.size} corners | listed eutectics (K) {[[round(a, 1), round(b)] for a, b in eut]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("json_dir", nargs="?", help="raw MPDS JSON folder, for report [2]")
    ap.add_argument("--n", type=int, default=8, help="systems to trace per cause")
    a = ap.parse_args()

    ds = load_dataset(a.processed_dir)
    rows = classify(ds)
    c = Counter(r["kind"] for r in rows)
    print(f"\n[1] Causes, over {len(rows)} eutectic systems:")
    for kind in ("ok", "element end", "-273 K", "diagram floor", "other", "above eutectic"):
        print(f"  {kind:15s} {c.get(kind, 0):5d}  ({100 * c.get(kind, 0) / max(len(rows), 1):.0f}%)")

    for kind in ("diagram floor", "other", "-273 K"):
        sel = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["d"])[:a.n]
        if not sel:
            continue
        print(f"\n  worst '{kind}' systems:")
        for r in sel:
            print(f"    {r['pair']:10s} lowest solidus {r['smin']:6.0f} K at x={r['x']:3.0f} | "
                  f"eutectic {r['eut']:6.0f} K | lower Tm {r['tm_low']:6.0f} K")

    if a.json_dir:
        print("\n[2] Tracing to source files and regions:")
        pick = []
        for kind in ("diagram floor", "other", "-273 K"):
            pick += [r["pair"] for r in sorted((r for r in rows if r["kind"] == kind),
                                               key=lambda r: r["d"])[:a.n]]
        trace(Path(a.json_dir), pick, ds.comp_grid)


if __name__ == "__main__":
    main()