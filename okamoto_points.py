"""
okamoto_points.py — liquidus points of binary systems from ORCHID's assessed
phase assemblages.

Source: ORCHID-Assemblages v2026.10 (data.orchid-project.org, CC BY 4.0;
doi in its CITATION.cff), one point deep inside each phase field of the
expert-drawn binary diagrams of H. Okamoto, Desk Handbook: Phase Diagrams for
Binary Alloys, 2nd ed. (ASM International, 2010), 2,178 diagrams. No handbook
drawing is used, only these points.

A point in a field "liquid + solid" carries the tie line through it: the
compositions of the liquid and of the solid at that temperature. The liquid
end lies exactly on the liquidus, so every such point gives one liquidus
point (x_liquid, T). Liquids are labelled L, L#2, L#3 ...; a field with two
liquids marks a liquid miscibility gap. On 1,080 systems shared with
processed_v8 these liquidus points agree with our liquidus to a median of
6 K (75 % within 29 K). They are sparse: a median of 3 per system.

    python okamoto_points.py /home/alibi/lab/external/orchid/assemblages/records.parquet
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

LIQUID = re.compile(r"^(L(#\d+)?|Liquid\d?)$")


def _pair(system: str) -> str | None:
    parts = system.split("-")
    return "_".join(sorted(p.capitalize() for p in parts)) if len(parts) == 2 else None


def liquidus_points(records: str) -> dict[str, dict]:
    """{pair id: dict(x=at% of the second element, T=K, gap=bool, n_points)}."""
    r = pd.read_parquet(records)
    r["pair"] = r.system.map(_pair)
    r = r[r.pair.notna()]
    labels = r.phase_labels.fillna("").str.split(";")
    r["n_liquid"] = labels.map(lambda ls: sum(bool(LIQUID.match(t.strip())) for t in ls))
    gap = r.groupby("pair").n_liquid.max() >= 2
    tl = r[(r.n_phases == 2) & (r.n_liquid == 1) & r.tie_line_x_left.notna() & r.tie_line_x_right.notna()]
    left_is_liquid = tl.tie_line_phase_left.fillna("").str.strip().str.match(LIQUID.pattern)
    x_liq = np.where(left_is_liquid, tl.tie_line_x_left, tl.tie_line_x_right).astype(float)
    second = tl.pair.str.split("_").str[1]
    x_b = np.where(tl.x_component.str.capitalize() == second, x_liq, 1.0 - x_liq) * 100.0
    pts = pd.DataFrame({"pair": tl.pair.to_numpy(), "x": x_b, "T": tl.T_K.to_numpy(dtype=float)})
    pts = pts[(pts.x > 0.5) & (pts.x < 99.5) & np.isfinite(pts["T"])]
    out = {}
    for p, g in pts.groupby("pair"):
        out[p] = dict(x=g.x.to_numpy(), T=g["T"].to_numpy(), gap=bool(gap.get(p, False)), n_points=len(g))
    return out


def on_grid(points: dict, grid: np.ndarray) -> np.ndarray:
    """Liquidus on the composition grid: each point on its nearest grid point
    (averaged where several fall together), NaN elsewhere."""
    out = np.full(grid.shape, np.nan)
    j = np.clip(np.rint(np.interp(points["x"], grid, np.arange(grid.size))).astype(int), 0, grid.size - 1)
    for k in np.unique(j):
        out[k] = points["T"][j == k].mean()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records")
    a = ap.parse_args()
    s = liquidus_points(a.records)
    n = np.array([v["n_points"] for v in s.values()])
    print(f"{len(s)} systems with liquidus points; {n.sum()} points; median {np.median(n):.0f} per system; "
          f"{(n >= 3).sum()} systems with >= 3; {sum(v['gap'] for v in s.values())} with a liquid miscibility gap")
