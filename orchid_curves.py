"""
orchid_curves.py — liquidus and solidus curves from the ORCHID calculated
binary phase diagrams.

Source: ORCHID-Diagrams v2026.10 (Arroyave, Saal, Shin; doi:10.5281/zenodo.23735859,
https://github.com/orchid-calphad/orchid-data, folder phase-diagrams/, CC BY 4.0),
calculated with pycalphad from the SGTE binary collection of assessed
thermodynamic databases (Hallstedt 2025, CC BY 4.0). Cite both, and the
original assessment named in each file.

Each file holds the stable phases on a grid of 101 compositions x 10 K steps.
Two coexisting liquids are both called LIQUID, so

    liquidus(x) = top of the highest cell that contains a solid phase
    solidus(x)  = bottom of the lowest cell that contains LIQUID or GAS

which is the convention of processed_v8 (under a two-liquid dome the liquidus
is the monotectic line). The grid step makes both curves exact to about 5 K.
A system's curves are the median over its assessments.

    python orchid_curves.py /home/alibi/lab/external/orchid/repo/phase-diagrams \
        --compare /home/alibi/lab/processed_v8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLUID = ("LIQUID", "GAS")


def decode(grid: dict) -> np.ndarray:
    """(n_T, n_x) array of label indices from the run-length rows."""
    return np.array([np.repeat([r[0] for r in row], [r[1] for r in row]) for row in grid["rows"]])


def curves(diagram: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(liquidus, solidus) in K on 0..100 at% of the alphabetically second
    element; NaN where the grid does not reach the boundary or a cell is empty."""
    grid = diagram.get("grid")
    if diagram.get("status") != "ok" or not isinstance(grid, dict) or grid.get("x_points") != 101:
        return None
    cells = decode(grid)
    if cells.ndim != 2 or cells.shape[1] != 101:
        return None
    labels = [set(l.split("+")) if l else None for l in grid["labels"]]
    empty = np.array([l is None for l in labels])[cells]
    has_solid = np.array([l is not None and any(p not in FLUID for p in l) for l in labels])[cells]
    has_fluid = np.array([l is not None and any(p in FLUID for p in l) for l in labels])[cells]
    t0, dt, n = float(grid["T0"]), float(grid["dT"]), cells.shape[0]
    liq, sol = np.full(101, np.nan), np.full(101, np.nan)
    for j in range(101):
        solid_rows, fluid_rows = np.flatnonzero(has_solid[:, j]), np.flatnonzero(has_fluid[:, j])
        if solid_rows.size and solid_rows[-1] < n - 1 and not empty[solid_rows[-1] + 1, j]:
            liq[j] = t0 + (solid_rows[-1] + 0.5) * dt
        if fluid_rows.size and fluid_rows[0] > 0 and not empty[fluid_rows[0] - 1, j]:
            sol[j] = t0 + (fluid_rows[0] - 0.5) * dt
    if diagram["x_component"].capitalize() != sorted(e.capitalize() for e in diagram["components"])[1]:
        liq, sol = liq[::-1], sol[::-1]
    return liq, np.where(sol > liq, liq, sol)


def system_curves(folder: str | Path) -> dict[str, dict]:
    """{pair id 'A_B': dict(liquidus, solidus, n_assessments, citations)}."""
    folder = Path(folder)
    index = json.loads((folder / "index.json").read_text())["systems"]
    out = {}
    for entries in index.values():
        liqs, sols, cites, pid = [], [], [], None
        for e in entries:
            path = folder / "sgte" / e["file"]
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            c = curves(d)
            if c is None or np.isfinite(c[0]).sum() < 30:
                continue
            pid = "_".join(sorted(x.capitalize() for x in d["components"]))
            liqs.append(c[0])
            sols.append(c[1])
            cites.append(e.get("citation") or e["file"])
        if liqs:
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[pid] = dict(liquidus=np.nanmedian(liqs, axis=0), solidus=np.nanmedian(sols, axis=0),
                                    n_assessments=len(liqs), citations=cites)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="ORCHID phase-diagrams folder (index.json and sgte/)")
    ap.add_argument("--compare", default=None, help="processed dataset to compare the shared systems with")
    a = ap.parse_args()
    systems = system_curves(a.folder)
    print(f"{len(systems)} systems with a usable liquidus, "
          f"{sum(s['n_assessments'] for s in systems.values())} assessments")
    if a.compare:
        from mpds_models.base import load_dataset, mask_clipped_tops
        ds = load_dataset(a.compare)
        mask_clipped_tops(ds)
        dl, dsol = [], []
        for i, pid in enumerate(ds.pair_ids):
            if pid not in systems:
                continue
            m = ds.T_liquidus_mask[i] & np.isfinite(systems[pid]["liquidus"])
            if m.sum() >= 30:
                dl.append(np.abs(systems[pid]["liquidus"] - ds.T_liquidus[i])[m].mean())
            ms = np.isfinite(ds.T_solidus[i]) & np.isfinite(systems[pid]["solidus"])
            if ms.sum() >= 30:
                dsol.append(np.abs(systems[pid]["solidus"] - ds.T_solidus[i])[ms].mean())
        for name, d in (("liquidus", np.array(dl)), ("solidus", np.array(dsol))):
            print(f"{name}: {len(d)} shared systems; mean |ORCHID - {Path(a.compare).name}| per system: "
                  f"median {np.median(d):.0f} K, 75th percentile {np.percentile(d, 75):.0f} K, "
                  f"90th {np.percentile(d, 90):.0f} K; within 25 K: {100 * (d < 25).mean():.0f} %, "
                  f"within 50 K: {100 * (d < 50).mean():.0f} %")
        new = [p for p in systems if p not in set(ds.pair_ids)]
        full = sum(np.isfinite(systems[p]["liquidus"]).sum() >= 95 for p in new)
        print(f"{len(new)} systems are not in {Path(a.compare).name}; {full} of them have a full-range liquidus")


if __name__ == "__main__":
    main()
