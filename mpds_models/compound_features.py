"""
EXPERIMENTAL features: how stable the compounds of A and B are, from computed
(DFT, zero kelvin) formation energies. Not part of v14.

Why: the largest part of the liquidus error is each system's overall height
above or below the melting-point line, which is set by how strongly the two
elements bind (a very stable compound melts far above both elements). The
Miedema enthalpies in the other features estimate this for metals only and
have no parameters for S, Se and Te.

Source: Materials Project snapshot of 18 Oct 2018 as distributed by matminer
(dataset "mp_nostruct_20181018", 83,989 entries with formula, formation energy
and energy above the convex hull; Jain et al., APL Materials 1 (2013) 011002;
Materials Project data are CC BY 4.0). Download once with

    from matminer.datasets import load_dataset
    load_dataset("mp_nostruct_20181018", data_home="/home/alibi/lab/external")

CAVEAT, read before using these in a headline number: most Materials Project
entries are calculations of compounds that are known experimentally. Whether a
system has entries, and how many are stable, therefore tells the model which
compounds have been observed, which is knowledge of the phase diagram itself.
For a system nobody has studied these features would be nearly empty. The
honest version of this idea computes the same quantities from a database that
enumerates hypothetical structures for every pair (OQMD prototypes, AFLOW).

Features of the ordered pair (A, B), x = atomic fraction of B:
  cmp__n_entries / n_stable / n_near   binary entries; on the hull; within 30 meV of it
  cmp__min_eform                       lowest formation energy (eV/atom; may be > 0)
  cmp__x_at_min                        its composition
  cmp__hull_XX                         depth of the convex hull at XX at% B (<= 0)
  cmp__hull_mean                       mean depth over composition
  cmp__x_first_stable / x_last_stable  stable compounds closest to A and to B
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HULL_POINTS = (10, 25, 33, 50, 67, 75, 90)
STABLE_EV, NEAR_EV = 1e-3, 0.03


def load_binary_entries(path: str | Path) -> dict[frozenset, np.ndarray]:
    """{frozenset({A, B}): array of rows (fraction of the alphabetically
    second element, e_form, e_hull)} from the matminer json.gz file."""
    import pandas as pd
    from pymatgen.core import Composition
    path = Path(path)
    df = pd.read_json(path, orient="split") if path.suffix != ".gz" else pd.read_json(
        path, orient="split", compression="gzip")
    out: dict[frozenset, list] = {}
    for formula, e_form, e_hull in zip(df["formula"], df["e_form"], df["e_hull"]):
        try:
            comp = Composition(formula).fractional_composition.as_dict()
        except Exception:
            continue
        if len(comp) != 2 or not np.isfinite(e_form):
            continue
        a, b = sorted(comp)
        out.setdefault(frozenset((a, b)), []).append((comp[b], e_form, e_hull))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def lower_hull(x: np.ndarray, e: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Lower convex hull of the points (x, e) together with (0, 0) and (1, 0),
    evaluated at the fractions `at`."""
    pts = sorted(set([(0.0, 0.0), (1.0, 0.0)] + [(float(a), float(b)) for a, b in zip(x, e) if b < 0]))
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(at, hx, hy)


def pair_features(a: str, b: str, entries: dict) -> dict[str, float]:
    rows = entries.get(frozenset((a, b)))
    f: dict[str, float] = {}
    at = np.array(HULL_POINTS) / 100.0
    if rows is None or len(rows) == 0:
        f.update(cmp__n_entries=0.0, cmp__n_stable=0.0, cmp__n_near=0.0, cmp__min_eform=0.0,
                 cmp__x_at_min=50.0, cmp__hull_mean=0.0, cmp__x_first_stable=-1.0, cmp__x_last_stable=-1.0)
        f.update({f"cmp__hull_{k}": 0.0 for k in HULL_POINTS})
        return f
    x = rows[:, 0] if sorted((a, b))[1] == b else 1.0 - rows[:, 0]        # fraction of B
    e_form, e_hull = rows[:, 1], rows[:, 2]
    j = int(np.argmin(e_form))
    stable = x[e_hull <= STABLE_EV]
    f["cmp__n_entries"] = float(len(rows))
    f["cmp__n_stable"] = float(len(np.unique(np.round(stable, 3))))
    f["cmp__n_near"] = float(len(np.unique(np.round(x[e_hull <= NEAR_EV], 3))))
    f["cmp__min_eform"] = float(e_form[j])
    f["cmp__x_at_min"] = 100.0 * float(x[j])
    f["cmp__hull_mean"] = float(lower_hull(x, e_form, np.linspace(0, 1, 101)).mean())
    f["cmp__x_first_stable"] = 100.0 * float(stable.min()) if stable.size else -1.0
    f["cmp__x_last_stable"] = 100.0 * float(stable.max()) if stable.size else -1.0
    for k, v in zip(HULL_POINTS, lower_hull(x, e_form, at)):
        f[f"cmp__hull_{k}"] = float(v)
    return f


def build_features(pair_ids: list[str], path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, X_swap, names): the features of each system and of the same system
    with A and B exchanged."""
    entries = load_binary_entries(path)
    rows, rows_swap = [], []
    for pid in pair_ids:
        a, b = pid.split("_")[:2]
        rows.append(pair_features(a, b, entries))
        rows_swap.append(pair_features(b, a, entries))
    names = list(rows[0])
    to_matrix = lambda rs: np.array([[r[n] for n in names] for r in rs], dtype=np.float32)
    n_with = sum(r["cmp__n_entries"] > 0 for r in rows)
    logger.info("Compound features: %d of %d systems have computed entries", n_with, len(pair_ids))
    return to_matrix(rows), to_matrix(rows_swap), names
