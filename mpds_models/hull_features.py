"""
Compound-stability features from the computed (DFT, 0 K) binary convex hulls
published by ORCHID (ORCHID-Hulls v2026.10, doi:10.5281/zenodo.23735891, CC BY 4.0):

  oqmd   hulls/dft-oqmd: OQMD v1.8 (Saal et al., JOM 65 (2013) 1501; Kirklin et
         al., npj Comput. Mater. 1 (2015) 15010; CC BY 4.0). OQMD calculates
         hypothetical prototype compounds for every element pair as well as the
         known ones, so these features exist for systems nobody has studied
         (3,916 binary systems).
  mp26   hulls/dft-mp: Materials Project release 2026.04.13 (Jain et al., APL
         Mater. 1 (2013) 011002; CC BY 4.0). Mostly compounds known from
         experiment (the caveat of compound_features.py applies).

Features of the ordered pair (A, B), x = atomic fraction of B, prefix "<source>__":
  hull_XX            depth of the lower convex hull at XX at% B (eV/atom, <= 0)
  hull_mean          its mean over composition
  min_energy, x_at_min   the deepest hull vertex and its composition
  n_vertices         stable compounds (hull vertices strictly between the elements)
  x_first, x_last    stable compounds closest to A and to B (-1 if none)
Missing systems get zeros. The number of calculated structures and whether the
stable ones are hypothetical are left out on purpose: they mostly measure how
well a system has been studied, not its chemistry.

    python -m mpds_models.hull_features /home/alibi/lab/external/orchid/repo/hulls Ag_Al Cu_Pb
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HULLS_DIR = "/home/alibi/lab/external/orchid/repo/hulls"     # env MPDS_HULLS_DIR overrides
SOURCES = {"oqmd": "dft-oqmd", "mp26": "dft-mp"}
POINTS = (10, 25, 33, 50, 67, 75, 90)


def _dir() -> Path:
    return Path(os.environ.get("MPDS_HULLS_DIR", HULLS_DIR))


@lru_cache(maxsize=None)
def _vertices(source: str) -> dict:
    """{frozenset({A, B}): (x_element, [(x, energy, theoretical)], n_entries)}."""
    root = _dir() / SOURCES[source]
    out = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sub.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except ValueError:
                continue
            els = [e.capitalize() for e in (d.get("elements") or [])]
            if len(els) != 2 or not d.get("isotherms"):
                continue
            iso = d["isotherms"][0]
            verts = [(float(v["x"]), float(v["energy"]), bool(v.get("theoretical", False)))
                     for v in iso.get("vertices", []) if v.get("x") is not None and v.get("energy") is not None]
            key = frozenset(els)
            if key not in out:
                out[key] = (str(d.get("x_element", "")).capitalize(), verts,
                            int((d.get("counts") or {}).get("entries", 0)))
    logger.info("%s hulls: %d binary systems", source, len(out))
    return out


def _features(source: str, a: str, b: str) -> dict[str, float]:
    p = f"{source}__"
    names = [f"hull_{k}" for k in POINTS] + ["hull_mean", "min_energy", "x_at_min", "n_vertices",
                                             "x_first", "x_last"]
    hit = _vertices(source).get(frozenset((a, b)))
    if hit is None:
        return {p + n: (-1.0 if n in ("x_first", "x_last") else 0.0) for n in names}
    x_el, verts, n_entries = hit
    pts = sorted(((x if x_el == b else 1.0 - x), e, t) for x, e, t in verts)
    xs = np.array([v[0] for v in pts])
    es = np.minimum(np.array([v[1] for v in pts]), 0.0)
    if xs.size == 0 or xs[0] > 1e-6:
        xs, es = np.r_[0.0, xs], np.r_[0.0, es]
    if xs[-1] < 1 - 1e-6:
        xs, es = np.r_[xs, 1.0], np.r_[es, 0.0]
    hull = np.interp(np.linspace(0, 1, 101), xs, es)
    inner = [(x, e, t) for x, e, t in pts if 1e-6 < x < 1 - 1e-6]
    f = {p + f"hull_{k}": float(np.interp(k / 100.0, xs, es)) for k in POINTS}
    j = int(np.argmin(hull))
    f.update({p + "hull_mean": float(hull.mean()), p + "min_energy": float(hull[j]), p + "x_at_min": float(j),
              p + "n_vertices": float(len(inner)),
              p + "x_first": 100.0 * inner[0][0] if inner else -1.0,
              p + "x_last": 100.0 * inner[-1][0] if inner else -1.0})
    return f


def build_features(pair_ids: list[str], source: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, X_swap, names) for pair ids such as 'Ag_Al'."""
    rows, rows_swap = [], []
    for pid in pair_ids:
        a, b = pid.split("_")[:2]
        rows.append(_features(source, a, b))
        rows_swap.append(_features(source, b, a))
    names = list(rows[0])
    mat = lambda rs: np.array([[r[n] for n in names] for r in rs], dtype=np.float32)
    n_hit = sum(frozenset(p.split("_")[:2]) in _vertices(source) for p in pair_ids)
    logger.info("%s hull features: %d of %d systems covered", source, n_hit, len(pair_ids))
    return mat(rows), mat(rows_swap), names


if __name__ == "__main__":
    import sys
    os.environ["MPDS_HULLS_DIR"] = sys.argv[1]
    for src in SOURCES:
        X, _, names = build_features(sys.argv[2:], src)
        for pid, row in zip(sys.argv[2:], X):
            print(src, pid, {n.split("__")[1]: round(float(v), 3) for n, v in zip(names, row)})
