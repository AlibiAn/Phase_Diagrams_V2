"""Out-of-fold v16 / v17 / v18 liquidus from the caches of explore_v18.py (tag
pre-cleanup), so experiments can be compared with v18 without the GPU. The
same 5 folds as cross_validate.py; checked against the recorded 96.1 / 91.7 / 89.2 K."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mpds_models.curves import fill_curves, melting_point_line

DATA, FEATURES, EXTRA = "/home/alibi/lab/processed_v8", "extended+mp", "/home/alibi/lab/extra_v8_all"
CACHE = Path("/home/alibi/lab/cache_v18")


def _key(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def v18_oof(ds) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{"v16": (liq, sol), "v17": ..., "v18": ...} for the rows of `ds` (processed_v8, clipped tops masked)."""
    c = np.load(CACHE / f"curves_{_key(DATA, FEATURES, EXTRA, {})}.npz")
    liq, sol = c["liq"], c["sol"]
    frac = ds.comp_grid.astype(float) / 100.0
    tm_a, tm_b = ds.melting_points()
    line = melting_point_line(tm_a, tm_b, frac)
    h = np.mean([np.load(CACHE / f"fm_height_{m}_{_key(DATA, FEATURES, 'height', True, False)}.npy")
                 for m in ("tabpfn", "tabicl")], axis=0)
    w = frac * (1 - frac)
    w /= w.mean()
    l17 = (liq + 0.5 * (h - (liq - line).mean(axis=1))[:, None] * w[None]).astype(np.float32)
    s17 = np.minimum(sol, l17)
    fm = np.load(CACHE / f"pca_fm_10_{_key(DATA, FEATURES)}.npy")
    l18 = (0.5 * l17 + 0.5 * (line + fm)).astype(np.float32)
    return {"v16": (liq, sol), "v17": (l17, s17), "v18": (l18, np.minimum(s17, l18))}


if __name__ == "__main__":
    import logging
    from cross_validate import folds
    from mpds_models.data import curve_scores, load_dataset, mask_clipped_tops
    logging.basicConfig(level=logging.WARNING)
    ds = load_dataset(DATA, FEATURES)
    mask_clipped_tops(ds)
    for name, (l, s) in v18_oof(ds).items():
        f = [curve_scores(ds, te, l[te], s[te]) for _, te in folds(ds.n_systems)]
        print(name, f"{np.mean([x['liquidus_MAE_K'] for x in f]):.1f} / {np.mean([x['solidus_MAE_K'] for x in f]):.1f} K")
