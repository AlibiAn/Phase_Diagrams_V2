"""
Train every model of one version into a folder, then score it on the test split.

    MPDS_DEVICE=cpu python -m mpds_models.train /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18
    ... --height 0                v16 (trees only)
    ... --blend 0                 v17 (trees + height correction)
    ... --features extended       without the Materials Project compound features (v14 inputs)
    ... --no-extra-data           without the 404 extra curve systems

MPDS_DEVICE=cpu keeps XGBoost on the CPU (it builds different trees on the
GPU; every version since v13 was trained on the CPU); TabPFN and TabICL use
the GPU when one is visible. Writes the models, config.json and metrics.json
(the test-split report of predict.py, which scores the same way).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .classifiers import xgb_device
from .data import FEATURE_SETS, fix_end_points, load_dataset, mask_clipped_tops, system_quality
from .evaluation import evaluate, print_summary
from .pipeline import PhaseDiagramModel

EXTRA_DATA = "/home/alibi/lab/extra_v8_all"
logger = logging.getLogger("mpds_models.train")


def train(processed_dir: str, out_dir: str, features: str = "extended+mp", extra_data: str | None = EXTRA_DATA,
          height: float = 0.5, blend: float = 0.5, n_components: int = 10, seed: int = 0) -> dict:
    t0 = time.time()
    ds = load_dataset(processed_dir, features)
    ds = replace(ds, train_idx=np.sort(np.concatenate([ds.train_idx, ds.val_idx])))
    quality = system_quality(ds)                        # before masking
    mask_clipped_tops(ds)
    logger.info("Training on %d systems (train + validation); XGBoost on %s", len(ds.train_idx), xgb_device()["device"])
    model = PhaseDiagramModel(height, blend, n_components, seed).fit(ds, extra_data)
    model.save(out_dir)

    score = replace(ds, T_liquidus_mask=ds.T_liquidus_mask.copy(), T_solidus=ds.T_solidus.copy())
    fix_end_points(score)
    idx = ds.test_idx
    pred = model.predict(ds.X[idx])
    diagrams = model.diagrams(pred, [ds.pair_ids[i] for i in idx])
    ev = evaluate(score, idx, pred, quality)
    n_bad = sum(bool(d.inconsistencies) for d in diagrams)
    metrics = dict(config=model.config, xgboost_device=xgb_device()["device"], minutes=round((time.time() - t0) / 60, 1),
                   test=ev, n_inconsistent_diagrams=n_bad)
    (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print_summary(ev, len(idx), n_bad)
    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("processed_dir")
    p.add_argument("out_dir")
    p.add_argument("--features", choices=FEATURE_SETS, default="extended+mp")
    p.add_argument("--extra-data", default=EXTRA_DATA, help="extra systems for the curve model")
    p.add_argument("--no-extra-data", action="store_true")
    p.add_argument("--height", type=float, default=0.5, help="height correction strength (0 = trees only, v16)")
    p.add_argument("--blend", type=float, default=0.5, help="share of the foundation-model curve (0 = v17)")
    p.add_argument("--components", type=int, default=10, help="principal components the foundation models predict")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    train(a.processed_dir, a.out_dir, a.features, None if a.no_extra_data else a.extra_data,
          a.height, a.blend if a.height else 0.0, a.components, a.seed)


if __name__ == "__main__":
    main()
