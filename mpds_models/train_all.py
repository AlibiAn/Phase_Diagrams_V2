"""
Training driver: fits M1, M2/M3, M4, M5 and M6 (solo + stacked meta) on the
preprocessing artefacts and saves every model plus metrics.json.

Usage
-----
    from mpds_models.train_all import train_all
    artefacts = train_all("/data/processed/", "/data/models/")

Or as a script:
    python -m mpds_models.train_all /data/processed /data/models
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .base import load_dataset, detect_device
from .m1_phase import PhaseClassifier
from .m4_eutectic import EutecticDetector
from .m5_miscibility import MiscibilityGapDetector
from .m6_topology import TopologyClassifier
from .deep_curve_net import LiquidusSolidusModel, CurveNetConfig
from .meta_learner import StackedTopologyClassifier, FullDiagramReconstructor

logger = logging.getLogger(__name__)


def _val_test(model, ds, *args) -> dict:
    return {split: model.evaluate(ds, *args, split=split) for split in ("val", "test")}


def train_all(
    processed_dir: str,
    out_dir: str,
    *,
    curve_epochs: int = 200,
    train_stacked: bool = True,
) -> dict:
    """Train all specialists and (optionally) the stacked meta-learner.

    Writes the model files and metrics.json to `out_dir`; returns the metrics.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(processed_dir)
    device = detect_device()
    logger.info("Compute device: %s", device)
    metrics: dict = {"_device": device, "_n_systems": int(ds.n_systems)}

    logger.info("=== M1: phase classifier ===")
    m1 = PhaseClassifier().fit(ds)
    m1.save(out / "m1_phase.joblib")
    metrics["M1"] = _val_test(m1, ds)

    logger.info("=== M2+M3: joint liquidus/solidus curve net ===")
    m23 = LiquidusSolidusModel(CurveNetConfig(
        epochs=curve_epochs,
        batch_size=min(32, max(8, len(ds.train_idx) // 4)),
        lambda_endpoints=10.0,
        lambda_smooth=0.1,
    )).fit(ds)
    m23.save(out / "m23_curves.pt")
    metrics["M2M3"] = _val_test(m23, ds)

    logger.info("=== M4: eutectic detector ===")
    m4 = EutecticDetector().fit(ds)
    m4.save(out / "m4_eutectic.joblib")
    metrics["M4"] = _val_test(m4, ds)

    logger.info("=== M5: miscibility gap ===")
    m5 = MiscibilityGapDetector().fit(ds)
    m5.save(out / "m5_miscibility.joblib")
    metrics["M5"] = _val_test(m5, ds)

    logger.info("=== M6 (solo): topology classifier ===")
    m6_solo = TopologyClassifier().fit(ds)
    m6_solo.save(out / "m6_topology_solo.joblib")
    metrics["M6_solo"] = _val_test(m6_solo, ds)

    meta = None
    if train_stacked and len(ds.train_idx) >= 20:
        logger.info("=== Stacked meta-learner ===")
        meta = StackedTopologyClassifier().fit(ds)
        meta.save(out / "m6_topology_stacked.joblib")
        metrics["M6_stacked"] = _val_test(meta, ds, {"m1": m1, "m23": m23, "m4": m4, "m5": m5})
    elif train_stacked:
        logger.info("Skipping stacked meta-learner (train size %d < 20)", len(ds.train_idx))

    # Sanity demo: reconstruct full diagrams for the test split
    if len(ds.test_idx) > 0:
        recon = FullDiagramReconstructor(
            m1=m1, m23=m23, m4=m4, m5=m5,
            m6_meta=meta, m6_solo=m6_solo,
            topology_classes=ds.topology_classes,
            phase_vocab=ds.phase_vocab, comp_grid=ds.comp_grid,
        )
        test_pairs = [ds.pair_ids[i] for i in ds.test_idx]
        diagrams = recon.reconstruct(ds.X[ds.test_idx], test_pairs, with_uncertainty=False)
        metrics["reconstruction_demo"] = {
            "n_diagrams": len(diagrams),
            "n_with_inconsistencies": sum(bool(d.inconsistencies) for d in diagrams),
            "examples": [
                {
                    "pair": d.pair,
                    "topology": d.topology,
                    "eutectic": (d.eutectic_comp, d.eutectic_temp) if d.eutectic_exists else None,
                    "inconsistencies": d.inconsistencies,
                }
                for d in diagrams[:5]
            ],
        }

    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Wrote metrics + model files to %s", out)
    return metrics


def _cli() -> None:
    p = argparse.ArgumentParser(description="Train all M1-M6 + meta-learner.")
    p.add_argument("processed_dir", help="output directory from preprocessing pipeline")
    p.add_argument("out_dir", help="where to write trained model files + metrics.json")
    p.add_argument("--curve-epochs", type=int, default=200)
    p.add_argument("--no-stacked", action="store_true", help="skip meta-learner training")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s : %(message)s",
        datefmt="%H:%M:%S",
    )
    metrics = train_all(
        args.processed_dir, args.out_dir,
        curve_epochs=args.curve_epochs,
        train_stacked=not args.no_stacked,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    _cli()
