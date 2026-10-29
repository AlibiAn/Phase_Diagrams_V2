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
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .base import detect_device, fix_end_points, load_dataset, mask_clipped_tops
from .m1_phase import PhaseClassifier
from .m4_eutectic import EutecticDetector, EutecticDetectorConfig
from .m5_miscibility import MiscibilityGapDetector
from .m6_topology import TopologyClassifier
from .deep_curve_net import LiquidusSolidusModel, CurveNetConfig
from .tree_curves import CURVE_BACKENDS, TreeCurveConfig, TreeCurveModel
from .curve_ensemble import CurveEnsemble, CurveEnsembleConfig
from .meta_learner import FullDiagramReconstructor, StackedMetaConfig, StackedTopologyClassifier

logger = logging.getLogger(__name__)


def _val_test(model, ds, *args) -> dict:
    return {split: model.evaluate(ds, *args, split=split) for split in ("val", "test")}


def train_all(
    processed_dir: str,
    out_dir: str,
    *,
    curve_epochs: int = 200,
    train_stacked: bool = True,
    endpoint_mode: str = "penalty",
    solidus_mode: str = "gap",
    lambda_endpoints: float = 1.0,
    lambda_smooth: float = 0.1,
    seed: int = 0,
    m4_backend: str = "xgboost",
    curve_model: str = "curvenet",
    mask_end_points: bool = False,
    mask_clipped: bool = False,
    m4_comp_from_lower_tm: bool = False,
    fix_ends: bool = False,
    swap_augment: bool = False,
) -> dict:
    """Train all specialists and (optionally) the stacked meta-learner.

    Writes the model files and metrics.json to `out_dir`; returns the metrics.
    `endpoint_mode="hard"` makes every curve pass exactly through Tm(A) and
    Tm(B), so the endpoint penalty (`lambda_endpoints`) no longer applies.
    `solidus_mode="gap"` keeps the solidus below the liquidus by construction.
    `seed` makes the curve networks reproducible (exactly so on CPU; GPU
    kernels can still differ slightly between runs).
    `m4_backend` picks the eutectic models: "xgboost", "random_forest" or "extra_trees".
    `curve_model` is "curvenet", a tree ensemble ("extra_trees", "random_forest",
    "xgboost"; see tree_curves.py) or "ensemble" (the average of CurveNet and
    ExtraTrees, see curve_ensemble.py), saved as m23_curves.pt /
    m23_curves.joblib / both plus m23_ensemble.json.
    `mask_end_points` leaves CurveNet's x = 0 / 100 data values out of its data
    loss (tree curves always pin their ends to Tm). `mask_clipped` treats flat
    liquidus tops cut off by the source chart as unobserved (base.mask_clipped_tops).
    `m4_comp_from_lower_tm` makes M4 predict the eutectic composition measured
    from the lower-melting element.
    `fix_ends` trains the curve model on targets whose wrong pure-element end
    points are set to Tm (base.fix_end_points); its scores in metrics.json
    leave those points out. `swap_augment` also trains the curve model on
    every training system with A and B swapped (base.swap_augment).
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(processed_dir)
    device = detect_device()
    logger.info("Compute device: %s", device)
    metrics: dict = {"_device": device, "_n_systems": int(ds.n_systems)}
    if mask_clipped:
        metrics["_clipped_systems"] = mask_clipped_tops(ds)
    # The curve model trains on ds_curves; every score uses ds
    ds_curves = ds
    if fix_ends:
        ds_curves = replace(ds, T_liquidus=ds.T_liquidus.copy(), T_solidus=ds.T_solidus.copy(),
                            T_liquidus_mask=ds.T_liquidus_mask.copy())
        metrics["_fixed_end_points"] = fix_end_points(ds_curves, inject=True)
        fix_end_points(ds, inject=False)

    logger.info("=== M1: phase classifier ===")
    m1 = PhaseClassifier().fit(ds)
    m1.save(out / "m1_phase.joblib")
    metrics["M1"] = _val_test(m1, ds)

    curvenet_cfg = CurveNetConfig(
        epochs=curve_epochs,
        batch_size=min(32, max(8, len(ds.train_idx) // 4)),
        lambda_endpoints=lambda_endpoints,
        lambda_smooth=lambda_smooth,
        endpoint_mode=endpoint_mode,
        solidus_mode=solidus_mode,
        mask_end_points=mask_end_points,
        swap_augment=swap_augment,
    )
    if curve_model == "curvenet":
        logger.info("=== M2+M3: joint liquidus/solidus curve net ===")
        curve_cfg = curvenet_cfg
        m23, curve_files = LiquidusSolidusModel(curve_cfg), {"m23_curves.pt"}
    elif curve_model in CURVE_BACKENDS:
        logger.info("=== M2+M3: %s tree-ensemble curves ===", curve_model)
        curve_cfg = TreeCurveConfig(backend=curve_model, seed=seed, swap_augment=swap_augment)
        m23, curve_files = TreeCurveModel(curve_cfg), {"m23_curves.joblib"}
    elif curve_model == "ensemble":
        logger.info("=== M2+M3: CurveNet + ExtraTrees curve ensemble ===")
        curve_cfg = CurveEnsembleConfig(
            curvenet=curvenet_cfg,
            trees=TreeCurveConfig(backend="extra_trees", seed=seed, swap_augment=swap_augment))
        m23 = CurveEnsemble(curve_cfg)
        curve_files = {"m23_curves.pt", "m23_curves.joblib", "m23_ensemble.json"}
    else:
        raise ValueError(f"curve_model must be 'curvenet', 'ensemble' or one of {CURVE_BACKENDS}")
    metrics["_seed"] = seed
    metrics["_m4_backend"] = m4_backend
    metrics["_curve_model"] = curve_model
    m23.fit(ds_curves)
    if curve_model == "ensemble":
        # The stacked model's fold ensembles have no validation split: give them this weight
        curve_cfg = replace(curve_cfg, weight=m23.weight)
        m23.save(out)
    else:
        m23.save(out / next(iter(curve_files)))
    metrics["_curve_config"] = asdict(curve_cfg)
    for stale in {"m23_curves.pt", "m23_curves.joblib", "m23_ensemble.json"} - curve_files:
        if (out / stale).exists():
            # from_dir would load another model type instead of this run's
            logger.warning("removing %s left by an earlier run in %s", stale, out)
            (out / stale).unlink()
    metrics["M2M3"] = _val_test(m23, ds)

    logger.info("=== M4: eutectic detector ===")
    m4_cfg = EutecticDetectorConfig(backend=m4_backend, seed=seed,
                                    comp_from_lower_tm=m4_comp_from_lower_tm)
    m4 = EutecticDetector(m4_cfg).fit(ds)
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
        meta = StackedTopologyClassifier(StackedMetaConfig(seed=seed), curve_config=curve_cfg,
                                         eutectic_config=m4_cfg).fit(ds_curves)
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
    p.add_argument("--endpoint-mode", choices=("penalty", "hard"), default="penalty",
                   help="'hard' makes every curve pass exactly through Tm(A) and Tm(B)")
    p.add_argument("--solidus-mode", choices=("gap", "free"), default="gap",
                   help="'gap' keeps solidus <= liquidus by construction; 'free' is the v8 behaviour")
    p.add_argument("--seed", type=int, default=0, help="random seed for reproducible runs")
    p.add_argument("--lambda-endpoints", type=float, default=1.0,
                   help="endpoint penalty weight (penalty mode only)")
    p.add_argument("--lambda-smooth", type=float, default=0.1)
    p.add_argument("--m4-backend", choices=("xgboost", "random_forest", "extra_trees"),
                   default="xgboost", help="tree ensemble used by the eutectic models")
    p.add_argument("--curve-model", choices=("curvenet", "ensemble", *CURVE_BACKENDS),
                   default="curvenet",
                   help="liquidus/solidus model: CurveNet, a tree ensemble, or their average")
    p.add_argument("--mask-end-points", action="store_true",
                   help="CurveNet: leave the x = 0 / 100 data values out of the data loss")
    p.add_argument("--mask-clipped", action="store_true",
                   help="treat flat liquidus tops cut off by the source chart as unobserved")
    p.add_argument("--fix-end-points", action="store_true",
                   help="train the curves with wrong pure-element end points set to Tm "
                        "(and score without them)")
    p.add_argument("--swap-augment", action="store_true",
                   help="also train the curve model on every system with A and B swapped")
    p.add_argument("--m4-comp-from-lower-tm", action="store_true",
                   help="M4 predicts the eutectic composition measured from the lower-melting element")
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
        endpoint_mode=args.endpoint_mode,
        solidus_mode=args.solidus_mode,
        seed=args.seed,
        lambda_endpoints=args.lambda_endpoints,
        lambda_smooth=args.lambda_smooth,
        m4_backend=args.m4_backend,
        curve_model=args.curve_model,
        mask_end_points=args.mask_end_points,
        mask_clipped=args.mask_clipped,
        m4_comp_from_lower_tm=args.m4_comp_from_lower_tm,
        fix_ends=args.fix_end_points,
        swap_augment=args.swap_augment,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    _cli()
