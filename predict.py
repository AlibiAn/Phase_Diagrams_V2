"""
predict.py — Run trained models on the test split and produce a results report.

Usage
-----
    # Evaluate on the held-out test split (ground truth available)
    python predict.py ~/data/processed ~/data/models --report results.json

    # Predict a specific element pair by name
    python predict.py ~/data/processed ~/data/models --pair Ag_Al

    # Predict all systems, save full CSV + report
    python predict.py ~/data/processed ~/data/models --all --report results.json

    # Predict from a new preprocessed directory (no ground truth)
    python predict.py ~/data/new_processed ~/data/models --all --no-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── make sure mpds_models is importable from the same directory ──────────────
sys.path.insert(0, str(Path(__file__).parent))

from mpds_models.base import load_dataset, PhaseDataset
from mpds_models.curves_io import load_curves_model
from mpds_models.meta_learner import FullDiagramReconstructor, PredictedDiagram


# ─────────────────────────────────────────────────────────────────────────────
# Load trained models
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_dir: str | Path) -> FullDiagramReconstructor:
    p = Path(model_dir)
    m1      = joblib.load(p / "m1_phase.joblib")
    m23     = load_curves_model(p / "m23_curves.pt")
    m4      = joblib.load(p / "m4_eutectic.joblib")
    m5      = joblib.load(p / "m5_miscibility.joblib")
    m6_solo = joblib.load(p / "m6_topology_solo.joblib")
    # stacked is optional — fall back to solo if missing
    stacked_path = p / "m6_topology_stacked.joblib"
    m6_meta = joblib.load(stacked_path) if stacked_path.exists() else None
    return FullDiagramReconstructor(
        m1=m1, m23=m23, m4=m4, m5=m5,
        m6_meta=m6_meta, m6_solo=m6_solo,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topology_name(ds: PhaseDataset, idx: int) -> str:
    return ds.topology_classes[int(ds.topology[idx])]


def evaluate_topology(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    """Accuracy + per-class F1 for topology predictions."""
    y_true = [_topology_name(ds, i) for i in indices]
    y_pred = [d.topology for d in predictions]
    classes = list(ds.topology_classes)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)

    per_class = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[cls] = {
            "support": sum(t == cls for t in y_true),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_correct": correct,
        "n_total": len(y_true),
        "per_class": per_class,
    }


def evaluate_eutectic(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    """F1 for existence + MAE for composition and temperature."""
    true_exists = ds.eutectic_exists[indices]
    pred_exists = np.array([d.eutectic_exists for d in predictions])

    tp = int(((pred_exists == 1) & (true_exists == 1)).sum())
    fp = int(((pred_exists == 1) & (true_exists == 0)).sum())
    fn = int(((pred_exists == 0) & (true_exists == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    # Position errors — only where both true and predicted say "exists"
    both_pos = true_exists & pred_exists
    comp_errs, temp_errs = [], []
    for i, (idx, d) in enumerate(zip(indices, predictions)):
        if both_pos[i] and d.eutectic_comp is not None:
            true_comp = float(ds.eutectic_comp[idx])
            true_temp = float(ds.eutectic_temp[idx])
            if np.isfinite(true_comp):
                comp_errs.append(abs(d.eutectic_comp - true_comp))
            if np.isfinite(true_temp):
                temp_errs.append(abs(d.eutectic_temp - true_temp))

    return {
        "existence_f1": round(f1, 4),
        "existence_precision": round(prec, 4),
        "existence_recall": round(rec, 4),
        "n_true_positive": tp,
        "comp_MAE_at_pct": round(float(np.mean(comp_errs)), 2) if comp_errs else None,
        "temp_MAE_K": round(float(np.mean(temp_errs)), 2) if temp_errs else None,
        "n_position_pairs": len(comp_errs),
    }


def evaluate_curves(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    """MAE / RMSE for liquidus and solidus curves."""
    liq_errs, sol_errs = [], []
    for i, (idx, d) in enumerate(zip(indices, predictions)):
        liq_true = ds.T_liquidus[idx]
        sol_true = ds.T_solidus[idx]
        mask_liq = ds.T_liquidus_mask[idx]
        mask_sol = ~np.isnan(sol_true)

        if mask_liq.any():
            liq_errs.extend(np.abs(d.liquidus_T - liq_true)[mask_liq].tolist())
        if mask_sol.any():
            sol_errs.extend(np.abs(d.solidus_T - np.nan_to_num(sol_true))[mask_sol].tolist())

    liq_errs = np.array(liq_errs)
    sol_errs = np.array(sol_errs)
    return {
        "liquidus_MAE_K": round(float(liq_errs.mean()), 2) if liq_errs.size else None,
        "liquidus_RMSE_K": round(float(np.sqrt((liq_errs**2).mean())), 2) if liq_errs.size else None,
        "solidus_MAE_K": round(float(sol_errs.mean()), 2) if sol_errs.size else None,
        "solidus_RMSE_K": round(float(np.sqrt((sol_errs**2).mean())), 2) if sol_errs.size else None,
        # Patched: counted on the RAW model output. The returned curves are
        # clamped (solidus <= liquidus), so comparing them always gave 0.
        "order_violation_fraction": round(
            float(np.mean([
                getattr(d, "n_order_violations", 0) / max(len(d.liquidus_T), 1)
                for d in predictions
            ])), 6,
        ),
        "n_liquidus_points_evaluated": int(liq_errs.size),
    }


def evaluate_miscibility(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    true_gap = ds.miscibility_gap[indices]
    pred_gap = np.array([d.miscibility_gap for d in predictions])
    tp = int(((pred_gap == 1) & (true_gap == 1)).sum())
    fp = int(((pred_gap == 1) & (true_gap == 0)).sum())
    fn = int(((pred_gap == 0) & (true_gap == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "accuracy": round(float((pred_gap == true_gap).mean()), 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "n_true_gaps": int(true_gap.sum()),
        "n_predicted_gaps": int(pred_gap.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-system prediction table (human-readable)
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_table(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
    include_ground_truth: bool = True,
) -> pd.DataFrame:
    rows = []
    for i, (idx, d) in enumerate(zip(indices, predictions)):
        row = {
            "pair": d.pair,
            "predicted_topology": d.topology,
            "topology_confidence": round(d.topology_proba[d.topology], 3),
        }
        if include_ground_truth:
            true_topo = _topology_name(ds, idx)
            row["true_topology"] = true_topo
            row["topology_correct"] = (d.topology == true_topo)

        row["eutectic_predicted"] = d.eutectic_exists
        row["eutectic_comp_pct"] = round(d.eutectic_comp, 1) if d.eutectic_comp is not None else None
        row["eutectic_temp_K"] = round(d.eutectic_temp, 0) if d.eutectic_temp is not None else None

        if include_ground_truth:
            row["eutectic_true"] = bool(ds.eutectic_exists[idx])
            true_comp = float(ds.eutectic_comp[idx])
            true_temp = float(ds.eutectic_temp[idx])
            row["eutectic_true_comp_pct"] = round(true_comp, 1) if np.isfinite(true_comp) else None
            row["eutectic_true_temp_K"] = round(true_temp, 0) if np.isfinite(true_temp) else None

        row["miscibility_gap_predicted"] = d.miscibility_gap
        row["miscibility_proba"] = round(d.miscibility_proba, 3)
        if include_ground_truth:
            row["miscibility_gap_true"] = bool(ds.miscibility_gap[idx])

        row["liquidus_MAE_K"] = None
        row["solidus_MAE_K"] = None
        if include_ground_truth:
            liq_true = ds.T_liquidus[idx]
            sol_true = ds.T_solidus[idx]
            mask_l = ds.T_liquidus_mask[idx]
            mask_s = ~np.isnan(sol_true)
            if mask_l.any():
                row["liquidus_MAE_K"] = round(float(np.abs(d.liquidus_T - liq_true)[mask_l].mean()), 1)
            if mask_s.any():
                row["solidus_MAE_K"] = round(float(np.abs(d.solidus_T - np.nan_to_num(sol_true))[mask_s].mean()), 1)

        row["top_phases"] = ", ".join(f"{ph}({prob:.2f})" for ph, prob in d.phases[:3])
        row["n_inconsistencies"] = len(d.inconsistencies)
        row["inconsistencies"] = "; ".join(d.inconsistencies) if d.inconsistencies else ""

        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(report: dict) -> None:
    print("\n" + "=" * 65)
    print("  PHASE DIAGRAM PREDICTION — RESULTS SUMMARY")
    print("=" * 65)

    ev = report.get("evaluation")
    if ev:
        t = ev["topology"]
        print(f"\n  TOPOLOGY CLASSIFICATION  ({t['n_total']} systems)")
        print(f"    Accuracy  : {t['accuracy']:.1%}  ({t['n_correct']}/{t['n_total']} correct)")
        print(f"    Macro-F1  : {t['macro_f1']:.3f}")
        print()
        print(f"    {'Class':<35} {'Support':>7} {'Precision':>10} {'Recall':>8} {'F1':>6}")
        print(f"    {'-'*35} {'-'*7} {'-'*10} {'-'*8} {'-'*6}")
        for cls, v in t["per_class"].items():
            print(f"    {cls:<35} {v['support']:>7} {v['precision']:>10.3f} {v['recall']:>8.3f} {v['f1']:>6.3f}")

        c = ev["curves"]
        print(f"\n  LIQUIDUS / SOLIDUS CURVES  ({c['n_liquidus_points_evaluated']:,} grid points)")
        print(f"    Liquidus MAE   : {c['liquidus_MAE_K']} K")
        print(f"    Liquidus RMSE  : {c['liquidus_RMSE_K']} K")
        print(f"    Solidus MAE    : {c['solidus_MAE_K']} K")
        print(f"    Order violations: {c['order_violation_fraction']:.6f}  (T_sol > T_liq)")

        e = ev["eutectic"]
        print(f"\n  EUTECTIC DETECTION  ({e['n_true_positive']} true positives found)")
        print(f"    Existence F1        : {e['existence_f1']:.3f}")
        print(f"    Existence Precision : {e['existence_precision']:.3f}")
        print(f"    Existence Recall    : {e['existence_recall']:.3f}")
        if e["comp_MAE_at_pct"] is not None:
            print(f"    Composition MAE     : {e['comp_MAE_at_pct']} at%  ({e['n_position_pairs']} pairs)")
            print(f"    Temperature MAE     : {e['temp_MAE_K']} K")

        m = ev["miscibility"]
        print(f"\n  MISCIBILITY GAP  ({m['n_true_gaps']} true / {m['n_predicted_gaps']} predicted)")
        print(f"    Accuracy  : {m['accuracy']:.1%}")
        print(f"    F1        : {m['f1']:.3f}  (precision={m['precision']:.3f}, recall={m['recall']:.3f})")

        n_inc = report.get("n_inconsistent_diagrams", 0)
        n_tot = report.get("n_diagrams", 1)
        print(f"\n  CONSISTENCY CHECKS")
        print(f"    Internally consistent diagrams: {n_tot - n_inc}/{n_tot} ({(n_tot-n_inc)/n_tot:.1%})")

    print("\n" + "=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    print(f"Loading dataset from {args.processed_dir} ...")
    ds = load_dataset(args.processed_dir)

    print(f"Loading models from {args.model_dir} ...")
    recon = load_models(args.model_dir)
    # wire the dataset vocab/grid into the reconstructor
    recon.phase_vocab = ds.phase_vocab
    recon.comp_grid   = ds.comp_grid
    recon.topology_classes = ds.topology_classes

    # ── determine which systems to predict ──────────────────────────────────
    if args.pair:
        if args.pair not in ds.pair_ids:
            print(f"ERROR: '{args.pair}' not found. Available pairs include: "
                  f"{', '.join(ds.pair_ids[:10])} ...")
            sys.exit(1)
        idx_list = np.array([ds.pair_ids.index(args.pair)], dtype=np.int64)
        label = f"single pair: {args.pair}"
    elif args.all:
        idx_list = np.arange(ds.n_systems, dtype=np.int64)
        label = f"all {ds.n_systems} systems"
    else:
        idx_list = ds.test_idx
        label = f"test split ({len(idx_list)} systems)"

    print(f"Running predictions on {label} ...")
    pairs = [ds.pair_ids[i] for i in idx_list]
    X     = ds.X[idx_list]

    diagrams = recon.reconstruct(X, pairs, with_uncertainty=not args.no_uncertainty)

    # ── evaluation against ground truth ─────────────────────────────────────
    eval_results = None
    if not args.no_eval:
        print("Evaluating against ground truth ...")
        eval_results = {
            "topology":    evaluate_topology(diagrams, ds, idx_list),
            "curves":      evaluate_curves(diagrams, ds, idx_list),
            "eutectic":    evaluate_eutectic(diagrams, ds, idx_list),
            "miscibility": evaluate_miscibility(diagrams, ds, idx_list),
        }

    # ── build output table ───────────────────────────────────────────────────
    table = build_prediction_table(
        diagrams, ds, idx_list,
        include_ground_truth=not args.no_eval,
    )

    # ── assemble report ──────────────────────────────────────────────────────
    n_inconsistent = sum(len(d.inconsistencies) > 0 for d in diagrams)
    report = {
        "n_diagrams": len(diagrams),
        "n_inconsistent_diagrams": n_inconsistent,
        "evaluation": eval_results,
        "predictions": json.loads(table.to_json(orient="records")),
    }

    print_summary(report)

    # ── save outputs ─────────────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Full report saved to {report_path}")

    csv_path = Path(args.report).with_suffix(".csv") if args.report else Path("predictions.csv")
    table.to_csv(csv_path, index=False)
    print(f"Prediction table saved to {csv_path}")

    # ── single-pair detailed output ──────────────────────────────────────────
    if args.pair:
        d = diagrams[0]
        print(f"\n{'─'*55}")
        print(f"  DETAILED PREDICTION: {d.pair}")
        print(f"{'─'*55}")
        print(f"  Topology : {d.topology}")
        print(f"  Confidence breakdown:")
        for cls, prob in sorted(d.topology_proba.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:<35} {prob:.3f}  {bar}")
        print(f"\n  Eutectic exists : {d.eutectic_exists} (p={d.eutectic_proba:.3f})")
        if d.eutectic_exists:
            print(f"  Eutectic position: {d.eutectic_comp:.1f} at%  {d.eutectic_temp:.0f} K")
        print(f"\n  Miscibility gap : {d.miscibility_gap} (p={d.miscibility_proba:.3f})")
        print(f"\n  Top predicted phases:")
        for ph, prob in d.phases[:5]:
            print(f"    {ph:<40} p={prob:.3f}")
        print(f"\n  Liquidus range  : {d.liquidus_T.min():.0f} K – {d.liquidus_T.max():.0f} K")
        print(f"  Solidus range   : {d.solidus_T.min():.0f} K – {d.solidus_T.max():.0f} K")
        if d.liquidus_T_std is not None:
            print(f"  Liquidus uncertainty (1σ): {d.liquidus_T_std.mean():.1f} K avg")
        if d.inconsistencies:
            print(f"\n  ⚠ Inconsistencies:")
            for msg in d.inconsistencies:
                print(f"    - {msg}")
        print()


def _cli() -> None:
    p = argparse.ArgumentParser(description="Predict binary phase diagrams from trained models.")
    p.add_argument("processed_dir", help="output directory from preprocessing pipeline")
    p.add_argument("model_dir",     help="directory containing trained model files")
    p.add_argument("--pair",        help="predict a specific element pair, e.g. Ag_Al")
    p.add_argument("--all",         action="store_true", help="predict all systems (not just test split)")
    p.add_argument("--report",      help="path to save JSON report (also saves a .csv)")
    p.add_argument("--no-eval",     action="store_true", help="skip ground-truth comparison")
    p.add_argument("--no-uncertainty", action="store_true",
                   help="skip MC-Dropout uncertainty (faster)")
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
