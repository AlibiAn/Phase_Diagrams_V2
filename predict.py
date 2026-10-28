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

import numpy as np
import pandas as pd

from mpds_models.base import PhaseDataset, load_dataset, mask_clipped_tops, precision_recall_f1
from mpds_models.meta_learner import FullDiagramReconstructor, PredictedDiagram


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth comparison
# ─────────────────────────────────────────────────────────────────────────────

def curve_errors(d: PredictedDiagram, ds: PhaseDataset, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute liquidus / solidus errors at the grid points that have ground truth."""
    sol_true = ds.T_solidus[idx]
    liq_err = np.abs(d.liquidus_T - ds.T_liquidus[idx])[ds.T_liquidus_mask[idx]]
    sol_err = np.abs(d.solidus_T - sol_true)[~np.isnan(sol_true)]
    return liq_err, sol_err


def liquidus_mae(d: PredictedDiagram, ds: PhaseDataset, idx: int) -> float:
    """Liquidus MAE in K, NaN if the system has no liquidus data."""
    err, _ = curve_errors(d, ds, idx)
    return float(err.mean()) if err.size else float("nan")


def evaluate_topology(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    """Accuracy, macro-F1 and per-class precision / recall / F1."""
    y_true = np.array([ds.topology_name(i) for i in indices])
    y_pred = np.array([d.topology for d in predictions])
    correct = int((y_true == y_pred).sum())

    per_class = {}
    for cls in ds.topology_classes:
        prec, rec, f1 = precision_recall_f1(y_true == cls, y_pred == cls)
        per_class[cls] = {
            "support": int((y_true == cls).sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    return {
        "accuracy": round(correct / len(y_true), 4),
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
    """Existence precision / recall / F1, plus composition and temperature MAE
    where both truth and prediction have a eutectic."""
    true_exists = ds.eutectic_exists[indices]
    pred_exists = np.array([d.eutectic_exists for d in predictions])
    prec, rec, f1 = precision_recall_f1(true_exists, pred_exists)

    comp_errs, temp_errs = [], []
    for idx, d, both in zip(indices, predictions, true_exists & pred_exists):
        if both and d.eutectic_comp is not None:
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
        "n_true_positive": int((true_exists & pred_exists).sum()),
        "comp_MAE_at_pct": round(float(np.mean(comp_errs)), 2) if comp_errs else None,
        "temp_MAE_K": round(float(np.mean(temp_errs)), 2) if temp_errs else None,
        "n_position_pairs": len(comp_errs),
    }


def evaluate_curves(
    predictions: list[PredictedDiagram],
    ds: PhaseDataset,
    indices: np.ndarray,
) -> dict:
    """MAE / RMSE over every liquidus and solidus grid point with ground truth."""
    errs = [curve_errors(d, ds, idx) for idx, d in zip(indices, predictions)]
    liq_errs = np.concatenate([e[0] for e in errs]).astype(np.float64)
    sol_errs = np.concatenate([e[1] for e in errs]).astype(np.float64)

    def mae(e):
        return round(float(e.mean()), 2) if e.size else None

    def rmse(e):
        return round(float(np.sqrt((e ** 2).mean())), 2) if e.size else None

    return {
        "liquidus_MAE_K": mae(liq_errs),
        "liquidus_RMSE_K": rmse(liq_errs),
        "solidus_MAE_K": mae(sol_errs),
        "solidus_RMSE_K": rmse(sol_errs),
        # From the raw model output: the returned curves are clamped
        # (solidus <= liquidus), so comparing them would always give 0
        "order_violation_fraction": round(
            float(np.mean([
                d.n_order_violations / max(len(d.liquidus_T), 1)
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
    prec, rec, f1 = precision_recall_f1(true_gap, pred_gap)
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
    for idx, d in zip(indices, predictions):
        row = {
            "pair": d.pair,
            "predicted_topology": d.topology,
            "topology_confidence": round(d.topology_proba[d.topology], 3),
        }
        if include_ground_truth:
            true_topo = ds.topology_name(idx)
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
            liq_err, sol_err = curve_errors(d, ds, idx)
            if liq_err.size:
                row["liquidus_MAE_K"] = round(float(liq_err.mean()), 1)
            if sol_err.size:
                row["solidus_MAE_K"] = round(float(sol_err.mean()), 1)

        row["top_phases"] = ", ".join(f"{ph}({prob:.2f})" for ph, prob in d.phases[:3])
        row["n_inconsistencies"] = len(d.inconsistencies)
        row["inconsistencies"] = "; ".join(d.inconsistencies) if d.inconsistencies else ""

        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Console output
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
        print("\n  CONSISTENCY CHECKS")
        print(f"    Internally consistent diagrams: {n_tot - n_inc}/{n_tot} ({(n_tot-n_inc)/n_tot:.1%})")

    print("\n" + "=" * 65 + "\n")


def print_diagram(d: PredictedDiagram) -> None:
    """Detailed console view of a single prediction."""
    print(f"\n{'─'*55}")
    print(f"  DETAILED PREDICTION: {d.pair}")
    print(f"{'─'*55}")
    print(f"  Topology : {d.topology}")
    print("  Confidence breakdown:")
    for cls, prob in sorted(d.topology_proba.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<35} {prob:.3f}  {bar}")
    print(f"\n  Eutectic exists : {d.eutectic_exists} (p={d.eutectic_proba:.3f})")
    if d.eutectic_exists:
        print(f"  Eutectic position: {d.eutectic_comp:.1f} at%  {d.eutectic_temp:.0f} K")
    print(f"\n  Miscibility gap : {d.miscibility_gap} (p={d.miscibility_proba:.3f})")
    print("\n  Top predicted phases:")
    for ph, prob in d.phases[:5]:
        print(f"    {ph:<40} p={prob:.3f}")
    print(f"\n  Liquidus range  : {d.liquidus_T.min():.0f} K – {d.liquidus_T.max():.0f} K")
    print(f"  Solidus range   : {d.solidus_T.min():.0f} K – {d.solidus_T.max():.0f} K")
    if d.liquidus_T_std is not None:
        print(f"  Liquidus uncertainty (1σ): {d.liquidus_T_std.mean():.1f} K avg")
    if d.inconsistencies:
        print("\n  ⚠ Inconsistencies:")
        for msg in d.inconsistencies:
            print(f"    - {msg}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> None:
    print(f"Loading dataset from {args.processed_dir} ...")
    ds = load_dataset(args.processed_dir)
    if args.mask_clipped:
        print(f"Not scoring clipped liquidus tops in {len(mask_clipped_tops(ds))} systems")

    print(f"Loading models from {args.model_dir} ...")
    recon = FullDiagramReconstructor.from_dir(args.model_dir, ds)

    if args.pair:
        if args.pair not in ds.pair_ids:
            print(f"ERROR: '{args.pair}' not found. Available pairs include: "
                  f"{', '.join(ds.pair_ids[:10])} ...")
            sys.exit(1)
        idx_list = ds.indices_of([args.pair])
        label = f"single pair: {args.pair}"
    elif args.all:
        idx_list = np.arange(ds.n_systems, dtype=np.int64)
        label = f"all {ds.n_systems} systems"
    else:
        idx_list = ds.test_idx
        label = f"test split ({len(idx_list)} systems)"

    print(f"Running predictions on {label} ...")
    pairs = [ds.pair_ids[i] for i in idx_list]
    diagrams = recon.reconstruct(ds.X[idx_list], pairs, with_uncertainty=not args.no_uncertainty)

    eval_results = None
    if not args.no_eval:
        print("Evaluating against ground truth ...")
        eval_results = {
            "topology":    evaluate_topology(diagrams, ds, idx_list),
            "curves":      evaluate_curves(diagrams, ds, idx_list),
            "eutectic":    evaluate_eutectic(diagrams, ds, idx_list),
            "miscibility": evaluate_miscibility(diagrams, ds, idx_list),
        }

    table = build_prediction_table(diagrams, ds, idx_list, include_ground_truth=not args.no_eval)
    report = {
        "n_diagrams": len(diagrams),
        "n_inconsistent_diagrams": sum(bool(d.inconsistencies) for d in diagrams),
        "evaluation": eval_results,
        "predictions": json.loads(table.to_json(orient="records")),
    }
    print_summary(report)

    if args.report:
        report_path = Path(args.report)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Full report saved to {report_path}")

    csv_path = Path(args.report).with_suffix(".csv") if args.report else Path("predictions.csv")
    table.to_csv(csv_path, index=False)
    print(f"Prediction table saved to {csv_path}")

    if args.pair:
        print_diagram(diagrams[0])


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
    p.add_argument("--mask-clipped", action="store_true",
                   help="leave liquidus tops cut off by the source chart out of the scores")
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
