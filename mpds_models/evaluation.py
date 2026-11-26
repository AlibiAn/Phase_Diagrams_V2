"""
Scores of a PhaseDiagramModel against the data, in the report format of
predict.py (read by compare_reports.py): diagram type, curves (overall and
per data group), eutectic, miscibility gap, and one row per system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DATA_GROUPS, TOPOLOGY_CLASSES, PhaseDataset, curve_errors, group_rows, precision_recall_f1

r4 = lambda v: round(float(v), 4)


def topology_scores(ds: PhaseDataset, idx, pred) -> dict:
    y_true = np.array([ds.topology_name(i) for i in idx])
    y_pred = np.array([TOPOLOGY_CLASSES[t] for t in pred["topology"]])
    per_class = {}
    for c in TOPOLOGY_CLASSES:
        p, r, f = precision_recall_f1(y_true == c, y_pred == c)
        per_class[c] = {"support": int((y_true == c).sum()), "precision": r4(p), "recall": r4(r), "f1": r4(f)}
    correct = int((y_true == y_pred).sum())
    return {"accuracy": r4(correct / len(y_true)), "macro_f1": r4(np.mean([v["f1"] for v in per_class.values()])),
            "n_correct": correct, "n_total": len(y_true), "per_class": per_class}


def curves_scores(ds: PhaseDataset, idx, liq, sol) -> dict:
    e_liq, e_sol = (e.astype(np.float64) for e in curve_errors(ds, idx, liq, sol))
    r2 = lambda v: round(float(v), 2) if np.size(v) else None
    return {"liquidus_MAE_K": r2(e_liq.mean()) if e_liq.size else None,
            "liquidus_RMSE_K": r2(np.sqrt((e_liq ** 2).mean())) if e_liq.size else None,
            "solidus_MAE_K": r2(e_sol.mean()) if e_sol.size else None,
            "solidus_RMSE_K": r2(np.sqrt((e_sol ** 2).mean())) if e_sol.size else None,
            "order_violation_fraction": 0.0,           # the solidus is capped at the liquidus by construction
            "n_liquidus_points_evaluated": int(e_liq.size)}


def eutectic_scores(ds: PhaseDataset, idx, pred) -> dict:
    true, exists = ds.eutectic_exists[idx], pred["eutectic_exists"]
    p, r, f = precision_recall_f1(true, exists)
    both = true & exists
    comp = np.abs(pred["eutectic_comp"] - ds.eutectic_comp[idx])[both]
    temp = np.abs(pred["eutectic_temp"] - ds.eutectic_temp[idx])[both]
    comp, temp = comp[np.isfinite(comp)], temp[np.isfinite(temp)]
    return {"existence_f1": r4(f), "existence_precision": r4(p), "existence_recall": r4(r),
            "n_true_positive": int(both.sum()),
            "comp_MAE_at_pct": round(float(comp.mean()), 2) if comp.size else None,
            "temp_MAE_K": round(float(temp.mean()), 2) if temp.size else None, "n_position_pairs": int(comp.size)}


def gap_scores(ds: PhaseDataset, idx, pred) -> dict:
    true, gap = ds.miscibility_gap[idx], pred["gap"]
    p, r, f = precision_recall_f1(true, gap)
    return {"accuracy": r4((gap == true).mean()), "f1": r4(f), "precision": r4(p), "recall": r4(r),
            "n_true_gaps": int(true.sum()), "n_predicted_gaps": int(gap.sum())}


def evaluate(ds: PhaseDataset, idx: np.ndarray, pred: dict, quality: pd.DataFrame) -> dict:
    pos = {int(i): k for k, i in enumerate(idx)}
    by_group = {}
    for g, rows in group_rows(quality, np.asarray(idx)).items():
        if len(rows):
            k = [pos[int(i)] for i in rows]
            by_group[g] = {"n_systems": int(len(rows)), **curves_scores(ds, rows, pred["liquidus"][k], pred["solidus"][k])}
    return {"topology": topology_scores(ds, idx, pred), "curves": curves_scores(ds, idx, pred["liquidus"], pred["solidus"]),
            "eutectic": eutectic_scores(ds, idx, pred), "miscibility": gap_scores(ds, idx, pred),
            "curves_by_group": by_group}


def prediction_table(ds: PhaseDataset, idx, diagrams, quality: pd.DataFrame | None, truth: bool = True) -> pd.DataFrame:
    rnd = lambda v, k: round(v, k) if v is not None else None
    fin = lambda v, k: round(float(v), k) if np.isfinite(v) else None
    rows = []
    for i, d in zip(idx, diagrams):
        row = {"pair": d.pair}
        if quality is not None:
            row["data_group"] = quality["group"].iloc[i]
        row.update(predicted_topology=d.topology, topology_confidence=round(d.topology_proba[d.topology], 3))
        if truth:
            row.update(true_topology=ds.topology_name(i), topology_correct=d.topology == ds.topology_name(i))
        row.update(eutectic_predicted=d.eutectic_exists, eutectic_comp_pct=rnd(d.eutectic_comp, 1),
                   eutectic_temp_K=rnd(d.eutectic_temp, 0))
        if truth:
            row.update(eutectic_true=bool(ds.eutectic_exists[i]), eutectic_true_comp_pct=fin(ds.eutectic_comp[i], 1),
                       eutectic_true_temp_K=fin(ds.eutectic_temp[i], 0))
        row.update(miscibility_gap_predicted=d.miscibility_gap, miscibility_proba=round(d.miscibility_proba, 3))
        if truth:
            row["miscibility_gap_true"] = bool(ds.miscibility_gap[i])
        row.update(liquidus_MAE_K=None, solidus_MAE_K=None)
        if truth:
            e_liq, e_sol = curve_errors(ds, [i], d.liquidus_T[None], d.solidus_T[None])
            row["liquidus_MAE_K"] = round(float(e_liq.mean()), 1) if e_liq.size else None
            row["solidus_MAE_K"] = round(float(e_sol.mean()), 1) if e_sol.size else None
        row.update(top_phases=", ".join(f"{ph}({p:.2f})" for ph, p in d.phases[:3]),
                   n_inconsistencies=len(d.inconsistencies), inconsistencies="; ".join(d.inconsistencies))
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(ev: dict, n_diagrams: int, n_inconsistent: int) -> None:
    t, c, e, m = ev["topology"], ev["curves"], ev["eutectic"], ev["miscibility"]
    nan = lambda v: float("nan") if v is None else v
    print(f"\nDiagram type ({t['n_total']} systems): accuracy {t['accuracy']:.1%}, macro-F1 {t['macro_f1']:.3f}")
    for cls, v in t["per_class"].items():
        print(f"    {cls:<30} n={v['support']:<4} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}")
    print(f"Curves ({c['n_liquidus_points_evaluated']:,} liquidus points): liquidus MAE {c['liquidus_MAE_K']} K "
          f"(RMSE {c['liquidus_RMSE_K']} K), solidus MAE {c['solidus_MAE_K']} K")
    print(f"    {'data group':<18} {'systems':>7} {'liquidus':>10} {'solidus':>10}")
    for g in DATA_GROUPS:
        if g in ev["curves_by_group"]:
            v = ev["curves_by_group"][g]
            print(f"    {g:<18} {v['n_systems']:>7} {nan(v['liquidus_MAE_K']):>8.1f} K {nan(v['solidus_MAE_K']):>8.1f} K")
    print(f"Eutectic: existence F1 {e['existence_f1']:.3f} (precision {e['existence_precision']:.3f}, recall "
          f"{e['existence_recall']:.3f}); position where both have one ({e['n_position_pairs']} systems): "
          f"{e['comp_MAE_at_pct']} at%, {e['temp_MAE_K']} K")
    print(f"Miscibility gap: F1 {m['f1']:.3f} ({m['n_true_gaps']} true, {m['n_predicted_gaps']} predicted)")
    print(f"Internally consistent diagrams: {n_diagrams - n_inconsistent}/{n_diagrams}")
