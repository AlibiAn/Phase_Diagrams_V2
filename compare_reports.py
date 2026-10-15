"""
compare_reports.py — Compare two predict.py reports (e.g. v6 vs v7 models).

Prints headline metrics side by side with the change, per-class topology
scores, and which systems' topology predictions were fixed or broken.
Optionally saves a bar chart of the per-class F1 scores.

Usage
-----
    python compare_reports.py results/results_v6.json results/results_v7.json
    python compare_reports.py old.json new.json --names v6 v7 --plot topology_f1.png

Both reports should come from the same data split (by default predict.py
evaluates the test split) and, for repeatable numbers, from runs with
--no-uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (section, key, label, higher_is_better)
HEADLINE = [
    ("topology", "accuracy", "Topology accuracy", True),
    ("topology", "macro_f1", "Topology macro-F1", True),
    ("curves", "liquidus_MAE_K", "Liquidus MAE (K)", False),
    ("curves", "liquidus_RMSE_K", "Liquidus RMSE (K)", False),
    ("curves", "solidus_MAE_K", "Solidus MAE (K)", False),
    ("curves", "order_violation_fraction", "Order-violation fraction", False),
    ("eutectic", "existence_f1", "Eutectic existence F1", True),
    ("eutectic", "comp_MAE_at_pct", "Eutectic comp. MAE (at%)", False),
    ("eutectic", "temp_MAE_K", "Eutectic temp. MAE (K)", False),
    ("miscibility", "f1", "Miscibility-gap F1", True),
]


def load(path: str) -> dict:
    report = json.loads(Path(path).read_text())
    if not report.get("evaluation"):
        raise SystemExit(f"{path} has no evaluation (was it run with --no-eval?)")
    return report


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"


def delta(a, b, higher_is_better: bool) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    if abs(d) < 1e-12:
        return "  ="
    better = (d > 0) == higher_is_better
    return f"{'+' if d > 0 else ''}{fmt(d)}  {'better' if better else 'worse'}"


def print_headline(a: dict, b: dict, names: tuple[str, str]) -> None:
    ea, eb = a["evaluation"], b["evaluation"]
    print(f"\n{'Metric':<28} {names[0]:>10} {names[1]:>10}   Change")
    print(f"{'-' * 28} {'-' * 10} {'-' * 10}   {'-' * 18}")
    for section, key, label, hib in HEADLINE:
        va, vb = ea[section].get(key), eb[section].get(key)
        print(f"{label:<28} {fmt(va):>10} {fmt(vb):>10}   {delta(va, vb, hib)}")
    for r, name in zip((a, b), names):
        n, bad = r["n_diagrams"], r["n_inconsistent_diagrams"]
        print(f"{'Inconsistent diagrams':<28} {name}: {bad}/{n}")


def print_per_class(a: dict, b: dict, names: tuple[str, str]) -> None:
    pa = a["evaluation"]["topology"]["per_class"]
    pb = b["evaluation"]["topology"]["per_class"]
    print(f"\n{'Topology class':<30} {'n':>4}   {'F1 ' + names[0]:>8} {'F1 ' + names[1]:>8}   "
          f"{'recall ' + names[0]:>10} {'recall ' + names[1]:>10}")
    for cls in pa:
        print(f"{cls:<30} {pb[cls]['support']:>4}   {pa[cls]['f1']:>8.3f} {pb[cls]['f1']:>8.3f}   "
              f"{pa[cls]['recall']:>10.3f} {pb[cls]['recall']:>10.3f}")


def print_flips(a: dict, b: dict, names: tuple[str, str], limit: int) -> None:
    """Systems whose topology prediction became right or wrong between runs."""
    rows_a = {r["pair"]: r for r in a["predictions"]}
    rows_b = {r["pair"]: r for r in b["predictions"]}
    common = sorted(rows_a.keys() & rows_b.keys())
    if len(common) < max(len(rows_a), len(rows_b)):
        print(f"\nNOTE: only {len(common)} systems appear in both reports "
              f"({len(rows_a)} vs {len(rows_b)}); were they run on the same split?")

    fixed = [p for p in common if not rows_a[p]["topology_correct"] and rows_b[p]["topology_correct"]]
    broken = [p for p in common if rows_a[p]["topology_correct"] and not rows_b[p]["topology_correct"]]
    print(f"\nTopology: {len(fixed)} systems fixed, {len(broken)} broken "
          f"(net {len(fixed) - len(broken):+d})")
    for title, pairs in ((f"Fixed in {names[1]}", fixed), (f"Broken in {names[1]}", broken)):
        if pairs:
            print(f"  {title}:")
            for p in pairs[:limit]:
                ra, rb = rows_a[p], rows_b[p]
                print(f"    {p:<10} true {rb['true_topology']:<30} "
                      f"{names[0]}: {ra['predicted_topology']:<30} {names[1]}: {rb['predicted_topology']}")
            if len(pairs) > limit:
                print(f"    ... and {len(pairs) - limit} more")

    # Largest liquidus changes, either direction
    diffs = [(rows_b[p]["liquidus_MAE_K"] - rows_a[p]["liquidus_MAE_K"], p) for p in common
             if rows_a[p]["liquidus_MAE_K"] is not None and rows_b[p]["liquidus_MAE_K"] is not None]
    if diffs:
        diffs.sort()
        d = np.array([x for x, _ in diffs])
        print(f"\nLiquidus MAE per system: improved in {(d < 0).sum()}, worse in {(d > 0).sum()}, "
              f"median change {np.median(d):+.1f} K")
        print("  Most improved: " + ", ".join(f"{p} ({x:+.0f} K)" for x, p in diffs[:5]))
        print("  Most worsened: " + ", ".join(f"{p} ({x:+.0f} K)" for x, p in diffs[::-1][:5]))


def plot_per_class(a: dict, b: dict, names: tuple[str, str], out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pa = a["evaluation"]["topology"]["per_class"]
    pb = b["evaluation"]["topology"]["per_class"]
    classes = list(pa)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [pa[c]["f1"] for c in classes], 0.4, label=names[0], color="#9AA5B1")
    ax.bar(x + 0.2, [pb[c]["f1"] for c in classes], 0.4, label=names[1], color="#2A78D6")
    ax.set_xticks(x, [f"{c}\n(n={pb[c]['support']})" for c in classes], fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"Topology F1 by class — macro-F1 {a['evaluation']['topology']['macro_f1']:.3f} "
                 f"→ {b['evaluation']['topology']['macro_f1']:.3f}", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two predict.py reports.")
    p.add_argument("old_report")
    p.add_argument("new_report")
    p.add_argument("--names", nargs=2, default=("old", "new"), help="labels for the two runs")
    p.add_argument("--plot", help="save a per-class topology F1 bar chart to this PNG")
    p.add_argument("--limit", type=int, default=15, help="max systems listed per group")
    a = p.parse_args()

    old, new = load(a.old_report), load(a.new_report)
    names = tuple(a.names)
    print_headline(old, new, names)
    print_per_class(old, new, names)
    print_flips(old, new, names, a.limit)
    if a.plot:
        plot_per_class(old, new, names, a.plot)


if __name__ == "__main__":
    main()
