"""
predict.py — score a model folder (mpds_models.train) on the test split, or
predict chosen systems.

    python predict.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --report results/v18/report_v18.json
    python predict.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --pairs Ag_Al Cu_Ni
    python predict.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --split all --report all.json

Scoring leaves out liquidus tops cut off by the source chart and pure-element
end points that disagree with the melting point (data.mask_clipped_tops,
data.fix_end_points), as for every version since v10. The report (JSON plus a
CSV of the per-system table) is what compare_reports.py reads.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mpds_models.data import fix_end_points, load_dataset, mask_clipped_tops, system_quality
from mpds_models.evaluation import evaluate, prediction_table, print_summary
from mpds_models.pipeline import PhaseDiagramModel


def load_for_scoring(processed_dir: str, model: PhaseDiagramModel):
    """(dataset with the scoring masks, data groups from before masking)."""
    ds = load_dataset(processed_dir, model.features)
    quality = system_quality(ds)
    mask_clipped_tops(ds)
    fix_end_points(ds)
    return ds, quality


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("processed_dir")
    p.add_argument("model_dir")
    p.add_argument("--split", choices=("test", "val", "train", "all"), default="test")
    p.add_argument("--pairs", nargs="+", help="predict these systems instead, e.g. Ag_Al")
    p.add_argument("--report", help="JSON report path (a .csv table is written next to it)")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    model = PhaseDiagramModel.load(a.model_dir)
    ds, quality = load_for_scoring(a.processed_dir, model)
    if a.pairs:
        missing = [x for x in a.pairs if x not in ds.pair_ids]
        if missing:
            raise SystemExit(f"not in the dataset: {', '.join(missing)}")
        idx = np.array([ds.pair_ids.index(x) for x in a.pairs])
    else:
        idx = np.arange(ds.n_systems) if a.split == "all" else ds.split(a.split)
    pred = model.predict(ds.X[idx])
    diagrams = model.diagrams(pred, [ds.pair_ids[i] for i in idx])
    ev = evaluate(ds, idx, pred, quality)
    table = prediction_table(ds, idx, diagrams, quality)
    n_bad = sum(bool(d.inconsistencies) for d in diagrams)
    print_summary(ev, len(idx), n_bad)
    if a.pairs:
        print(table.drop(columns=["inconsistencies"]).T.to_string(header=False))
    if a.report:
        report = {"model_dir": a.model_dir, "n_diagrams": len(diagrams), "n_inconsistent_diagrams": n_bad,
                  "evaluation": ev, "predictions": json.loads(table.to_json(orient="records"))}
        Path(a.report).write_text(json.dumps(report, indent=2, default=str))
        table.to_csv(Path(a.report).with_suffix(".csv"), index=False)
        print(f"Report saved to {a.report} (and .csv)")


if __name__ == "__main__":
    main()
