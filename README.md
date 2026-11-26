# Binary phase diagrams from element properties

Predicts the liquidus and solidus curves, the diagram type, the eutectic
point, the phases present and whether the two liquids separate, for a pair
of elements, from properties of the two elements alone. Trained on MPDS
diagrams. The research record (what was tried, what helped, the numbers) is
in `CLAUDE.md`, `docs/research_review.md` and `results/vNN/`.

## Layout

```
mpds_models/            the model package
  data.py               dataset loading, extra systems, A/B swap, data fixes, data groups, scores
  features.py           features from the element symbols (element data, compound energies)
  curves.py             liquidus + solidus from ExtraTrees
  foundation.py         TabPFN + TabICL corrections of the liquidus (v17 height, v18 whole curve)
  classifiers.py        phases, eutectic, miscibility gap, diagram type (XGBoost)
  pipeline.py           PhaseDiagramModel: every model trained, saved and loaded together
  evaluation.py         scores and the report format
  train.py              training command
predict.py              score a model folder on the test split / predict chosen systems
cross_validate.py       5-fold cross-validation (the tool for deciding between versions)
error_analysis.py       diagram galleries and error breakdowns for a model folder
compare_reports.py      two predict.py reports side by side
data_build/             how the datasets were made (processed_v6 -> v8, extra systems)
results/vNN/            logs, reports and figures of each version
```

Everything before this layout (CurveNet, the v9-v15 comparison scripts, the
v18/v19 experiments, the thermodynamic engine, the MPDS downloader) is at
git tag `pre-cleanup`; `git checkout pre-cleanup` re-runs any older result.

## Commands

Run from the repository root in the `ml` conda environment.
`MPDS_DEVICE=cpu` keeps XGBoost on the CPU, as for every version since v13
(it builds different trees on the GPU); TabPFN and TabICL use the GPU.

```bash
# train v18 (about 15 min), then score it, then draw the error galleries
MPDS_DEVICE=cpu python -m mpds_models.train /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18
python predict.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --report results/v18/report_v18.json
python error_analysis.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --out results/v18/error_analysis

# earlier versions: --height 0 (v16, trees only), --blend 0 (v17, height correction only)
MPDS_DEVICE=cpu python -m mpds_models.train /home/alibi/lab/processed_v8 /home/alibi/lab/models_v16 --height 0

# predict particular systems
python predict.py /home/alibi/lab/processed_v8 /home/alibi/lab/models_v18 --pairs Ag_Cu Al_Si

# cross-validation of v16, v17 and v18 in the same folds (about 15 min)
MPDS_DEVICE=cpu python cross_validate.py /home/alibi/lab/processed_v8 --out results/v18/cv \
    --groups-from /home/alibi/lab/processed_v6

# compare two reports
python compare_reports.py results/v17/report_v17.json results/v18/report_v18.json --names v17 v18
```

From Python:

```python
from mpds_models.data import load_dataset
from mpds_models.pipeline import PhaseDiagramModel

model = PhaseDiagramModel.load("/home/alibi/lab/models_v18")
ds = load_dataset("/home/alibi/lab/processed_v8", model.features)
pred = model.predict(ds.X[ds.test_idx])     # liquidus, solidus, topology, eutectic_*, gap, ...
```

## Data

- `/home/alibi/lab/processed_v8`: 1,380 systems (1,104 train / 138 validation /
  138 test), built from the raw MPDS files by `data_build/run_preprocessing.py`
  (processed_v6) and `data_build/build_two_liquid_dataset.py --solid-floor`.
- `/home/alibi/lab/extra_v8_all`: 404 systems with curves only, for training the
  curve model (`data_build/build_extra_systems.py`; ORCHID calculated diagrams,
  Okamoto handbook points, rejected MPDS systems).
- Features computed from the element symbols are cached in
  `/home/alibi/lab/cache_features` (`MPDS_FEATURE_CACHE`); delete it to recompute.

## Licences of the inputs

TabPFN v2 (Prior Labs Licence: Apache 2.0 + "Built with TabPFN"), TabICL v2
(BSD-3), matminer element tables (BSD-style), Materials Project 2018 snapshot
(CC BY 4.0), ORCHID and its Okamoto-derived records (CC BY 4.0).
