# Phase diagram prediction from element properties

We predict binary (two-element) phase diagrams from properties of the two
elements: the liquidus and solidus curves, the diagram type, the eutectic
point, the phases present and whether the two liquids mix. The models learn
from diagrams in the MPDS database. This is a research project; results are
reported to the supervising professor, so every claim needs evidence from a
run we can repeat.

## Environment

- Work on the owner's Ubuntu machine: repository `/home/alibi/lab_new/Phase_Diagrams_V2`,
  conda environment `ml` (`conda activate ml`; without it XGBoost is missing).
- Processed data: `/home/alibi/lab/processed_v6` (features.parquet, targets_*.npz,
  targets_scalar.parquet, weights.parquet, splits.json). It is the input to
  everything; never write into it. Put derived data in a new folder.
- The preprocessing package that builds `processed_v6` from the raw MPDS
  diagrams (`mpds_preprocess`, called by `run_preprocessing.py`) is not in this
  repository; it lives on the owner's machine. Ask where it is before changing it.
- Trained models live outside the repository, in `/home/alibi/lab/models_<version>`
  (an ExtraTrees curve file is about 230 MB). Never commit model files.
- XGBoost runs on the GPU when CUDA is visible and on the CPU otherwise; the two
  build slightly different trees, so compare versions trained on the same device.

## Data

- 1,380 systems, 207 features each (`phys__*` element properties with __A / __B /
  __ratio / __abs_diff variants, `comp__MagpieData *` statistics of the 50/50
  mixture, `thermo__*` Miedema enthalpies and Yang Ω/δ, `meta__*` counts).
  Split 1,104 train / 138 validation / 138 test (`splits.json`).
- Targets: liquidus and solidus on a 101-point composition grid (0-100 at% B,
  with masks and the spread between sources), diagram type (isomorphous,
  simple_eutectic, eutectic_with_intermetallics, monotectic, complex), eutectic
  existence / composition / temperature, miscibility gap, phase set.
- Known data problems (all found and measured; see results/v9_experiments,
  v10, v12):
  - Melting-point features are correct (all within 50 K of the CRC Handbook).
  - A third of solidus end points sit at the eutectic temperature instead of Tm
    (the grid is too coarse for narrow terminal solid solutions); 11 % of
    liquidus end points are off. `fix_end_points` handles both.
  - 76 diagrams (63 monotectic) are cut off at the source chart's top edge: the
    liquidus runs flat at round Celsius values. `mask_clipped_tops` masks them.
  - Monotectic systems record the two-liquid dome inconsistently (dome, flat
    monotectic line, or clipped). This is the main reason they are hard.
  - Data groups (`base.system_quality`): 883 clean, 356 with a volatile element
    (boils below 1,300 K), 132 incomplete (under half the range covered or a
    gap of 10 at% or more), 144 monotectic. Groups overlap.
  - 315 systems have liquidus points above an element's boiling point; most are
    chalcogenide/pnictide systems measured under pressure, not errors.
    `results/v12/audit/check_by_hand.csv` lists the 57 worth checking by hand.
  - Where several sources describe a system they agree closely (median spread
    2.6 K on the liquidus), so the model, not label noise, limits the error.

## Code

`mpds_models/` (the package):
- `base.py`: `load_dataset`, metrics, data fixes (`mask_clipped_tops`,
  `fix_end_points`), data groups (`system_quality`), A/B swap augmentation
  (`swap_augment`, `swap_rules`).
- `tree_curves.py`: `TreeCurveModel`, the current curve model. ExtraTrees
  predicts the deviation from the straight line between the two melting points
  (curve ends exactly at Tm) and a non-negative solidus gap; `aggregate="median"`
  takes the median over the trees.
- `deep_curve_net.py` (CurveNet MLP), `curve_ensemble.py` (CurveNet + trees
  average), `gated_curves.py` (miscibility-gap expert + rest): alternatives
  that lost to `tree_curves.py`; kept for comparison.
- `m1_phase.py` (one XGBoost per phase), `m4_eutectic.py` (eutectic exists /
  composition / temperature; XGBoost, or forests via `backend`),
  `m5_miscibility.py`, `m6_topology.py` (solo topology classifier),
  `meta_learner.py` (stacked topology classifier on out-of-fold M1-M5
  predictions, and `FullDiagramReconstructor`, which loads a model folder).
- `train_all.py`: trains everything into one model folder.

Scripts (repository root):
- `predict.py`: scores a model folder on the test split; writes a JSON report
  with per-data-group curve errors.
- `compare_curve_models.py`: curve models over K-fold cross-validation, with
  variants (aggregate, swap, end-point fix, incomplete weight, gated).
- `error_analysis.py`: every test diagram vs the truth (gallery pages, worst
  first) and where/why the error occurs (composition, eutectic, steepness,
  data quality, data groups, drivers).
- `check_endpoints.py` (data audit and flags), `noise_floor.py` (source spread),
  `analyze_eutectic.py` (eutectic models over repeated CV), `compare_reports.py`,
  `plot_results.py` (charts across versions), `compare_diagrams.py`,
  `plot_diagrams.py`, `benchmark_models.py` (LazyPredict screen).
- The README's "Expected performance" section predates this work; trust
  `results/` instead.

## How to evaluate

- Score every model the same way:
  `python predict.py $D <model_dir> --report results/vNN/report_vNN.json --mask-clipped --fix-end-points --no-uncertainty`
- Decide on changes with 5-fold cross-validation over all 1,380 systems
  (`compare_curve_models.py --cv 5 ...`), not the 138-system test split, where
  one system is 0.7 % of accuracy. A change is real if it wins in at least 4 of
  5 folds. Keep the test split for final reporting.
- Headline numbers: liquidus and solidus MAE (overall, and on clean systems),
  diagram-type accuracy and macro-F1, eutectic existence F1 and position error.
- With no new options, `train_all` must reproduce the previous results exactly;
  check this before and after changing shared code.

## Current best: v13

```
python -m mpds_models.train_all $D /home/alibi/lab/models_v13 \
    --curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped
```

| Version | Change | Liquidus MAE (test) | Solidus MAE (test) |
|---|---|---|---|
| v9 | CurveNet MLP | 166 K | 217 K |
| v10 | ExtraTrees curves, clipped tops masked | 141 K | 199 K |
| v11 | + A/B swap augmentation | 133 K | 189 K |
| v13 | + median over trees | 126 K | 185 K |

v13 on the test split: clean systems 114 K liquidus / 175 K solidus, monotectic
250 / 386 K, incomplete 164 / 276 K, volatile 145 / 175 K; diagram type 74.6 %
accuracy, macro-F1 0.62 ("complex" F1 0.29, simple eutectic 0.55); eutectic
existence F1 0.89, composition error 20 at%, temperature error 189 K;
miscibility-gap F1 0.63. In 5-fold CV the v13 curve model scores 119 K liquidus
overall, 97 K on clean systems, about 170 K solidus.

## What we learned (do not repeat without a new reason)

- Model family matters little: LazyPredict and repeated CV put tree ensembles,
  XGBoost and HistGradientBoosting level; only linear models are clearly worse.
- What helped: predicting the deviation from the melting-point line (-18 to
  -22 K), A/B swap augmentation (-8 K, wins every fold), the median over trees
  (-4.5 K, every fold; removed invented bumps such as Au-Co, Au-Ni), masking
  clipped tops.
- What did not help: CurveNet (worse than trees even after fixes), down-weighting
  or dropping incomplete diagrams (worse in every fold), separate experts for
  miscibility-gap systems (`gated_curves`, worse on monotectic systems: too few
  examples and gate errors), measuring the eutectic composition from the
  lower-melting element (within noise), fixing end points for the tree model
  (no effect; it already pins its ends to Tm).
- Eutectic existence looks easy only because 80 % of systems have one ("always
  yes" scores F1 0.89); ROC-AUC 0.85 is the honest score.
- The solidus error is close to the error of predicting invariant temperatures
  (eutectic temperature 189 K vs solidus 185 K): the solidus steps sit at each
  system's own eutectic/peritectic temperatures, which neighbours don't share.
- Averaging neighbour curves smears sharp features (V-shaped eutectics, steps);
  under positional uncertainty the smearing also lowers MAE, so sharper curves
  only pay off with better information on where the features are.
- Better curves did not improve the diagram-type classifier.
- Error is concentrated: worst 10 % of test systems carry 38 % of the liquidus
  error; diagrams of the wrong predicted type carry 40 %.

## Ideas not yet tried

Align neighbour curves on their eutectic before averaging; build curves from
predicted landmarks (eutectic point, compound melting points, monotectic line);
a regular-solution thermodynamic curve as the base instead of the straight line;
CALPHAD-style parameter prediction (first check how well fitted parameters can
reproduce the simple and monotectic diagrams); better eutectic/peritectic
temperature models; element crystal structure and heat of fusion as features;
a clear preprocessing rule for the liquidus in two-liquid systems; redefining
the "complex" topology class; more systems.

## Conventions

- One branch per change, merged to `main` by pull request; results go in
  `results/vNN/` with the console output saved (`2>&1 | tee ...`) and committed.
- New behaviour goes behind an option until cross-validation shows it wins;
  then make it the default and record it here.
- Keep this file current: update "Current best", "What we learned" and "Ideas
  not yet tried" when results change, and delete notes that turn out wrong.
