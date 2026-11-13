# Phase diagram prediction from element properties

We predict binary (two-element) phase diagrams from properties of the two
elements: the liquidus and solidus curves, the diagram type, the eutectic
point, the phases present and whether the two liquids mix. The models learn
from diagrams in the MPDS database. This is a research project; results are
reported to the supervising professor, so every claim needs evidence from a
run we can repeat. The full review behind v14 is `docs/research_review.md`.

## Environment

- Work on the owner's Ubuntu machine (WSL, 20 cores; the GPU is not always
  visible, see below): repository
  `/home/alibi/lab_new/Phase_Diagrams_V2`, conda environment `ml`
  (`conda activate ml`; without it XGBoost is missing).
- Processed data: `/home/alibi/lab/processed_v6` (features.parquet, targets_*.npz,
  targets_scalar.parquet, weights.parquet, splits.json). It is the input to
  everything; never write into it. Put derived data in a new folder.
- `/home/alibi/lab/processed_v7` is derived from it by `build_two_liquid_dataset.py`:
  same systems, rows and splits, with two-liquid regions treated as liquid (see
  Data). `changes_vs_reference.csv` in it lists what changed per system.
- Raw MPDS diagrams: `/home/alibi/lab/mpds_data` (9,300 JSON files, 1,671 systems).
  The preprocessing package `mpds_preprocess` is in `/home/alibi/lab/` (not in this
  repository). Re-running it reproduces `processed_v6` file for file (checked
  November 2026). Do not edit it in place; `build_two_liquid_dataset.py` shows how
  to wrap its functions at run time instead.
- Trained models live outside the repository, in `/home/alibi/lab/models_<version>`
  (an ExtraTrees curve file is about 480 MB). Never commit model files.
- XGBoost runs on the GPU when CUDA is visible and on the CPU otherwise; the two
  build different trees, so compare versions trained on the same device. The
  GPU (RTX 5070 Ti) comes and goes with WSL restarts: it was invisible on
  12 November and visible on the 13th, and the same v13 command then gave gap
  F1 0.59 instead of 0.63 and eutectic temperature error 184 K instead of 189 K
  on the test split (the ExtraTrees curves do not change). v13 and v14 were
  trained on the CPU. To stay comparable, train with `MPDS_DEVICE=cpu` in the
  environment (`base.detect_device` reads it; `results/v14/train_and_score.sh`
  sets it) and check the "Compute device" line of the training log. Do not hide
  the GPU with `CUDA_VISIBLE_DEVICES=""`: XGBoost 3.2 then fails with
  cudaErrorNoDevice. The number of threads does not change the results.
- Do not run several all-core jobs at once: XGBoost with 20 threads against other
  jobs slows down tenfold (a scalar-model fold took 205 s instead of 62 s). For
  runs side by side, pin each to its own cores with `taskset -c 0-5 python ...`.

## Data

- 1,380 systems, 207 features each (`phys__*` element properties with __A / __B /
  __ratio / __abs_diff variants, `comp__MagpieData *` statistics of the 50/50
  mixture, `thermo__*` Miedema enthalpies and Yang Ω/δ, `meta__*` counts).
  `load_dataset(..., features="extended")` adds 147 more (`el__*`, `mied__*`,
  `phys__ideal_*` / `phys__reg_*`; see `element_features.py`).
  Split 1,104 train / 138 validation / 138 test (`splits.json`).
- 67 elements, so 1,380 of 2,211 possible pairs are present; every element
  occurs in 8 to 61 systems.
- Targets: liquidus and solidus on a 101-point composition grid (0-100 at% B,
  with masks and the spread between sources), diagram type (isomorphous,
  simple_eutectic, eutectic_with_intermetallics, monotectic, complex), eutectic
  existence / composition / temperature, miscibility gap, phase set.
- Known data problems (all found and measured; see results/v9_experiments,
  v10, v12, v14):
  - Melting-point features are correct (all within 50 K of the CRC Handbook).
  - A third of solidus end points sit at the eutectic temperature instead of Tm
    (the grid is too coarse for narrow terminal solid solutions); 11 % of
    liquidus end points are off. `fix_end_points` handles both.
  - 76 diagrams (63 monotectic) are cut off at the source chart's top edge: the
    liquidus runs flat at round Celsius values. `mask_clipped_tops` masks them.
  - **Two-liquid systems (the largest problem, fixed in processed_v7).** The raw
    files label the two-liquid region ("L1 + L2"), which the preprocessing never
    read. In `processed_v6` the "liquidus" there is the top of the dome in some
    systems, the monotectic line in others and the chart edge in the rest, and
    the gap flag marks 144 systems where 245 have two liquids (101 are labelled
    simple_eutectic / eutectic_with_intermetallics / complex). These 245 systems
    carry 32 % of the v13 liquidus error. In `processed_v7` the liquidus is the
    monotectic line, the solidus counts regions under the gap as liquid + solid,
    the gap flag and diagram type are corrected, and the dome is a new target
    (`targets_binodal.npz`, 235 systems, not validated, not modelled yet).
  - The eutectic target is inconsistent with the liquidus: in 38 % of eutectic
    systems it lies more than 50 K from the liquidus at its own composition, in
    15 % its composition is within 2 at% of a pure element
    (`results/v14/eutectic_target_check.txt`). The eutectic position errors we
    report measure this as much as the model.
  - Some diagrams are schematic: a straight line between the melting points
    (In-Nb, Eu-Ta, Mo-Y, Cr-Eu). They dominate the worst-case lists.
  - Data groups (`base.system_quality`): 883 clean, 356 with a volatile element
    (boils below 1,300 K), 132 incomplete (under half the range covered or a
    gap of 10 at% or more), 144 monotectic (245 in processed_v7). Groups overlap.
  - 315 systems have liquidus points above an element's boiling point; most are
    chalcogenide/pnictide systems measured under pressure, not errors.
    `results/v12/audit/check_by_hand.csv` lists the 57 worth checking by hand.
  - Where several sources describe a system they agree closely (median spread
    2.6 K on the liquidus), so the model, not label noise, limits the error.
    (True for scatter between sources; the two-liquid problem above is a
    definition problem that the spread does not see.)

## Code

`mpds_models/` (the package):
- `base.py`: `load_dataset` (`features="v6"` or `"extended"`), metrics, data fixes
  (`mask_clipped_tops`, `fix_end_points`), data groups (`system_quality`), A/B swap
  augmentation (`swap_augment`; uses `ds.X_swap` when the dataset carries it,
  else `swap_rules`), `load_dataset_for_models` (reads the feature set a model
  folder was trained on from its metrics.json).
- `element_features.py`: the extended features, each a function of the ordered
  pair (A, B) so the swapped system is just (B, A). Sources and licences in its
  docstring (all tables ship with matminer).
- `tree_curves.py`: `TreeCurveModel`, the current curve model. ExtraTrees
  predicts the deviation from the straight line between the two melting points
  (curve ends exactly at Tm); `aggregate="median"` takes the median over the
  trees; `solidus="direct"` predicts the solidus as its own deviation from that
  line (v14), `solidus="gap"` as a non-negative gap below the liquidus (v13).
- `deep_curve_net.py` (CurveNet MLP), `curve_ensemble.py` (CurveNet + trees
  average), `gated_curves.py` (miscibility-gap expert + rest): alternatives
  that lost to `tree_curves.py`; kept for comparison.
- `m1_phase.py` (one XGBoost per phase), `m4_eutectic.py` (eutectic exists /
  composition / temperature; XGBoost, or forests via `backend`; the temperature
  regressor keeps the v6 columns when the features are extended),
  `m5_miscibility.py`, `m6_topology.py` (solo topology classifier),
  `meta_learner.py` (stacked topology classifier on out-of-fold M1-M5
  predictions, and `FullDiagramReconstructor`, which loads a model folder).
- `compound_features.py`: EXPERIMENTAL, not in v14. Formation energies of known
  compounds from a Materials Project snapshot. Read its caveat before use.
- `train_all.py`: trains everything into one model folder.

Scripts (repository root):
- `predict.py`: scores a model folder on the test split; writes a JSON report
  with per-data-group curve errors.
- `cross_validate.py`: 5-fold cross-validation of named curve configurations
  (`v13`, `v13+direct`, `v14-features`, `v14`) and of the scalar models with
  either feature set. The tool for deciding between versions.
- `explore_curve_variants.py`: the experiments behind v14 (variants that lost,
  upper bounds, learning curve), each against v14 in cross-validation.
- `build_two_liquid_dataset.py`: derives `processed_v7` from the raw files.
- `compare_curve_models.py`: older curve comparison (CurveNet, ensemble, gated,
  end-point fix, incomplete weight); trains on 90 % of each training fold.
- `error_analysis.py`: every test diagram vs the truth (gallery pages, worst
  first) and where/why the error occurs (composition, eutectic, steepness,
  data quality, data groups, drivers).
- `check_endpoints.py` (data audit and flags), `noise_floor.py` (source spread),
  `analyze_eutectic.py` (eutectic models over repeated CV), `compare_reports.py`,
  `plot_results.py` (charts across versions), `compare_diagrams.py`,
  `plot_diagrams.py`, `benchmark_models.py` (LazyPredict screen),
  `make_review_figures.py` (figures of the review).

## How to evaluate

- Score every model the same way:
  `python predict.py $D <model_dir> --report results/vNN/report_vNN.json --mask-clipped --fix-end-points --no-uncertainty`
- Decide on changes with 5-fold cross-validation over all 1,380 systems
  (`cross_validate.py $D --out results/vNN/cv --configs v14 <new>`), not the
  138-system test split, where one system is 0.7 % of accuracy. A change is real
  if it wins in at least 4 of 5 folds. Keep the test split for final reporting.
  The test split can disagree with cross-validation: the v14 solidus gain
  (14 K in CV, every fold) is 0.8 K on the test split because of three systems.
- `cross_validate.py` trains on all systems of the training folds, so its
  numbers are about 2 K better than those of `compare_curve_models.py`
  (v13: 116.7 K vs the 119.1 K recorded in results/v13).
- State which targets a number refers to. Errors on `processed_v7` are against a
  different (consistent) liquidus and must not be mixed with `processed_v6` ones.
- Headline numbers: liquidus and solidus MAE (overall, and on clean systems),
  diagram-type accuracy and macro-F1, eutectic existence F1 and position error.
- With no new options, `train_all` must reproduce the previous results exactly;
  check this before and after changing shared code
  (`bash results/v14/train_and_score.sh <tmp folder> "v13_rerun"`, then compare
  `results/v14/report_v13_rerun.json` with `results/v13/report_v13.json`).

## Current best: v14

```
python -m mpds_models.train_all $D /home/alibi/lab/models_v14 \
    --curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped \
    --features extended --curve-solidus direct --train-with-val
```

The same command on `/home/alibi/lab/processed_v7` gives
`/home/alibi/lab/models_v14_two_liquid`. `results/v14/train_and_score.sh` trains
and scores every model of the comparison.

| Version | Change | Liquidus MAE (test) | Solidus MAE (test) |
|---|---|---|---|
| v9 | CurveNet MLP | 166 K | 217 K |
| v10 | ExtraTrees curves, clipped tops masked | 141 K | 199 K |
| v11 | + A/B swap augmentation | 133 K | 189 K |
| v13 | + median over trees | 126 K | 185 K |
| v14 | + extended features, direct solidus, validation systems used for training | 113 K | 184 K |
| v14 on processed_v7 | two-liquid targets (not comparable with the rows above) | 104 K | 161 K |

5-fold cross-validation, liquidus / solidus MAE in K (results/v14/cv_v6_curves.txt,
cv_v7_curves.txt); v14 wins every fold on both curves and both datasets:

| | all | clean | incomplete | volatile | monotectic (144) |
|---|---|---|---|---|---|
| v13, processed_v6 | 116.7 / 167.7 | 95 / 150 | 224 / 321 | 136 / 178 | 232 / 284 |
| v14, processed_v6 | 112.0 / 153.4 | 93 / 139 | 215 / 281 | 133 / 157 | 205 / 276 |
| v13 settings, processed_v7 | 102.4 / 157.6 | 85 / 137 | 228 / 355 | 126 / 169 | 158 / 225 |
| v14, processed_v7 | 98.9 / 137.1 | 84 / 125 | 210 / 272 | 122 / 142 | 136 / 184 |

v14 on the test split (processed_v6): clean systems 99 K liquidus / 173 K
solidus, monotectic 253 / 415 K, incomplete 166 / 290 K, volatile 139 / 163 K;
diagram type 75.4 % accuracy, macro-F1 0.65; eutectic existence F1 0.89,
composition error 18 at%, temperature error 179 K; miscibility-gap F1 0.61
(v13: 0.63; 14 true cases, see docs/research_review.md section 4.4).
On processed_v7: miscibility-gap F1 0.89 (v13 settings 0.77), diagram type 76.8 %.
Of the 12.8 K liquidus gain on the test split, about half comes from training on
the validation systems (v13 with them: 119.3 K) and half from the features.

## What we learned (do not repeat without a new reason)

- Model family matters little: LazyPredict and repeated CV put tree ensembles,
  XGBoost and HistGradientBoosting level; only linear models are clearly worse.
- What helped: predicting the deviation from the melting-point line (-18 to
  -22 K), A/B swap augmentation (-8 K, wins every fold), the median over trees
  (-4.5 K, every fold; removed invented bumps such as Au-Co, Au-Ni), masking
  clipped tops, the extended features (-4.7 K liquidus, every fold; most of it
  from the regular-solution liquidus features), the direct solidus (-12 to
  -14 K solidus, every fold), training on the validation systems (-2.4 K), and
  the consistent two-liquid targets (-14 K liquidus for the same model).
- The error still falls with more systems: 146 / 130 / 119 / 112 K liquidus with
  25 / 50 / 75 / 100 % of the training systems. More data is the largest lever.
- A third of the liquidus error is each system's overall height above the
  melting-point line (112 -> 75 K if it were known). A separate model for that
  number is no better than the curve model's own value.
- Upper bound: the TRUE diagram type, gap flag and eutectic point as extra
  inputs do not improve the liquidus (and the solidus by 4 K). Building curves
  from predicted landmarks cannot pay off with this model.
- What did not help (results/v14/explore_v6.txt unless noted): CurveNet (worse
  than trees even after fixes), down-weighting or dropping incomplete diagrams
  (worse in every fold), separate experts for miscibility-gap systems
  (`gated_curves`), the gap probability as an extra input, measuring the
  eutectic composition from the lower-melting element (within noise), fixing
  end points for the tree model (no effect; it already pins its ends to Tm), a
  regular-solution or ideal-solution curve as the base instead of the straight
  line (worse in every fold; the regular-solution curve alone is off by 758 K),
  the deviation relative to the mean melting point, curves borrowed from
  systems sharing an element weighted by measured element similarity (132 K),
  LightGBM on one row per composition point (121 K), the eutectic point read off
  the predicted liquidus (195 K vs 165 K from the eutectic model), other tree
  settings (all within 1 K).
- Passed the 4-of-5 rule but deliberately not adopted: knots every 1 at%
  (-0.2 / -0.9 K, doubles the model file); a 30:70 blend of the per-point
  LightGBM with v14 (-0.7 K, weight picked on the same folds); formation
  energies of known compounds (-1.4 K; they tell the model which compounds
  have been observed, which is the professor's call).
- The extended features help the gap classifier (F1 0.56 -> 0.61, every fold)
  and the solo type classifier (74.7 -> 76.1 %, 4 of 5) and the eutectic
  composition (19.8 -> 18.9 at%, every fold), but made the eutectic temperature
  regressor worse (164.8 -> 166.2 K, 3 of 5 folds), so that one keeps the v6
  columns.
- Eutectic existence looks easy only because 80 % of systems have one ("always
  yes" scores F1 0.89); ROC-AUC 0.86 is the honest score.
- The eutectic position scores in predict.py are taken where truth AND
  prediction have a eutectic, so they move when the existence classifier
  changes even if the regressors are identical. Use the `*_all_true` columns of
  `cross_validate.py --scalars` to compare regressors.
- The solidus error is close to the error of predicting invariant temperatures:
  the solidus steps sit at each system's own eutectic/peritectic temperatures,
  which neighbours don't share.
- Averaging neighbour curves smears sharp features (V-shaped eutectics, steps);
  under positional uncertainty the smearing also lowers MAE, so sharper curves
  only pay off with better information on where the features are.
- Better curves did not improve the diagram-type classifier. On the test split
  the identical v14 command gave 75.4 % type accuracy on the CPU and 79.0 % on
  the GPU: differences of that size on 138 systems mean nothing.
- Error is concentrated: the worst 10 % of systems carry a third of the liquidus
  error (compounds of P, As, S, Se, Te melting far above both elements; carbon
  systems; schematic straight-line diagrams; strongly immiscible pairs).

## Ideas not yet tried

Ranked by expected gain (see docs/research_review.md, section 7): more systems
(291 in the raw files were rejected by the preprocessing filters, 88 of them
only because the temperature unit was undetermined; the liquidus curves of 466
assessed systems of Deffrennes et al. 2023 are CC BY 4.0); flag or remove
schematic straight-line diagrams and treat carbon separately; model the
two-liquid dome (`targets_binodal.npz`); calibrate the spread between trees into
an error bar per diagram; redefine the eutectic target from the liquidus
itself; scores for the features a metallurgist reads (lowest and highest
liquidus point); compound stability from a database of hypothetical structures
(OQMD prototypes, AFLOW) instead of known compounds; align neighbour curves on
their eutectic before averaging; redefining the "complex" topology class.
Not worth pursuing without a new reason: CALPHAD-style parameter prediction
(813 systems have intermetallic compounds whose stability it would have to
predict first), curves from predicted landmarks, other model families.

## Conventions

- One branch per change, merged to `main` by pull request; results go in
  `results/vNN/` with the console output saved (`2>&1 | tee ...`) and committed.
- New behaviour goes behind an option until cross-validation shows it wins;
  then make it the default and record it here. (v14's options are passed
  explicitly, as v13's were; `train_all` defaults are still the v9 CurveNet.)
- Keep this file current: update "Current best", "What we learned" and "Ideas
  not yet tried" when results change, and delete notes that turn out wrong.
