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
- `/home/alibi/lab/processed_v7` is derived from it by `data_build/build_two_liquid_dataset.py`:
  same systems, rows and splits, with two-liquid regions treated as liquid (see
  Data). `changes_vs_reference.csv` in it lists what changed per system.
- `/home/alibi/lab/processed_v8` (same script with `--solid-floor`) supersedes
  processed_v7: the monotectic line is never below a solid phase under the dome
  (v7 cut 17 systems too low, e.g. Cs-Ga to 303 K) and the solidus is capped at
  the liquidus. Use v8 for new work; v7 is kept so the v14 numbers reproduce.
- Extra systems for training the curve model only (`data_build/build_extra_systems.py`,
  `data.add_extra_systems`): `/home/alibi/lab/extra_v6` and `extra_v8`, 90
  systems (29 rejected MPDS systems, 61 CALPHAD liquidus curves of Deffrennes
  et al. 2023 in `/home/alibi/lab/external/deffrennes2023`, CC BY 4.0).
- `/home/alibi/lab/extra_v8_orchid`: 300 extra systems for curve training (271 from
  the ORCHID calculated diagrams in `/home/alibi/lab/external/orchid`, read by
  `orchid_curves.py`; 29 recovered MPDS systems). ORCHID follows the processed_v8
  convention and agrees with it to a median of 24 K on 848 shared systems.
- `/home/alibi/lab/extra_v8_all`: 404 extra systems = the 300 above + 104 from
  liquidus points of Okamoto's handbook diagrams (ORCHID-Assemblages,
  `/home/alibi/lab/external/orchid/assemblages`, read by `okamoto_points.py`:
  the liquid end of each "liquid + solid" tie line; median 6 K from processed_v8,
  about 3 points per system). This is the set v16 uses.
- `/home/alibi/lab/mpds_open`: keyless MPDS download (`download_mpds.py` and
  `convert_mpds_open.py`, at tag pre-cleanup; resumable; stopped after about 1,000
  of 3,655 pairs); geometry only (curves, no labels), not used by any model.
- Speed: the foundation models (TabPFN, TabICL) take 5-10 s per fit or predict
  call and are CPU-bound: running them in parallel threads does not help. Model
  folders store the fitted ones in `curves_foundation/`; TabPFN predictions after
  loading differ by < 0.06 K. (`thermo_engine.py`, at tag pre-cleanup, is compiled
  with numba: 0.4 ms per diagram.)
- XGBoost uses 6 threads on the CPU (`MPDS_XGB_THREADS`); 20 threads were three
  to a hundred times slower on our 1,200 rows and give identical trees.
- Raw MPDS diagrams: `/home/alibi/lab/mpds_data` (9,300 JSON files, 1,671 systems).
  The preprocessing package `mpds_preprocess` is in `/home/alibi/lab/` (not in this
  repository). Re-running it reproduces `processed_v6` file for file (checked
  November 2026). Do not edit it in place; `data_build/build_two_liquid_dataset.py` shows how
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
  environment (`classifiers.xgb_device` reads it) and check the "XGBoost on" line of
  the training log. Do not hide
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

Cleaned up on 26 November 2026 (branch `cleanup`): only what v16-v18 use is
left. Everything older (CurveNet, gated / ensemble curves, `train_all.py`, the
v9-v15 comparison scripts, `explore_v18.py`, the thermodynamic engine of
results/v19, the MPDS downloader, plotting scripts) is at git tag
`pre-cleanup`; check it out to re-run an older result. Model folders saved
before the cleanup (models_v16-v18) only load at that tag.

`mpds_models/`:
- `data.py`: `PhaseDataset`, `load_dataset(dir, "extended" | "extended+mp")`,
  `add_extra_systems` (curve-only rows, `extra_idx`), A/B swap (`swap_rules`,
  `swap_features`; `X_swap` holds every system with A and B exchanged), data
  fixes (`mask_clipped_tops`, `fix_end_points` for scoring), data groups
  (`system_quality`, `group_rows`), `curve_scores`.
- `features.py`: the extended element features (v14) and the Materials Project
  compound features (v15, read the caveat there), cached on disk in
  `/home/alibi/lab/cache_features` (35 s -> 0.1 s per load).
- `curves.py`: `TreeCurves`, ExtraTrees with the v14 settings fixed (deviation
  from the melting-point line, median over trees, direct solidus, A/B swap).
- `foundation.py`: `FoundationCurves` = trees -> height correction (v17) ->
  blend with the 10-component curve (v18); `predict_stages` gives all three.
  Predictions are remembered per input array (each of the 22 TabPFN/TabICL
  models takes 5-10 s per call; the old code called them ~8 times on the same
  rows, about 25 of the 34 training minutes).
- `classifiers.py`: phases, eutectic, miscibility gap, solo and stacked
  diagram type (XGBoost, v6 settings).
- `pipeline.py`: `PhaseDiagramModel` (fit / predict / save / load of all models;
  `height=0` is v16, `blend=0` v17, defaults v18).
- `evaluation.py`: scores in the report format of predict.py.
- `train.py`: `python -m mpds_models.train PROCESSED MODEL_DIR [--height 0] [--blend 0]
  [--features extended] [--no-extra-data]`.

Scripts: `predict.py` (report of a model folder), `cross_validate.py` (v16 / v17
/ v18 in the same folds; `--against NAME=oof_curves.npz` compares with an
earlier run, `--scalars` the classifiers), `error_analysis.py` (galleries and
error breakdowns), `compare_reports.py`. `data_build/`: `run_preprocessing.py`
(processed_v6), `build_two_liquid_dataset.py` (v7 / v8), `build_extra_systems.py`
+ `orchid_curves.py` + `okamoto_points.py` (extra_v8_all; the rebuild with the
command in its docstring reproduces extra_v8_all file for file).

## How to evaluate

- Score every model the same way: `python predict.py $D <model_dir> --report
  results/vNN/report_vNN.json` (it always leaves clipped tops and end points
  that disagree with Tm out of the scores, as `--mask-clipped --fix-end-points`
  did before the cleanup).
- Decide on changes with 5-fold cross-validation over all 1,380 systems
  (`MPDS_DEVICE=cpu python cross_validate.py $D --out results/vNN/cv --groups-from
  /home/alibi/lab/processed_v6`; v16 / v17 / v18 come out of one run, and
  `--against v18=<earlier oof_curves.npz>` compares a new idea with them), not the
  138-system test split, where one system is 0.7 % of accuracy. A change is real
  if it wins in at least 4 of 5 folds. Keep the test split for final reporting.
  The test split can disagree with cross-validation: the v14 solidus gain
  (14 K in CV, every fold) is 0.8 K on the test split because of three systems.
- The data groups of the v14-v18 tables come from processed_v6 (`--groups-from`);
  in processed_v8 the monotectic group has 245 systems instead of 144.
- State which targets a number refers to. Errors on `processed_v7` / `v8` are
  against a different (consistent) liquidus and must not be mixed with
  `processed_v6` ones.
- Headline numbers: liquidus and solidus MAE (overall, and on clean systems),
  diagram-type accuracy and macro-F1, eutectic existence F1 and position error.
- With default options, `python -m mpds_models.train` must reproduce
  results/v18/report_v18.json; check this after changing shared code
  (results/cleanup has the check made for the cleanup).

## Results by version (commands with `train_all` need tag `pre-cleanup`)

v14:

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

### v15 candidate (16 November 2026, results/v15)

v14 plus formation energies of known compounds, on processed_v8:

```
MPDS_DEVICE=cpu python -m mpds_models.train_all /home/alibi/lab/processed_v8 /home/alibi/lab/models_v15 \
    --curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped \
    --features extended+mp --curve-solidus direct --train-with-val
```

5-fold CV on processed_v8: v13 settings 102.2 / 157.1 K, v14 98.8 / 136.1 K,
v15 96.9 / 133.9 K (wins every fold on both curves). Test split of processed_v8:
v14 105.7 / 160.5 K, v15 105.8 / 154.9 K (no liquidus gain there). Scalar models
unchanged within noise. The compound energies describe compounds that are
mostly known from experiment, so report v15 separately from v14 until the
professor decides whether such inputs are in scope.

### v16 candidate (17 November 2026, results/v16)

v15 with the curve model also trained on the 300 extra systems:

```
MPDS_DEVICE=cpu python -m mpds_models.train_all /home/alibi/lab/processed_v8 /home/alibi/lab/models_v16 \
    --curve-model extra_trees --curve-aggregate median --swap-augment --mask-clipped \
    --features extended+mp --curve-solidus direct --train-with-val \
    --extra-data /home/alibi/lab/extra_v8_all
```

5-fold CV on processed_v8: v14 98.8 / 136.1 K, v14 + extras 97.9 / 132.6 K (4 of 5
on each curve), v15 96.9 / 133.9 K, v16 96.1 / 130.4 K (against v15: 5 of 5 liquidus,
4 of 5 solidus). Test split: v14 105.7 / 160.5 K, v15 105.8 / 154.9 K, v16 100.4 /
140.2 K. With only the 300 ORCHID + MPDS extras the gain was 3 of 5 folds; the
handbook points tipped it. Not adopted: ORCHID copies of systems we already have
(liquidus worse, solidus better).
Sources checked and not usable: NIMS CPDDB (terms forbid bulk acquisition), the
non-SGTE databases in ORCHID's catalogue (10 binary systems), the MPDS web GUI
(API access disabled for the university account; do not automate the GUI).

### v17 experiment: tabular foundation models (19 November 2026, results/v17)

`compare_foundation_models.py` (TabPFN v2, Prior Labs Licence = Apache 2.0 + "Built
with TabPFN" attribution; TabICL v2, BSD-3; both on the GPU) against the current
models, 5-fold CV on processed_v8 with the v16 features. The newer TabPFN versions
need a registered account and licence acceptance; not used.
- Curve height (the largest find): predicting each system's mean liquidus height
  above the melting-point line, both beat the v16 curve model's own height (TabPFN
  72.7 K, TabICL 74.0 K, their average 71.4 K, against 76.1 K). Shifting the v16 curve
  halfway towards the average lowers the liquidus from 96.1 to 91.7 K, better in all
  5 folds and in every data group (monotectic 134.9 -> 124.0 K); on the test split
  100.4 -> 96.1 K. The 50 % shift was picked after seeing the results (100 % also
  wins 4 of 5). In the pipeline as `train_all --height-correction 0.5`
  (`mpds_models/height_correction.py`: wraps the tree curve model, refits the
  foundation models from stored training rows when it first predicts; the stacked
  type classifier keeps reading the uncorrected curves via `.base`). Height from
  one A/B order is as good as averaging both (91.7 K either way).
  v17 (`results/v17/train_and_score.sh`) = v16 + `--height-correction 0.5`:
  test split 100.4 / 140.2 K (v16) -> 96.0 / 140.1 K; liquidus better in 75 systems,
  worse in 57 (In-Nb +136 K, a schematic straight-line diagram); scalar models
  unchanged; clean systems 82 K mean, 64 K median. v13 still reproduces exactly.
- Classifiers: small gains. Eutectic ROC-AUC 0.864 -> 0.874-0.879 (5 of 5), gap
  average precision 0.848 -> 0.858 (4 of 5), type accuracy 79.3 -> 80.8 % with TabPFN
  (4 of 5); F1 scores within noise. Dividing by class frequency ("balanced") hurts.

### v18 (20 November 2026, results/v18) and the thermodynamic-engine test (results/v19)

v18 = v17 + the liquidus blended 50 % with a curve whose 10 principal components
TabPFN + TabICL predict (`mpds_models/foundation_curves.py`, `train_all
--foundation-blend 0.5 --foundation-components 10`; `results/v18/train_and_score.sh`).
5-fold CV on processed_v8: 91.7 -> 89.2 K (every fold; 6 components 89.7, 16 89.1);
test split 96.0 -> 94.6 K; solidus and scalar models unchanged.
Did not help after v17 (`explore_v18.py`, results/v18): OQMD / Materials Project 2026
hull inputs alone (at most -0.6 K; with the v18 blend -0.8 K, 4 of 5 folds, not
adopted), regular-solution equilibrium inputs (`regular_solution.py`; it places
monotectic plateaus within 54 K median from element data, yet the trees gain
nothing), learning deviations in units of the melting-point gap (worse), averaging
both A/B orders, height-correction variants, a plateau correction (upper bound with
the TRUE plateau level 82.9 K, but predicted levels are too poor), 67 keyless MPDS
systems. Pattern: new inputs barely move the tree model; a different learner aimed
at one quantity does.
Thermodynamic engine (`thermo_engine.py`: regular-solution liquid with 2 terms,
solid solution, OQMD compounds; 4 parameters per system). Fitted to each system's
own liquidus it reproduces it to 57.8 K (median system 42 K; isomorphous 16 K,
simple eutectic 24 K, with compounds 70 K), against 89.2 K for v18. But with the
parameters PREDICTED from element properties (TabPFN + TabICL, out of fold) it scores
150.4 K: the diagram is very sensitive to the parameters and the fitted parameters
are partly interchangeable. Blending it into v18 makes v18 worse.
Follow-ups (`engine_projection.py`, `engine_learn.py`, results/v19), none beats v18:
- The engine fitted to v18's own predicted curve scores 106-108 K. Blended with v18
  it reaches 89.3 K at best.
- The median curve over 64 parameter sets (predicted + out-of-fold residuals)
  scores 151 K.
- A regularised refit (fit + 2 x distance to the predicted parameters, then retrain
  the predictor) improves the engine itself, 150 -> 138 -> 134 K over two rounds.
  It also fits the data better (52 K), but the best blend with v18 is 90.5 K.
- Why: the four parameters only work as a set. Fitted values for any 1-3 of them
  with the rest predicted give 136-185 K; all four give 57.8 K.
Not worth more rounds without a richer engine or new inputs that pin the
parameters jointly.

### v20 experiment: one row per measured point (26 November 2026, results/v20)

The professor's suggestion: inputs = the features + the composition x, output
= the liquidus temperature at x, one training row per measured point, so a
diagram covering only part of 0-100 at% adds fewer rows and nothing is filled
in (`experiments/pointwise_liquidus.py`, LightGBM, both A/B orders, the 404
extra systems, the same 5 folds; v16-v18 rebuilt from the explore_v18 caches by
`experiments/v18_oof.py`, which reproduces 96.1 / 91.7 / 89.2 K). Liquidus MAE:
output T directly 112.2 K, output T relative to the melting-point line 105.6 K,
against v16 96.1 and v18 89.2 K (worse in every fold). Not better on
incomplete diagrams either (209 / 216 K against v16 207, v18 191 K). Blended
into v18 it gains nothing (10 %: 89.3 K, 2 of 5 folds). Parity plots
(parity_liquidus.png: R^2 0.935 / 0.938 against 0.947 for v18; within +-100 K:
61 / 64 % against 71 %) and example curves (the per-point curves are jagged and
do not end at Tm). Scores count only measured points for every model, so the
curve model is not rewarded for the parts it fills in.

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
- More systems are hard to get from what we have: the raw files and the
  published CALPHAD set yield only 90 extra systems, worth 0.3 to 0.8 K on the
  liquidus (3 or 4 of 5 folds) and nothing on the solidus; not adopted. Real
  growth needs MPDS diagrams for elements beyond the present 67.
- An automatic rule for schematic straight-line diagrams does not work: 186
  systems have a long straight sloping liquidus and most are genuine.
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
  LightGBM on one row per composition point (121 K; again in v20: 105.6 K against v18 89.2 K), the eutectic point read off
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
