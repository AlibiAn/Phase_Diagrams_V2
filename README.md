# MPDS Specialist Models

Six specialist ML models + a stacked meta-learner that predict binary phase
diagrams from the feature vectors produced by the `mpds_preprocess` pipeline.

---

## Quick start

Run everything from the repository root.

```bash
# Train all models end-to-end
python -m mpds_models.train_all /data/processed /data/models --curve-epochs 200

# Reproducible run with a fixed seed (default 0); --solidus-mode free restores the v8 behaviour
python -m mpds_models.train_all /data/processed /data/models --seed 1

# Curves that pass exactly through both melting points
python -m mpds_models.train_all /data/processed /data/models --endpoint-mode hard

# Verbose + skip meta-learner (faster for iteration)
python -m mpds_models.train_all /data/processed /data/models --curve-epochs 200 --no-stacked -v

# Evaluate on the test split, then plot predicted vs true diagrams
python predict.py /data/processed /data/models --report results/results.json
python compare_diagrams.py /data/processed /data/models --out comparisons
python plot_diagrams.py /data/processed /data/models --out diagrams

# Eutectic models from a random forest / extra trees instead of XGBoost
python -m mpds_models.train_all /data/processed /data/models_et --m4-backend extra_trees

# ExtraTrees curves instead of CurveNet (saved as m23_curves.joblib), with
# liquidus tops cut off by the source chart left out of training and scoring
python -m mpds_models.train_all /data/processed /data/models_trees --curve-model extra_trees --mask-clipped
python predict.py /data/processed /data/models_trees --report results/trees.json --mask-clipped

# CurveNet without the x = 0 / 100 data values in its data loss; M4 predicting the
# eutectic composition measured from the lower-melting element
python -m mpds_models.train_all /data/processed /data/models_cn --mask-end-points --m4-comp-from-lower-tm

# Average of CurveNet and ExtraTrees, trained on corrected end points and on A/B-swapped copies
python -m mpds_models.base /data/processed      # first check how each feature swaps
python -m mpds_models.train_all /data/processed /data/models_ens --curve-model ensemble \
    --fix-end-points --swap-augment --mask-clipped --mask-end-points
python predict.py /data/processed /data/models_ens --report results/ens.json --mask-clipped --fix-end-points

# How much do the sources disagree, and how close are the models to that?
python noise_floor.py /data/processed --model-dir /data/models_ens --mask-clipped

# Judge changes over 5 cross-validation folds instead of one 138-system split
python compare_curve_models.py /data/processed --out curve_cv --cv 5 --backends extra_trees \
    --train-curvenet --ensemble --mask-clipped --swap-augment both --fix-end-points both

# Experiments: end-point audit, tree-ensemble curves vs CurveNet, eutectic study
python check_endpoints.py /data/processed --out endpoint_audit --curve-model /data/models/m23_curves.pt
python compare_curve_models.py /data/processed --out curve_cmp --curvenet /data/models/m23_curves.pt --seeds 0 1 2
python analyze_eutectic.py /data/processed --out eutectic_study
```

```python
from mpds_models.train_all import train_all
metrics = train_all("/data/processed/", "/data/models/", curve_epochs=200)

from mpds_models.base import load_dataset
from mpds_models.meta_learner import FullDiagramReconstructor
ds = load_dataset("/data/processed/")
recon = FullDiagramReconstructor.from_dir("/data/models/", ds)
```

Writes model files + `metrics.json` to the output dir. Training the full stack
on the ~9,200-system MPDS corpus takes roughly 2–4 hours on a single GPU
(mostly CurveNet epochs). Use `--curve-epochs 60` during development.

---

## Architecture overview

```
Features (207 columns from preprocessing)
        │
        ├──► M1: PhaseClassifier          ─ XGBoost per phase  (multi-label)
        ├──► M2+M3: LiquidusSolidusModel  ─ Shared-trunk MLP   (2 × 101-D curves)
        ├──► M4: EutecticDetector         ─ XGBoost cls + 2× XGBoost reg
        ├──► M5: MiscibilityGapDetector   ─ XGBoost binary cls
        └──► M6: TopologyClassifier       ─ XGBoost 5-class (standalone baseline)
                 │
                 ▼
        StackedTopologyClassifier  ←  M1-M5 out-of-fold predictions
        (meta XGBoost, the "M6 final")
                 │
                 ▼
        FullDiagramReconstructor
        → PredictedDiagram per system (with consistency checks)
```

---

## Model-by-model details

### M1 — Phase Classifier (`m1_phase.py`)
**What it predicts:** which stable phases exist anywhere in the system (multi-hot
vector over the global phase vocabulary, typically 200–400 phases after pruning).

**Why one XGBoost per phase:** phase occurrence is severely class-imbalanced.
`Liquid` appears in 100 % of systems; a specific intermetallic in maybe 1–2 %.
A single multi-label deep net with sigmoid heads handles this poorly without
careful per-output reweighting. Independent classifiers with per-phase
`scale_pos_weight` give better recall on the rare intermetallics that drive
topology classification.

Special cases:
- Phases positive in **every** training system (e.g. `Liquid`) → always-true mask,
  no model trained.
- Phases with fewer than `min_positive_count=3` positives → always-false mask,
  no model trained.

---

### M2 + M3 — Liquidus / Solidus Predictor (`deep_curve_net.py`)
**What it predicts:** two 101-point temperature curves (liquidus and solidus)
over the composition grid `[0, 100]` mol% B.

**Architecture:** shared-trunk MLP → two independent linear heads (one per
curve). Sharing the trunk is justified because the solidus tracks the liquidus
everywhere — a shared representation is more sample-efficient.

**Physics-informed loss (`physics_loss.py`):**

| Penalty | Equation | Physics |
|---------|----------|---------|
| Solidus ≤ Liquidus | `ReLU(T_sol − T_liq) / T_ref` | Cannot be liquid below solidus |
| Endpoint consistency | `((T(x=0) − Tm_A) / T_ref)²` | Pure elements constrain both curves |
| Smoothness | `‖d²T/dx²‖² / T_ref²` | Curves are piecewise smooth (kinks only at invariant points) |
| Bounded magnitude | `ReLU(T − T_max)² + ReLU(T_min − T)²` | Prevents unphysical blow-ups |

All penalties are normalised by `T_ref = 100 K` so they're dimensionless and
comparable in scale to the Huber data loss.

**Uncertainty:** MC-Dropout at inference (`predict_with_uncertainty(X, n_samples=30)`).

**Why MLP, not CGCNN/MEGNet:** the preprocessing pipeline outputs system-level
compositional + thermodynamic features — not per-atom crystal graphs. A GNN
operating on this input is just an MLP with extra machinery. When MPDS S-entry
structures are added to the pipeline, replace the `CurveNet` trunk with a real
`CGCNN/M3GNet` that emits per-system embeddings concatenated to the matminer
features — the "hybrid GNN" design described in the project paper.

---

### M4 — Eutectic Detector (`m4_eutectic.py`)
Three tree-ensemble sub-models on the same feature vector: XGBoost by default,
or a random forest / extra trees with `--m4-backend` (`EutecticDetectorConfig.backend`):

1. **M4a** (classifier): does a eutectic exist? (`scale_pos_weight` for imbalance)
2. **M4b** (regressor): eutectic composition, at% B — trained on positive subset only
3. **M4c** (regressor): eutectic temperature, K — trained on positive subset only

M4b and M4c output `NaN` when M4a predicts no eutectic.

---

### M5 — Miscibility Gap Detector (`m5_miscibility.py`)
Binary XGBoost classifier. Only ~5–10 % of binary systems show a liquid
miscibility gap, so `scale_pos_weight` is critical. If 0 positive examples
appear in training the model correctly degenerates to "always predict False."

---

### M6 — Topology Classifier (`m6_topology.py` + `meta_learner.py`)
**Standalone M6** (`m6_topology.py`): XGBoost 5-class on raw features alone.
Use as a baseline to measure how much the stacking step helps.

**Stacked M6** (`StackedTopologyClassifier` in `meta_learner.py`):
- K-fold over training systems; train fresh M1-M5 copies on K-1 folds,
  predict on held-out fold → out-of-fold (OOF) predictions for every training row.
- Final meta-XGBoost trained on `[original features ∥ OOF curve summaries ∥
  eutectic proba ∥ miscibility proba ∥ phase count stats]` (228 features total).
- OOF step prevents leakage: the meta-learner never sees predictions on the
  same data it was trained on.

Both classifiers handle the case where the training set is missing one or more
topology classes: `seen_classes_` is tracked and probability outputs are padded
with zeros for unseen classes.

---

### Full Diagram Reconstructor (`meta_learner.py`)
`FullDiagramReconstructor.reconstruct(X, pair_ids)` runs all six models and
returns a list of `PredictedDiagram` dataclasses. Each contains:
- `topology` and `topology_proba` (from stacked M6 or solo M6 fallback)
- `liquidus_T`, `solidus_T`, `comp_grid` (from M2/M3)
- `liquidus_T_std`, `solidus_T_std` (MC-Dropout, when `with_uncertainty=True`)
- `eutectic_exists`, `eutectic_comp`, `eutectic_temp` (from M4)
- `miscibility_gap`, `miscibility_proba` (from M5)
- `phases` — top-5 predicted phases with probabilities (from M1)
- `inconsistencies` — list of flagged contradictions, e.g.:
  - `M6 said 'isomorphous' but M4 predicts eutectic at x=42.3%, T=847K`
  - `M6 said 'monotectic' but M5 says no miscibility gap`
  - `solidus > liquidus at N grid points (clamped at inference)`

Inconsistencies are **reported, not corrected** — exposing them is diagnostically
more valuable than silently smoothing them away.

---

## GPU support

XGBoost models auto-detect CUDA via `xgb_device_kwargs()` in `base.py`:
```python
# CUDA available  →  {"device": "cuda", "tree_method": "hist"}
# No CUDA         →  {"device": "cpu",  "tree_method": "hist", "n_jobs": -1}
```

PyTorch (M2/M3 CurveNet) auto-selects `cuda` via `torch_device()`. The `.to(device)`
calls ensure all tensors live on the same device as the model.

On an A100 or H100, the full 9,200-system training run with 200 epochs
takes approximately 90–120 minutes. XGBoost tree building on GPU is typically
5–10× faster than CPU for the feature dimensionalities here.

---

## File reference

`mpds_models/` (the package):

| File | Contents |
|------|----------|
| `base.py` | `PhaseDataset`, `load_dataset()`, device detection, metric helpers, `SpecialistModel` base |
| `physics_loss.py` | Differentiable physics penalties (order, endpoints, smoothness, bounds) |
| `deep_curve_net.py` | `CurveNet` MLP, `LiquidusSolidusModel` (M2+M3), MC-Dropout uncertainty |
| `tree_curves.py` | `TreeCurveModel`: M2+M3 from ExtraTrees / RandomForest / XGBoost, exact end points |
| `curve_ensemble.py` | `CurveEnsemble`: validation-weighted average of CurveNet and ExtraTrees curves |
| `m1_phase.py` | `PhaseClassifier`: one XGBoost per phase |
| `m4_eutectic.py` | `EutecticDetector`: M4a/b/c classifier + regressors |
| `m5_miscibility.py` | `MiscibilityGapDetector`: binary XGBoost |
| `m6_topology.py` | `TopologyClassifier`: standalone multiclass XGBoost |
| `meta_learner.py` | `StackedTopologyClassifier`, `FullDiagramReconstructor` (+ `from_dir` loader), `PredictedDiagram` |
| `train_all.py` | End-to-end training CLI |

Scripts (repository root):

| File | Contents |
|------|----------|
| `run_preprocessing.py` | CLI for the `mpds_preprocess` pipeline (package not included in this repo) |
| `predict.py` | Predict + evaluate against ground truth; writes a JSON report and CSV table |
| `compare_diagrams.py` | Predicted-vs-true figures: full grid, best/median/worst page, per-pair panels |
| `plot_diagrams.py` | Per-system diagrams, summary grid, accuracy summary chart |
| `compare_reports.py` | Side-by-side comparison of two `predict.py` reports (e.g. v6 vs v7) |
| `benchmark_models.py` | LazyPredict screen of ~30-40 model families per task |
| `check_endpoints.py` | Audit of the pure-element end points: features vs reference Tm, data vs features, models |
| `compare_curve_models.py` | Curve models (trees, CurveNet, ensemble) on one split or K-fold CV, with/without swap and end-point fix |
| `noise_floor.py` | Spread between sources (the floor under the curve error) and the model error by that spread |
| `analyze_eutectic.py` | Repeated-CV eutectic study: XGBoost vs forests, significance, learning curves, importance |
| `results/` | Saved prediction report from the latest evaluation run |

---

## Expected performance on real MPDS data

Based on comparable benchmarks (Derennes et al., Mansouri Tehrani et al.):

| Model | Metric | Expected |
|-------|--------|---------|
| M1 phases | micro-F1 | 0.75–0.85 |
| M2 liquidus | MAE | 50–80 K |
| M3 solidus | MAE | 60–100 K |
| M4 eutectic detection | F1 | 0.80–0.90 |
| M4 eutectic comp | MAE | 5–10 at% |
| M4 eutectic temp | MAE | 30–60 K |
| M6 solo | accuracy | 0.70–0.80 |
| M6 stacked | accuracy | 0.75–0.85 |

The stacked M6 should outperform M6 solo by 3–8 pp. If it doesn't, check the
OOF pipeline — the most common failure mode is data leakage from a too-small
training split collapsing the K-fold.

---

## Dependencies

```
xgboost >= 2.0     # for device="cuda" GPU support
torch >= 2.0
scikit-learn
pandas, numpy, pyarrow
joblib             # model serialisation for XGBoost-based models
matplotlib         # plotting scripts
lazypredict        # benchmark_models.py only
```
