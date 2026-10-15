# MPDS Specialist Models

Six specialist ML models + a stacked meta-learner that predict binary phase
diagrams from the feature vectors produced by the `mpds_preprocess` pipeline.

---

## Quick start

```bash
# Train all models end-to-end
python -m mpds_models.train_all /data/processed /data/models --curve-epochs 200

# Verbose + skip meta-learner (faster for iteration)
python -m mpds_models.train_all /data/processed /data/models --curve-epochs 200 --no-stacked -v
```

```python
from mpds_models.train_all import train_all
metrics = train_all("/data/processed/", "/data/models/", curve_epochs=200)
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

**SHAP:** use `shap_analysis.explain(m1.models[j], X, feature_names)` to
explain any individual phase classifier.

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
comparable in scale to the Huber data loss. The `lever_rule_penalty` in
`physics_loss.py` is exposed for future models that predict phase fractions.

**Uncertainty:** MC-Dropout at inference (`predict_with_uncertainty(X, n_samples=30)`).

**Why MLP, not CGCNN/MEGNet:** the preprocessing pipeline outputs system-level
compositional + thermodynamic features — not per-atom crystal graphs. A GNN
operating on this input is just an MLP with extra machinery. When MPDS S-entry
structures are added to the pipeline, replace the `CurveNet` trunk with a real
`CGCNN/M3GNet` that emits per-system embeddings concatenated to the matminer
features — the "hybrid GNN" design described in the project paper.

---

### M4 — Eutectic Detector (`m4_eutectic.py`)
Three XGBoost sub-models trained jointly on the same feature vector:

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

## Active learning / uncertainty sampling (`uncertainty.py`)

### System-level: which binary to target next?
```python
from mpds_models.uncertainty import rank_systems_for_acquisition

scores = rank_systems_for_acquisition(
    reconstructor, X_candidates, candidate_pair_ids,
    top_k=25, n_mc_samples=30,
)
# scores is a list of SystemAcquisitionScore, ranked most-uncertain first
# Fields: pair, total_score, topology_entropy, eutectic_ambiguity,
#         miscibility_ambiguity, mean_liquidus_std, mean_solidus_std
```

Composite score: `H(topology) + 0.5·|p(eut)−0.5| + 0.5·|p(misc)−0.5| + ...`
Higher = more uncertain = better candidate for the next CALPHAD assessment or
experimental run.

### Intra-system: where on this diagram to measure next?
```python
from mpds_models.uncertainty import rank_grid_points_within_system

pts = rank_grid_points_within_system(diagram, top_k=20)
# pts = list of (x_composition, T_Kelvin, score)
# Gaussian kernel centred on liquidus/solidus boundaries
```

Implements the Terayama et al. (2019) uncertainty-sampling strategy: target
(x, T) points nearest the predicted phase boundaries, scaled by local
MC-Dropout std.

---

## SHAP analysis (`shap_analysis.py`)

```python
from mpds_models.shap_analysis import explain, top_features_for_prediction

# Global feature importance for M6 topology
ex = explain(m6_solo, X_train, feature_names=feature_names)
# ex['global_importance']['class_2']  →  top-20 features for eutectic_with_intermetallics

# Per-prediction explanation
df = top_features_for_prediction(m6_solo, X_row, feature_names, k=10, class_index=2)
# Returns DataFrame with columns: feature, value, shap, abs_shap
```

Works with M5, M6 solo, M6 stacked meta, and M4a/b/c. Pass `class_index` for
any multiclass model. Handles both the older list-of-arrays SHAP API and the
newer `(N, F, C)` 3D array.

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

| File | Contents |
|------|----------|
| `base.py` | `PhaseDataset`, `load_dataset()`, GPU detection, `SpecialistModel` base |
| `physics_loss.py` | Differentiable physics penalties (order, endpoints, smoothness, bounds, lever rule) |
| `deep_curve_net.py` | `CurveNet` MLP, `LiquidusSolidusModel` (M2+M3), MC-Dropout uncertainty |
| `m1_phase.py` | `PhaseClassifier` — one XGBoost per phase |
| `m4_eutectic.py` | `EutecticDetector` — M4a/b/c classifer+regressors |
| `m5_miscibility.py` | `MiscibilityGapDetector` — binary XGBoost |
| `m6_topology.py` | `TopologyClassifier` — standalone multiclass XGBoost |
| `meta_learner.py` | `StackedTopologyClassifier`, `FullDiagramReconstructor`, `PredictedDiagram` |
| `uncertainty.py` | `rank_systems_for_acquisition`, `rank_grid_points_within_system` |
| `shap_analysis.py` | `explain()`, `top_features_for_prediction()` |
| `train_all.py` | End-to-end CLI driver |

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
shap >= 0.45
scikit-learn
pandas, numpy, pyarrow
joblib             # model serialisation for XGBoost-based models
```
