"""
PhaseDiagramModel: every model of one version, trained, saved and loaded together.

    model = PhaseDiagramModel(height=0.5, blend=0.5, n_components=10).fit(ds, extra_dir)
    model.save("/home/alibi/lab/models_v18")
    model = PhaseDiagramModel.load("/home/alibi/lab/models_v18")
    pred = model.predict(ds.X[rows])        # dict of arrays, see predict()

Versions (all on processed_v8, features "extended+mp", extra systems
/home/alibi/lab/extra_v8_all for the curve model):
    v16   height=0                  ExtraTrees curves
    v17   height=0.5, blend=0       + height correction by TabPFN + TabICL
    v18   height=0.5, blend=0.5     + their 10-component whole curve (default)

The scalar models train on the labelled systems only (train + validation
split; nothing here uses the validation split for model selection, so it
would only waste a ninth of the data). The stacked diagram-type classifier
reads the trees' curves, not the corrected ones, as it was trained on them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classifiers import (EutecticDetector, MiscibilityGapDetector, PhaseClassifier, StackedTopologyClassifier,
                          TopologyClassifier)
from .curves import TreeCurves
from .data import TOPOLOGY_CLASSES, PhaseDataset, add_extra_systems

logger = logging.getLogger(__name__)
PARTS = ("phases", "eutectic", "gap", "topology_solo", "topology")


class PhaseDiagramModel:
    def __init__(self, height: float = 0.5, blend: float = 0.5, n_components: int = 10, seed: int = 0):
        self.height, self.blend, self.n_components, self.seed = height, blend, n_components, seed
        self.config: dict = {}

    def fit(self, ds: PhaseDataset, extra_dir: str | None = None) -> "PhaseDiagramModel":
        """`ds`: the dataset with clipped tops masked and the training split set;
        `extra_dir`: extra systems for the curve model (or None)."""
        import torch
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        ds_curves = add_extra_systems(ds, extra_dir) if extra_dir else ds
        self.phase_vocab, self.comp_grid = list(ds.phase_vocab), np.asarray(ds.comp_grid)
        self.config = dict(features=ds.features, height=self.height, blend=self.blend, n_components=self.n_components,
                           seed=self.seed, extra_data=extra_dir, n_train=int(len(ds.train_idx)),
                           n_extra=int(len(ds_curves.extra_idx)))
        logger.info("=== phases ===")
        self.phases = PhaseClassifier().fit(ds)
        logger.info("=== curves ===")
        self.trees = TreeCurves(seed=self.seed).fit(ds_curves)
        self.curves = self.trees
        if self.height:
            from .foundation import FoundationCurves
            self.curves = FoundationCurves(self.trees, self.height, self.blend, self.n_components,
                                           seed=self.seed).fit(ds_curves)
            self.curves._ensure_fitted()
        logger.info("=== eutectic, miscibility gap, diagram type ===")
        self.eutectic = EutecticDetector(self.seed).fit(ds)
        self.gap = MiscibilityGapDetector().fit(ds)
        self.topology_solo = TopologyClassifier().fit(ds)
        self.topology = StackedTopologyClassifier().fit(ds)
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """liquidus, solidus (N, 101) K; topology (index), topology_proba (N, 5);
        phases_proba (N, V); eutectic_exists / _proba / _comp (at% B) / _temp (K);
        gap, gap_proba."""
        liq, sol = self.curves.predict(X)
        eut = self.eutectic.predict(X)
        gap_proba = self.gap.predict_proba(X)
        topo = self.topology.predict_proba(X, self.phases, self.trees, self.eutectic, self.gap)
        return dict(liquidus=liq, solidus=sol, topology=topo.argmax(axis=1), topology_proba=topo,
                    phases_proba=self.phases.predict_proba(X), eutectic_exists=eut["exists"],
                    eutectic_proba=eut["exists_proba"], eutectic_comp=eut["comp"], eutectic_temp=eut["temp"],
                    gap=gap_proba >= self.gap.threshold, gap_proba=gap_proba)

    def diagrams(self, pred: dict, pair_ids: list[str]) -> list["PredictedDiagram"]:
        return [PredictedDiagram.from_prediction(pred, i, p, self.phase_vocab, self.comp_grid)
                for i, p in enumerate(pair_ids)]

    # ---- saving ------------------------------------------------------------
    def save(self, out_dir) -> None:
        import joblib
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in PARTS:
            joblib.dump(getattr(self, name), out / f"{name}.joblib")
        if self.curves is self.trees:
            joblib.dump(self.trees, out / "curves.joblib", compress=3)
        else:
            self.curves.save(out / "curves.joblib")          # holds the trees, plus curves_foundation/
        meta = dict(self.config, phase_vocab=self.phase_vocab, comp_grid=self.comp_grid.tolist())
        (out / "config.json").write_text(json.dumps(meta, indent=1))
        logger.info("saved to %s", out)

    @classmethod
    def load(cls, model_dir) -> "PhaseDiagramModel":
        import joblib
        p = Path(model_dir)
        meta = json.loads((p / "config.json").read_text())
        self = cls(meta["height"], meta["blend"], meta["n_components"], meta["seed"])
        self.phase_vocab, self.comp_grid = meta.pop("phase_vocab"), np.asarray(meta.pop("comp_grid"), dtype=np.float32)
        self.config = meta
        for name in PARTS:
            setattr(self, name, joblib.load(p / f"{name}.joblib"))
        self.curves = joblib.load(p / "curves.joblib")
        self.trees = getattr(self.curves, "trees", self.curves)
        return self

    @property
    def features(self) -> str:
        return self.config["features"]


@dataclass
class PredictedDiagram:
    """One predicted system, for tables and plots."""
    pair: str
    topology: str
    topology_proba: dict
    liquidus_T: np.ndarray
    solidus_T: np.ndarray
    comp_grid: np.ndarray
    phases: list            # (phase, probability), the 5 most probable above 10 %
    eutectic_exists: bool
    eutectic_proba: float
    eutectic_comp: float | None
    eutectic_temp: float | None
    miscibility_gap: bool
    miscibility_proba: float
    inconsistencies: list = field(default_factory=list)

    @classmethod
    def from_prediction(cls, pred: dict, i: int, pair: str, phase_vocab, comp_grid) -> "PredictedDiagram":
        num = lambda v: float(v) if np.isfinite(v) else None
        pp = pred["phases_proba"][i]
        d = cls(pair=pair, topology=TOPOLOGY_CLASSES[pred["topology"][i]],
                topology_proba={c: float(p) for c, p in zip(TOPOLOGY_CLASSES, pred["topology_proba"][i])},
                liquidus_T=pred["liquidus"][i], solidus_T=pred["solidus"][i], comp_grid=comp_grid,
                phases=[(phase_vocab[j], float(pp[j])) for j in np.argsort(-pp)[:5] if pp[j] > 0.1],
                eutectic_exists=bool(pred["eutectic_exists"][i]), eutectic_proba=float(pred["eutectic_proba"][i]),
                eutectic_comp=num(pred["eutectic_comp"][i]), eutectic_temp=num(pred["eutectic_temp"][i]),
                miscibility_gap=bool(pred["gap"][i]), miscibility_proba=float(pred["gap_proba"][i]))
        # Contradictions between the models are reported, not corrected
        if d.topology == "isomorphous" and d.eutectic_exists:
            d.inconsistencies.append("type 'isomorphous' but a eutectic is predicted")
        if d.topology == "monotectic" and not d.miscibility_gap:
            d.inconsistencies.append("type 'monotectic' but no miscibility gap")
        if d.miscibility_gap and d.topology not in ("monotectic", "complex"):
            d.inconsistencies.append(f"miscibility gap but type '{d.topology}'")
        return d
