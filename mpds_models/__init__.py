"""
Binary phase diagrams predicted from the properties of the two elements.

  data.py         the dataset: loading, extra systems, A/B swap, data fixes, data groups
  features.py     features computed from the element symbols (element data, compound energies)
  curves.py       liquidus + solidus from ExtraTrees
  foundation.py   TabPFN + TabICL corrections of the liquidus (height, whole curve)
  classifiers.py  phases, eutectic, miscibility gap, diagram type (XGBoost)
  pipeline.py     PhaseDiagramModel: all of them trained, saved, loaded together
  evaluation.py   scores and the report format
  train.py        python -m mpds_models.train PROCESSED_DIR MODEL_DIR

    from mpds_models.data import load_dataset
    from mpds_models.pipeline import PhaseDiagramModel
    model = PhaseDiagramModel.load("/home/alibi/lab/models_v18")
    ds = load_dataset("/home/alibi/lab/processed_v8", model.features)
    pred = model.predict(ds.X[ds.test_idx])
"""
