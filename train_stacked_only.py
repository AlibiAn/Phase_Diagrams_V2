import logging
import joblib
from mpds_models.base import load_dataset
from mpds_models.deep_curve_net import LiquidusSolidusModel
from mpds_models.meta_learner import StackedTopologyClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

ds = load_dataset("processed_v2")
m23 = LiquidusSolidusModel.load("models_v2/m23_curves.pt")
specialists = {
    "m1": joblib.load("models_v2/m1_phase.joblib"),
    "m23": m23,
    "m4": joblib.load("models_v2/m4_eutectic.joblib"),
    "m5": joblib.load("models_v2/m5_miscibility.joblib"),
}

# Same curve settings as the saved curve model
meta = StackedTopologyClassifier(curve_config=m23.cfg).fit(ds)
meta.save("models_v2/m6_topology_stacked.joblib")
print("val :", meta.evaluate(ds, specialists, "val"))
print("test:", meta.evaluate(ds, specialists, "test"))