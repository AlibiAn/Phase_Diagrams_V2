"""
Drop-in driver for the MPDS preprocessing pipeline.

Usage (as a script):
    python run_preprocessing.py /path/to/mpds_jsons /path/to/out

Usage (programmatic):
    from mpds_preprocess.pipeline import run_pipeline
    manifest = run_pipeline("/data/mpds", "/data/processed")

Outputs (in /path/to/out/)
    features.parquet        # ~207 columns: 132 magpie + ~11 thermo + ~56 phys + 8 meta
    targets_scalar.parquet  # M4, M5, M6 + provenance
    targets_liquidus.npz    # M2: (N, 101) consensus liquidus + per-point MAD + mask
    targets_solidus.npz     # M3: (N, 101) consensus solidus + per-point MAD
    targets_phases.npz      # M1: (N, V) multi-hot phase vector + vocab
    weights.parquet         # sample weights for rare-element upweighting
    splits.json             # pair-disjoint train/val/test indices
    manifest.json           # counts, drop reasons, long-tail summary

Recommended invocation for the full ~9,200-entry MPDS dataset:
    python run_preprocessing.py /data/mpds_jsons /data/processed \
        --val-frac 0.10 --test-frac 0.10 --phase-min-occurrences 5

Tips
----
* On 9,200 entries the bottleneck is matminer featurisation (~5-10 minutes
  on a single core). The featurisation loop is trivially parallelisable —
  see the comment in features.featurize_dataset().

* Featurisation is NOT wrapped in a try/except: a system that fails
  featurisation stops the run with an error (it is not silently dropped).
  This keeps every output array row-aligned with features.parquet; the
  pipeline also checks that alignment and stores `pair_ids` in each .npz.

* Every entry is converted to Kelvin with a consistent composition axis
  before consensus (mpds_preprocess/units.py). Entries whose unit can't be
  determined are dropped by default; pass --keep-unknown-units to keep them.
  See unit_report.csv and the "units" block of manifest.json.

* The default phase_min_occurrences=5 means a phase must appear in at least
  5 different systems to get its own vocabulary slot. Lower this if you
  want finer-grained M1 labels at the cost of training-set sparsity.
"""
import logging
import sys
from pathlib import Path

# Ensure the local package is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from mpds_preprocess.pipeline import _cli  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s : %(message)s",
        datefmt="%H:%M:%S",
    )
    _cli()
