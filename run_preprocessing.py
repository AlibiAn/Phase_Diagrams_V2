"""
Driver for the MPDS preprocessing pipeline (the `mpds_preprocess` package).

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
* The bottleneck is matminer featurisation (~5-10 minutes on one core for
  9,200 entries); see features.featurize_dataset() for parallelising it.

* A system that fails featurisation stops the run rather than being
  silently dropped, which keeps every output row-aligned with
  features.parquet (the pipeline also checks this and stores `pair_ids` in
  each .npz).

* Every entry is converted to Kelvin on a consistent composition axis
  before consensus (mpds_preprocess/units.py). Entries with an undetermined
  unit are dropped unless --keep-unknown-units is passed; see
  unit_report.csv and the "units" block of manifest.json.

* --phase-min-occurrences (default 5) is how many systems a phase must
  appear in to get its own vocabulary slot. Lower it for finer-grained M1
  labels at the cost of sparser training targets.
"""
import logging

from mpds_preprocess.pipeline import _cli

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s : %(message)s",
        datefmt="%H:%M:%S",
    )
    _cli()
