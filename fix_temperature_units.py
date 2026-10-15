"""
fix_temperature_units.py — convert liquidus/solidus targets that are still in °C to K.

Idea (per system):
  Near x=0 the liquidus should be close to Tm(A); near x=100, close to Tm(B).
  The Tm features are in Kelvin. So for each end we ask:
      is the curve close to Tm        -> that end "votes" Kelvin
      is the curve close to Tm - 273  -> that end "votes" Celsius
  If the available ends agree, we trust the vote. If they disagree, or the curve
  has no points near either end, the system is marked "?" (uncertain).

Systems voted Celsius get +273.15 K on liquidus, solidus and eutectic temperature.
The original folder is NOT modified; everything is written to a new folder.

Usage:
    python fix_temperature_units.py ~/data/processed ~/data/processed_fixed
    python fix_temperature_units.py ~/data/processed ~/data/processed_fixed --drop-uncertain

--drop-uncertain hides the curves of "?" systems from M2/M3 training (they stay
in the dataset for M1/M4/M5/M6). A unit_report.csv lists every system's vote.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

C2K = 273.15
EDGE = 8           # look at valid points within 8 grid steps of each end
MAX_RESID = 120.0  # K: how far from Tm the end of a liquidus may reasonably sit


def edge_value(T: np.ndarray, mask: np.ndarray, left: bool) -> float:
    """Median of the (up to) 3 valid points closest to one end.
    Using a median instead of the single end point ignores one bad edge point."""
    valid = np.flatnonzero(mask & np.isfinite(T))
    near = valid[valid <= EDGE] if left else valid[valid >= len(T) - 1 - EDGE]
    if near.size == 0:
        return np.nan
    near = near[:3] if left else near[-3:]
    return float(np.median(T[near]))


def vote(resid_k: float) -> str:
    """resid_k = curve end - Tm (Tm in Kelvin)."""
    if not np.isfinite(resid_k):
        return "?"
    resid_c = resid_k + C2K           # residual if the curve were in Celsius
    if abs(resid_c) < MAX_RESID and abs(resid_c) < abs(resid_k):
        return "C"
    if abs(resid_k) < MAX_RESID:
        return "K"
    return "?"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("processed_dir")
    p.add_argument("out_dir")
    p.add_argument("--drop-uncertain", action="store_true")
    a = p.parse_args()

    src, dst = Path(a.processed_dir), Path(a.out_dir)
    if src.resolve() == dst.resolve():
        raise SystemExit("out_dir must be a different folder from processed_dir")
    shutil.copytree(src, dst, dirs_exist_ok=True)

    feats = pd.read_parquet(src / "features.parquet")
    pairs = feats.index.tolist()
    TmA = feats["phys__melting_point__A"].to_numpy(float)
    TmB = feats["phys__melting_point__B"].to_numpy(float)

    liq = dict(np.load(src / "targets_liquidus.npz"))
    sol = dict(np.load(src / "targets_solidus.npz"))
    T, mask = liq["T"].astype(float), liq["mask"].astype(bool)
    assert T.shape[0] == len(pairs), "liquidus rows don't match features rows"

    # ── 1. vote per system ───────────────────────────────────────────────────
    report = []
    for i, pair in enumerate(pairs):
        r_left = edge_value(T[i], mask[i], left=True) - TmA[i]
        r_right = edge_value(T[i], mask[i], left=False) - TmB[i]
        votes = {v for v in (vote(r_left), vote(r_right)) if v != "?"}
        unit = votes.pop() if len(votes) == 1 else "?"
        report.append({"pair": pair, "unit": unit,
                       "resid_left_K": r_left, "resid_right_K": r_right})
    report = pd.DataFrame(report)
    is_c = (report["unit"] == "C").to_numpy()
    unknown = (report["unit"] == "?").to_numpy()

    # ── 2. convert Celsius systems to Kelvin ─────────────────────────────────
    liq["T"][is_c] += C2K
    sol["T"][is_c] += C2K
    scalar = pd.read_parquet(src / "targets_scalar.parquet")
    c_pairs = report.loc[is_c, "pair"]
    scalar.loc[scalar.index.isin(c_pairs), "m4_eutectic_temp"] += C2K

    # ── 3. optionally hide uncertain curves from curve training ─────────────
    if a.drop_uncertain:
        liq["mask"][unknown] = False
        sol["T"][unknown] = np.nan

    # ── 4. save ──────────────────────────────────────────────────────────────
    np.savez(dst / "targets_liquidus.npz", **liq)
    np.savez(dst / "targets_solidus.npz", **sol)
    scalar.to_parquet(dst / "targets_scalar.parquet")
    report.to_csv(dst / "unit_report.csv", index=False)

    n = len(report)
    print(f"Systems: {n}")
    print(f"  already Kelvin      : {(report['unit'] == 'K').sum()}")
    print(f"  converted from °C   : {is_c.sum()}")
    print(f"  uncertain ('?')     : {unknown.sum()}"
          + ("  -> curves hidden from M2/M3" if a.drop_uncertain else "  -> left unchanged"))
    print(f"Wrote fixed data to {dst}  (details in unit_report.csv)")


if __name__ == "__main__":
    main()
