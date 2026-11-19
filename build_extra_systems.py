"""
build_extra_systems.py — extra systems for TRAINING the curve model only.

The 1,380 systems of the processed dataset keep their targets, rows, folds and
test split. This script builds a second, small dataset of systems that are
NOT among them, from two sources:

  mpds      systems the preprocessing rejected. Its filters ask for 80 at% of
            composition range and a `liquidus_line` of 8 points; here a system
            is kept if its liquidus (from `liquidus_line` or the outline of
            the liquid region) covers at least --min-span at% with at least 8
            points, in a diagram of 3 regions and 200 K. Only entries whose
            temperature unit the preprocessing could determine are used.
  calphad   liquidus curves computed from published CALPHAD assessments
            (Deffrennes et al., Materials & Design 2023,
            doi:10.1016/j.matdes.2023.112111; data doi:10.17632/rj6gnmnnmt.2,
            CC BY 4.0), folder "Raw liquidus data". Liquidus only, in K, every
            0.5 at%. On the 362 systems they share with our data they agree
            with the MPDS liquidus to a median of 22 K.

  orchid    liquidus AND solidus read from the ORCHID calculated binary phase
            diagrams (orchid_curves.py; doi:10.5281/zenodo.23735859, CC BY 4.0,
            calculated from the SGTE binary collection). They follow the
            processed_v8 convention, so they are used with --two-liquid only.
            On 848 shared systems they agree with processed_v8 to a median of
            24 K on the liquidus.

  okamoto   liquidus POINTS (no solidus) from the tie lines of ORCHID's
            assessed phase assemblages (okamoto_points.py; Okamoto's Desk
            Handbook, CC BY 4.0 derived data), for systems with at least
            --okamoto-min-points of them; the curve model fills between them
            and the two melting points. Median 6 K from processed_v8 on
            1,080 shared systems, but only about 3 points per system.

  mpds-open diagrams from the keyless MPDS download (download_mpds.py, then
            convert_mpds_open.py): geometry only, so the two-liquid dome of a
            new monotectic system cannot be cut down; lowest priority.

A system found in several sources is taken from the first of mpds, orchid,
calphad that has it.

The extra systems have curves but no diagram type, eutectic or phase labels,
so only the curve model can use them (base.add_extra_curve_systems).

With --two-liquid the curves follow the rule of processed_v8 (the liquidus
under a two-liquid dome is the monotectic line, never below a solid phase
under the dome); without it they are as the sources draw them, like processed_v6.

Usage
-----
    python build_extra_systems.py /home/alibi/lab/extra_v8 --reference /home/alibi/lab/processed_v8 \\
        --two-liquid --mpds /home/alibi/lab/mpds_data --preprocess-dir /home/alibi/lab \\
        --orchid /home/alibi/lab/external/orchid/repo/phase-diagrams \\
        --calphad /home/alibi/lab/external/deffrennes2023/raw_liquidus
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_two_liquid_dataset import flatten_dome, install

logger = logging.getLogger("extra")
GRID = np.arange(0.0, 100.5, 1.0)


def mpds_systems(json_dir: str, known: set[str], min_span: float, two_liquid: bool, pre) -> dict:
    """{pair id: (liquidus, mask, solidus, info)} for rejected MPDS systems."""
    loader, units, consensus, targets = pre
    if two_liquid and not getattr(consensus, "_two_liquid_installed", False):
        import build_two_liquid_dataset
        build_two_liquid_dataset.SOLID_FLOOR = True          # the processed_v8 rule
        install({"consensus": consensus, "targets": targets})
        consensus._two_liquid_installed = True
    entries, reports = [], []
    for entry in loader.load_directory(Path(json_dir)):
        els = entry.get("elements") or []
        if len(els) != 2 or "_".join(sorted(els)) in known:
            continue
        rep = loader.quality_filter(entry, min_comp_span_pct=0.0, require_liquidus=False)
        if not rep.passed:
            continue
        pts = units.liquidus_points(entry)
        if pts.shape[0] < 8 or np.ptp(pts[:, 0]) < min_span:
            continue
        if units.normalise_entry(entry).unit == "?":
            continue
        entries.append(entry)
        reports.append(rep)
    out = {}
    for key, sources in consensus.group_by_element_pair(entries, reports=reports).items():
        c = consensus.build_consensus(key, sources)
        seen = c.liquidus_mask & np.isfinite(c.liquidus_T)
        if seen.sum() < min_span * 0.6:
            continue
        sol, _ = targets.m3_solidus_vector(c)
        out[f"{key[0]}_{key[1]}"] = (c.liquidus_T, seen, sol,
                                     dict(source="mpds", n_sources=c.n_sources, gap=bool(c.miscibility_gap)))
    return out


def calphad_systems(folder: str, known: set[str], two_liquid: bool) -> dict:
    out = {}
    for path in sorted(Path(folder).glob("*.csv")):
        m = re.match(r"^([A-Z]+)-([A-Z]+)(_L\+L)?\.csv$", path.name)
        if not m:
            continue
        a, b = sorted((m.group(1).capitalize(), m.group(2).capitalize()))
        pid = f"{a}_{b}"
        if pid in known or pid in out:
            continue
        d = pd.read_csv(path)
        x = d[f"x({b.upper()})"].to_numpy(dtype=float)
        order = np.argsort(x)
        x, t = x[order], d["T"].to_numpy(dtype=float)[order]
        dome = (d["phase_name"].to_numpy()[order] == "LIQUID+LIQUID")
        liq = np.full(GRID.shape, np.nan)
        inside = (GRID >= x[0]) & (GRID <= x[-1])
        liq[inside] = np.interp(GRID[inside], x, t)
        if two_liquid and dome.any():
            # the label of a two-liquid region: mid-composition, just under the top of the dome
            xs, ts = x[dome], t[dome]
            liq, _ = flatten_dome(liq, GRID, float(xs[np.argmax(ts)]), float(ts.max()) - 1.0)
        out[pid] = (liq, np.isfinite(liq), np.full(GRID.shape, np.nan),
                    dict(source="calphad", n_sources=1, gap=bool(dome.any())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--reference", required=True, help="processed dataset whose systems are excluded")
    ap.add_argument("--preprocess-dir", required=True, help="folder that contains mpds_preprocess")
    ap.add_argument("--mpds", default=None, help="raw MPDS diagram files")
    ap.add_argument("--orchid", default=None, help="ORCHID phase-diagrams folder (index.json and sgte/)")
    ap.add_argument("--mpds-open", default=None, help="converted keyless MPDS diagrams (convert_mpds_open.py)")
    ap.add_argument("--okamoto", default=None, help="ORCHID assessed-phase-assemblages records.parquet")
    ap.add_argument("--okamoto-min-points", type=int, default=3)
    ap.add_argument("--calphad", default=None, help="folder 'Raw liquidus data' of Deffrennes et al. 2023")
    ap.add_argument("--min-span", type=float, default=30.0, help="MPDS: liquidus coverage needed, at%%")
    ap.add_argument("--two-liquid", action="store_true", help="curves follow the processed_v7 rule")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out, ref = Path(a.out_dir), Path(a.reference)
    if out.resolve() == ref.resolve():
        raise SystemExit("out_dir must be a new folder")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.preprocess_dir).resolve()))
    from mpds_preprocess import consensus, features, loader, rare_elements, targets, units

    ref_features = pd.read_parquet(ref / "features.parquet")
    known = set(ref_features.index)
    systems = {}
    if a.mpds:
        systems.update(mpds_systems(a.mpds, known, a.min_span, a.two_liquid, (loader, units, consensus, targets)))
    if a.orchid:
        if not a.two_liquid:
            raise SystemExit("--orchid curves follow the processed_v8 convention; use them with --two-liquid")
        from orchid_curves import system_curves
        for pid, c in system_curves(a.orchid).items():
            if pid not in known and pid not in systems:
                systems[pid] = (c["liquidus"], np.isfinite(c["liquidus"]), c["solidus"],
                                dict(source="orchid", n_sources=c["n_assessments"], gap=False))
    if a.okamoto:
        from okamoto_points import liquidus_points, on_grid
        for pid, pts in liquidus_points(a.okamoto).items():
            if pid in known or pid in systems or pts["n_points"] < a.okamoto_min_points:
                continue
            liq = on_grid(pts, GRID)
            systems[pid] = (liq, np.isfinite(liq), np.full(GRID.shape, np.nan),
                            dict(source="okamoto", n_sources=1, gap=pts["gap"]))
    if a.mpds_open:
        for pid, v in mpds_systems(a.mpds_open, known | set(systems), a.min_span, a.two_liquid,
                                   (loader, units, consensus, targets)).items():
            v[3]["source"] = "mpds-open"
            systems.setdefault(pid, v)
    if a.calphad:
        for pid, v in calphad_systems(a.calphad, known, a.two_liquid).items():
            systems.setdefault(pid, v)                     # MPDS first: it has a solidus too
    from pymatgen.core import Element
    def known_element(sym: str) -> bool:                  # element_features covers Z <= 94
        try:
            return Element(sym).Z <= 94
        except ValueError:
            return False
    systems = {p: v for p, v in systems.items() if all(known_element(e) for e in p.split("_"))}
    pair_ids = sorted(systems)
    pairs = [tuple(p.split("_")) for p in pair_ids]

    feats = features.featurize_dataset(pairs, show_progress=False)
    all_pairs = [tuple(p.split("_")) for p in ref_features.index] + pairs
    meta = rare_elements.rarity_meta_features(all_pairs).iloc[len(ref_features):]
    meta.index = feats.index
    full = pd.concat([feats, meta], axis=1)[list(ref_features.columns)]
    full.to_parquet(out / "features.parquet")
    weights = rare_elements.compute_sample_weights(all_pairs, method="sqrt_inverse")[len(ref_features):]
    pd.DataFrame({"weight": weights}, index=full.index).to_parquet(out / "weights.parquet")

    ids = np.asarray(pair_ids, dtype=str)
    liq = np.vstack([systems[p][0] for p in pair_ids]).astype(np.float32)
    np.savez_compressed(out / "targets_liquidus.npz", T=liq,
                        mask=np.vstack([systems[p][1] for p in pair_ids]).astype(bool),
                        comp_grid=GRID.astype(np.float32), pair_ids=ids)
    np.savez_compressed(out / "targets_solidus.npz",
                        T=np.vstack([systems[p][2] for p in pair_ids]).astype(np.float32),
                        comp_grid=GRID.astype(np.float32), pair_ids=ids)
    info = pd.DataFrame([systems[p][3] for p in pair_ids], index=pd.Index(pair_ids, name="pair"))
    info["liquidus_points"] = [int(systems[p][1].sum()) for p in pair_ids]
    info["solidus_points"] = [int(np.isfinite(systems[p][2]).sum()) for p in pair_ids]
    ref_elements = {e for p in known for e in p.split("_")}
    info["new_element"] = [" ".join(e for e in p.split("_") if e not in ref_elements) for p in pair_ids]
    info.to_csv(out / "systems.csv")
    manifest = {"reference": str(ref), "two_liquid_rule": a.two_liquid, "min_span_at_pct": a.min_span,
                "n_systems": len(pair_ids), "by_source": info.source.value_counts().to_dict(),
                "with_miscibility_gap": int(info.gap.sum()),
                "with_an_element_outside_the_reference": int((info.new_element != "").sum()),
                "full_range_liquidus (>= 95 points)": int((info.liquidus_points >= 95).sum())}
    with (out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
