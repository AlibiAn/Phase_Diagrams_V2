"""
convert_mpds_open.py — turn the keyless MPDS diagram files (download_mpds.py)
into the raw format that mpds_preprocess reads (the format of mpds_data).

The keyless JSON gives each region as an SVG path with the number of phases
and a solid / not-solid flag. What is rebuilt here:

  phases[].x / y     the path's points (composition in at% of the second
                     element, temperature in the chart's own unit)
  phases[].label     "Liquid" for a single-phase region that is not solid
                     ("G" if it lies entirely above another such region);
                     "(A)" / "(B)" for a single-phase solid touching a pure
                     element; "Unknown" for everything else
  liquidus_line      empty: mpds_preprocess then takes the liquidus from the
                     lower boundary of the liquid region

What cannot be rebuilt: compound names, "L1 + L2" labels, the eutectic list.
Systems converted from these files are therefore good for liquidus / solidus
curves only (build_extra_systems.py), not for diagram type or eutectic labels,
and their two-liquid domes cannot be cut down to the monotectic line.

Usage
-----
    python convert_mpds_open.py /home/alibi/lab/mpds_open/raw /home/alibi/lab/mpds_open/converted
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def path_points(svgpath: str) -> tuple[list[float], list[float]]:
    """All points of an SVG path made of M / L commands with absolute x,y pairs."""
    values = [float(v) for v in NUMBER.findall(svgpath or "")]
    return values[0::2], values[1::2][:len(values[0::2])]


def convert(raw: dict) -> dict | None:
    if raw.get("arity") != 2 or raw.get("naxes") != 2:
        return None
    a, b = raw.get("title_a"), raw.get("title_b")
    if not a or not b:
        return None
    phases = []
    for s in raw.get("shapes") or []:
        if s.get("kind") != "phase":
            continue
        x, y = path_points(s.get("svgpath", ""))
        if len(x) < 2 or len(x) != len(y):
            continue
        n = s.get("nphases")
        solid = bool(s.get("is_solid", True))
        label = "Unknown"
        if n == 1 and not solid:
            label = "Liquid"
        elif n == 1 and min(x) <= 2.0 and max(x) < 50.0:
            label = f"({a})"
        elif n == 1 and max(x) >= 98.0 and min(x) > 50.0:
            label = f"({b})"
        phases.append({"type": "phase", "label": label, "phases": n, "is_solid": solid, "x": x, "y": y})
    # A second non-solid single-phase region lying above the liquid is the gas
    liquids = [p for p in phases if p["label"] == "Liquid"]
    for p in liquids:
        for q in liquids:
            overlap = min(max(p["x"]), max(q["x"])) - max(min(p["x"]), min(q["x"]))
            if p is not q and overlap > 20.0 and min(p["y"]) >= max(q["y"]) - 5.0:
                p["label"] = "G"
    search = raw.get("_search") or []
    return {"entry": raw.get("entry"), "temp_range": raw.get("temp"), "comp_range": raw.get("comp_range"),
            "elements": [a, b], "phases": phases, "labels": {}, "liquidus_line": [], "eutectics": [],
            "reference": {"year": str(search[6]) if len(search) > 6 else None,
                          "title": search[2] if len(search) > 2 else None,
                          "source": "MPDS open API, " + str((raw.get("reference") or {}).get("release", ""))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    for f in sorted(Path(a.raw_dir).glob("*.json")):
        entry = convert(json.loads(f.read_text()))
        if entry is None or not entry["phases"] or not np.all(np.isfinite(entry["temp_range"] or [np.nan])):
            skipped += 1
            continue
        (out / f.name).write_text(json.dumps(entry))
        n += 1
    print(f"Converted {n} diagrams into {out} ({skipped} skipped: not a binary T-x diagram or no regions)")


if __name__ == "__main__":
    main()
