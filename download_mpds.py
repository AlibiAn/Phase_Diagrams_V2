"""
download_mpds.py — download every binary phase diagram of the MPDS open API
that we do not have yet.

MPDS serves the search index and the geometry of its phase diagrams without
an API key (https://api.mpds.io/v0; the open-access part of the PAULING FILE
data is CC BY 4.0, cite Villars & Cenzual / MPDS). The keyless diagram JSON
holds the outline of every region, the number of phases in it and whether it
is solid, but NOT the region names, the "L1 + L2" labels, the eutectic list
or the reference that the files in mpds_data have. It is enough to extract a
liquidus and a solidus (convert_mpds_open.py), not the diagram type.

What it does
------------
1. For every pair of the chosen elements, asks the search endpoint for the
   binary phase-diagram entries (at most 150 per query, which is why it goes
   pair by pair). Answers are cached in <out>/index/A-B.json.
2. Downloads each entry that is not already in --have (matched by entry id in
   the file names) to <out>/raw/A-B_C123456.json, exactly as served.

It waits --pause seconds between requests, retries with a growing delay, and
can be stopped and started again: cached searches and files are skipped.

Usage
-----
    python download_mpds.py /home/alibi/lab/mpds_open --have /home/alibi/lab/mpds_data
    python download_mpds.py /home/alibi/lab/mpds_open --elements B Lu Tm Yb --with-elements-of /home/alibi/lab/mpds_data
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://api.mpds.io/v0"
# Z = 1..94 without the noble gases and the short-lived elements nobody has diagrams for
ELEMENTS = ("H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho "
            "Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po Th Pa U Np Pu").split()
HEADERS = {"User-Agent": "python-requests (binary phase diagram research download)"}


def get(session: requests.Session, url: str, params: dict, pause: float, tries: int = 6):
    """GET with a pause after every request and a growing delay on failure.
    Returns the response, or None after `tries` failures or on 4xx other than 429."""
    delay = 5.0
    for _ in range(tries):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=90)
        except requests.RequestException:
            r = None
        time.sleep(pause)
        if r is not None and r.status_code == 200:
            return r
        if r is not None and r.status_code in (400, 403, 404):
            return None
        time.sleep(delay)
        delay = min(delay * 2, 300.0)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", help="new folder; gets index/ and raw/")
    ap.add_argument("--have", default=None, help="folder of diagrams we already have (names like Ag-Al_C101001.json)")
    ap.add_argument("--elements", nargs="+", default=None, help="only pairs containing one of these elements")
    ap.add_argument("--with-elements-of", default=None,
                    help="with --elements: pair them with the elements found in this folder's file names")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--limit-pairs", type=int, default=0, help="stop after this many pairs (for a test run)")
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    have = set()
    if a.have:
        have = {m.group(1) for f in Path(a.have).iterdir() if (m := re.search(r"_(C\d+)\.json$", f.name))}
    partners = list(ELEMENTS)
    if a.with_elements_of:
        partners = sorted({e for f in Path(a.with_elements_of).iterdir()
                           if (m := re.match(r"^([A-Z][a-z]?)-([A-Z][a-z]?)_C", f.name)) for e in m.groups()})
    if a.elements:
        pairs = sorted({tuple(sorted((x, y))) for x in a.elements for y in set(partners) | set(a.elements) if x != y})
    else:
        pairs = list(itertools.combinations(sorted(ELEMENTS), 2))
    if a.limit_pairs:
        pairs = pairs[:a.limit_pairs]

    session = requests.Session()
    n_found = n_new = n_failed = n_truncated = 0
    t0 = time.time()
    for k, (x, y) in enumerate(pairs):
        index_file = out / "index" / f"{x}-{y}.json"
        if index_file.exists():
            hits = json.loads(index_file.read_text())
        else:
            q = json.dumps({"elements": f"{x}-{y}", "classes": "binary", "props": "phase diagram"})
            r = get(session, f"{API}/search/facet", {"q": q}, a.pause)
            if r is None:
                n_failed += 1
                continue
            data = r.json()
            hits = [h for h in (data.get("out") or []) if str(h[0]).startswith("C")]
            if (data.get("estimated_count") or 0) > len(data.get("out") or []):
                n_truncated += 1
                print(f"  {x}-{y}: {data.get('estimated_count')} entries, only {len(hits)} listed", flush=True)
            index_file.write_text(json.dumps(hits))
        n_found += len(hits)
        for h in hits:
            entry = h[0]
            target = out / "raw" / f"{x}-{y}_{entry}.json"
            if entry in have or target.exists():
                continue
            r = get(session, f"{API}/download/c", {"q": entry, "fmt": "json"}, a.pause)
            if r is None:
                n_failed += 1
                continue
            try:
                diagram = r.json()
            except ValueError:
                n_failed += 1
                continue
            diagram["_search"] = h                           # title, calc./exp., journal code, year
            target.write_text(json.dumps(diagram))
            n_new += 1
        if (k + 1) % 100 == 0 or k + 1 == len(pairs):
            print(f"{k + 1}/{len(pairs)} pairs, {n_found} entries listed, {n_new} downloaded, "
                  f"{n_failed} failed, {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"Done: {n_found} entries listed in {len(pairs)} pairs, {n_new} new diagrams in {out / 'raw'}, "
          f"{n_failed} requests failed, {n_truncated} pairs had more entries than the search lists")
    if n_failed:
        print("Run the same command again to retry what failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
