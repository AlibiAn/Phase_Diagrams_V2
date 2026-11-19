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

--workers pairs are processed in parallel (default 2), each waiting --pause
seconds after every request. On HTTP 429 (too many requests) all workers pause
with a growing delay. It can be stopped and started again: cached searches and
files are skipped.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://api.mpds.io/v0"
# Z = 1..94 without the noble gases and the short-lived elements nobody has diagrams for
ELEMENTS = ("H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho "
            "Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po Th Pa U Np Pu").split()
HEADERS = {"User-Agent": "python-requests (binary phase diagram research download)"}
_LOCAL = threading.local()
_THROTTLE = {"until": 0.0}           # shared: set when the server answers 429
_LOCK = threading.Lock()


def _session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def get(session: requests.Session, url: str, params: dict, pause: float, tries: int = 6):
    """GET with a pause after every request and a growing delay on failure.
    Returns the response, or None after `tries` failures or on 4xx other than 429."""
    delay = 5.0
    for _ in range(tries):
        wait = _THROTTLE["until"] - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=90)
        except requests.RequestException:
            r = None
        time.sleep(pause)
        if r is not None and r.status_code == 200:
            return r
        if r is not None and r.status_code in (400, 403, 404):
            return None
        if r is not None and r.status_code == 429:
            with _LOCK:                                   # every worker pauses
                _THROTTLE["until"] = max(_THROTTLE["until"], time.time() + delay)
            print(f"  throttled (429); pausing {delay:.0f} s", flush=True)
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
    ap.add_argument("--pause", type=float, default=1.0, help="seconds each worker waits after a request")
    ap.add_argument("--workers", type=int, default=2,
                    help="element pairs processed in parallel (MPDS throttles at about 20 downloads a minute; more workers only cause more 429s)")
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

    stats = {"found": 0, "new": 0, "failed": 0, "truncated": 0, "done": 0}
    t0 = time.time()

    def do_pair(pair):
        x, y = pair
        session = _session()
        new = failed = truncated = 0
        index_file = out / "index" / f"{x}-{y}.json"
        hits = None
        if index_file.exists():
            try:
                hits = json.loads(index_file.read_text())
            except ValueError:                               # half-written by an interrupted run
                hits = None
        if hits is None:
            q = json.dumps({"elements": f"{x}-{y}", "classes": "binary", "props": "phase diagram"})
            r = get(session, f"{API}/search/facet", {"q": q}, a.pause)
            if r is None:
                return 0, 0, 1, 0
            data = r.json()
            hits = [h for h in (data.get("out") or []) if str(h[0]).startswith("C")]
            if (data.get("estimated_count") or 0) > len(data.get("out") or []):
                truncated = 1
                print(f"  {x}-{y}: {data.get('estimated_count')} entries, only {len(hits)} listed", flush=True)
            tmp_index = index_file.with_name(index_file.name + ".part")
            tmp_index.write_text(json.dumps(hits))
            tmp_index.rename(index_file)
        for h in hits:
            entry = h[0]
            target = out / "raw" / f"{x}-{y}_{entry}.json"
            if entry in have or target.exists():
                continue
            r = get(session, f"{API}/download/c", {"q": entry, "fmt": "json"}, a.pause)
            if r is None:
                failed += 1
                continue
            try:
                diagram = r.json()
            except ValueError:
                failed += 1
                continue
            diagram["_search"] = h                           # title, calc./exp., journal code, year
            tmp = target.with_name(target.name + ".part")
            tmp.write_text(json.dumps(diagram))
            tmp.rename(target)                               # never leave a half-written file
            new += 1
        return len(hits), new, failed, truncated

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for found, new, failed, truncated in pool.map(do_pair, pairs):
            stats["found"] += found
            stats["new"] += new
            stats["failed"] += failed
            stats["truncated"] += truncated
            stats["done"] += 1
            k = stats["done"]
            if k % 100 == 0 or k == len(pairs):
                print(f"{k}/{len(pairs)} pairs, {stats['found']} entries listed, {stats['new']} downloaded, "
                      f"{stats['failed']} failed, {(time.time() - t0) / 60:.0f} min", flush=True)
    n_found, n_new, n_failed, n_truncated = stats["found"], stats["new"], stats["failed"], stats["truncated"]
    print(f"Done: {n_found} entries listed in {len(pairs)} pairs, {n_new} new diagrams in {out / 'raw'}, "
          f"{n_failed} requests failed, {n_truncated} pairs had more entries than the search lists")
    if n_failed:
        print("Run the same command again to retry what failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
