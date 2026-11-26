"""
Features computed from the two element symbols (A, B), added to the 207
processed features by data.load_dataset. Every feature is a function of the
ORDERED pair, so the swapped system's features are those of (B, A).

1. Element features ("extended", v14, 147 columns; element_features)
   The processed features describe each element by a dozen pymatgen
   properties and the 50/50 mixture by Magpie statistics and Miedema enthalpies.
   These add what a thermodynamic estimate of the liquidus needs:

     el__*     per-element properties (__A, __B): heat of fusion, entropy of
               fusion, heat of vaporisation, bulk and shear modulus, cohesive
               energy, Mendeleev number, Miedema's parameters (work function phi,
               electron density n_ws^1/3, molar volume), crystal structure of the
               ground state, valence counts; plus absolute differences of the
               ones that drive mixing.
     mied__*   Miedema enthalpies at 10, 25, 50, 75 and 90 at% B (the processed
               features have 50 at% only), their asymmetry, and Miedema's
               tabulated enthalpy of mixing of the equiatomic liquid.
     phys__*   the liquidus of a regular solution with no solid solubility
               (Schroder-van Laar equation), for an ideal liquid and for one with
               the interaction parameter omega = 4 x the tabulated liquid
               enthalpy: its distance from the melting-point line at seven
               compositions, its eutectic point, and the critical temperature of
               the liquid miscibility gap (omega / 2R) relative to the higher
               melting point.

   Sources and licences
   --------------------
   All tables ship with matminer (BSD-style licence, Lawrence Berkeley National
   Laboratory; Ward et al., Comput. Mater. Sci. 152 (2018) 60):
     - magpie_elementdata/*.table: Magpie element properties (Ward et al., npj
       Comput. Mater. 2 (2016) 16028; most values from Wolfram's ElementData).
     - Miedema.csv: Miedema model parameters (de Boer et al., "Cohesion in
       Metals", North-Holland 1988; Zhang et al., Comput. Phys. Commun. 209
       (2016) 58). No parameters for S, Se, Te: those features are missing (0).
     - MiedemaLiquidDeltaHf.tsv: enthalpy of mixing of equiatomic liquid alloys
       (kJ/mol), Miedema's tables as distributed with matminer.
     - cohesive_energies.json: Kittel, "Introduction to Solid State Physics".
   Nothing here is fitted to the phase diagrams.

2. Compound features ("extended+mp", v15, 15 columns; compound_features)
   How stable the compounds of A and B are, from computed (DFT, zero kelvin)
   formation energies. Used from v15 on; read the caveat below.

   Why: the largest part of the liquidus error is each system's overall height
   above or below the melting-point line, which is set by how strongly the two
   elements bind (a very stable compound melts far above both elements). The
   Miedema enthalpies in the other features estimate this for metals only and
   have no parameters for S, Se and Te.

   Source: Materials Project snapshot of 18 Oct 2018 as distributed by matminer
   (dataset "mp_nostruct_20181018", 83,989 entries with formula, formation energy
   and energy above the convex hull; Jain et al., APL Materials 1 (2013) 011002;
   Materials Project data are CC BY 4.0). Download once with

       from matminer.datasets import load_dataset
       load_dataset("mp_nostruct_20181018", data_home="/home/alibi/lab/external")

   CAVEAT, read before using these in a headline number: most Materials Project
   entries are calculations of compounds that are known experimentally. Whether a
   system has entries, and how many are stable, therefore tells the model which
   compounds have been observed, which is knowledge of the phase diagram itself.
   For a system nobody has studied these features would be nearly empty. The
   honest version of this idea computes the same quantities from a database that
   enumerates hypothetical structures for every pair (OQMD prototypes, AFLOW).

   Features of the ordered pair (A, B), x = atomic fraction of B:
     cmp__n_entries / n_stable / n_near   binary entries; on the hull; within 30 meV of it
     cmp__min_eform                       lowest formation energy (eV/atom; may be > 0)
     cmp__x_at_min                        its composition
     cmp__hull_XX                         depth of the convex hull at XX at% B (<= 0)
     cmp__hull_mean                       mean depth over composition
     cmp__x_first_stable / x_last_stable  stable compounds closest to A and to B
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

R_KJ = 8.314462618e-3          # gas constant, kJ/(mol K)

MAGPIE_PROPERTIES = (
    "MendeleevNumber", "HeatFusion", "HeatVaporization", "BulkModulus", "ShearModulus",
    "FirstIonizationEnergy", "ElectronAffinity", "Polarizability", "HeatCapacityMolar",
    "MiracleRadius", "GSenergy_pa", "GSvolume_pa", "SpaceGroupNumber", "NdValence", "NfValence",
    "NValence", "NUnfilled", "IsMetal", "IsMetalloid", "IsNonmetal", "AllenElectronegativity",
    "phi", "n_ws^third", "MeltingT", "BoilingT", "CovalentRadius", "Electronegativity", "Row",
    "Column", "Number", "AtomicVolume")
MIEDEMA_PARAMETERS = ("molar_volume", "electron_density", "electronegativity", "valence_electrons",
                      "R_const", "H_trans", "structural_stability", "compressibility")
DIFFERENCES = ("MendeleevNumber", "phi", "n_ws^third", "mied_molar_volume", "MiracleRadius",
               "AllenElectronegativity", "cohesive", "BulkModulus", "NValence", "S_fus", "HeatFusion",
               "GSvolume_pa")
MIEDEMA_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 0.9)        # fraction of B
CURVE_POINTS = (10, 25, 40, 50, 60, 75, 90)            # at% B


def _data_dir() -> str:
    import matminer
    return os.path.join(os.path.dirname(matminer.__file__), "utils", "data_files")


@lru_cache(maxsize=None)
def element_table() -> pd.DataFrame:
    """One row per element (Z <= 94), one column per property; NaN = unknown."""
    from pymatgen.core import Element
    d = _data_dir()
    elements = [e for e in Element if e.Z <= 94]
    cols = {}
    for prop in MAGPIE_PROPERTIES:
        with open(os.path.join(d, "magpie_elementdata", prop + ".table")) as f:
            lines = f.read().split("\n")
        col = {}
        for e in elements:
            try:
                col[e.symbol] = float(lines[e.Z - 1])
            except (ValueError, IndexError):
                col[e.symbol] = np.nan
        cols[prop] = col
    t = pd.DataFrame(cols)
    mied = pd.read_csv(os.path.join(d, "Miedema.csv"), index_col=0).set_index("element")
    for p in MIEDEMA_PARAMETERS:
        t["mied_" + p] = mied[p].reindex(t.index)
    with open(os.path.join(d, "cohesive_energies.json")) as f:
        t["cohesive"] = pd.Series(json.load(f)).reindex(t.index)
    t["S_fus"] = 1000.0 * t["HeatFusion"] / t["MeltingT"]            # J/(mol K)
    t["Tb_over_Tm"] = t["BoilingT"] / t["MeltingT"]
    for name, sg in (("is_bcc", 229), ("is_fcc", 225), ("is_hcp", 194)):
        t[name] = (t["SpaceGroupNumber"] == sg).astype(float)
    return t


@lru_cache(maxsize=None)
def liquid_enthalpy_table() -> dict:
    """{frozenset({A, B}): enthalpy of mixing of the equiatomic liquid, kJ/mol}."""
    out = {}
    with open(os.path.join(_data_dir(), "MiedemaLiquidDeltaHf.tsv")) as f:
        for line in f.read().split("\n")[1:]:
            parts = line.split()
            if len(parts) == 3:
                try:
                    out[frozenset(parts[:2])] = float(parts[2])
                except ValueError:
                    pass
    return out


_MIEDEMA = None


def miedema_enthalpies(a: str, b: str, frac_b: float) -> list[float]:
    """[intermetallic, amorphous, solid solution] enthalpies (eV/atom) of
    A(1-x)B(x) from matminer's Miedema featurizer; NaN where it has no parameters."""
    global _MIEDEMA
    from pymatgen.core import Composition
    if _MIEDEMA is None:
        from matminer.featurizers.composition import Miedema
        _MIEDEMA = Miedema()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [float(v) for v in _MIEDEMA.featurize(Composition({a: 1.0 - frac_b, b: frac_b}))]
    except Exception:
        return [np.nan, np.nan, np.nan]


def interaction_parameter(a: str, b: str) -> float:
    """Regular-solution interaction parameter of the liquid, kJ/mol: 4 x the
    tabulated enthalpy of mixing at 50 at% (NaN if the pair is not tabulated)."""
    h = liquid_enthalpy_table().get(frozenset((a, b)), np.nan)
    return 4.0 * h


def schroder_van_laar(x_solvent: np.ndarray, tm: float, h_fus: float, omega: float) -> np.ndarray:
    """Liquidus temperature at which the pure solid of one element is in
    equilibrium with a regular-solution liquid holding a fraction x_solvent of it:

        T = (H_fus + omega (1 - x)^2) / (H_fus / Tm - R ln x)

    (H_fus, omega in kJ/mol). omega = 0 gives the ideal-solution liquidus."""
    x = np.clip(x_solvent, 1e-9, 1.0)
    return (h_fus + omega * (1.0 - x) ** 2) / (h_fus / tm - R_KJ * np.log(x))


def regular_solution_liquidus(a: str, b: str, omega: float = 0.0,
                              frac_b: np.ndarray | None = None) -> np.ndarray:
    """Liquidus of A-B with no solid solubility: the higher of the two
    Schroder-van Laar branches, capped at the higher melting point (above it
    this simple model has no solid to be in equilibrium with). Falls back to
    the straight line between the melting points where a value is undefined."""
    t = element_table()
    frac_b = np.linspace(0.0, 1.0, 101) if frac_b is None else np.asarray(frac_b, dtype=float)
    tm_a, tm_b = t.at[a, "MeltingT"], t.at[b, "MeltingT"]
    line = (1.0 - frac_b) * tm_a + frac_b * tm_b
    with np.errstate(all="ignore"):
        curve = np.maximum(schroder_van_laar(1.0 - frac_b, tm_a, t.at[a, "HeatFusion"], omega),
                           schroder_van_laar(frac_b, tm_b, t.at[b, "HeatFusion"], omega))
        curve = np.minimum(curve, max(tm_a, tm_b))
    return np.where(np.isfinite(curve), curve, line)


@lru_cache(maxsize=None)
def element_pair_features(a: str, b: str) -> dict[str, float]:
    """All extra features of the ordered pair (A, B); see the module docstring."""
    t = element_table()
    f: dict[str, float] = {}
    for c in t.columns:
        f[f"el__{c}__A"] = t.at[a, c]
        f[f"el__{c}__B"] = t.at[b, c]
    for c in DIFFERENCES:
        f[f"el__{c}__absdiff"] = abs(t.at[a, c] - t.at[b, c])
    ra, rb = t.at[a, "MiracleRadius"], t.at[b, "MiracleRadius"]
    f["el__radius_ratio"] = min(ra, rb) / max(ra, rb)
    f["el__same_structure"] = float(t.at[a, "SpaceGroupNumber"] == t.at[b, "SpaceGroupNumber"])

    h_liquid = liquid_enthalpy_table().get(frozenset((a, b)), np.nan)
    f["mied__liquid_dH_table"] = h_liquid
    amor, inter = {}, {}
    for x in MIEDEMA_FRACTIONS:
        inter[x], amor[x], ss = miedema_enthalpies(a, b, x)
        f[f"mied__inter_{x}"] = inter[x]
        f[f"mied__amor_{x}"] = amor[x]
        f[f"mied__ss_{x}"] = ss
    f["mied__amor_asym"] = amor[0.75] - amor[0.25]
    f["mied__inter_asym"] = inter[0.75] - inter[0.25]

    omega = interaction_parameter(a, b)
    tm_a, tm_b = t.at[a, "MeltingT"], t.at[b, "MeltingT"]
    f["phys__omega"] = omega
    f["phys__Tc_minus_Tmax"] = omega / (2.0 * R_KJ) - max(tm_a, tm_b)
    f["phys__omega_over_RTmean"] = omega / (R_KJ * 0.5 * (tm_a + tm_b))
    frac = np.linspace(0.0, 1.0, 101)
    line = (1.0 - frac) * tm_a + frac * tm_b
    for name, om in (("ideal", 0.0), ("reg", omega if np.isfinite(omega) else 0.0)):
        curve = regular_solution_liquidus(a, b, om, frac)
        for k in CURVE_POINTS:
            f[f"phys__{name}_dev_{k}"] = curve[k] - line[k]
        j = int(np.argmin(curve))
        f[f"phys__{name}_eut_x"] = 100.0 * frac[j]
        f[f"phys__{name}_eut_T"] = curve[j]
        f[f"phys__{name}_eut_depth"] = min(tm_a, tm_b) - curve[j]
        f[f"phys__{name}_mean_dev"] = float((curve - line).mean())
    return f


def _cached_table(kind: str, pair_ids: list[str], compute) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of compute(a, b) (a dict of floats) for every ordered pair and its
    swap, as float64 tables indexed like `pair_ids`. Computed once and kept in
    $MPDS_FEATURE_CACHE/<kind>.parquet (default /home/alibi/lab/cache_features):
    the Miedema enthalpies take a few seconds per hundred systems."""
    pairs = [tuple(p.split("_")[:2]) for p in pair_ids]
    keys, keys_swap = [f"{a}_{b}" for a, b in pairs], [f"{b}_{a}" for a, b in pairs]
    path = Path(os.environ.get("MPDS_FEATURE_CACHE", "/home/alibi/lab/cache_features")) / f"{kind}.parquet"
    table = pd.read_parquet(path) if path.exists() else pd.DataFrame(dtype=np.float64)
    missing = sorted(set(keys + keys_swap) - set(table.index))
    if missing:
        new = pd.DataFrame([compute(*k.split("_")) for k in missing], index=missing, dtype=np.float64)
        table = new if table.empty else pd.concat([table, new[table.columns]])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        table.to_parquet(tmp)
        os.replace(tmp, path)
    return table.loc[keys], table.loc[keys_swap]


def element_features(pair_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, X_swap, names) for pair ids such as 'Ag_Al': the element features of
    each system and of the same system with A and B exchanged. Unknown values are 0."""
    rows, rows_swap = _cached_table("element_features", pair_ids, element_pair_features)
    matrix = lambda t: np.nan_to_num(t.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    logger.info("Element features: %d columns for %d systems", rows.shape[1], len(pair_ids))
    return matrix(rows), matrix(rows_swap), list(rows.columns)


# ---------------------------------------------------------------------------
# Compound features (Materials Project 2018)
# ---------------------------------------------------------------------------


HULL_POINTS = (10, 25, 33, 50, 67, 75, 90)
STABLE_EV, NEAR_EV = 1e-3, 0.03


def load_binary_entries(path: str | Path) -> dict[frozenset, np.ndarray]:
    """{frozenset({A, B}): array of rows (fraction of the alphabetically
    second element, e_form, e_hull)} from the matminer json.gz file."""
    import pandas as pd
    from pymatgen.core import Composition
    path = Path(path)
    df = pd.read_json(path, orient="split") if path.suffix != ".gz" else pd.read_json(
        path, orient="split", compression="gzip")
    out: dict[frozenset, list] = {}
    for formula, e_form, e_hull in zip(df["formula"], df["e_form"], df["e_hull"]):
        try:
            comp = Composition(formula).fractional_composition.as_dict()
        except Exception:
            continue
        if len(comp) != 2 or not np.isfinite(e_form):
            continue
        a, b = sorted(comp)
        out.setdefault(frozenset((a, b)), []).append((comp[b], e_form, e_hull))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def lower_hull(x: np.ndarray, e: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Lower convex hull of the points (x, e) together with (0, 0) and (1, 0),
    evaluated at the fractions `at`."""
    pts = sorted(set([(0.0, 0.0), (1.0, 0.0)] + [(float(a), float(b)) for a, b in zip(x, e) if b < 0]))
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(at, hx, hy)


def compound_pair_features(a: str, b: str, entries: dict) -> dict[str, float]:
    rows = entries.get(frozenset((a, b)))
    f: dict[str, float] = {}
    at = np.array(HULL_POINTS) / 100.0
    if rows is None or len(rows) == 0:
        f.update(cmp__n_entries=0.0, cmp__n_stable=0.0, cmp__n_near=0.0, cmp__min_eform=0.0,
                 cmp__x_at_min=50.0, cmp__hull_mean=0.0, cmp__x_first_stable=-1.0, cmp__x_last_stable=-1.0)
        f.update({f"cmp__hull_{k}": 0.0 for k in HULL_POINTS})
        return f
    x = rows[:, 0] if sorted((a, b))[1] == b else 1.0 - rows[:, 0]        # fraction of B
    e_form, e_hull = rows[:, 1], rows[:, 2]
    j = int(np.argmin(e_form))
    stable = x[e_hull <= STABLE_EV]
    f["cmp__n_entries"] = float(len(rows))
    f["cmp__n_stable"] = float(len(np.unique(np.round(stable, 3))))
    f["cmp__n_near"] = float(len(np.unique(np.round(x[e_hull <= NEAR_EV], 3))))
    f["cmp__min_eform"] = float(e_form[j])
    f["cmp__x_at_min"] = 100.0 * float(x[j])
    f["cmp__hull_mean"] = float(lower_hull(x, e_form, np.linspace(0, 1, 101)).mean())
    f["cmp__x_first_stable"] = 100.0 * float(stable.min()) if stable.size else -1.0
    f["cmp__x_last_stable"] = 100.0 * float(stable.max()) if stable.size else -1.0
    for k, v in zip(HULL_POINTS, lower_hull(x, e_form, at)):
        f[f"cmp__hull_{k}"] = float(v)
    return f


def compound_features(pair_ids: list[str], path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, X_swap, names): the features of each system and of the same system
    with A and B exchanged."""
    entries = {}
    def compute(a, b):
        if not entries:
            entries.update(load_binary_entries(path))
        return compound_pair_features(a, b, entries)
    rows, rows_swap = _cached_table(f"compound_features_{Path(path).name.split('.')[0]}", pair_ids, compute)
    logger.info("Compound features: %d of %d systems have computed entries", int((rows["cmp__n_entries"] > 0).sum()),
                len(pair_ids))
    return rows.to_numpy(np.float32), rows_swap.to_numpy(np.float32), list(rows.columns)
