"""
Extra features computed from the two element symbols alone (added in v14).

The processed_v6 features describe each element by a dozen pymatgen
properties and the 50/50 mixture by Magpie statistics and Miedema enthalpies.
This module adds what a thermodynamic estimate of the liquidus needs:

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

Every feature is a function of the ORDERED pair (A, B), so the features of
the system with A and B exchanged are simply `pair_features(B, A)`; no swap
rules are needed (base.swap_augment uses PhaseDataset.X_swap).

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
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from functools import lru_cache

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
def pair_features(a: str, b: str) -> dict[str, float]:
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


def build_features(pair_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X, X_swap, names) for pair ids such as 'Ag_Al': the extra features of
    each system and of the same system with A and B exchanged. Unknown values
    are 0, as in base.load_dataset."""
    rows, rows_swap = [], []
    for pid in pair_ids:
        a, b = pid.split("_")[:2]
        rows.append(pair_features(a, b))
        rows_swap.append(pair_features(b, a))
    names = list(rows[0])
    def matrix(rs):
        m = np.array([[r[n] for n in names] for r in rs], dtype=np.float64)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    logger.info("Extra element features: %d columns for %d systems", len(names), len(pair_ids))
    return matrix(rows), matrix(rows_swap), names
