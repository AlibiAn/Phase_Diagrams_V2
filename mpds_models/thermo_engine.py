"""
A small CALPHAD-style phase-diagram engine for one binary system, to test
whether predicting a few thermodynamic parameters (and computing the diagram)
can beat predicting the curve directly.

Phases, Gibbs energy per mole of atoms relative to the pure LIQUID elements:
  liquid      R T [x ln x + (1-x) ln(1-x)] + x (1-x) [L0 + L1 (1 - 2x)]
  solid sol.  (1-x) gA + x gB + R T [x ln x + (1-x) ln(1-x)] + Ls x (1-x)
              with gA = -H_A (1 - T / Tm_A), gB likewise (the pure solids);
              Ls large -> no mutual solubility, Ls small -> isomorphous lens
  compounds   at each OQMD convex-hull vertex x_k (formation enthalpy H_k of
              the 0 K DFT calculation, from the solid elements):
              (1 - x_k) gA + x_k gB + c * H_k   (stoichiometric, no entropy)
Parameters per system: L0, L1, Ls (kJ/mol) and the compound scale c.
At each temperature the stable state at x is the lower convex hull of all
phases; the liquidus T_L(x) is the lowest temperature above which no solid
phase is on the hull at x (the convention of processed_v8: under a two-liquid
dome it is the monotectic line).

Element data: melting point and heat of fusion from the Magpie tables
(element_features.py); compounds from ORCHID's OQMD hulls (hull_features.py).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                      # numba is optional: same results, about 30 times slower
    def njit(*args, **kwargs):
        return (lambda f: f) if not (args and callable(args[0])) else args[0]

R = 8.314462618e-3                  # kJ / (mol K)
EV = 96.485                         # kJ/mol per eV/atom
GRID = np.linspace(0.0, 1.0, 101)
_IN = GRID[1:-1]
_MIX = _IN * np.log(_IN) + (1 - _IN) * np.log(1 - _IN)


def _lower_hull(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of points sorted by x (monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (x[b] - x[a]) * (g[i] - g[a]) - (g[b] - g[a]) * (x[i] - x[a]) <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull)


@njit(cache=True)
def _liquidus_kernel(xs_o, solid_o, order, grid, mix_in, temps, t_step, tm_a, tm_b, h_a, h_b,
                     comp_x, comp_h, L0, L1, Ls, c):
    """Compiled core of System.liquidus (same steps, see there)."""
    n = grid.shape[0]
    m = n - 2
    nc = comp_x.shape[0]
    npts = 2 * n + nc
    T_L = np.full(n, np.nan)
    found = np.zeros(n, dtype=np.bool_)
    n_found = 0
    g = np.empty(npts)
    g_o = np.empty(npts)
    hull = np.empty(npts, dtype=np.int64)
    for it in range(temps.shape[0]):
        T = temps[it]
        gA = -h_a * (1.0 - T / tm_a)
        gB = -h_b * (1.0 - T / tm_b)
        g[0] = 0.0
        g[n - 1] = 0.0
        g[n] = gA
        g[2 * n - 1] = gB
        for k in range(m):
            xk = grid[k + 1]
            mixk = R * T * mix_in[k]
            g[k + 1] = mixk + xk * (1.0 - xk) * (L0 + L1 * (1.0 - 2.0 * xk))
            g[n + k + 1] = (1.0 - xk) * gA + xk * gB + mixk + Ls * xk * (1.0 - xk)
        for k in range(nc):
            g[2 * n + k] = (1.0 - comp_x[k]) * gA + comp_x[k] * gB + c * comp_h[k]
        for k in range(npts):
            g_o[k] = g[order[k]]
        # lower convex hull (monotone chain)
        h = 0
        for i in range(npts):
            while h >= 2:
                a = hull[h - 2]
                b = hull[h - 1]
                if (xs_o[b] - xs_o[a]) * (g_o[i] - g_o[a]) - (g_o[b] - g_o[a]) * (xs_o[i] - xs_o[a]) <= 1e-12:
                    h -= 1
                else:
                    break
            hull[h] = i
            h += 1
        # segment of the hull over each grid composition
        seg = 0
        for j in range(n):
            x = grid[j]
            while seg + 1 < h and xs_o[hull[seg + 1]] <= x:
                seg += 1
            s_ = seg if seg <= h - 2 else (h - 2 if h >= 2 else 0)
            left = hull[s_]
            right = hull[s_ + 1] if s_ + 1 < h else hull[h - 1]
            at_left = abs(xs_o[left] - x) <= 1e-8 + 1e-5 * abs(x)
            solid_here = solid_o[left] if at_left else (solid_o[left] or solid_o[right])
            if solid_here and not found[j]:
                T_L[j] = T + t_step / 2.0
                found[j] = True
                n_found += 1
        if n_found == n:
            break
    for j in range(n):
        if not found[j]:
            T_L[j] = temps[temps.shape[0] - 1]
    T_L[0] = tm_a
    T_L[n - 1] = tm_b
    return T_L


class System:
    """Fixed data of one binary system: melting points, heats of fusion, compounds."""

    def __init__(self, tm_a, tm_b, h_a, h_b, compounds: list[tuple[float, float]], t_step: float = 10.0):
        self.tm_a, self.tm_b, self.h_a, self.h_b = float(tm_a), float(tm_b), float(h_a), float(h_b)
        self.comp_x = np.array([c[0] for c in compounds], dtype=float)
        self.comp_h = np.array([c[1] * EV for c in compounds], dtype=float)        # kJ/mol, <= 0
        # compounds can melt far above both elements (AlP: 2,800 K from 933 and 317 K)
        self.temps = np.arange(5000.0, 150.0, -t_step)
        self.t_step = t_step

    def liquidus(self, L0: float, L1: float, Ls: float, c: float = 1.0) -> np.ndarray:
        """T_L on the 101-point grid (K); compiled with numba when available."""
        if not hasattr(self, "_order"):
            n = len(GRID)
            xs = np.r_[GRID, GRID, self.comp_x]
            self._order = np.argsort(xs, kind="stable").astype(np.int64)
            self._xs_o = xs[self._order]
            self._solid_o = np.r_[np.zeros(n, bool), np.ones(n, bool), np.ones(len(self.comp_x), bool)][self._order]
        return _liquidus_kernel(self._xs_o, self._solid_o, self._order, GRID, _MIX, self.temps, float(self.t_step),
                                self.tm_a, self.tm_b, self.h_a, self.h_b, self.comp_x.astype(float),
                                self.comp_h.astype(float), float(L0), float(L1), float(Ls), float(c))

    def liquidus_python(self, L0: float, L1: float, Ls: float, c: float = 1.0) -> np.ndarray:
        """The uncompiled reference implementation of liquidus()."""
        n = len(GRID)
        nc = len(self.comp_x)
        # points: liquid at all grid x (0..100), solid solution at all grid x, compounds
        xs = np.r_[GRID, GRID, self.comp_x]
        is_solid = np.r_[np.zeros(n, bool), np.ones(n, bool), np.ones(nc, bool)]
        order = np.argsort(xs, kind="stable")
        xs_o, solid_o = xs[order], is_solid[order]
        T_L = np.full(n, np.nan)
        found = np.zeros(n, dtype=bool)
        idx = np.arange(n)
        rk = _IN * (1 - _IN) * (L0 + L1 * (1 - 2 * _IN))
        ss_int = Ls * _IN * (1 - _IN)
        for T in self.temps:
            gA = -self.h_a * (1 - T / self.tm_a)
            gB = -self.h_b * (1 - T / self.tm_b)
            g_liq = np.r_[0.0, R * T * _MIX + rk, 0.0]
            g_ss = np.r_[gA, (1 - _IN) * gA + _IN * gB + R * T * _MIX + ss_int, gB]
            g_cmp = (1 - self.comp_x) * gA + self.comp_x * gB + c * self.comp_h
            g = np.r_[g_liq, g_ss, g_cmp][order]
            hull = _lower_hull(xs_o, g)
            hx = xs_o[hull]
            # which hull segment covers each grid composition, and does it involve a solid
            seg = np.clip(np.searchsorted(hx, GRID, side="right") - 1, 0, max(len(hull) - 2, 0))
            left, right = hull[seg], hull[np.minimum(seg + 1, len(hull) - 1)]
            at_left = np.isclose(xs_o[left], GRID)
            solid_here = np.where(at_left, solid_o[left], solid_o[left] | solid_o[right])
            new = solid_here & ~found
            T_L[new] = T + self.t_step / 2
            found |= solid_here
            if found.all():
                break
        T_L[~found] = self.temps[-1]
        T_L[0], T_L[-1] = self.tm_a, self.tm_b
        return T_L


def system_for(a: str, b: str, use_compounds: bool = True, t_step: float = 10.0) -> System | None:
    """The System of the ordered pair (A, B) with x = fraction of B; None if data are missing."""
    from .element_features import element_table
    from .hull_features import _vertices
    t = element_table()
    vals = [t.at[a, "MeltingT"], t.at[b, "MeltingT"], t.at[a, "HeatFusion"], t.at[b, "HeatFusion"]]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        return None
    comps: list[tuple[float, float]] = []
    if use_compounds:
        hit = _vertices("oqmd").get(frozenset((a, b)))
        if hit:
            x_el, verts, _ = hit
            for x, e, _t in verts:
                xb = x if x_el == b else 1.0 - x
                if 1e-6 < xb < 1 - 1e-6 and e < 0:
                    comps.append((xb, e))
    return System(*vals, comps, t_step=t_step)
