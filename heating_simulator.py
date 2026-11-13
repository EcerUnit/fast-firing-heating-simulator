#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heating of a ceramic pellet dropped into a preheated furnace
============================================================
Transient axisymmetric conduction in (r, z) by finite volumes, with
radiation and convection on the free surfaces, solved on a two-layer mesh
that carries the pellet and the setter it rests on.

Written for fast firing. The question it answers is the practical one: once
the pellet goes into a furnace already at temperature, how long before its
coldest point — the centre — is actually hot enough to sinter? Ovens are
quick, ceramics are not, and the gap between the two is where fast firing
succeeds or cracks the sample.

Everything the program knows is in the Theory tab of the interface, written
out in full. The same material is in the PDF report that ships with it.

Requires numpy and matplotlib.
Run with:  python heating_simulator.py
"""

import os
import sys
import csv
import threading
import queue
import subprocess
import numpy as np

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# The interface needs Tk; the solver does not. Importing this module from a
# headless script — a batch sweep, a CI job, a server — has to keep working
# on a machine with no Tk at all, so the GUI imports are optional and their
# absence is only reported if someone actually asks for a window.
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    HAVE_TK = True
    _TK_ERROR = None
except Exception as _err:                                    # noqa: BLE001
    HAVE_TK, _TK_ERROR = False, _err
    ttk = filedialog = messagebox = None
    FigureCanvasTkAgg = NavigationToolbar2Tk = None

    class _NoTk:
        """Stand-in so the classes below can still be defined.

        Instantiating one explains what is missing instead of failing with
        a NameError from somewhere deep in the widget tree.
        """

        def __init__(self, *a, **k):
            raise RuntimeError(
                "The graphical interface needs Tkinter, which is not "
                f"available here ({_TK_ERROR}). On Debian or Ubuntu: "
                "sudo apt install python3-tk. The solver itself works "
                "without it — see examples/headless_run.py.")

    class tk:                                            # noqa: N801
        Tk = Toplevel = _NoTk

SIG = 5.670374419e-8          # Stefan-Boltzmann constant [W/m²K⁴]
K0 = 273.15

# Running this file as a program registers it under the name "__main__". If
# sensitivity.py then does "from heating_simulator import ...", Python loads
# the SAME file a second time under its real name and builds a second copy
# of every class. Objects made by one copy are then not recognised by the
# other, which surfaces as baffling type errors a long way from the cause.
# Registering the alias here prevents the second load.
if __name__ == "__main__":
    sys.modules.setdefault("heating_simulator", sys.modules[__name__])

# Bumped whenever something sensitivity.py relies on changes. It is there to
# catch the case of two files from different versions sitting in the same
# folder, which otherwise fails with type errors that are hard to read.
VERSION = 7


def is_property(x):
    """Is x a Property? Checked by interface rather than with isinstance.

    If this file does end up loaded twice (as __main__ and as an imported
    module) there are two distinct Property classes, and isinstance between
    them fails even though the objects are equivalent. The alias above
    avoids that situation; duck typing covers it even if something slips
    past.
    """
    return all(hasattr(x, a) for a in ("grid", "evaluate", "variable",
                                       "representative"))

# --------------------------------------------------------------------------
# Measured properties, as functions of T in kelvin
# --------------------------------------------------------------------------
# Validity range 373-1473 K. Outside it the polynomials misbehave: k and cp
# of LATP go negative above roughly 1500 K, and both emissivities climb past
# unity above roughly 1570 K. Clamping at the ends is therefore not a
# convenience, it is what keeps an extrapolated fit from quietly wrecking
# the solution.


_T_VAL = (373.0, 1473.0)

_LATP_K = ("-2.016e-17*T**6 + 1.278e-13*T**5 - 3.236e-10*T**4 "
           "+ 4.090e-7*T**3 - 2.673e-4*T**2 + 8.585e-2*T - 9.700e0")
_LATP_CP = ("-6.414e-14*T**6 + 3.680e-10*T**5 - 8.507e-7*T**4 "
            "+ 9.983e-4*T**3 - 6.179e-1*T**2 + 1.899e2*T - 2.091e4")
_LATP_EPS = "5.600e-4*T + 1.210e-1"
_LATP_ALPHA = "9.774e-11*T + 1.486e-7"         # measured diffusivity, m²/s

_AL_K = "1.912e-5*T**2 - 5.025e-2*T + 3.866e1"
_AL_CP = "9.052e2 + 3.678e-1*T - 2.142e-3/T**2"
_AL_EPS = "7.314e-4*T - 1.774e-1"


def props_latp():
    """LATP with its measured properties.

    The density, 2990 kg/m³, is not an independent datum. It falls out of
    requiring k = α·ρ·cₚ to hold between the three measured functions, and
    it comes out constant to within 3 % from 400 to 1200 K — which is the
    useful part: it says the three measurements agree with each other.
    """
    a, b = _T_VAL
    return dict(k=Property(expr=_LATP_K, T_min=a, T_max=b,
                           name="LATP conductivity"),
                cp=Property(expr=_LATP_CP, T_min=a, T_max=b,
                            name="LATP specific heat"),
                eps=Property(expr=_LATP_EPS, T_min=a, T_max=b,
                             name="LATP emissivity"))


def props_alumina():
    """Setter alumina: conductivity and specific heat from the literature,
    emissivity measured in the lab with a pyrometer."""
    a, b = _T_VAL
    return dict(k=Property(expr=_AL_K, T_min=a, T_max=b,
                           name="alumina conductivity"),
                cp=Property(expr=_AL_CP, T_min=a, T_max=b,
                            name="alumina specific heat"),
                eps=Property(expr=_AL_EPS, T_min=a, T_max=b,
                             name="alumina emissivity"))


MATERIALS = {
    "LATP, measured f(T)": dict(rho=2990.0, f=props_latp),
    "LATP, green (~65 % dens.)": dict(k=0.8, rho=1900.0, cp=950.0, eps=0.85),
    "LATP, densified":             dict(k=1.5, rho=2800.0, cp=950.0, eps=0.85),
    "NZSP, green":                 dict(k=0.9, rho=2000.0, cp=900.0, eps=0.85),
    "NZSP, densified":             dict(k=1.8, rho=3200.0, cp=900.0, eps=0.85),
    "Dense alumina":               dict(k=6.0, rho=3900.0, cp=1200.0, eps=0.70),
    "Zirconia (YSZ)":              dict(k=2.2, rho=6000.0, cp=600.0, eps=0.80),
}

SUPPORTS = {
    "Alumina, measured f(T)": dict(rho=3900.0, f=props_alumina),
    "Dense alumina":  dict(k=6.0, rho=3900.0, cp=1200.0, eps=0.70),
    "Porous alumina": dict(k=2.0, rho=2300.0, cp=1100.0, eps=0.80),
    "Zirconia (YSZ)": dict(k=2.2, rho=6000.0, cp=600.0, eps=0.80),
    "Mullite":        dict(k=3.5, rho=2800.0, cp=1100.0, eps=0.75),
}

# Cells across the pellet thickness. Everything else in the mesh is sized
# from this, so it is the one knob that trades accuracy against runtime.
RESOLUTION = {"Coarse": 8, "Normal": 14, "Fine": 22}

PROGRAMMES = {
    "Stabilised furnace (door opened)": "stabilised",
    "Programmed ramp": "ramp",
}

# Temperature grid the properties are tabulated on [K]. One kelvin per
# point is far finer than any property varies, so interpolating on it costs
# nothing in accuracy and saves evaluating the expressions in the inner
# loop, which would otherwise dominate the runtime.
TGRID = np.arange(250.0, 2400.0 + 1.0, 1.0)

# What a user may write when entering a property as f(T). The environment is
# deliberately small: no builtins, no imports, no attribute access.
_ENV = {
    "__builtins__": {},
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "tanh": np.tanh, "pi": np.pi, "e": np.e,
    "min": np.minimum, "max": np.maximum,
}


class Property:
    """A thermal property: either a constant, or a function of T in kelvin.

    The expression is written in terms of T, in kelvin, using ordinary
    Python syntax: '900 + 0.12*T - 2.1e7/T**2'. exp, log, log10, sqrt, tanh,
    abs, min and max are available.

    OUTSIDE THE VALIDITY RANGE the value is held constant at the nearest
    endpoint. That is deliberate. A fit obtained between 800 and 1000 °C,
    extrapolated down to 100 °C, can return something absurd or negative,
    and a negative specific heat breaks the model without saying a word.
    Clamping turns a silent catastrophe into a visible approximation.
    """

    def __init__(self, value=None, expr=None, T_min=None, T_max=None,
                 name="property"):
        self.name = name
        self.expr = (expr or "").strip() or None
        self.T_min = T_min
        self.T_max = T_max
        self.value = None if self.expr else float(value)
        self._grid = None

    # ----------------------------------------------------------- factory
    @staticmethod
    def of(x, name="property"):
        """Accept either a Property or a bare number, return a Property."""
        if is_property(x):
            return x
        return Property(value=float(x), name=name)

    @property
    def variable(self):
        return self.expr is not None

    # ---------------------------------------------------------- evaluate
    def evaluate(self, T):
        T = np.asarray(T, dtype=float)
        if not self.variable:
            return np.full(T.shape, self.value)
        Tc = T
        if self.T_min is not None:
            Tc = np.maximum(Tc, self.T_min)
        if self.T_max is not None:
            Tc = np.minimum(Tc, self.T_max)
        try:
            v = eval(self.expr, dict(_ENV), {"T": Tc})           # noqa: S307
        except Exception as err:                                 # noqa: BLE001
            raise ValueError(
                f"Could not evaluate the function for “{self.name}”: {err}")
        v = np.asarray(v, dtype=float)
        if v.shape != Tc.shape:              # a constant expression returns
            v = np.full(Tc.shape, float(v))  # a scalar; broadcast it
        return v

    def grid(self, zero_ok=False):
        """Values over TGRID, validated and cached.

        zero_ok=True allows zero, which is meaningful for an emissivity (it
        switches radiation off) but not for a conductivity or a specific
        heat, where it would make the model singular.

        Validating here rather than mid-solve is worth the up-front cost: a
        bad fit is caught the moment it is entered, with the temperatures
        where it misbehaves, instead of surfacing as a nan a thousand time
        steps later.
        """
        if self._grid is None:
            v = self.evaluate(TGRID)
            if not np.all(np.isfinite(v)):
                raise ValueError(
                    f"The function for “{self.name}” returns non-finite "
                    f"values inside the computed range.")
            bad = v < 0 if zero_ok else v <= 0
            if np.any(bad):
                T_bad = TGRID[bad]
                adj = "negative" if zero_ok else "zero or negative"
                raise ValueError(
                    f"The function for “{self.name}” returns {adj} values "
                    f"between {T_bad.min():.0f} and {T_bad.max():.0f} K. "
                    f"Check the fit, or narrow its validity range.")
            self._grid = v
        return self._grid

    def representative(self, T0, T1):
        """Mean value over an interval, for the dimensionless numbers.

        Those numbers and the compact formula are estimates and cannot take
        a varying property, so they get a single number averaged over the
        temperature range the run will actually cover.
        """
        if not self.variable:
            return self.value
        m = (TGRID >= min(T0, T1)) & (TGRID <= max(T0, T1))
        return float(self.grid()[m].mean()) if m.any() else self.value

    # -------------------------------------------------------- arithmetic
    def scaled(self, factor):
        """Multiply the property by a factor, keeping its nature.

        A constant gives another constant; a function of T gives the same
        function scaled, over the same validity range. Scaling the whole
        curve is the perturbation that means something in a sensitivity
        analysis: it asks what happens if every value were a few per cent
        higher, without distorting the shape.
        """
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            return NotImplemented
        if not self.variable:
            return Property(value=self.value * factor, name=self.name)
        return Property(expr=f"({self.expr})*{factor!r}", T_min=self.T_min,
                        T_max=self.T_max, name=self.name)

    __mul__ = scaled
    __rmul__ = scaled

    def describe(self):
        if not self.variable:
            return f"{self.value:g}"
        r = ""
        if self.T_min is not None or self.T_max is not None:
            a = f"{self.T_min:.0f}" if self.T_min is not None else "—"
            b = f"{self.T_max:.0f}" if self.T_max is not None else "—"
            r = f"  [{a}–{b} K]"
        return f"f(T) = {self.expr}{r}"


class Material:
    """One material. Density is taken constant; k, cₚ and ε may depend on T.

    The important consequence of a temperature-dependent cₚ: the time
    advance can NOT be done on temperature. It has to be done on ENTHALPY.
    Otherwise the scheme stops conserving energy, because the heat capacity
    changes within the time step itself and the error accumulates through
    the transient. So u(T) = ρ·∫cₚ dT is tabulated once here and inverted
    by interpolation when the solver needs a temperature back. A pleasant
    side effect: a latent heat would appear as a step in u(T) and the
    solver would handle it with no changes at all.
    """

    def __init__(self, k, rho, cp, eps, name=""):
        self.name = name
        self.rho = float(rho)
        self.k = Property.of(k, f"{name} conductivity" if name else "k")
        self.cp = Property.of(cp, f"{name} specific heat" if name else "cp")
        self.eps = Property.of(eps, f"{name} emissivity"
                               if name else "emissivity")
        self.k_g = self.k.grid()
        self.cp_g = self.cp.grid()
        self.eps_g = self.eps.grid(zero_ok=True)
        # cumulative volumetric enthalpy [J/m³]. Monotonically increasing
        # because cₚ > 0, which is what makes np.interp usable to invert it.
        self.u_g = self.rho * np.concatenate(
            ([0.0], np.cumsum(0.5 * (self.cp_g[1:] + self.cp_g[:-1])
                              * np.diff(TGRID))))

    @property
    def variable(self):
        return self.k.variable or self.cp.variable or self.eps.variable

    def u_of_T(self, T):
        return np.interp(T, TGRID, self.u_g)

    def T_of_u(self, u):
        return np.interp(u, self.u_g, TGRID)

    def k_of_T(self, T):
        return np.interp(T, TGRID, self.k_g)

    def eps_of_T(self, T):
        return np.interp(T, TGRID, self.eps_g)


def material_from(d, name=""):
    """Build a Material from the parameter dictionary."""
    if isinstance(d, Material):
        return d
    return Material(d["k"], d["rho"], d["cp"], d["eps"], name)


# --------------------------------------------------------------------------
# Furnace thermal programme
# --------------------------------------------------------------------------
def furnace_programme(p):
    """Return T_furnace(t) in K for the chosen programme.

    'stabilised': the furnace is already at the setpoint. Opening the door
       drops it by a given amount and it recovers exponentially. This is
       fast firing.
    'ramp': the furnace starts cold and climbs at a constant rate to the
       setpoint, then holds. This is conventional sintering, and having it
       here lets both modes be compared inside the same framework rather
       than across two different tools.
    """
    T_h = p["T_furnace"]
    if p.get("programme", "stabilised") == "ramp":
        T_i = p.get("T_furnace_initial", 25 + K0)
        v = max(1e-6, p.get("ramp_K_min", 300.0)) / 60.0      # K/s

        def T_env(t):
            return min(T_h, T_i + v * t)
        return T_env
    drop, tau = p["drop"], p["tau_recovery"]
    if drop <= 0:
        return lambda t: T_h
    return lambda t: T_h - drop * np.exp(-t / tau)


# --------------------------------------------------------------------------
# Post-processing: heating rate and thermal stress
# --------------------------------------------------------------------------
def heating_rate(t, T):
    """dT/dt in K/s, by central differences on the saved curve."""
    if len(t) < 3:
        return np.zeros_like(T)
    return np.gradient(T, t)


def thermal_stress(T_mean, T_centre, T_surface, mech):
    """Thermal stress in a plate with a through-thickness gradient.

    What drives the stress is the difference between the local temperature
    and the MEAN temperature of the piece, not the face-to-centre
    difference. A body with a gradient deforms as a whole, and each point
    ends up in tension or compression according to how far it sits from
    that mean. Getting this wrong is easy and expensive: using face minus
    centre overstates the stress by roughly a factor of two.

        σ = E·α·(T_mean − T_local) / (1 − ν)

    While heating, the centre lags behind the mean and goes into TENSION,
    which is how ceramics fail; the surface is left in compression.
    Returns (σ_centre, σ_surface) in Pa, positive meaning tension.
    """
    f = mech["E"] * mech["alpha"] / (1.0 - mech["nu"])
    return f * (T_mean - T_centre), f * (T_mean - T_surface)


# --------------------------------------------------------------------------
# Numerical core
# --------------------------------------------------------------------------
class Model:
    """Axisymmetric finite-volume mesh with two layers: support and sample.

    The mesh is built once in the constructor and then reused for the whole
    run. Cells belong to one layer or to neither: the region outside the
    support radius but inside the mesh rectangle is simply marked inactive,
    which is how a stepped geometry is represented without an unstructured
    mesh.
    """

    def __init__(self, p, n_thick):
        self.p = p
        e_m, r_m = p["e_sample"], p["r_sample"]
        self.with_support = p["with_support"]

        # One uniform dz has to span both layers, so when a support is
        # present the requested cell count is adjusted to a spacing that
        # divides both thicknesses reasonably. At least two cells across
        # the support, or its thermal mass is badly resolved.
        dz = e_m / n_thick
        if self.with_support:
            e_s = p["e_support"]
            n_s = max(2, int(round(e_s / dz)))
            dz = (e_m + e_s) / (n_thick + n_s)
            H, self.z0 = e_m + e_s, e_s
            R = max(p["r_support"], r_m * 1.05)
        else:
            H, self.z0, R = e_m, 0.0, r_m

        # Radial cells are coarser than axial ones on purpose: the gradient
        # lives across the thickness, and the disc is far wider than thick.
        # The cap at 60 keeps a thin, wide pellet from exploding the mesh.
        n_rad = max(6, int(round(n_thick * r_m / e_m / 3)))
        n_rad = min(n_rad, 60)
        dr = R / n_rad

        self.nr, self.nz = int(round(R / dr)), int(round(H / dz))
        self.dr, self.dz = dr, dz
        self.r = (np.arange(self.nr) + 0.5) * dr
        self.z = (np.arange(self.nz) + 0.5) * dz
        rp = (np.arange(self.nr) + 1.0) * dr

        self.V = (2 * np.pi * self.r * dr * dz)[:, None] * np.ones(self.nz)
        self.Az = (2 * np.pi * self.r * dr)[:, None] * np.ones(self.nz)
        self.Ar = (2 * np.pi * rp * dz)[:, None] * np.ones(self.nz)

        # Material index per cell: 0 support, 1 sample, -1 inactive.
        self.mat_idx = np.full((self.nr, self.nz), -1, dtype=int)
        self.act = np.zeros((self.nr, self.nz), dtype=bool)
        self.materials = [material_from(p["support"], "support"),
                          material_from(p["sample"], "sample")]

        if self.with_support:
            self._layer(0.0, self.z0, 0, p["r_support"])
        self._layer(self.z0, self.z0 + e_m, 1, r_m)

        # Cells belonging to the sample, for the mean temperature and the
        # stress. The support must not be averaged in: it is hotter, and
        # including it would flatter the numbers.
        self.mask_sample = (((self.z[None, :] >= self.z0) &
                             (self.r[:, None] <= r_m)) & self.act)

        # Indices of the probe points reported on the curves.
        self.j_centre = int(np.argmin(np.abs(self.z - (self.z0 + e_m / 2))))
        self.j_top = int(np.argmin(np.abs(self.z - (self.z0 + e_m - dz / 2))))
        self.j_bottom = int(np.argmin(np.abs(self.z - (self.z0 + dz / 2))))
        self.i_rim = max(0, int(np.argmin(np.abs(self.r - (r_m - dr / 2)))))
        self.j_support = max(0, int(np.argmin(np.abs(self.z - self.z0 / 2)))) \
            if self.with_support else 0
        self.j_int = int(np.ceil(self.z0 / dz)) if self.with_support else 0

    def _layer(self, z0, z1, idx, rmax):
        m = ((self.z[None, :] >= z0) & (self.z[None, :] < z1)
             & (self.r[:, None] <= rmax))
        self.mat_idx[m] = idx
        self.act |= m

    # ----------------------------------------------------- local properties
    def _fields(self, T):
        """Conductivity and emissivity of every cell at its current T."""
        k = np.zeros_like(T)
        eps = np.zeros_like(T)
        for idx, mat in enumerate(self.materials):
            m = self.mat_idx == idx
            if m.any():
                k[m] = mat.k_of_T(T[m])
                eps[m] = mat.eps_of_T(T[m])
        return k, eps

    def _T_of_u(self, u):
        T = np.zeros_like(u)
        for idx, mat in enumerate(self.materials):
            m = self.mat_idx == idx
            if m.any():
                T[m] = mat.T_of_u(u[m])
        return T

    def _u_of_T(self, T):
        u = np.zeros_like(T)
        for idx, mat in enumerate(self.materials):
            m = self.mat_idx == idx
            if m.any():
                u[m] = mat.u_of_T(T[m])
        return u

    # ---------------------------------------------------------------- solve
    def solve(self, t_end, cb=None, stop=None, n_save=160):
        p = self.p
        act = self.act
        nr, nz = self.nr, self.nz
        variable = any(m.variable for m in self.materials)

        def harmonic(a, b):
            """Harmonic mean of the conductivities on either side of a face.

            Not the arithmetic mean. At the sample/support interface the two
            conductivities differ by a factor of several, and the arithmetic
            mean lets the better conductor dominate a resistance that is
            physically in series. Patankar makes the case in §4.2.3; the
            practical consequence here is a heat flux across the interface
            that is right rather than optimistic.
            """
            out = np.zeros_like(a)
            m = (a > 0) & (b > 0)
            out[m] = 2 * a[m] * b[m] / (a[m] + b[m])
            return out

        Rc_col = None
        if self.with_support and p["R_contact"] > 0 and 0 < self.j_int <= nz - 1:
            Rc_col = (self.j_int - 1,
                      p["R_contact"] / np.maximum(self.Az[:, self.j_int - 1],
                                                  1e-30))

        def conductances(k):
            """Conductances between neighbouring cells, from the k field.

            A face with an inactive cell on either side gets zero, which is
            how the stepped geometry is enforced without special-casing the
            boundary anywhere else in the solver.
            """
            Gr = harmonic(k[:-1, :], k[1:, :]) * self.Ar[:-1, :] / self.dr
            Gr[~(act[:-1, :] & act[1:, :])] = 0.0
            Gz = harmonic(k[:, :-1], k[:, 1:]) * self.Az[:, :-1] / self.dz
            Gz[~(act[:, :-1] & act[:, 1:])] = 0.0
            if Rc_col is not None:          # contact resistance, in series
                j, Rc = Rc_col
                col = Gz[:, j].copy()
                m = col > 0
                col[m] = 1.0 / (1.0 / col[m] + Rc[m])
                Gz[:, j] = col
            return Gr, Gz

        # Areas exposed to the furnace, and the distance from each surface
        # cell centre to its own face. A corner cell is exposed on two
        # sides, so the areas accumulate and the distance is area-weighted.
        A_exp = np.zeros((nr, nz))
        Ad = np.zeros((nr, nz))

        def expose(mask, area, d):
            A_exp[mask] += area[mask]
            Ad[mask] += area[mask] * d

        nb = np.zeros((nr, nz), dtype=bool)      # outward in r: the rim
        nb[:-1, :] = ~act[1:, :]
        nb[-1, :] = True
        expose(nb & act, self.Ar, self.dr / 2)

        nb = np.zeros((nr, nz), dtype=bool)      # upward in z: the top face
        nb[:, :-1] = ~act[:, 1:]
        nb[:, -1] = True
        expose(nb & act, self.Az, self.dz / 2)

        nb = np.zeros((nr, nz), dtype=bool)      # downward: the bottom face
        nb[:, 1:] = ~act[:, :-1]
        nb[:, 0] = True
        expose(nb & act, self.Az, self.dz / 2)

        surf = A_exp > 0
        d_ef = np.ones((nr, nz))
        d_ef[surf] = Ad[surf] / A_exp[surf]

        T_h, T_0 = p["T_furnace"], p["T_initial"]
        h_c = p["h_conv"]
        T_env = furnace_programme(p)

        # Initial fields. The state the solver actually advances is u, the
        # volumetric enthalpy; T is derived from it every step.
        T = np.where(act, T_0, 0.0)
        u = self._u_of_T(T)
        k, eps = self._fields(T)
        Gr, Gz = conductances(k)
        G_half = np.zeros((nr, nz))
        G_half[surf] = k[surf] * A_exp[surf] / d_ef[surf]

        # ---- Stable time step.
        # The scheme is explicit, so the step is bounded by stability, not
        # by accuracy. With temperature-dependent properties the bound
        # itself moves during the run, and recomputing it every step would
        # be both slow and fragile. Instead it is fixed once from the worst
        # case over the whole thermal range the run will visit: the highest
        # conductivity it will ever see against the lowest ρ·cₚ. That is
        # conservative by construction and cannot be violated later.
        k_bound = np.zeros((nr, nz))
        rc_bound = np.ones((nr, nz))
        eps_bound = np.zeros((nr, nz))
        lo = min(T_0, 250.0)
        high = max(T_h, T_0) + 50.0
        span = (TGRID >= lo) & (TGRID <= high)
        for idx, mat in enumerate(self.materials):
            m = self.mat_idx == idx
            if m.any():
                k_bound[m] = mat.k_g[span].max()
                rc_bound[m] = mat.rho * mat.cp_g[span].min()
                eps_bound[m] = mat.eps_g[span].max()
        Grc, Gzc = conductances(k_bound)
        Gs = np.zeros((nr, nz))
        Gs[:-1, :] += Grc
        Gs[1:, :] += Grc
        Gs[:, :-1] += Gzc
        Gs[:, 1:] += Gzc
        G_half_bound = np.zeros((nr, nz))
        G_half_bound[surf] = k_bound[surf] * A_exp[surf] / d_ef[surf]
        G_film = A_exp * (h_c + 4 * eps_bound * SIG * T_h ** 3)
        with np.errstate(divide="ignore"):
            Gs += np.where(
                surf,
                1.0 / (1.0 / np.maximum(G_half_bound, 1e-30)
                       + 1.0 / np.maximum(G_film, 1e-30)),
                0.0)
        C_bound = rc_bound * self.V
        C_bound[~act] = 1.0
        # 0.4 rather than the formal 0.5: margin for the non-linear
        # radiative term, which the linearised bound above understates.
        dt = 0.4 * np.min(C_bound[act] / np.maximum(Gs[act], 1e-12))
        nstep = int(np.ceil(t_end / dt))
        dt = t_end / nstep

        Ts = np.full((nr, nz), T_0)
        hc_s = np.where(surf, k / d_ef, 0.0)
        V = self.V
        # Refreshing the property fields every step would roughly double
        # the runtime for no visible gain: over four steps the surface
        # moves by a fraction of a kelvin.
        every_prop = 4 if variable else 0

        every = max(1, nstep // n_save)
        ts, Tp, snaps, Tmed = [], [], [], []
        # The centre temperature is also kept at EVERY step, not just at
        # the saved ones. The peak dT/dt measured on the saved curve
        # depends on how often it was saved, which made the reported peak
        # heating rate an artefact of the output interval rather than a
        # property of the run.
        T_centre_fine = []
        jc = self.j_centre

        probes = [(0, self.j_centre), (0, self.j_top), (0, self.j_bottom),
                  (self.i_rim, self.j_centre), (0, self.j_support)]
        mm = self.mask_sample
        V_mm = self.V[mm]
        V_tot = V_mm.sum()

        for n in range(nstep + 1):
            if stop is not None and stop.is_set():
                break
            T_centre_fine.append(T[0, jc])
            if n % every == 0 or n == nstep:
                ts.append(n * dt)
                Tp.append([T[i, j] for i, j in probes])
                Tmed.append(float((T[mm] * V_mm).sum() / V_tot))
                snaps.append(T.copy())
                if cb and n % (every * 4) == 0:
                    cb(n / nstep)
            if n == nstep:
                break

            if every_prop and n % every_prop == 0 and n > 0:
                k, eps = self._fields(T)
                Gr, Gz = conductances(k)
                G_half[surf] = k[surf] * A_exp[surf] / d_ef[surf]
                hc_s = np.where(surf, k / d_ef, 0.0)

            Te = T_env(n * dt)
            q = np.zeros((nr, nz))
            f = Gr * (T[1:, :] - T[:-1, :])
            q[:-1, :] += f
            q[1:, :] -= f
            f = Gz * (T[:, 1:] - T[:, :-1])
            q[:, :-1] += f
            q[:, 1:] -= f

            # Wall temperature, by Newton-Raphson on the surface balance
            #     k/d·(Tc − Ts) = h·(Ts − Te) + ε·σ·(Ts⁴ − Te⁴)
            # Applying the flux at the cell-centre temperature instead
            # overstates the driving difference and biased the heating time
            # by about 1.4 %. Three iterations are enough: the previous
            # step's Ts is an excellent initial guess.
            Tc = np.where(surf, T, Te)
            for _ in range(3):
                fn = (hc_s * (Tc - Ts) - h_c * (Ts - Te)
                      - eps * SIG * (Ts ** 4 - Te ** 4))
                fp = -hc_s - h_c - 4 * eps * SIG * Ts ** 3
                Ts = np.where(surf, Ts - fn / fp, Te)
            q += G_half * (Ts - T) * surf

            # The advance is on ENTHALPY, not on temperature. See Material.
            u = u + dt * q / V
            T = self._T_of_u(u)
            T[~act] = 0.0

        tcf = np.array(T_centre_fine)
        v_fine = np.gradient(tcf, dt) if len(tcf) > 2 else np.zeros_like(tcf)
        t_fine = np.arange(len(tcf)) * dt
        ts = np.array(ts)
        return dict(t=ts, T=np.array(Tp), snaps=snaps, dt=dt,
                    T_mean=np.array(Tmed),
                    v_centre=np.interp(ts, t_fine, v_fine),
                    v_max=float(v_fine.max()) if len(v_fine) else 0.0,
                    T_env=np.array([T_env(x) for x in ts]), model=self)


# --------------------------------------------------------------------------
# Compact formula (lumped model with radiation)
# --------------------------------------------------------------------------
def compact_formula(p, t_end):
    """Plate of half-thickness L = e/2 radiating from both faces, with the
    internal resistance L/3k in series.

    This is the back-of-the-envelope version, kept so the full solution can
    be checked against something that fits on one line. It assumes a
    quasi-steady parabolic profile through the thickness, which is where
    the 1/3 and the 1/6 come from.

    It holds while the disc is thin: the error is 0.1 % at e/d = 0.075 and
    10 % at e/d = 0.15, so read it as good below e/d ≈ 0.10 and indicative
    beyond. It has no support in it at all, so it represents the
    point-contact case, not a pellet sitting on a setter.
    """
    k_, cp_, eps_, rho_ = _representative_sample(p)
    L = p["e_sample"] / 2
    Rint = L / (3 * k_)
    C = rho_ * cp_ * L
    T_h, h_c = p["T_furnace"], p["h_conv"]

    def flux(Tb):
        """Surface flux for a given mean temperature.

        The wall temperature is not known in advance, so it is relaxed:
        damped fixed-point iteration, half old and half new. Undamped it
        oscillates, because the T⁴ term is stiff.
        """
        Tw = Tb
        for _ in range(60):
            q = eps_ * SIG * (T_h ** 4 - Tw ** 4) + h_c * (T_h - Tw)
            Tw = 0.5 * Tw + 0.5 * (Tb + q * Rint)
        return eps_ * SIG * (T_h ** 4 - Tw ** 4) + h_c * (T_h - Tw)

    n = 3000
    t = np.linspace(0, t_end, n)
    dt = t[1] - t[0]
    Tm = np.empty(n)
    Tm[0] = p["T_initial"]
    for i in range(1, n):                      # Heun, predictor-corrector
        q1 = flux(Tm[i - 1])
        pred = Tm[i - 1] + dt * q1 / C
        Tm[i] = Tm[i - 1] + 0.5 * dt * (q1 + flux(pred)) / C
    q = np.array([flux(x) for x in Tm])
    return t, Tm - q * L / (6 * k_)           # mean back to centre


def _representative_sample(p):
    """Single representative values for the sample over the run.

    A property given as f(T) is averaged between the initial temperature
    and the setpoint. The dimensionless numbers and the compact formula are
    estimates by nature and cannot carry a varying property, so they get
    one number each rather than being silently denied to anyone using
    measured properties.
    """
    m = p["sample"]
    T0, T1 = p["T_initial"], p["T_furnace"]

    def rep(x):
        return x.representative(T0, T1) if is_property(x) else float(x)

    return rep(m["k"]), rep(m["cp"]), rep(m["eps"]), float(m["rho"])


def dimensionless(p):
    """Radiative coefficient, total h, Biot number and time constant.

    h_rad comes from factorising T⁴ − Te⁴ exactly, evaluated at the mean of
    the initial and furnace temperatures. Bi says whether the pellet is
    thermally thin (below ~0.1 the centre barely lags the surface) and τ
    sets the scale of the whole transient.
    """
    k_, cp_, eps_, rho_ = _representative_sample(p)
    L = p["e_sample"] / 2
    Tm = 0.5 * (p["T_initial"] + p["T_furnace"])
    h_rad = eps_ * SIG * (p["T_furnace"] ** 2 + Tm ** 2) * (p["T_furnace"] + Tm)
    h = h_rad + p["h_conv"]
    Bi = h * L / k_
    tau = rho_ * cp_ * L / h
    return h_rad, h, Bi, tau


def t_fraction(t, T, frac, T0, Tinf):
    """Time to reach a given fraction of the total temperature rise.

    Returns nan rather than the end of the window if the target was never
    reached, so a run that fell short is visible instead of being reported
    as a suspiciously round number.
    """
    target = T0 + frac * (Tinf - T0)
    if len(T) == 0 or T[-1] < target:
        return float("nan")
    return float(np.interp(target, T, t))


# --------------------------------------------------------------------------
# Embedded documentation
# --------------------------------------------------------------------------
# Markup: "## " heading, "> " formula (monospaced), "- " bullet, "" blank.
DOCS = [
("Overview", """
## What this simulator solves

A cylindrical ceramic pellet, initially at uniform temperature, is placed in
a furnace already stabilised at its setpoint. The simulator computes how the
temperature evolves throughout the body and, in particular, at its coldest
point: the geometric centre.

The question it answers is how long one must wait, from the moment the
furnace is closed, until the centre of the sample reaches the sintering
temperature. That interval is what separates the programmed thermal cycle
from the one the material actually experiences.

## Geometry

The domain is a cylinder of radius R and thickness e, resting on a coaxial
support of radius Rs and thickness es. Neither the geometry nor the boundary
conditions depend on the azimuthal angle, so the problem is two-dimensional
in the coordinates (r, z).

The top face and the rim see the furnace walls directly; the bottom face
does not, because the support stands in the way. That asymmetry dominates
the result.
"""),

("Governing equation", """
## From energy conservation to the heat equation

The starting point is the first law applied to a fixed control volume, with
no internal generation, no work and no convective transport within the
solid. The rate of change of stored internal energy equals the net heat flux
through its boundary. Closure comes from Fourier's law, q = -k grad(T).

Fourier's law assumes the energy carriers, phonons in a ceramic insulator,
reach local equilibrium in times far shorter than the process. The phonon
relaxation time in a ceramic is of the order of picoseconds, against seconds
for the transient studied here, so the assumption is excellent and no
hyperbolic formulation is warranted.

In cylindrical coordinates with axial symmetry:

> ρ·cₚ·dT/dt = (1/r)·d/dr( r·k·dT/dr ) + d/dz( k·dT/dz )

The conductivity stays inside the derivatives deliberately: the domain holds
two materials with different k, and taking it out is only legitimate within
each homogeneous region.

## Time scales

Where k is uniform the equation takes its classical form with the thermal
diffusivity α = k/(ρ·cₚ), which sets the time scale of the problem.
Diffusion times scale as L²/α, so for the reference pellet:

> τ_R / τ_L = (R/L)² = (5 / 0.75)² ≈ 44

Radial diffusion is some forty times slower than axial. The pellet therefore
behaves essentially as a plate, with appreciable gradients across the
thickness and a nearly flat field in radius except near the rim.
"""),

("Boundary conditions", """
## Initial condition and symmetry

The piece enters at uniform temperature T0. On the axis, symmetry forces the
radial flux to vanish. This is a geometric consequence, not an assumption: a
non-zero flux at r = 0 would require a line source on the axis.

## Free surfaces

On every surface facing the furnace, the heat arriving by conduction from
within equals what the surface exchanges with its environment by radiation
and convection. It is a balance on a massless surface, with no storage:

> -k·dT/dn = ε·σ·(Th⁴ - Ts⁴) + h·(Th - Ts)

Here Ts is the temperature of the surface itself, not that of the material
just beneath it. The distinction has numerical consequences.

## The radiative term

The radiative term follows from the Stefan-Boltzmann law, corrected by the
emissivity of a real surface. Its form rests on two assumptions: a grey
surface, whose emissivity does not depend on wavelength over the range where
emission matters; and a small body in a large cavity, so that the view
factor is unity and the net exchange reduces to the expression above.

## Why radiation is not linearised

It is common to replace the radiative term by an equivalent coefficient:

> h_rad = ε·σ·(Th + Ts)·(Th² + Ts²)

The factorisation is exact; its use is the problem. h_rad depends on Ts,
which is precisely the unknown. Between 100 and 1200 degrees C it goes from
about 90 to about 580 W/m²K, a factor of six. Freezing it at a mean value
introduces errors of tens of percent early in the transient, which is where
the heating time is decided. The calculation therefore keeps the fourth
power, and h_rad is used only afterwards, to interpret results.

## Interface and furnace programme

At the interface between sample and support, continuity of normal flux and
of temperature are imposed. A thermal contact resistance Rc replaces the
second by a jump proportional to the flux. The furnace itself may follow
either an exponential recovery after the door is opened, or a programmed
ramp at constant rate up to the setpoint.
"""),

("Discretisation", """
## Why finite volumes

Finite volumes are used rather than finite differences. The reason is
conservation, not accuracy. The method integrates the equation over each
cell, so the flux leaving one cell is identically the one entering its
neighbour. Energy is conserved exactly, including across the interface
between materials with very different properties, and at any mesh size.

> ρ·cₚ·V·(Tⁿ⁺¹ − Tⁿ)/Δt = Σ_faces G·(T_neighbour − T) + q·A

G is the conductance between cell centres, in W/K. In cylindrical
coordinates the volumes and areas carry the radius.

## The harmonic mean at the interface

This is where a careless implementation fails. The conductivity to use on
the face shared by two cells of different materials is not the arithmetic
mean of the two conductivities but the harmonic one:

> k_face = 2·k1·k2 / (k1 + k2)

The reason is physical. Two half-cells in series oppose resistances that
add, and resistance is inversely proportional to conductivity. For a
LATP-alumina interface with k = 0.8 and 6 W/m·K, the arithmetic mean gives
3.4 and the harmonic one 1.41. Using the arithmetic mean would overestimate
the thermal coupling to the support by a factor of 2.4.

## Stability

Time advance is explicit, and therefore conditionally stable. The criterion
follows from requiring the coefficient multiplying the cell's own
temperature to stay non-negative:

> dt ≤ ρ·cₚ·V / ( Σ G + A·(h + 4·ε·σ·Th³) )

Its meaning is transparent: a cell cannot give up more energy in one step
than it stores. The code scans every cell, takes the minimum and applies a
40 % margin. With temperature-dependent properties the bound is evaluated at
the most demanding point of the whole thermal path: maximum conductivity,
which maximises the conductances, and minimum heat capacity, which minimises
the stored energy.
"""),

("Surface treatment", """
## The wall temperature

A finite-volume scheme knows the temperature at the centre of each cell, not
at its surface. Evaluating the radiative flux with the nodal temperature
ignores the thermal drop across the half-cell separating that node from the
wall, which biases the result towards faster heating.

The correct treatment solves for the wall temperature Ts by imposing the
surface balance in discrete form:

> (k/d)·(T_node - Ts) = ε·σ·(Ts⁴ - Th⁴) + h·(Ts - Th)

where d is the distance from the cell centre to the wall. The equation is
non-linear in Ts and is solved by Newton-Raphson, whose derivative is
available in closed form:

> df/dTs = -k/d - h - 4·ε·σ·Ts³

Three iterations suffice per time step, because Ts changes very little
between consecutive steps and the previous value is an excellent start.

## How much it matters

This is not cosmetic. Omitting the treatment makes the model overestimate
the heating rate by 1.4 % of the total thermal span, and the bias does not
vanish with mesh refinement at the expected rate, because the structure of
the surface error persists. Solving the wall removes the bias and allows
coarse meshes to be used with confidence, which is what makes the
sensitivity analysis affordable.
"""),

("Temperature-dependent properties", """
## Why enthalpy, not temperature

When the specific heat depends on temperature, the time advance cannot be
carried out on temperature. The heat capacity changes within the time step
itself, so the scheme stops conserving energy and the error accumulates
through the transient.

The model therefore advances on the volumetric enthalpy:

> u(T) = ρ · ∫ cₚ(T') dT'

The balance is written on u, and the temperature is recovered by inverting
u(T), which is monotonically increasing because cₚ > 0:

> V·(uⁿ⁺¹ − uⁿ)/Δt = Σ fluxes       Tⁿ⁺¹ = u⁻¹(uⁿ⁺¹)

Energy is then conserved exactly whatever the variation of cₚ. This has been
verified: the accumulated energy matches the theoretical value to within
10⁻¹⁴ %, that is, to machine precision. The formulation also admits a
latent heat without modification, since that would appear as a step in u(T).

## Validity range

Conductivity and emissivity are re-evaluated periodically through the
calculation, and the conductances with them.

Outside the declared validity range the value is held constant and equal to
that of the corresponding endpoint, rather than extrapolated. This is
deliberate: a fit obtained between 800 and 1000 degrees C, extrapolated down
to 100, can return absurd or negative values, and a negative specific heat
breaks the calculation silently. The program also rejects any function
returning null, negative or non-finite values inside the computational
range.
"""),

("Lumped model", """
## The Biot number

The full calculation gives the answer but gives no criterion. For that, a
reduced form is useful. The quantity deciding whether a body may be treated
as isothermal is the ratio of internal to external resistance:

> Bi = (L/k) / (1/h) = h·L/k        with L = e/2

The classical criterion for lumped capacitance is Bi < 0.1, with an error
below 5 %. The path forward is to keep the lumped equation and add a profile
correction.

## The parabolic profile

Assume the temperature profile across the thickness is quasi-steady, that
is, the whole plate heats at the same rate. With symmetric heating the
conduction equation reduces to a constant second derivative, and integrating
twice with zero flux at the mid-plane gives a parabolic profile. Evaluating
at the surface and averaging over the thickness:

> Ts - Tc = q·L/(2k)      Tm - Tc = q·L/(6k)      Ts - Tm = q·L/(3k)

The coefficients 1/3 and 1/6 are not empirical adjustments: they follow from
integrating a parabola. The first gives the equivalent internal resistance
placed in series with the surface; the second converts the mean temperature
into that of the centre.

## The compact formula

> ρ·cₚ·L·dTm/dt = q
> q  = ε·σ·(Th⁴ - Ts⁴) + h·(Th - Ts)
> Ts = Tm + q·L/(3k)
> Tc = Tm - q·L/(6k)

A single ordinary differential equation with two algebraic corrections,
integrable in any spreadsheet. The governing time constant is
τ = ρ·cₚ·L/h with h = h_conv + h_rad.

## Where it holds

The formula treats the pellet as an infinite plate radiating from both
faces: it ignores the rim and does not contemplate a support. Against the
two-dimensional calculation with constant properties, the deviation stays
below 2 % while e/d < 0.10, and below 5 % while e/d < 0.12. At e/d = 0.15 it
already reaches 10 %.

With a support the formula does not apply at all, and with strongly
temperature-dependent properties the discrepancy grows further, since the
reduction presumes constant properties. In those cases it serves only as an
instrument of reasoning, not of calculation.
"""),

("Rate and stress", """
## Heating rate

The fast-firing literature is expressed in K/min or K/s, so the rate at the
coldest point is the quantity that lets an experiment be placed against the
bibliography.

A numerical caution: if this derivative is evaluated on the curve stored for
plotting, the peak value depends on the sampling interval and is
appreciably underestimated. The derivative must be computed on the
integrator's own time step, and that is what the program does.

## Thermal stress

A body with a thermal gradient expands as a whole. Each point is put in
tension or compression according to how it departs from the mean strain, so
the driver of stress is not the difference between face and centre but
between each point and the mean temperature of the piece:

> σ(x) = E·α/(1 - ν) · ( Tm - T(x) )

The factor 1/(1 - ν) appears because expansion is restrained in both
in-plane directions, not just one. During heating the centre lags the mean
and goes into tension, which is how ceramics fail, while the surface is
compressed.

The value should be read as an order of magnitude. In the green state
Young's modulus is far below that of the dense piece, so the real stress is
lower than computed with dense properties, but so is the strength. The
usefulness of the calculation lies in comparing set-ups, not in predicting
fracture.
"""),

("Validation", """
## Against the analytical solution

The test case is the infinite flat plate with surface convection, whose
exact series solution is known. The eigenvalues are the roots of the
transcendental equation imposed by the convective boundary condition, and
the coefficients follow from the orthogonality of the modes:

> λ·tan(λ) = Bi        Fo = α·t/L²

The test runs at Bi = 1, with radiation disabled and a cylinder wide enough
for the axis to behave as a plate. Summing twenty terms of the series, the
maximum error for Fo > 0.05 is:

- 10 cells across the half-thickness: 0.338 %
- 20 cells: 0.084 %
- 40 cells: 0.021 %
- 80 cells: 0.0053 %

The error is divided by exactly four each time the cell count doubles. That
factor is the signature of a second-order scheme in space, which is what a
centred finite-volume discretisation should deliver. The observed
convergence confirms the implementation hides no first-order errors, which
no single verified calculation could ever demonstrate.

## Mesh independence

Verifying convergence against an exact solution is not enough, because the
real case involves two materials and a radiative condition. Four
configurations repeated at three resolutions, from 8 to 22 cells across the
thickness, show a scatter of 0.2 to 4 %. The coarsest mesh remains adequate
for exploring trends.

## What this does not prove

Validation confirms the equations posed are solved correctly. It does not
confirm they are the right equations for a particular furnace. That requires
a measurement: a fine thermocouple embedded in a sacrificial pellet,
recording the introduction into the furnace. Such a measurement calibrates
the effective emissivity, the radiative geometry and the furnace recovery
all at once.
"""),

("Sensitivity analysis", """
## Elasticity

A model with seventeen input variables is of little use without knowing
which ones govern the result. The right indicator for comparing them is not
the partial derivative, whose units depend on the variable, but the
elasticity, the derivative on a logarithmic scale and therefore
dimensionless:

> S = d(ln t99) / d(ln x) = (dt99/t99) / (dx/x)

S = 1 means proportionality, so doubling the variable doubles the time;
S = 0 means no influence; a negative value means the time falls as the
variable grows. Equivalently, doubling a variable multiplies the time by
2ˢ.

## A methodological caution

The perturbation used is +-22 %, deliberately wide. With small
perturbations, of the order of +-5 %, the weak variables are buried in
discretisation noise: changing a thickness changes the number of cells
representing the support, and since the support mass has an elasticity near
0.6, that rounding introduces jumps of about 1 % in t99, the same order as
the signal sought. The elasticities reported are therefore secant over a
finite interval, not point derivatives.

## What governs the result

With the measured properties, the support and the furnace dominate, but the
sample is no longer irrelevant. Density and specific heat of each material
return exactly the same elasticity, because they act only through their
product, the volumetric heat capacity. That is the model confirming its own
structure.

Three negative results spare useless work: the thermal contact resistance is
irrelevant over the whole realistic range; convection does not count, since
at 1200 degrees C the radiative coefficient is some twenty-six times the
convective one; and the conductivity of both materials barely matters when
the Biot number is low.
"""),

("Assumptions and limits", """
## What the model does account for

- Simultaneous gradients across thickness and radius.
- Two different materials in contact, each with its own properties.
- Thermal conductivity, specific heat and emissivity as functions of T.
- Radiation in its non-linear form.
- The thermal recovery dynamics of the furnace.
- A thermal contact resistance at the interface, when declared.

## The principal limitation: radiative geometry

The assumption of unit view factor is the model's serious weakness. The
support protrudes several millimetres beyond the rim of the pellet, so it
shades that rim at grazing angles, and the rim partly sees the cooler top
surface of the support instead of the furnace walls. The rim is no
geometric detail: for a 10 mm diameter and 1.5 mm thickness it accounts for
37 % of the exposed area.

> A_rim / A_total = 2·π·R·e / (π·R² + 2·π·R·e) = e / (R/2 + e)

The model heats that rim too much. A rigorous solution requires a radiosity
method with view factors between the four surfaces involved. It is the only
modelling change capable of moving the result by more than 10 %.

## Other limitations

- Base of the support. The model assumes the lower face radiates to the
  furnace. If it rests on the hearth it is conducting into a far larger
  mass, and the thermal sink is considerably greater.
- Opacity. At 1200 degrees C the emission peak falls at 2 micrometres, where
  many oxides are partly transparent. Strong scattering in a porous body
  limits penetration, but the effect has not been quantified.
- Furnace programme. The drop and recovery parameters are estimates. They
  matter little for the total time, but they govern the early part of the
  curve entirely.
- Sintering heat, shrinkage and the coupling between densification and
  conductivity are not included.
"""),

("Material properties", """
## LATP

> k(T)   = -2.016e-17·T⁶ + 1.278e-13·T⁵ - 3.236e-10·T⁴
>          + 4.090e-7·T³ - 2.673e-4·T² + 8.585e-2·T - 9.700
> cₚ(T)  = -6.414e-14·T⁶ + 3.680e-10·T⁵ - 8.507e-7·T⁴
>          + 9.983e-4·T³ - 6.179e-1·T² + 1.899e2·T - 2.091e4
> ε(T) = 5.600e-4·T + 1.210e-1
> α(T) = 9.774e-11·T + 1.486e-7

Emissivity and specific heat come from own measurements: the former with a
pyrometer in the laboratory, the latter as reported in the fast-firing
article. The conductivity is not an independent measurement: it is obtained
from the linearisation of the measured diffusivity values and a subsequent
fit to a sixth-order polynomial.

## Alumina

> k(T)   = 1.912e-5·T² - 5.025e-2·T + 3.866e1
> cₚ(T)  = 9.052e2 + 3.678e-1·T - 2.142e-3/T²
> ε(T) = 7.314e-4·T - 1.774e-1

Specific heat from Tablas ceramicas (Enrique Navarro, 1984) and conductivity
from the CRC Handbook of Materials Science and Engineering. Emissivity
measured in the laboratory with a pyrometer, as for the LATP.

## Internal consistency

The three measured LATP functions are not independent, since conductivity
was derived from diffusivity. That dependence allows a check: if they are
mutually consistent, the ratio k/(α·cₚ) must be constant. Evaluated
between 400 and 1200 K it gives 2990 kg/m³ with a 3 % scatter, a value
consistent with dense LATP. That is the density used by the model, and it is
not an independent datum.

## Two reservations about the data

The specific heat of LATP exceeds the classical limit. The maximum value the
specific heat of a crystalline solid can reach through thermal agitation is
set by the Dulong-Petit law, 3R per mole of atoms:

> cp_max = 3·R·n_atoms / M = 3·8.314·18.3 / 0.3834 ≈ 1190 J/kg·K

The fit used returns between 1.4 and 2.2 times that limit across the working
range, peaking at 2598 J/kg·K at 1073 K. A sensible specific heat cannot
exceed that ceiling. The reading compatible with the datum is that it is an
apparent specific heat, obtained by differential calorimetry during heating,
incorporating the enthalpy of the transformations occurring over that
interval. If so the datum is legitimate and the model handles it correctly,
since the enthalpy formulation absorbs any latent heat without
modification.

The T⁻² term in the specific heat of alumina is numerically null. With the
coefficient 2.142e-3 that term is worth −1.5·10⁻⁸ at 373 K, ten orders of
magnitude below the rest of the expression. Maier–Kelley fits for alumina
carry a coefficient of order 10⁷ in that position; with it, the specific
heat at 373 K would go from 1042 to 888 J/kg·K, the commonly tabulated
value. The exponent should be checked against the original source.
"""),

("Nomenclature", """
## Symbols

- T: temperature, K
- Tc, Ts, Tm: centre, wall and mean temperature, K
- Th: furnace wall temperature, K
- T0: initial sample temperature, K
- e, L: thickness and half-thickness, m
- R, Rs: sample and support radius, m
- k: thermal conductivity, W/m·K
- ρ: density, kg/m³
- cₚ: specific heat, J/kg·K
- α: thermal diffusivity, k/(ρ·cₚ), m²/s
- ε: surface emissivity, dimensionless
- σ: Stefan-Boltzmann constant, 5.670·10⁻⁸ W/m²K⁴
- h: convection coefficient, W/m²K
- h_rad: equivalent radiative coefficient, W/m²K
- Rc: thermal contact resistance, m²K/W
- E, ν: Young's modulus and Poisson's ratio
- Bi: Biot number, h·L/k
- Fo: Fourier number, α·t/L²
- τ: time constant, ρ·cₚ·L/h, s
- q: heat flux per unit area, W/m²
- G: conductance between neighbouring cells, W/K
- t99: time to 99 % of the thermal span, s

## A note on diffusivity

Diffusivity is an output, not an input. It cannot be the input either: it is
a ratio, and forming it discards the absolute scale. Three materials with
identical α but conductivities spanning two orders of magnitude give
heating times differing by a factor of thirty, because the boundary
condition is a flux balance in which k appears on its own. The two
independent quantities of the problem are k and ρ·cₚ.
"""),

("References", """
## Heat transfer

- F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine. Fundamentals
  of Heat and Mass Transfer, 7th ed. Wiley, 2011. Heat diffusion equation in
  cylindrical coordinates, thermal contact resistance, lumped capacitance,
  Biot number, series solution, explicit schemes and their stability
  criterion.
- H. S. Carslaw, J. C. Jaeger. Conduction of Heat in Solids, 2nd ed. Oxford
  University Press, 1959. Analytical series solution used as the validation
  benchmark.
- M. N. Ozisik. Heat Conduction, 2nd ed. Wiley, 1993. Approximate methods
  and the heat balance integral, the basis of the parabolic profile
  correction.
- A. F. Mills. Heat and Mass Transfer. Irwin, 1995.

## Numerical methods

- S. V. Patankar. Numerical Heat Transfer and Fluid Flow. Hemisphere, 1980.
  Finite volume method, harmonic mean interface conductivity, explicit
  scheme stability.
- H. K. Versteeg, W. Malalasekera. An Introduction to Computational Fluid
  Dynamics: The Finite Volume Method, 2nd ed. Pearson, 2007.

## Radiation

- M. F. Modest. Radiative Heat Transfer, 3rd ed. Academic Press, 2013.
  Exchange between grey surfaces; small body in a large cavity.
- R. Siegel, J. R. Howell. Thermal Radiation Heat Transfer, 4th ed. Taylor
  and Francis, 2002.

## Material data

- Tablas ceramicas, Enrique Navarro, 1984. Specific heat of alumina.
- CRC Handbook of Materials Science and Engineering. Thermal conductivity
  of alumina.
"""),
]


# --------------------------------------------------------------------------
# Visual identity
# --------------------------------------------------------------------------
PAL = dict(
    bg="#eef1f4", card="#ffffff", ink="#10212b", muted="#5b7183",
    line="#d4dce3", accent="#13507e", accent_dk="#0d3c60", head="#16303f",
    centre="#c0392b", top="#e07b1d", bottom="#1f6fb2", edge="#2e8b57",
    support="#7c8d9c", oven="#394b59", warn="#b03a2e",
)

SERIES = [PAL["centre"], PAL["top"], PAL["bottom"], PAL["edge"], PAL["support"]]


def _matplotlib_style():
    """Make the plots look like part of the window rather than guests.

    Same background as the cards they sit on, same ink, top and right
    spines dropped.
    """
    matplotlib.rcParams.update({
        "figure.facecolor": PAL["card"], "axes.facecolor": PAL["card"],
        "axes.edgecolor": PAL["line"], "axes.labelcolor": PAL["ink"],
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": PAL["ink"], "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": PAL["muted"], "ytick.color": PAL["muted"],
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": PAL["line"], "grid.alpha": 0.9, "axes.grid": True,
        "legend.fontsize": 8, "legend.frameon": True,
        "legend.edgecolor": PAL["line"], "legend.framealpha": 0.95,
        "font.size": 9,
    })


# --------------------------------------------------------------------------
# Property dialog
# --------------------------------------------------------------------------
class PropertyDialog(tk.Toplevel):
    """Asks, property by property, whether its variation with temperature is
    available. If it is, a function of T in kelvin is entered; otherwise the
    mean value from the input panel is kept."""

    FIELDS = [("k", "Thermal conductivity  k", "W/m·K"),
              ("cp", "Specific heat  cₚ", "J/kg·K"),
              ("eps", "Emissivity  ε", "–")]

    def __init__(self, parent, title, current, constants, T_range):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=PAL["bg"])
        self.resizable(False, False)
        self.result = None
        self.T_range = T_range
        self.v = {}

        head = tk.Frame(self, bg=PAL["head"])
        head.pack(fill="x")
        tk.Label(head, text=title, bg=PAL["head"], fg="#ffffff",
                 font=("Segoe UI", 11, "bold"), anchor="w",
                 padx=14, pady=9).pack(fill="x")

        intro = ttk.Frame(self, padding=(14, 10, 14, 2))
        intro.pack(fill="x")
        ttk.Label(intro, text="Is the temperature dependence available?",
                  style="H2.TLabel").pack(anchor="w")
        ttk.Label(intro, style="Hint.TLabel", justify="left", wraplength=560,
                  text="Tick the box and write the function of T, with T in "
                       "kelvin — for example  1046 + 0.1742*T - 2.796e7/T**2. "
                       "The functions exp, log, log10, sqrt, tanh and abs are "
                       "available. If left unticked, the constant value from "
                       "the input panel is used.\nOutside the validity range "
                       "the value is held at the endpoint; leave the range "
                       "empty if the fit covers the whole path."
                  ).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(self, padding=(14, 8))
        body.pack(fill="both", expand=True)
        for row, (key, label, unit) in enumerate(self.FIELDS):
            g = ttk.Labelframe(body, text=f"  {label}   [{unit}]  ",
                               padding=(10, 8))
            g.grid(row=row, column=0, sticky="ew", pady=5)
            body.columnconfigure(0, weight=1)
            cur = current.get(key)
            use = tk.BooleanVar(value=cur is not None)
            expr = tk.StringVar(value=cur.expr if cur else "")
            tmin = tk.StringVar(value=("" if not cur or cur.T_min is None
                                       else f"{cur.T_min:g}"))
            tmax = tk.StringVar(value=("" if not cur or cur.T_max is None
                                       else f"{cur.T_max:g}"))
            info = tk.StringVar(value=f"constant = {constants.get(key, 0):g}")

            ttk.Checkbutton(g, text="Use a function of T (K)", variable=use,
                            command=lambda c=key: self._toggle(c)
                            ).grid(row=0, column=0, columnspan=6, sticky="w")
            ent = ttk.Entry(g, textvariable=expr, width=58,
                            font=("Consolas", 9))
            ent.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(5, 3))
            ttk.Label(g, text="valid from").grid(row=2, column=0, sticky="w")
            e1 = ttk.Entry(g, textvariable=tmin, width=7, justify="right")
            e1.grid(row=2, column=1, sticky="w", padx=3)
            ttk.Label(g, text="to").grid(row=2, column=2)
            e2 = ttk.Entry(g, textvariable=tmax, width=7, justify="right")
            e2.grid(row=2, column=3, sticky="w", padx=3)
            ttk.Label(g, text="K", style="Hint.TLabel").grid(row=2, column=4,
                                                             padx=(2, 12))
            ttk.Button(g, text="Check",
                       command=lambda c=key: self._check(c)
                       ).grid(row=2, column=5, sticky="e")
            ttk.Label(g, textvariable=info, style="Mono.TLabel"
                      ).grid(row=3, column=0, columnspan=6, sticky="w",
                             pady=(6, 0))
            g.columnconfigure(5, weight=1)
            self.v[key] = dict(use=use, expr=expr, tmin=tmin, tmax=tmax,
                               info=info, widgets=[ent, e1, e2])
            self._toggle(key)

        foot = ttk.Frame(self, padding=(14, 4, 14, 14))
        foot.pack(fill="x")
        ttk.Button(foot, text="Apply", style="Accent.TButton",
                   command=self._accept).pack(side="right")
        ttk.Button(foot, text="Cancel", command=self.destroy).pack(
            side="right", padx=8)

        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def _toggle(self, key):
        d = self.v[key]
        st = "normal" if d["use"].get() else "disabled"
        for w in d["widgets"]:
            w.config(state=st)

    def _build(self, key, name):
        d = self.v[key]
        if not d["use"].get():
            return None

        def lim(s):
            s = s.get().strip().replace(",", ".")
            return float(s) if s else None

        return Property(expr=d["expr"].get(), T_min=lim(d["tmin"]),
                        T_max=lim(d["tmax"]), name=name)

    def _check(self, key):
        d = self.v[key]
        try:
            pr = self._build(key, key)
            if pr is None:
                d["info"].set("the constant value will be used")
                return
            pr.grid(zero_ok=(key == "eps"))
            T0, T1 = self.T_range
            Ts = np.linspace(T0, T1, 5)
            vals = pr.evaluate(Ts)
            d["info"].set("   ".join(f"{t:.0f} K: {v:.4g}"
                                     for t, v in zip(Ts, vals)))
        except ValueError as err:
            d["info"].set(f"⚠  {err}")

    def _accept(self):
        out = {}
        for key, label, _ in self.FIELDS:
            try:
                pr = self._build(key, label)
                if pr is not None:
                    pr.grid(zero_ok=(key == "eps"))
                out[key] = pr
            except ValueError as err:
                messagebox.showerror("Invalid function", str(err), parent=self)
                return
        self.result = out
        self.destroy()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class App(tk.Tk):
    """The main window.

    Inputs on the left, results on the right. The solve runs in a worker
    thread and talks back through a queue that _pump drains on the Tk event
    loop — Tk is not thread-safe, so no widget is ever touched from the
    worker. That is also what makes the Cancel button and the progress bar
    possible: the window keeps responding while a long run is in progress.
    """

    def __init__(self):
        super().__init__()
        self.title("Ceramic Pellet Heating Simulator")
        self.geometry("1360x900")
        self.minsize(1180, 760)
        self.configure(bg=PAL["bg"])

        self.res = None
        self.queue_ui = queue.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.cbar = None
        self.props = {"sample": {"k": None, "cp": None, "eps": None},
                      "support": {"k": None, "cp": None, "eps": None}}

        _matplotlib_style()
        self._styles()
        self._build()
        self.after(120, self._pump)

    # ----------------------------------------------------------- styling
    def _styles(self):
        s = ttk.Style(self)
        if "clam" in s.theme_names():
            s.theme_use("clam")
        base = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"
        mono = "Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono"
        self.font_base = (base, 9)
        self.font_mono = (mono, 9)

        s.configure(".", background=PAL["bg"], foreground=PAL["ink"],
                    font=self.font_base)
        s.configure("TFrame", background=PAL["bg"])
        s.configure("Card.TFrame", background=PAL["card"])
        s.configure("TLabel", background=PAL["bg"], foreground=PAL["ink"])
        s.configure("H1.TLabel", font=(base, 13, "bold"), foreground=PAL["ink"])
        s.configure("H2.TLabel", font=(base, 10, "bold"), foreground=PAL["ink"])
        s.configure("Hint.TLabel", font=(base, 8), foreground=PAL["muted"])
        s.configure("Mono.TLabel", font=(mono, 9), foreground=PAL["ink"])
        s.configure("Unit.TLabel", font=(base, 8), foreground=PAL["muted"])
        s.configure("Status.TLabel", font=(base, 9), foreground=PAL["muted"])

        s.configure("TLabelframe", background=PAL["bg"],
                    bordercolor=PAL["line"], relief="solid", borderwidth=1)
        s.configure("TLabelframe.Label", background=PAL["bg"],
                    foreground=PAL["accent"], font=(base, 9, "bold"))

        s.configure("TNotebook", background=PAL["bg"], borderwidth=0,
                    tabmargins=(2, 6, 2, 0))
        s.configure("TNotebook.Tab", padding=(14, 7), font=(base, 9),
                    background="#dfe6ec", foreground=PAL["muted"],
                    borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", PAL["card"])],
              foreground=[("selected", PAL["accent"])],
              font=[("selected", (base, 9, "bold"))])

        s.configure("Side.TNotebook", background=PAL["bg"], borderwidth=0,
                    tabmargins=(1, 5, 1, 0))
        s.configure("Side.TNotebook.Tab", padding=(9, 6), font=(base, 9),
                    background="#dfe6ec", foreground=PAL["muted"],
                    borderwidth=0)
        s.map("Side.TNotebook.Tab",
              background=[("selected", PAL["card"])],
              foreground=[("selected", PAL["accent"])],
              font=[("selected", (base, 9, "bold"))])

        s.configure("TEntry", fieldbackground=PAL["card"],
                    bordercolor=PAL["line"], insertcolor=PAL["ink"])
        s.configure("TCombobox", fieldbackground=PAL["card"],
                    background=PAL["card"], bordercolor=PAL["line"])
        s.configure("TButton", padding=(10, 5), font=(base, 9),
                    background="#dfe6ec", bordercolor=PAL["line"])
        s.map("TButton", background=[("active", "#cfd9e2")])
        s.configure("Accent.TButton", padding=(10, 7),
                    font=(base, 10, "bold"), foreground="#ffffff",
                    background=PAL["accent"], bordercolor=PAL["accent"])
        s.map("Accent.TButton",
              background=[("active", PAL["accent_dk"]),
                          ("disabled", "#9fb3c4")])
        s.configure("TCheckbutton", background=PAL["bg"])
        s.configure("Horizontal.TProgressbar", background=PAL["accent"],
                    troughcolor="#dfe6ec", bordercolor=PAL["line"],
                    lightcolor=PAL["accent"], darkcolor=PAL["accent"])
        s.configure("Treeview", fieldbackground=PAL["card"],
                    background=PAL["card"], borderwidth=0, rowheight=24)
        s.map("Treeview", background=[("selected", PAL["accent"])],
              foreground=[("selected", "#ffffff")])

    # ------------------------------------------------------------ layout
    def _build(self):
        head = tk.Frame(self, bg=PAL["head"], height=56)
        head.pack(fill="x")
        head.pack_propagate(False)
        inner = tk.Frame(head, bg=PAL["head"])
        inner.pack(side="left", padx=18)
        tk.Label(inner, text="Ceramic Pellet Heating Simulator",
                 bg=PAL["head"], fg="#ffffff",
                 font=(self.font_base[0], 13, "bold")).pack(anchor="w",
                                                            pady=(9, 0))
        tk.Label(inner, text="Transient conduction with radiative and "
                             "convective boundaries  ·  axisymmetric finite "
                             "volumes", bg=PAL["head"], fg="#9fc2db",
                 font=(self.font_base[0], 8)).pack(anchor="w")

        # The status bar must be packed BEFORE the expanding body: pack
        # allocates in call order, so an expanding widget packed first
        # claims all the remaining space and leaves the bar invisible.
        bar = tk.Frame(self, bg=PAL["bg"])
        bar.pack(fill="x", side="bottom")
        ttk.Separator(bar).pack(fill="x")
        self.v_status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.v_status, style="Status.TLabel",
                  anchor="w").pack(side="left", padx=16, pady=5)
        ttk.Label(bar, text=f"v{VERSION}", style="Hint.TLabel").pack(
            side="right", padx=16)

        body = ttk.Frame(self, padding=(12, 10, 12, 6))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=318)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._inputs(left)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._outputs(right)

    def _field(self, parent, row, label, value, unit=""):
        lab = ttk.Label(parent, text=label)
        lab.grid(row=row, column=0, sticky="w", padx=(2, 8), pady=3)
        v = tk.StringVar(value=str(value))
        ent = ttk.Entry(parent, textvariable=v, width=10, justify="right")
        ent.grid(row=row, column=1, sticky="ew", pady=3)
        uni = ttk.Label(parent, text=unit, style="Unit.TLabel")
        uni.grid(row=row, column=2, sticky="w", padx=(5, 2))
        if not hasattr(self, "_wid"):
            self._wid = {}
        self._wid[str(v)] = (lab, ent, uni)
        return v

    def _show_field(self, var, visible):
        for w in self._wid.get(str(var), ()):
            try:
                w.grid() if visible else w.grid_remove()
            except tk.TclError:
                pass

    def _inputs(self, p):
        nb = ttk.Notebook(p, style="Side.TNotebook")
        nb.pack(fill="both", expand=True)
        self.nb_in = nb

        # ---- sample
        f1 = ttk.Frame(nb, padding=10)
        nb.add(f1, text="Sample")
        ttk.Label(f1, text="Material", style="H2.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.v_mat = tk.StringVar(value="LATP, measured f(T)")
        cb = ttk.Combobox(f1, textvariable=self.v_mat, values=list(MATERIALS),
                          state="readonly", width=25)
        cb.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        cb.bind("<<ComboboxSelected>>", self._load_material)
        self.v_d = self._field(f1, 2, "Diameter", 10.0, "mm")
        self.v_e = self._field(f1, 3, "Thickness", 1.5, "mm")
        ttk.Separator(f1).grid(row=4, column=0, columnspan=3, sticky="ew",
                               pady=8)
        self.v_k = self._field(f1, 5, "Conductivity  k", 0.8, "W/m·K")
        self.v_rho = self._field(f1, 6, "Density  ρ", 1900.0, "kg/m³")
        self.v_cp = self._field(f1, 7, "Specific heat  cₚ", 950.0, "J/kg·K")
        self.v_eps = self._field(f1, 8, "Emissivity  ε", 0.85, "–")
        ttk.Button(f1, text="Temperature dependence …",
                   command=lambda: self._props_dialog("sample")).grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.v_fT_m = tk.StringVar(value="all constant")
        ttk.Label(f1, textvariable=self.v_fT_m, style="Hint.TLabel",
                  wraplength=250, justify="left").grid(
            row=10, column=0, columnspan=3, sticky="w")
        ttk.Separator(f1).grid(row=11, column=0, columnspan=3, sticky="ew",
                               pady=8)
        ttk.Label(f1, text="Mechanical properties", style="H2.TLabel").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.v_E = self._field(f1, 13, "Young's modulus  E", 100.0, "GPa")
        self.v_alpha = self._field(f1, 14, "Expansion  α", 10.0,
                                   "10⁻⁶/K")
        self.v_nu = self._field(f1, 15, "Poisson  ν", 0.25, "–")
        ttk.Label(f1, style="Hint.TLabel", wraplength=250, justify="left",
                  text="Used only to estimate thermal stress. In the green "
                       "state E is far below the dense value."
                  ).grid(row=16, column=0, columnspan=3, sticky="w",
                         pady=(6, 0))
        f1.columnconfigure(1, weight=1)

        # ---- support
        f2 = ttk.Frame(nb, padding=10)
        nb.add(f2, text="Support")
        self.v_use_support = tk.BooleanVar(value=True)
        ttk.Checkbutton(f2, text="Sample resting on a support",
                        variable=self.v_use_support, command=self._toggle_support
                        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(f2, style="Hint.TLabel", wraplength=250, justify="left",
                  text="If unticked, the sample radiates from both faces "
                       "(point supports)."
                  ).grid(row=1, column=0, columnspan=3, sticky="w",
                         pady=(0, 8))
        self.v_mat_support = tk.StringVar(value="Alumina, measured f(T)")
        self.cb_support = ttk.Combobox(f2, textvariable=self.v_mat_support,
                                       values=list(SUPPORTS),
                                       state="readonly", width=25)
        self.cb_support.grid(row=2, column=0, columnspan=3, sticky="ew",
                             pady=(0, 8))
        self.cb_support.bind("<<ComboboxSelected>>", self._load_support)
        self.v_e_support = self._field(f2, 3, "Thickness", 2.0, "mm")
        self.v_r_support = self._field(f2, 4, "Radius", 10.0, "mm")
        ttk.Separator(f2).grid(row=5, column=0, columnspan=3, sticky="ew",
                               pady=8)
        self.v_k_support = self._field(f2, 6, "Conductivity  k", 6.0, "W/m·K")
        self.v_rho_support = self._field(f2, 7, "Density  ρ", 3900.0,
                                         "kg/m³")
        self.v_cp_support = self._field(f2, 8, "Specific heat  cₚ", 1200.0,
                                        "J/kg·K")
        self.v_eps_support = self._field(f2, 9, "Emissivity  ε", 0.70,
                                         "–")
        self.v_Rc = self._field(f2, 10, "Contact resist.  Rc", 0.0,
                                "m²K/W")
        ttk.Button(f2, text="Temperature dependence …",
                   command=lambda: self._props_dialog("support")).grid(
            row=11, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        self.v_fT_s = tk.StringVar(value="all constant")
        ttk.Label(f2, textvariable=self.v_fT_s, style="Hint.TLabel",
                  wraplength=250, justify="left").grid(
            row=12, column=0, columnspan=3, sticky="w")
        f2.columnconfigure(1, weight=1)

        # ---- furnace
        f3 = ttk.Frame(nb, padding=10)
        nb.add(f3, text="Furnace")
        ttk.Label(f3, text="Thermal programme", style="H2.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.v_programme = tk.StringVar(value="Stabilised furnace (door opened)")
        cbp = ttk.Combobox(f3, textvariable=self.v_programme, values=list(PROGRAMMES),
                           state="readonly", width=25)
        cbp.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        cbp.bind("<<ComboboxSelected>>", lambda e: self._toggle_programme())
        self.v_T_set = self._field(f3, 2, "Setpoint", 1200.0, "°C")
        self.v_T0 = self._field(f3, 3, "Sample initial T", 100.0, "°C")
        ttk.Separator(f3).grid(row=4, column=0, columnspan=3, sticky="ew",
                               pady=8)
        self.lbl_est = ttk.Label(f3, text="Door opening", style="H2.TLabel")
        self.lbl_est.grid(row=5, column=0, columnspan=3, sticky="w",
                          pady=(0, 4))
        self.v_drop = self._field(f3, 6, "Drop  ΔT", 40.0, "°C")
        self.v_tau = self._field(f3, 7, "Recovery  τ", 20.0, "s")
        self.nota_est = ttk.Label(
            f3, style="Hint.TLabel", wraplength=250, justify="left",
            text="Time constant of the exponential recovery. A drop of 0 "
                 "means an ideal furnace.")
        self.nota_est.grid(row=8, column=0, columnspan=3, sticky="w",
                           pady=(0, 4))
        self.v_ramp = self._field(f3, 9, "Ramp rate", 300.0, "°C/min")
        self.v_T_furnace0 = self._field(f3, 10, "Furnace initial T", 25.0, "°C")
        self.nota_ramp = ttk.Label(
            f3, style="Hint.TLabel", wraplength=250, justify="left",
            text="The furnace rises at constant rate to the setpoint and is "
                 "then held there.")
        self.nota_ramp.grid(row=11, column=0, columnspan=3, sticky="w",
                            pady=(0, 4))
        ttk.Separator(f3).grid(row=12, column=0, columnspan=3, sticky="ew",
                               pady=8)
        self.v_h = self._field(f3, 13, "Convection  h", 10.0, "W/m²K")
        f3.columnconfigure(1, weight=1)
        self._rows_est = [self.lbl_est, self.nota_est]
        self._rows_ramp = [self.nota_ramp]
        self._toggle_programme()

        # ---- numerics
        f4 = ttk.Frame(nb, padding=10)
        nb.add(f4, text="Numerics")
        self.v_t_end = self._field(f4, 0, "Simulated time", 200.0, "s")
        ttk.Label(f4, text="Mesh resolution").grid(row=1, column=0, sticky="w",
                                                   padx=(2, 8), pady=3)
        self.v_resolution = tk.StringVar(value="Normal")
        ttk.Combobox(f4, textvariable=self.v_resolution, values=list(RESOLUTION),
                     state="readonly", width=9).grid(row=1, column=1,
                                                     columnspan=2, sticky="w")
        ttk.Label(f4, style="Hint.TLabel", wraplength=250, justify="left",
                  text="A finer mesh sharpens the result and lengthens the "
                       "run. Between Coarse and Fine the difference stays "
                       "within a few percent."
                  ).grid(row=2, column=0, columnspan=3, sticky="w",
                         pady=(6, 6))
        ttk.Separator(f4).grid(row=3, column=0, columnspan=3, sticky="ew",
                               pady=8)
        self.v_compact = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f4, text="Compare with the lumped formula",
            variable=self.v_compact).grid(row=4, column=0, columnspan=3,
                                          sticky="w")
        f4.columnconfigure(1, weight=1)

        # ---- actions
        act = ttk.Frame(p, padding=(0, 10, 0, 0))
        act.pack(fill="x")
        self.btn = ttk.Button(act, text="Run simulation",
                              style="Accent.TButton", command=self._run)
        self.btn.pack(fill="x")
        self.prog = ttk.Progressbar(act, mode="determinate", maximum=1.0)
        self.prog.pack(fill="x", pady=(8, 0))
        exp = ttk.Frame(p, padding=(0, 8, 0, 0))
        exp.pack(fill="x")
        ttk.Button(exp, text="Export CSV", command=self._csv).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(exp, text="Save figure", command=self._png).pack(
            side="left", fill="x", expand=True, padx=(4, 0))

        self._load_material()
        self._load_support()

    def _canvas_tab(self, nb, title, subplots=1):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        fig = Figure(figsize=(8, 4.3), dpi=100)
        axes = ([fig.add_subplot(1, 1, 1)] if subplots == 1
                else [fig.add_subplot(1, subplots, i + 1)
                      for i in range(subplots)])
        for a in axes:
            self._blank(a, "")
        c = FigureCanvasTkAgg(fig, f)
        tb = NavigationToolbar2Tk(c, f, pack_toolbar=False)
        tb.config(background=PAL["bg"])
        for w in tb.winfo_children():
            try:
                w.config(background=PAL["bg"])
            except tk.TclError:
                pass
        tb.update()
        tb.pack(side="bottom", fill="x")
        c.get_tk_widget().pack(side="top", fill="both", expand=True)
        return f, fig, axes, c

    def _outputs(self, p):
        self.bar_t = ttk.Frame(p, padding=(10, 6))
        self.bar_t.pack(side="bottom", fill="x")
        ttk.Label(self.bar_t, text="Instant").pack(side="left")
        self.v_t = tk.DoubleVar(value=0.0)
        self.sl = ttk.Scale(self.bar_t, from_=0, to=1, variable=self.v_t,
                            command=lambda e: self._time_changed())
        self.sl.pack(side="left", fill="x", expand=True, padx=10)
        self.v_tlab = tk.StringVar(value="—")
        ttk.Label(self.bar_t, textvariable=self.v_tlab, width=9,
                  style="Mono.TLabel").pack(side="left")

        nb = ttk.Notebook(p)
        nb.pack(side="top", fill="both", expand=True)
        self.nb_out = nb

        # curves + results strip
        f1 = ttk.Frame(nb)
        nb.add(f1, text="Curves")
        res = ttk.Labelframe(f1, text="  Results  ", padding=(10, 8))
        res.pack(side="bottom", fill="x", padx=6, pady=(2, 6))
        self.v_res = tk.StringVar(
            value="Run a simulation to obtain the characteristic times.")
        ttk.Label(res, textvariable=self.v_res, style="Mono.TLabel",
                  justify="left").pack(anchor="w")
        self.fig1 = Figure(figsize=(8, 4.1), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self._blank(self.ax1, "Set the inputs and press Run simulation")
        self.c1 = FigureCanvasTkAgg(self.fig1, f1)
        tb1 = NavigationToolbar2Tk(self.c1, f1, pack_toolbar=False)
        tb1.update()
        tb1.pack(side="bottom", fill="x")
        self.c1.get_tk_widget().pack(side="top", fill="both", expand=True)

        _, self.fig2, (self.ax2a, self.ax2b), self.c2 = self._canvas_tab(
            nb, "Profiles", 2)
        _, self.fig3, (self.ax3,), self.c3 = self._canvas_tab(nb, "Field map")
        _, self.fig6, (self.ax6a, self.ax6b), self.c6 = self._canvas_tab(
            nb, "Rate & stress", 2)

        # sensitivity
        f5 = ttk.Frame(nb)
        nb.add(f5, text="Sensitivity")
        bar5 = ttk.Frame(f5, padding=(10, 8))
        bar5.pack(side="bottom", fill="x")
        self.btn_sens = ttk.Button(bar5, text="Compute sensitivity",
                                   command=self._sensitivity)
        self.btn_sens.pack(side="left")
        ttk.Label(bar5, style="Hint.TLabel",
                  text="Perturbs every input by ±22 % and measures the "
                       "effect on t₉₉. Takes a few minutes."
                  ).pack(side="left", padx=12)
        self.prog_s = ttk.Progressbar(bar5, mode="determinate", maximum=1.0,
                                      length=150)
        self.prog_s.pack(side="right")
        self.fig5 = Figure(figsize=(8, 4.3), dpi=100)
        self.ax5 = self.fig5.add_subplot(111)
        self._blank(self.ax5, "Press Compute sensitivity")
        self.c5 = FigureCanvasTkAgg(self.fig5, f5)
        tb5 = NavigationToolbar2Tk(self.c5, f5, pack_toolbar=False)
        tb5.update()
        tb5.pack(side="bottom", fill="x")
        self.c5.get_tk_widget().pack(side="top", fill="both", expand=True)

        self._docs_tab(nb)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())
        self._tab_changed()

    def _docs_tab(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Theory")
        top = ttk.Frame(f, padding=(10, 8, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Model documentation", style="H1.TLabel").pack(
            side="left")
        ttk.Button(top, text="Open full PDF report …",
                   command=self._open_pdf).pack(side="right")

        pane = ttk.PanedWindow(f, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left_pane = ttk.Frame(pane, width=272)
        pane.add(left_pane, weight=0)
        self.tree = ttk.Treeview(left_pane, show="tree", selectmode="browse")
        # the "#0" column defaults to 200 px and would clip the longest
        # section titles, so it is sized to the pane
        self.tree.column("#0", width=260, minwidth=260, stretch=True)
        self.tree.pack(fill="both", expand=True)
        for i, (title, _) in enumerate(DOCS):
            self.tree.insert("", "end", iid=str(i), text=f"  {title}")
        self.tree.bind("<<TreeviewSelect>>", self._show_doc)

        der = ttk.Frame(pane)
        pane.add(der, weight=1)
        sb = ttk.Scrollbar(der)
        sb.pack(side="right", fill="y")
        self.txt = tk.Text(der, wrap="word", yscrollcommand=sb.set,
                           relief="flat", background=PAL["card"],
                           foreground=PAL["ink"], padx=22, pady=16,
                           font=(self.font_base[0], 10), spacing1=2,
                           spacing3=6, cursor="arrow")
        self.txt.pack(fill="both", expand=True)
        sb.config(command=self.txt.yview)
        self.txt.tag_config("h", font=(self.font_base[0], 11, "bold"),
                            foreground=PAL["accent"], spacing1=12, spacing3=6)
        self.txt.tag_config("f", font=(self.font_mono[0], 9),
                            foreground="#0b3c5d", lmargin1=26, lmargin2=26,
                            spacing1=3, spacing3=3)
        self.txt.tag_config("b", lmargin1=18, lmargin2=30)
        self.tree.selection_set("0")

    def _show_doc(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        title, body_text = DOCS[int(sel[0])]
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        for line in body_text.strip("\n").split("\n"):
            if line.startswith("## "):
                self.txt.insert("end", line[3:] + "\n", "h")
            elif line.startswith("> "):
                self.txt.insert("end", line[2:] + "\n", "f")
            elif line.startswith("- "):
                self.txt.insert("end", "•  " + line[2:] + "\n", "b")
            else:
                self.txt.insert("end", line + "\n")
        self.txt.config(state="disabled")
        self.txt.yview_moveto(0.0)

    def _open_pdf(self):
        """Open the PDF report with whatever the system uses for PDFs.

        Looked for next to the program rather than at a fixed path, so the
        whole thing stays portable: copy the folder anywhere and the button
        still works. sys._MEIPASS is PyInstaller's marker, for the case of
        a frozen executable.
        """
        if hasattr(sys, "_MEIPASS"):
            folders = [os.path.dirname(os.path.abspath(sys.executable))]
        else:
            folders = [os.path.dirname(os.path.abspath(__file__)),
                       os.path.dirname(os.path.abspath(sys.argv[0])),
                       os.getcwd()]
        for here, name in ((f, n) for f in folders
                           for n in ("report.pdf", "informe.pdf")):
            path = os.path.join(here, name)
            if os.path.exists(path):
                try:
                    if sys.platform.startswith("win"):
                        os.startfile(path)          # noqa: S606
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", path])
                    else:
                        subprocess.Popen(["xdg-open", path])
                except OSError as err:
                    messagebox.showerror("Could not open", str(err))
                return
        messagebox.showinfo(
            "Report not found",
            "The full PDF report was not found next to this program.\n\n"
            "Place report.pdf in the same folder as heating_simulator.py, "
            "or use the Theory tab, which holds the same content.")

    @staticmethod
    def _blank(ax, msg):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if msg:
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color=PAL["muted"], fontsize=10)

    @staticmethod
    def _axes_on(ax):
        ax.clear()
        ax.grid(True)
        for name, s in ax.spines.items():
            s.set_visible(name in ("left", "bottom"))

    # --------------------------------------------------------- callbacks
    def _apply_preset(self, m, which, fields):
        fields["rho"].set(str(m["rho"]))
        if "f" in m:
            self.props[which] = m["f"]()
            Tm = 0.5 * sum(_T_VAL)
            for key in ("k", "cp", "eps"):
                val = float(self.props[which][key].evaluate(np.array([Tm]))[0])
                fields[key].set(f"{val:.4g}")
        else:
            self.props[which] = {"k": None, "cp": None, "eps": None}
            for key in ("k", "cp", "eps"):
                fields[key].set(str(m[key]))
        self._refresh_props()

    def _load_material(self, _=None):
        self._apply_preset(MATERIALS[self.v_mat.get()], "sample",
                           {"k": self.v_k, "rho": self.v_rho,
                            "cp": self.v_cp, "eps": self.v_eps})

    def _load_support(self, _=None):
        self._apply_preset(SUPPORTS[self.v_mat_support.get()], "support",
                           {"k": self.v_k_support, "rho": self.v_rho_support,
                            "cp": self.v_cp_support, "eps": self.v_eps_support})

    def _props_dialog(self, which):
        if which == "sample":
            cons = dict(k=self.v_k.get(), cp=self.v_cp.get(),
                        eps=self.v_eps.get())
            title = "Sample properties as a function of T"
        else:
            cons = dict(k=self.v_k_support.get(), cp=self.v_cp_support.get(),
                        eps=self.v_eps_support.get())
            title = "Support properties as a function of T"
        try:
            cons = {k: float(str(v).replace(",", ".")) for k, v in cons.items()}
        except ValueError:
            cons = {k: 0.0 for k in cons}
        try:
            T0 = float(str(self.v_T0.get()).replace(",", ".")) + K0
            T1 = float(str(self.v_T_set.get()).replace(",", ".")) + K0
        except ValueError:
            T0, T1 = 373.15, 1473.15
        d = PropertyDialog(self, title, self.props[which], cons, (T0, T1))
        if d.result is not None:
            self.props[which] = d.result
            self._refresh_props()

    def _refresh_props(self):
        for which, var, fields in (
                ("sample", self.v_fT_m,
                 {"k": self.v_k, "cp": self.v_cp, "eps": self.v_eps}),
                ("support", self.v_fT_s,
                 {"k": self.v_k_support, "cp": self.v_cp_support, "eps": self.v_eps_support})):
            lines_out = []
            for key, sym in (("k", "k"), ("cp", "cₚ"), ("eps", "ε")):
                pr = self.props[which].get(key)
                for w in self._wid.get(str(fields[key]), ())[1:2]:
                    w.config(state="disabled" if pr is not None else "normal")
                if pr is not None:
                    rng = ""
                    if pr.T_min is not None or pr.T_max is not None:
                        rng = f"  [{pr.T_min:.0f}–{pr.T_max:.0f} K]"
                    lines_out.append(f"{sym} = f(T){rng}")
            var.set("\n".join(lines_out) if lines_out else "all constant")

    def _toggle_support(self):
        self.cb_support.config(state="readonly" if self.v_use_support.get() else "disabled")

    def _toggle_programme(self):
        ramp = PROGRAMMES.get(self.v_programme.get(), "stabilised") == "ramp"
        for v in (self.v_drop, self.v_tau):
            self._show_field(v, not ramp)
        for v in (self.v_ramp, self.v_T_furnace0):
            self._show_field(v, ramp)
        for w in self._rows_est:
            w.grid() if not ramp else w.grid_remove()
        for w in self._rows_ramp:
            w.grid() if ramp else w.grid_remove()

    def _tab_changed(self):
        tab = self.nb_out.index(self.nb_out.select())
        activa = tab in (1, 2)
        for w in self.bar_t.winfo_children():
            try:
                w.config(state="normal" if activa else "disabled")
            except tk.TclError:
                pass
        if self.res:
            if tab == 1:
                self._draw_profiles()
            elif tab == 2:
                self._draw_map()

    def _time_changed(self):
        if not self.res:
            return
        tab = self.nb_out.index(self.nb_out.select())
        if tab == 1:
            self._draw_profiles()
        elif tab == 2:
            self._draw_map()

    # ------------------------------------------------------------- input
    def _read(self):
        def f(v, name, lo=None, hi=None):
            try:
                x = float(str(v.get()).replace(",", "."))
            except ValueError:
                raise ValueError(f"The field “{name}” is not a valid "
                                 f"number.")
            if lo is not None and x < lo:
                raise ValueError(f"“{name}” must be greater than or "
                                 f"equal to {lo}.")
            if hi is not None and x > hi:
                raise ValueError(f"“{name}” must be less than or "
                                 f"equal to {hi}.")
            return x

        def prop(which, key, var, name, lo, hi=None):
            pr = self.props[which].get(key)
            return pr if pr is not None else f(var, name, lo, hi)

        d = f(self.v_d, "Diameter", 0.1)
        e = f(self.v_e, "Thickness", 0.01)
        p = dict(
            r_sample=d / 2000.0,
            e_sample=e / 1000.0,
            sample=dict(k=prop("sample", "k", self.v_k, "Conductivity", 1e-3),
                        rho=f(self.v_rho, "Density", 1.0),
                        cp=prop("sample", "cp", self.v_cp,
                                "Specific heat", 1.0),
                        eps=prop("sample", "eps", self.v_eps,
                                 "Emissivity", 0.0, 1.0)),
            with_support=bool(self.v_use_support.get()),
            e_support=f(self.v_e_support, "Support thickness", 0.01) / 1000.0,
            r_support=f(self.v_r_support, "Support radius", 0.1) / 1000.0,
            support=dict(k=prop("support", "k", self.v_k_support,
                                "Support conductivity", 1e-3),
                         rho=f(self.v_rho_support, "Support density", 1.0),
                         cp=prop("support", "cp", self.v_cp_support,
                                 "Support specific heat", 1.0),
                         eps=prop("support", "eps", self.v_eps_support,
                                  "Support emissivity", 0.0, 1.0)),
            R_contact=f(self.v_Rc, "Contact resistance", 0.0),
            T_furnace=f(self.v_T_set, "Setpoint", 20.0) + K0,
            T_initial=f(self.v_T0, "Sample initial temperature", -200.0) + K0,
            drop=f(self.v_drop, "Drop on opening", 0.0),
            tau_recovery=max(0.5, f(self.v_tau, "Recovery time", 0.1)),
            h_conv=f(self.v_h, "Convection coefficient", 0.0),
            t_end=f(self.v_t_end, "Simulated time", 0.5),
            programme=PROGRAMMES.get(self.v_programme.get(), "stabilised"),
            ramp_K_min=f(self.v_ramp, "Ramp rate", 0.1),
            T_furnace_initial=f(self.v_T_furnace0,
                                "Furnace initial temperature", -200.0) + K0,
            mechanical=dict(
                E=f(self.v_E, "Young's modulus", 0.1) * 1e9,
                alpha=f(self.v_alpha, "Thermal expansion", 0.0) * 1e-6,
                nu=f(self.v_nu, "Poisson's ratio", 0.0, 0.49)),
        )
        if p["T_initial"] >= p["T_furnace"]:
            raise ValueError("The initial temperature must be below the "
                             "furnace setpoint.")
        if p["programme"] == "stabilised" and \
                p["drop"] >= p["T_furnace"] - p["T_initial"]:
            raise ValueError("The drop on opening is so large that the "
                             "furnace would start below the sample.")
        if p["programme"] == "ramp":
            if p["T_furnace_initial"] >= p["T_furnace"]:
                raise ValueError("The furnace initial temperature must be "
                                 "below the setpoint.")
            arrival = (p["T_furnace"] - p["T_furnace_initial"]) / \
                (p["ramp_K_min"] / 60.0)
            if arrival > p["t_end"]:
                raise ValueError(
                    f"At that rate the furnace needs {arrival:.0f} s to reach "
                    f"the setpoint, longer than the simulated time. Extend "
                    f"“Simulated time” or raise the ramp rate.")
        return p

    # --------------------------------------------------------------- run
    def _run(self):
        if self.thread and self.thread.is_alive():
            self.stop.set()
            self.v_status.set("Cancelling …")
            return
        try:
            p = self._read()
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return

        self.stop.clear()
        self.btn.config(text="Cancel")
        self.v_status.set("Running …")
        self.prog.config(value=0)
        n = RESOLUTION[self.v_resolution.get()]
        use_compact = bool(self.v_compact.get())

        def worker():
            try:
                mod = Model(p, n)
                r = mod.solve(p["t_end"],
                              cb=lambda x: self.queue_ui.put(("p", x)),
                              stop=self.stop)
                if self.stop.is_set():
                    self.queue_ui.put(("cancelled", None))
                    return
                r["p"] = p
                r["compact"] = None
                if use_compact:
                    try:
                        r["compact"] = compact_formula(p, p["t_end"])
                    except Exception:                      # noqa: BLE001
                        r["compact"] = None
                self.queue_ui.put(("ok", r))
            except Exception as err:                       # noqa: BLE001
                self.queue_ui.put(("error", err))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def _sensitivity(self):
        if getattr(self, "thread_sens", None) and self.thread_sens.is_alive():
            return
        try:
            p = self._read()
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return
        try:
            from sensitivity import elasticities
        except ImportError as err:
            messagebox.showerror(
                "Missing file",
                f"The sensitivity analysis needs sensitivity.py in the same "
                f"folder as this program.\n\n{err}")
            return

        self.btn_sens.config(state="disabled")
        self.v_status.set("Computing sensitivity …")
        self.prog_s.config(value=0)

        def worker():
            try:
                base, elas = elasticities(
                    p, cb=lambda x: self.queue_ui.put(("ps", x)))
                self.queue_ui.put(("sens", (base, elas)))
            except Exception as err:                       # noqa: BLE001
                self.queue_ui.put(("errors", err))

        self.thread_sens = threading.Thread(target=worker, daemon=True)
        self.thread_sens.start()

    def _pump(self):
        try:
            while True:
                tipo, dato = self.queue_ui.get_nowait()
                if tipo == "p":
                    self.prog.config(value=dato)
                elif tipo == "ok":
                    self.res = dato
                    self.prog.config(value=1.0)
                    self.btn.config(text="Run simulation")
                    m = dato["model"]
                    self.v_status.set(
                        f"Done  ·  mesh {m.nr}×{m.nz}  ·  step "
                        f"{dato['dt'] * 1000:.2f} ms")
                    self._draw_all()
                elif tipo == "cancelled":
                    self.btn.config(text="Run simulation")
                    self.v_status.set("Cancelled")
                    self.prog.config(value=0)
                elif tipo == "ps":
                    self.prog_s.config(value=dato)
                elif tipo == "sens":
                    self.prog_s.config(value=1.0)
                    self.btn_sens.config(state="normal")
                    self.v_status.set("Sensitivity computed")
                    self._draw_sensitivity(*dato)
                elif tipo == "errors":
                    self.btn_sens.config(state="normal")
                    self.prog_s.config(value=0)
                    self.v_status.set("Sensitivity failed")
                    messagebox.showerror("Sensitivity error", str(dato))
                elif tipo == "error":
                    self.btn.config(text="Run simulation")
                    self.v_status.set("Run failed")
                    self.prog.config(value=0)
                    messagebox.showerror("Simulation error", str(dato))
        except queue.Empty:
            pass
        self.after(120, self._pump)

    # ------------------------------------------------------------ drawing
    def _draw_all(self):
        r = self.res
        self.sl.config(to=max(1e-9, float(r["t"][-1])))
        self.v_t.set(float(r["t"][-1]))
        self._draw_curves()
        self._draw_rate()
        self._summary()
        self._tab_changed()

    def _draw_curves(self):
        r, p = self.res, self.res["p"]
        ax = self.ax1
        self._axes_on(ax)
        t, T = r["t"], r["T"] - K0
        lab = ["sample centre", "top face", "bottom face", "rim", "support"]
        n = 5 if p["with_support"] else 4
        for i in range(n):
            ax.plot(t, T[:, i], color=SERIES[i], lw=1.7, label=lab[i],
                    ls=((0, (5, 2)) if i == 3 else "-"))
        ax.plot(t, r["T_env"] - K0, color=PAL["oven"], lw=1.1, ls="--",
                label="furnace walls")
        if r.get("compact") is not None:
            tc, Tc = r["compact"]
            label = ("lumped formula" if not p["with_support"]
                     else "lumped formula (no support)")
            ax.plot(tc, Tc - K0, ":", color="#444444", lw=1.5, label=label)
        obj = p["T_initial"] + 0.99 * (p["T_furnace"] - p["T_initial"]) - K0
        ax.axhline(obj, color=PAL["muted"], lw=0.7, ls=":")
        ax.annotate("99 % of setpoint", (t[-1], obj), xytext=(-6, 5),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=PAL["muted"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("temperature (°C)")
        ax.set_title("Temperature evolution")
        ax.set_xlim(0, t[-1])
        ax.legend(loc="lower right", ncol=2)
        self.fig1.tight_layout()
        self.c1.draw_idle()

    def _index(self):
        return int(np.argmin(np.abs(self.res["t"] - float(self.v_t.get()))))

    def _draw_profiles(self):
        if not self.res:
            return
        r, p, m = self.res, self.res["p"], self.res["model"]
        i = self._index()
        self.v_tlab.set(f"{r['t'][i]:.1f} s")
        S = r["snaps"][i] - K0
        zz = (m.z - m.z0) * 1000
        sel = m.mask_sample[0, :]

        ax = self.ax2a
        self._axes_on(ax)
        ax.plot(S[0, sel], zz[sel], "-o", color=PAL["centre"], ms=3, lw=1.7)
        ax.set_ylim(0, p["e_sample"] * 1000)
        ax.set_xlabel("temperature (°C)")
        ax.set_ylabel("position across thickness (mm)")
        ax.set_title(f"Axial profile  ·  t = {r['t'][i]:.1f} s")

        ax = self.ax2b
        self._axes_on(ax)
        mr = m.r <= p["r_sample"]
        ax.plot(m.r[mr] * 1000, S[mr, m.j_centre], "-o", color=PAL["bottom"],
                ms=3, lw=1.7)
        ax.set_xlabel("radius (mm)")
        ax.set_ylabel("temperature (°C)")
        ax.set_title("Radial profile at mid-thickness")
        self.fig2.tight_layout()
        self.c2.draw_idle()

    def _draw_map(self):
        if not self.res:
            return
        r, p, m = self.res, self.res["p"], self.res["model"]
        i = self._index()
        self.v_tlab.set(f"{r['t'][i]:.1f} s")
        S = np.where(m.act, r["snaps"][i] - K0, np.nan)
        ax = self.ax3
        self._axes_on(ax)
        ax.grid(False)
        ax.set_facecolor("#e8ecef")
        ext = [0, m.r[-1] * 1000, (m.z[0] - m.z0) * 1000,
               (m.z[-1] - m.z0) * 1000]
        im = ax.imshow(S.T, origin="lower", aspect="auto", extent=ext,
                       cmap="inferno")
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:                              # noqa: BLE001
                pass
        self.cbar = self.fig3.colorbar(im, ax=ax)
        self.cbar.set_label("temperature (°C)")
        ax.add_patch(Rectangle((0, 0), p["r_sample"] * 1000,
                               p["e_sample"] * 1000, fill=False,
                               edgecolor="w", lw=1.4, ls="--"))
        ax.text(p["r_sample"] * 500, p["e_sample"] * 500, "sample",
                color="w", fontsize=9, ha="center", va="center", alpha=0.8)
        ax.set_xlabel("radius (mm)")
        ax.set_ylabel("thickness (mm; 0 = sample base)")
        ax.set_title(f"Temperature field  ·  t = {r['t'][i]:.1f} s")
        self.fig3.tight_layout()
        self.c3.draw_idle()

    def _draw_rate(self):
        r, p = self.res, self.res["p"]
        t = r["t"]
        s_c, s_s = thermal_stress(r["T_mean"], r["T"][:, 0], r["T"][:, 1],
                                  p["mechanical"])
        ax = self.ax6a
        self._axes_on(ax)
        ax.plot(t, r["v_centre"], color=PAL["centre"], lw=1.7, label="centre")
        ax.plot(t, heating_rate(t, r["T_env"]), color=PAL["support"], lw=1.1,
                ls="--", label="furnace")
        i = int(np.argmax(r["v_centre"]))
        ax.annotate(f"max {r['v_max']:.0f} K/s\n({r['v_max'] * 60:.0f} K/min)",
                    (t[i], r["v_centre"][i]), xytext=(14, -6),
                    textcoords="offset points", fontsize=8,
                    color=PAL["centre"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("dT/dt (K/s)")
        ax.set_title("Heating rate")
        ax.set_xlim(0, t[-1])
        ax.legend()

        ax = self.ax6b
        self._axes_on(ax)
        ax.plot(t, s_c / 1e6, color=PAL["centre"], lw=1.7,
                label="centre (tension)")
        ax.plot(t, s_s / 1e6, color=PAL["bottom"], lw=1.7,
                label="surface (compression)")
        ax.axhline(0, color=PAL["ink"], lw=0.8)
        j = int(np.argmax(s_c))
        ax.annotate(f"max {s_c[j] / 1e6:.0f} MPa", (t[j], s_c[j] / 1e6),
                    xytext=(12, 4), textcoords="offset points", fontsize=8,
                    color=PAL["centre"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("thermal stress σ (MPa)")
        ax.set_title("Thermal stress")
        ax.set_xlim(0, t[-1])
        ax.legend()
        self.fig6.tight_layout()
        self.c6.draw_idle()

    def _draw_sensitivity(self, base, elas):
        ax = self.ax5
        self._axes_on(ax)
        rows = [(e, S) for e, S, _, _ in elas
                if np.isfinite(S) and abs(S) > 5e-3]
        if not rows:
            self._blank(ax, "No variable has an appreciable effect")
            self.c5.draw_idle()
            return
        labels = [d[0] for d in rows]
        vals = [d[1] for d in rows]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, height=0.68,
                color=[PAL["centre"] if v > 0 else PAL["bottom"]
                       for v in vals])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color=PAL["ink"], lw=0.8)
        rng = max(vals) - min(vals)
        for yi, v in zip(y, vals):
            ax.text(v + 0.02 * rng * np.sign(v), yi, f"{v:+.2f}", va="center",
                    ha="left" if v > 0 else "right", fontsize=7.5,
                    color=PAL["muted"])
        ax.set_xlim(min(vals) - 0.18 * rng, max(vals) + 0.18 * rng)
        ax.set_xlabel("elasticity   S = d(ln t₉₉) / d(ln x)")
        ax.set_title(f"Effect of each input on t₉₉   "
                     f"(base case: {base:.0f} s)")
        ax.grid(axis="y", alpha=0)
        self.fig5.tight_layout()
        self.c5.draw_idle()

    def _summary(self):
        r, p = self.res, self.res["p"]
        t, Tc = r["t"], r["T"][:, 0]
        T0, Th = p["T_initial"], p["T_furnace"]
        h_rad, h, Bi, tau = dimensionless(p)
        e_d = p["e_sample"] / (2 * p["r_sample"])
        s_c, _ = thermal_stress(r["T_mean"], Tc, r["T"][:, 1], p["mechanical"])

        def fmt(x):
            return f"{x:7.1f} s" if np.isfinite(x) else "not reached"

        lag = np.max(r["T"][:, 1] - Tc)
        lines_out = [
            f"t₅₀ {fmt(t_fraction(t, Tc, 0.5, T0, Th))}     "
            f"Bi = {Bi:5.3f}  (h·L/k)        "
            f"h_rad = {h_rad:3.0f} W/m²K",
            f"t₉₀ {fmt(t_fraction(t, Tc, 0.9, T0, Th))}     "
            f"τ  = {tau:5.2f} s  (ρ·cₚ·L/h)   "
            f"h_total = {h:3.0f} W/m²K",
            f"t₉₉ {fmt(t_fraction(t, Tc, 0.99, T0, Th))}     "
            f"max face–centre lag = {lag:3.0f} K    "
            f"final centre = {Tc[-1] - K0:6.1f} °C",
            f"peak rate {r['v_max']:6.1f} K/s ({r['v_max'] * 60:5.0f} K/min)"
            f"    peak tension σ = {s_c.max() / 1e6:4.0f} MPa",
        ]
        avisos = []
        if not np.isfinite(t_fraction(t, Tc, 0.99, T0, Th)):
            avisos.append("the centre does not reach 99 % within the "
                          "simulated time")
        if e_d > 0.15:
            avisos.append(f"e/d = {e_d:.2f} > 0.15, the lumped formula loses "
                          f"validity (the 2-D result does not)")
        if avisos:
            lines_out.append("⚠  " + "   ".join(avisos))
        self.v_res.set("\n".join(lines_out))

    # ---------------------------------------------------------- exporting
    def _csv(self):
        if not self.res:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="heating.csv")
        if not path:
            return
        r, p = self.res, self.res["p"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh, delimiter=";")
                w.writerow(["# Ceramic pellet heating simulator"])
                w.writerow([f"# sample: d={2 * p['r_sample'] * 1000:.2f} mm; "
                            f"e={p['e_sample'] * 1000:.2f} mm; "
                            f"rho={p['sample']['rho']}"])
                w.writerow([f"# furnace: {p['T_furnace'] - K0:.0f} C; "
                            f"drop {p['drop']:.0f} C; "
                            f"recovery {p['tau_recovery']:.0f} s; "
                            f"h {p['h_conv']:.0f} W/m2K"])
                w.writerow(["# support: "
                            + (f"{p['e_support'] * 1000:.1f} mm; "
                               f"Rc={p['R_contact']} m2K/W"
                               if p["with_support"] else "point contact")])
                v_c = r["v_centre"]
                s_c, s_s = thermal_stress(r["T_mean"], r["T"][:, 0],
                                          r["T"][:, 1], p["mechanical"])
                w.writerow(["t (s)", "centre (C)", "top face (C)",
                            "bottom face (C)", "rim (C)", "support (C)",
                            "furnace (C)", "sample mean (C)",
                            "dT/dt centre (K/s)", "sigma centre (MPa)",
                            "sigma surface (MPa)"])
                for i, ti in enumerate(r["t"]):
                    w.writerow([f"{ti:.4f}"]
                               + [f"{x - K0:.3f}" for x in r["T"][i]]
                               + [f"{r['T_env'][i] - K0:.3f}",
                                  f"{r['T_mean'][i] - K0:.3f}",
                                  f"{v_c[i]:.4f}",
                                  f"{s_c[i] / 1e6:.3f}",
                                  f"{s_s[i] / 1e6:.3f}"])
            self.v_status.set(f"CSV saved: {os.path.basename(path)}")
        except OSError as err:
            messagebox.showerror("Could not save", str(err))

    def _png(self):
        if not self.res:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        tab = self.nb_out.index(self.nb_out.select())
        fig = {0: self.fig1, 1: self.fig2, 2: self.fig3,
               3: self.fig6, 4: self.fig5}.get(tab)
        if fig is None:
            messagebox.showinfo("Nothing to save",
                                "Switch to a tab containing a figure.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="heating.png")
        if not path:
            return
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight",
                        facecolor=PAL["card"])
            self.v_status.set(f"Figure saved: {os.path.basename(path)}")
        except OSError as err:
            messagebox.showerror("Could not save", str(err))


def main():
    if not HAVE_TK:
        print(f"Tkinter is not available here ({_TK_ERROR}).",
              file=sys.stderr)
        print("On Debian or Ubuntu: sudo apt install python3-tk. The solver "
              "works without it — see examples/headless_run.py.",
              file=sys.stderr)
        return 1
    try:
        app = App()
    except tk.TclError as err:
        print("Could not open the graphical interface:", err, file=sys.stderr)
        print("Check that a display is available.", file=sys.stderr)
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
