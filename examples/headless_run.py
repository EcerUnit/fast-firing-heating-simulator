#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Running the solver without the interface
========================================
The reference case: a 10 mm x 1.5 mm green LATP pellet on a 2 mm alumina
setter, dropped into a furnace stabilised at 1200 C, with the measured
temperature-dependent properties.

Run it from the repository root:

    python examples/headless_run.py

This is also the quickest regression check after touching the solver. The
numbers below are the ones it should print; if they move, something changed.

    mesh 16x33, step 0.39 ms
    t50 32.4 s, t90 58.2 s, t99 78.3 s
    Bi 0.141, tau 16.37 s, max face-centre lag 68 K
    peak heating rate 19.5 K/s, peak centre tension 70 MPa
"""

import os
import sys

import numpy as np

# Let the example run from anywhere without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heating_simulator import (                                  # noqa: E402
    K0, Model, dimensionless, props_alumina, props_latp,
    t_fraction, thermal_stress,
)


def reference_case():
    """Dimensions in metres, temperatures in kelvin."""
    return dict(
        r_sample=0.0050, e_sample=0.0015,
        sample=dict(rho=2990.0, **props_latp()),
        with_support=True, e_support=0.002, r_support=0.010,
        support=dict(rho=3900.0, **props_alumina()),
        R_contact=0.0,
        T_furnace=1200 + K0, T_initial=100 + K0,
        drop=40.0, tau_recovery=20.0, h_conv=10.0,
        t_end=200.0,
        mechanical=dict(E=100e9, alpha=10e-6, nu=0.25),
    )


def main():
    p = reference_case()

    # 14 cells across the pellet thickness is the "Normal" resolution in the
    # interface. Everything else in the mesh is sized from it.
    model = Model(p, 14)
    r = model.solve(p["t_end"])

    t = r["t"]
    T_centre = r["T"][:, 0]          # probe order: centre, top, bottom, rim,
    T_top = r["T"][:, 1]             # support
    h_rad, h, Bi, tau = dimensionless(p)
    sigma_centre, _ = thermal_stress(r["T_mean"], T_centre, T_top,
                                     p["mechanical"])

    def t_to(frac):
        return t_fraction(t, T_centre, frac, p["T_initial"], p["T_furnace"])

    print(f"mesh {model.nr}x{model.nz}, step {r['dt'] * 1000:.2f} ms")
    print(f"t50 {t_to(0.50):.1f} s, t90 {t_to(0.90):.1f} s, "
          f"t99 {t_to(0.99):.1f} s")
    print(f"Bi {Bi:.3f}, tau {tau:.2f} s, "
          f"max face-centre lag {np.max(T_top - T_centre):.0f} K")
    print(f"peak heating rate {r['v_max']:.1f} K/s, "
          f"peak centre tension {sigma_centre.max() / 1e6:.0f} MPa")
    print(f"centre at the end {T_centre[-1] - K0:.1f} C")


if __name__ == "__main__":
    main()
