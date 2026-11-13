#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for the pellet heating model
=================================================
Companion to heating_simulator.py. Keep both files in the same folder.

It answers one question: which inputs actually control how long the centre
of the pellet takes to reach temperature, and which ones can be measured
carelessly without hurting the answer.

Two complementary tools, because neither alone is enough:

1. ELASTICITIES (local sensitivity). For every input x, the elasticity of
   the heating time t99 with respect to x:

        S = d(ln t99) / d(ln x)

   S = +1 means doubling x doubles the time; S = 0 means the variable does
   not matter; a negative S means the time falls as x grows. Being
   dimensionless, it puts a thickness in millimetres and an emissivity on
   the same scale, which is the whole point.

2. SWEEPS (global sensitivity). An elasticity only describes the
   neighbourhood of the operating point, and several of these relations are
   not pure power laws, so the sweeps walk the full useful range of a
   variable and show the shape.

Two ways to use it
------------------
  python sensitivity.py
      Runs everything on the reference case below, prints a table, and
      writes fig_sensitivity.png and sensitivity.json next to the script.

  Inside the simulator
      The "Compute sensitivity" button on the Sensitivity tab calls
      elasticities() with whatever is currently typed into the form, so the
      bar chart you get there reflects your own case, not the reference one.

A warning about runtime: one elasticity costs two full simulations, so the
whole table is roughly 35 runs. Expect a couple of minutes, more if the
properties are temperature dependent.
"""

import json
import numpy as np

# The simulator is the single source of truth for the physics; nothing here
# re-implements it. If the import fails, the two files are out of step.
try:
    from heating_simulator import (Model, is_property, t_fraction,
                                   K0, VERSION)
except ImportError as _err:
    raise ImportError(
        "sensitivity.py does not match the heating_simulator.py in "
        "this folder: they are from different versions. Replace BOTH files "
        f"with the same version. (detail: {_err})") from None

_REQUIRED_VERSION = 7
if VERSION < _REQUIRED_VERSION:
    raise ImportError(
        f"heating_simulator.py is version {VERSION} and this "
        f"sensitivity.py needs version {_REQUIRED_VERSION}. Replace BOTH "
        f"files with the same version.")


# --------------------------------------------------------------------------
# Reference case
# --------------------------------------------------------------------------
# The parameter dictionary is the contract between this file and the
# simulator, which builds the same dictionary from its input form. Renaming
# a key here without renaming it there breaks the solver, so the full list
# lives here, once:
#
#   r_sample, e_sample      sample radius and thickness           [m]
#   sample, support         material dicts: k, rho, cp, eps       [SI]
#   with_support            False puts the pellet on point contact
#   e_support, r_support    support thickness and radius          [m]
#   R_contact               contact resistance, sample/support    [m²K/W]
#   T_furnace, T_initial    furnace setpoint, initial sample T    [K]
#   drop, tau_recovery      drop on opening [K], recovery time    [s]
#   h_conv                  convection coefficient                [W/m²K]
#   t_end                   length of the simulated window        [s]
#
# A property may be a plain number or a Property object from the simulator;
# both work everywhere below.
def base_case():
    """Green LATP pellet, 10 mm across and 1.5 mm thick, on a 2 mm dense
    alumina setter, dropped into a furnace stabilised at 1200 °C.

    Same geometry as the reference case in the report and in
    examples/headless_run.py, but with constant properties rather than the
    measured functions, so a standalone run finishes in a sane time.
    """
    return dict(
        r_sample=0.0050, e_sample=0.0015,
        sample=dict(k=0.8, rho=1900.0, cp=950.0, eps=0.85),
        with_support=True, e_support=0.002, r_support=0.010,
        support=dict(k=6.0, rho=3900.0, cp=1200.0, eps=0.70),
        R_contact=0.0,
        T_furnace=1200 + K0, T_initial=100 + K0,
        drop=40.0, tau_recovery=20.0, h_conv=10.0,
        t_end=180.0,
    )


# The variables that get an elasticity, as (label, path into the dict).
# The label is what ends up on the bar chart, so write it for a reader.
# To add a variable, append a line here: nothing else needs touching.
VARIABLES = [
    ("Sample thickness",         ("e_sample",)),
    ("Sample radius",            ("r_sample",)),
    ("Sample conductivity  k",   ("sample", "k")),
    ("Sample density  ρ",        ("sample", "rho")),
    ("Sample specific heat  cₚ", ("sample", "cp")),
    ("Sample emissivity  ε",     ("sample", "eps")),
    ("Support thickness",        ("e_support",)),
    ("Support radius",           ("r_support",)),
    ("Support conductivity  k",  ("support", "k")),
    ("Support density  ρ",       ("support", "rho")),
    ("Support specific heat  cₚ", ("support", "cp")),
    ("Support emissivity  ε",    ("support", "eps")),
    ("Furnace setpoint",         ("T_furnace",)),
    ("Initial temperature",      ("T_initial",)),
    ("Drop on opening",          ("drop",)),
    ("Recovery time",            ("tau_recovery",)),
    ("Convection coefficient",   ("h_conv",)),
]


def _get(p, path):
    d = p
    for key in path[:-1]:
        d = d[key]
    return d[path[-1]]


def _set(p, path, value):
    d = p
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def _copy_case(p):
    """Shallow copy with the two material dicts copied too, so that
    perturbing a property does not contaminate the reference case."""
    q = dict(p)
    q["sample"] = dict(p["sample"])
    q["support"] = dict(p["support"])
    return q


# --------------------------------------------------------------------------
# The metric: time for the centre to reach 99 % of the setpoint
# --------------------------------------------------------------------------
def t99(p, n_cells=10, t_end=180.0, frac=0.99, retries=2):
    """Time (s) for the centre to reach frac·(T_furnace − T_initial).

    If the target is not reached inside the window, the window is widened
    and the run repeated. A slow case with a thick support can easily need
    more than the default 180 s, and returning nan there would silently
    poison the elasticity it belongs to.
    """
    for _ in range(retries + 1):
        mod = Model(p, n_cells)
        r = mod.solve(t_end, n_save=400)
        t = t_fraction(r["t"], r["T"][:, 0], frac, p["T_initial"], p["T_furnace"])
        if np.isfinite(t):
            return t
        t_end *= 2.5
    return float("nan")


# --------------------------------------------------------------------------
# 1. Elasticities
# --------------------------------------------------------------------------
def elasticities(p=None, delta=0.22, n_cells=10, cb=None):
    """S = d(ln t99)/d(ln x) by central differences, for every variable.

    Returns (t99 of the unperturbed case, list of (label, S, t_minus,
    t_plus)) sorted by decreasing |S|.

    On the size of the perturbation. ±22 % looks crude for a derivative and
    it is deliberate. With a small perturbation (±5 %) the weak variables
    drown in discretisation noise: changing a thickness changes how many
    cells represent the support, and since the support carries an elasticity
    near 0.8, that rounding alone moves t99 by about 1 % — the same size as
    the signal being measured. I was caught by this once: a ±5 % run
    returned a negative elasticity for the sample thickness, which the wide
    sweep then flatly contradicted. At ±22 % the signal dominates. The price
    is that these are secant elasticities over a finite interval, not point
    derivatives, so read them as "what happens over a realistic spread of
    this variable", which is the useful question anyway.

    cb, if given, is called with the fraction done (0 to 1). The simulator
    uses it to drive its progress bar.
    """
    p = p or base_case()
    base = t99(p, n_cells)
    out = []
    for i, (label, path) in enumerate(VARIABLES):
        x0 = _get(p, path)
        if is_property(x0):
            # Defined as a function of temperature. Multiplying scales the
            # whole curve, which is the perturbation that means something:
            # every value a few per cent higher, the shape untouched.
            label = label + "  (f(T))"
        elif x0 == 0:
            # Nothing to scale: a relative perturbation of zero is zero.
            out.append((label, 0.0, base, base))
            continue
        p_lo, p_hi = _copy_case(p), _copy_case(p)
        _set(p_lo, path, x0 * (1 - delta))
        _set(p_hi, path, x0 * (1 + delta))
        t_lo, t_hi = t99(p_lo, n_cells), t99(p_hi, n_cells)
        if np.isfinite(t_lo) and np.isfinite(t_hi) and t_lo > 0 and t_hi > 0:
            S = ((np.log(t_hi) - np.log(t_lo))
                 / (np.log(1 + delta) - np.log(1 - delta)))
        else:
            S = float("nan")
        out.append((label, S, t_lo, t_hi))
        if cb:
            cb((i + 1) / len(VARIABLES))
    out.sort(key=lambda row: -abs(row[1] if np.isfinite(row[1]) else 0))
    return base, out


# --------------------------------------------------------------------------
# 2. Sweeps
# --------------------------------------------------------------------------
def sweep(path, values, p=None, n_cells=8, transform=None):
    """t99 across the full range of one variable.

    Pass either a path into the parameter dict, or a transform(case, value)
    for a variable that needs more than a single assignment — switching the
    support off when its thickness reaches zero, for instance.

    n_cells is lower than in the elasticities on purpose: a sweep is read
    for its shape, not for its third digit, and the runs are many.
    """
    p = p or base_case()
    ys = []
    for v in values:
        q = _copy_case(p)
        if transform:
            transform(q, v)
        else:
            _set(q, path, v)
        ys.append(t99(q, n_cells))
    return np.array(ys)


def _support_off_if_zero(q, v):
    """Zero support thickness means no support at all, not a support of
    zero thickness: the pellet then sits on point contact and sees the
    furnace from below as well."""
    if v <= 1e-9:
        q["with_support"] = False
    else:
        q["with_support"] = True
        q["e_support"] = v


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {}
    delta = 0.22
    print("Reference case …", flush=True)
    base, elas = elasticities(
        delta=delta,
        cb=lambda x: print(f"  elasticities {x * 100:3.0f} %", flush=True))
    res["t99_base"] = base
    res["elasticities"] = [(lbl, None if not np.isfinite(S) else S)
                           for lbl, S, _, _ in elas]
    print(f"\nt99 of the reference case = {base:.1f} s\n")
    lo, hi = 100 * delta, 100 * delta
    for label, S, t_lo, t_hi in elas:
        print(f"  {label:28s} S = {S:+6.3f}   "
              f"(−{lo:.0f} % → {t_lo:6.1f} s, +{hi:.0f} % → {t_hi:6.1f} s)")

    # --- sweeps
    print("\nSweeps …", flush=True)
    e_sample = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]) * 1e-3
    t_sample = sweep(("e_sample",), e_sample)
    print("  sample thickness:     ", np.round(t_sample, 1), flush=True)

    e_support = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]) * 1e-3
    t_support = sweep(None, e_support, transform=_support_off_if_zero)
    print("  support thickness:    ", np.round(t_support, 1), flush=True)

    emis = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    t_emis = sweep(("sample", "eps"), emis)
    print("  sample emissivity:    ", np.round(t_emis, 1), flush=True)

    rc = np.array([0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    t_rc = sweep(("R_contact",), rc)
    print("  contact resistance:   ", np.round(t_rc, 1), flush=True)

    k_support = np.array([1.0, 2.0, 3.5, 6.0, 10.0, 20.0])
    t_k_support = sweep(("support", "k"), k_support)
    print("  support conductivity: ", np.round(t_k_support, 1), flush=True)

    res["sweeps"] = dict(
        sample_thickness_mm=(e_sample * 1000).tolist(),
        t_sample_thickness=t_sample.tolist(),
        support_thickness_mm=(e_support * 1000).tolist(),
        t_support_thickness=t_support.tolist(),
        sample_emissivity=emis.tolist(), t_sample_emissivity=t_emis.tolist(),
        contact_resistance=rc.tolist(), t_contact_resistance=t_rc.tolist(),
        support_conductivity=k_support.tolist(),
        t_support_conductivity=t_k_support.tolist(),
    )
    with open("sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)

    # --- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2),
                           gridspec_kw={"width_ratios": [1.15, 1]})
    # Variables below 0.005 are indistinguishable from zero at this mesh;
    # plotting them would suggest a precision the numbers do not have.
    shown = [(lbl, S) for lbl, S, _, _ in elas
             if np.isfinite(S) and abs(S) > 5e-3]
    labels = [row[0] for row in shown]
    vals = [row[1] for row in shown]
    y = np.arange(len(labels))[::-1]
    colours = ["#c1272d" if v > 0 else "#2166ac" for v in vals]
    ax[0].barh(y, vals, color=colours, height=0.7)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("elasticity   S = d(ln t₉₉) / d(ln x)")
    ax[0].set_title(f"Local sensitivity  (reference t₉₉ = {base:.0f} s)")
    ax[0].grid(alpha=0.3, axis="x")
    for yi, v in zip(y, vals):
        ax[0].text(v + (0.012 if v > 0 else -0.012), yi, f"{v:+.2f}",
                   va="center", ha="left" if v > 0 else "right", fontsize=8)
    span = max(vals) - min(vals)
    ax[0].set_xlim(min(vals) - 0.16 * span, max(vals) + 0.16 * span)

    ax[1].plot(e_support * 1000, t_support, "o-", color="#c1272d", lw=2,
               label="support thickness")
    ax[1].plot(e_sample * 1000, t_sample, "s-", color="#2166ac", lw=2,
               label="sample thickness")
    ax[1].set_xlabel("thickness (mm)")
    ax[1].set_ylabel("t₉₉ (s)")
    ax[1].set_title("The support weighs far more than the pellet itself")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_sensitivity.png", dpi=160)
    print("\nfig_sensitivity.png and sensitivity.json written")


if __name__ == "__main__":
    main()
