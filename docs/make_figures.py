#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures and canonical numbers for the report
============================================
Produces the vector (PDF) figures and numbers.json, running the model with
the measured temperature-dependent properties.

This is what keeps the report honest. Every figure and every number quoted
in the text comes from here, so the document cannot drift away from the
code: change the model, rerun this, recompile the report.

Reference case: a 10 mm diameter, 1.5 mm thick LATP pellet on a 2 mm
alumina setter.

Run it from inside docs/, with the repository root importable:

    cd docs && PYTHONPATH=.. python make_figures.py

It takes a while — the tornado figure alone needs a full elasticity sweep.

The axis labels are in Spanish because the report is; everything else here
is in English, like the rest of the repository.

Needs scipy in addition to the project's own requirements.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from heating_simulator import (Model, t_fraction, thermal_stress, heating_rate,
                               dimensionless, compact_formula,
                               props_latp, props_alumina, K0, TGRID)
from sensitivity import elasticities

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "pdf.fonttype": 42, "figure.dpi": 150,
})
RED, BLUE, ORANGE, GREEN, GREY = "#c1272d", "#2166ac", "#e08214", "#1a9850", "#777777"
N = 14
num = {}


def base(**kw):
    pl, pa = props_latp(), props_alumina()
    p = dict(r_sample=0.005, e_sample=0.0015,
             sample=dict(k=pl["k"], rho=2990.0, cp=pl["cp"], eps=pl["eps"]),
             with_support=True, e_support=0.002, r_support=0.010,
             support=dict(k=pa["k"], rho=3900.0, cp=pa["cp"], eps=pa["eps"]),
             R_contact=0.0, T_furnace=1473.15, T_initial=373.15,
             drop=40.0, tau_recovery=20.0, h_conv=10.0,
             programme="stabilised",
             mechanical=dict(E=100e9, alpha=1e-5, nu=0.25))
    p.update(kw)
    return p


# --------------------------------------- fig 0: the measured properties
pl, pa = props_latp(), props_alumina()
T = np.linspace(373.0, 1473.0, 400)
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.5))
ax[0].plot(T, pl["k"].evaluate(T), color=RED, lw=1.5, label="LATP")
axb = ax[0].twinx()
axb.plot(T, pa["k"].evaluate(T), color=BLUE, lw=1.5, ls="--", label="alúmina")
axb.set_ylabel("alúmina (W/m·K)", fontsize=8)
axb.tick_params(labelsize=7)
ax[0].set_ylabel("LATP (W/m·K)"); ax[0].set_title("Conductividad")
ax[1].plot(T, pl["cp"].evaluate(T), color=RED, lw=1.5, label="LATP")
ax[1].plot(T, pa["cp"].evaluate(T), color=BLUE, lw=1.5, ls="--", label="alúmina")
ax[1].axhline(1190, color="k", lw=0.8, ls=":")
ax[1].text(930, 660, "límite de Dulong–Petit (LATP)", fontsize=6.5)
ax[1].set_ylabel("cp (J/kg·K)"); ax[1].set_title("Calor específico")
ax[1].legend(fontsize=7)
ax[2].plot(T, pl["eps"].evaluate(T), color=RED, lw=1.5, label="LATP")
ax[2].plot(T, pa["eps"].evaluate(T), color=BLUE, lw=1.5, ls="--", label="alúmina")
ax[2].axvspan(1073, 1273, color="#1a9850", alpha=0.12)
ax[2].text(1080, 0.15, "medida", fontsize=6.5, color="#1a9850")
ax[2].set_ylabel("emisividad"); ax[2].set_title("Emisividad")
ax[2].legend(fontsize=7, loc="upper left")
for a in ax:
    a.set_xlabel("T (K)")
fig.tight_layout(); fig.savefig("fig_props.pdf"); plt.close(fig)

# --------------------------------------- the reference case and its plots
p = base()
r = Model(p, N).solve(200.0, n_save=600)
t, Tp = r["t"], r["T"]
Tc = Tp[:, 0]
s_c, s_s = thermal_stress(r["T_mean"], Tc, Tp[:, 1], p["mechanical"])
for f, case_name in ((0.50, "t50"), (0.90, "t90"), (0.99, "t99")):
    num[case_name] = t_fraction(t, Tc, f, p["T_initial"], p["T_furnace"])
num["desfase"] = float(np.max(Tp[:, 1] - Tc))
num["dmedia"] = float(np.max(r["T_mean"] - Tc))
num["vmax"] = float(r["v_max"])
num["sigma"] = float(s_c.max() / 1e6)
h_rad, h_tot, Bi, tau = dimensionless(p)
num.update(h_rad=float(h_rad), h_tot=float(h_tot), Bi=float(Bi), tau=float(tau))
num["dt_ms"] = r["dt"] * 1000
num["grid"] = f"{r['model'].nr}x{r['model'].nz}"
num["rho_latp"] = 2990.0
print("base:", {k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in num.items()}, flush=True)

fig, ax = plt.subplots(figsize=(6.3, 3.4))
for i, (c, l, ls) in enumerate([(RED, "centro de la sample", "-"),
                                (ORANGE, "cara superior", "-"),
                                (BLUE, "cara inferior", "-"),
                                (GREEN, "borde", (0, (5, 2))),
                                (GREY, "support", "-")]):
    ax.plot(t, Tp[:, i] - K0, color=c, lw=1.5, ls=ls, label=l)
ax.plot(t, r["T_env"] - K0, "k--", lw=1.0, label="paredes del horno")
ax.axhline(p["T_initial"] + 0.99 * (p["T_furnace"] - p["T_initial"]) - K0,
           color="gray", lw=0.6, ls=":")
ax.set_xlim(0, 200); ax.set_xlabel("tiempo (s)"); ax.set_ylabel("temperatura (°C)")
ax.legend(loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig("fig_curvas.pdf"); plt.close(fig)

m = r["model"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
zz = (m.z - m.z0) * 1000
sel = m.mask_sample[0, :]
cols = plt.cm.inferno(np.linspace(0.15, 0.78, 5))
inst = [3, 8, 20, 50, 120]
for ts_, c in zip(inst, cols):
    i = int(np.argmin(np.abs(t - ts_)))
    ax[0].plot(r["snaps"][i][0, sel] - K0, zz[sel], color=c, lw=1.4,
               marker="o", ms=2.2, label=f"{ts_} s")
ax[0].set_ylim(0, p["e_sample"] * 1000)
ax[0].set_xlabel("temperatura (°C)")
ax[0].set_ylabel("posición en el espesor (mm)")
ax[0].set_title("Perfil axial en la sample"); ax[0].legend(fontsize=7)
mr = m.r <= p["r_sample"]
for ts_, c in zip(inst, cols):
    i = int(np.argmin(np.abs(t - ts_)))
    ax[1].plot(m.r[mr] * 1000, r["snaps"][i][mr, m.j_centre] - K0, color=c, lw=1.4)
ax[1].set_xlabel("radio (mm)"); ax[1].set_ylabel("temperatura (°C)")
ax[1].set_title("Perfil radial en el plano medio")
fig.tight_layout(); fig.savefig("fig_perfiles.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
ax[0].plot(t, r["v_centre"], color=RED, lw=1.6, label="centro")
ax[0].plot(t, heating_rate(t, r["T_env"]), color=GREY, lw=1.0, ls="--", label="horno")
ax[0].set_xlim(0, 150); ax[0].set_xlabel("tiempo (s)")
ax[0].set_ylabel("dT/dt (K/s)"); ax[0].set_title("Velocidad de calentamiento")
ax[0].legend()
ax[1].plot(t, s_c / 1e6, color=RED, lw=1.6, label="centro (tracción)")
ax[1].plot(t, s_s / 1e6, color=BLUE, lw=1.6, label="superficie (compresión)")
ax[1].axhline(0, color="k", lw=0.7)
ax[1].set_xlim(0, 150); ax[1].set_xlabel("tiempo (s)")
ax[1].set_ylabel("tensión (MPa)"); ax[1].set_title("Tensión de origen térmico")
ax[1].legend()
fig.tight_layout(); fig.savefig("fig_velocidad.pdf"); plt.close(fig)

# ------------------------------- the compact formula against the full run
pp = base(with_support=False, drop=0.0)
rp = Model(pp, N).solve(80.0, n_save=500)
tcf, Tcf = compact_formula(pp, 80.0)
num["t99_punt"] = t_fraction(rp["t"], rp["T"][:, 0], 0.99,
                             pp["T_initial"], pp["T_furnace"])
num["t99_comp"] = t_fraction(tcf, Tcf, 0.99, pp["T_initial"], pp["T_furnace"])
num["desv_comp"] = 100 * abs(num["t99_punt"] - num["t99_comp"]) / num["t99_punt"]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
ax.plot(rp["t"], rp["T"][:, 0] - K0, color=RED, lw=1.8, label="cálculo 2D (centro)")
ax.plot(tcf, Tcf - K0, "k--", lw=1.3, label="fórmula compacta")
ax.set_xlim(0, 80); ax.set_xlabel("tiempo (s)"); ax.set_ylabel("temperatura (°C)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("fig_compacta.pdf"); plt.close(fig)
print("compacta:", num["t99_punt"], num["t99_comp"], num["desv_comp"], flush=True)

# ------------------------ fast firing against a conventional ramp
pr = base(programme="ramp", ramp_K_min=600.0, T_furnace_initial=25 + K0)
rr = Model(pr, N).solve(360.0, n_save=600)
src, _ = thermal_stress(rr["T_mean"], rr["T"][:, 0], rr["T"][:, 1], pr["mechanical"])
num["t99_rampa"] = t_fraction(rr["t"], rr["T"][:, 0], 0.99,
                              pr["T_initial"], pr["T_furnace"])
num["v_rampa"] = float(rr["v_max"])
num["sigma_rampa"] = float(src.max() / 1e6)
num["desfase_rampa"] = float(np.max(rr["T"][:, 1] - rr["T"][:, 0]))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9))
ax[0].plot(t, Tc - K0, color=RED, lw=1.6, label="introducción directa")
ax[0].plot(rr["t"], rr["T"][:, 0] - K0, color=BLUE, lw=1.6, label="ramp 600 °C/min")
ax[0].set_xlim(0, 360); ax[0].set_xlabel("tiempo (s)")
ax[0].set_ylabel("temperatura del centro (°C)"); ax[0].legend(loc="lower right")
ax[0].set_title("Temperatura del centro")
ax[1].plot(t, s_c / 1e6, color=RED, lw=1.6, label="introducción directa")
ax[1].plot(rr["t"], src / 1e6, color=BLUE, lw=1.6, label="ramp 600 °C/min")
ax[1].set_xlim(0, 360); ax[1].set_xlabel("tiempo (s)")
ax[1].set_ylabel("tracción en el centro (MPa)"); ax[1].legend()
ax[1].set_title("Tensión en el centro")
fig.tight_layout(); fig.savefig("fig_rampa.pdf"); plt.close(fig)
print("ramp:", num["t99_rampa"], num["v_rampa"], num["sigma_rampa"], flush=True)

# ------------------------------------------------------ mesh independence
grid = {}
for case_name, con, es in (("Contacto puntual", False, 0.0),
                           ("Soporte de 1 mm", True, 0.001),
                           ("Soporte de 2 mm", True, 0.002),
                           ("Soporte de 5 mm", True, 0.005)):
    row = []
    for n in (8, 14, 22):
        q = base(with_support=con, e_support=(es if con else 0.002))
        rq = Model(q, n).solve(400.0, n_save=400)
        row.append(t_fraction(rq["t"], rq["T"][:, 0], 0.99,
                              q["T_initial"], q["T_furnace"]))
    grid[case_name] = row
    print(f"grid {case_name}: {[round(x, 1) for x in row]}", flush=True)
num["mesh_table"] = grid

# ------------------------- validation against the analytical slab series
L, k_, rho_, cp_ = 0.010, 1.0, 1000.0, 1000.0
al_ = k_ / (rho_ * cp_)
h_, T0_, Ti_ = 100.0, 300.0, 1000.0
lam = np.array([brentq(lambda x: x * np.sin(x) - 1.0 * np.cos(x),
                       i * np.pi + 1e-9, (i + 0.5) * np.pi - 1e-9)
                for i in range(20)])
A = 4 * np.sin(lam) / (2 * lam + np.sin(2 * lam))
conv = []
for n in (10, 20, 40, 80):
    pv = dict(r_sample=0.25, e_sample=2 * L,
              sample=dict(k=k_, rho=rho_, cp=cp_, eps=0.0),
              with_support=False, e_support=0.0, r_support=0.0,
              support=dict(k=1.0, rho=1.0, cp=1.0, eps=0.0), R_contact=0.0,
              T_furnace=Ti_, T_initial=T0_, drop=0.0, tau_recovery=1.0,
              h_conv=h_, programme="stabilised")
    rv = Model(pv, n).solve(400.0, n_save=400)
    Fo = al_ * rv["t"] / L ** 2
    Tan = Ti_ + (T0_ - Ti_) * (A[:, None] * np.exp(-lam[:, None] ** 2 * Fo)).sum(0)
    s = Fo > 0.05
    conv.append(float(100 * np.abs(rv["T"][:, 0] - Tan)[s].max() / (Ti_ - T0_)))
num["convergencia"] = conv
print("convergencia:", conv, flush=True)

# ------------------------------------------------------ sensitivity
print("sensitivity…", flush=True)
b_sens, elas = elasticities(base(), n_esp=8)
num["t99_sens"] = float(b_sens)
num["elasticities"] = [(e, (None if not np.isfinite(S) else float(S)))
                       for e, S, _, _ in elas]
shown = [(e, S) for e, S, _, _ in elas if np.isfinite(S) and abs(S) > 5e-3]
labels = [a for a, _ in shown]
val = [b for _, b in shown]
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(6.3, 4.0))
ax.barh(y, val, color=[RED if v > 0 else BLUE for v in val], height=0.68)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.axvline(0, color="k", lw=0.7)
for yi, v in zip(y, val):
    ax.text(v + 0.03 * np.sign(v), yi, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7)
rng = max(val) - min(val)
ax.set_xlim(min(val) - 0.20 * rng, max(val) + 0.20 * rng)
ax.set_xlabel(r"elasticidad  $S = \mathrm{d}(\ln t_{99})\,/\,\mathrm{d}(\ln x)$")
ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig("fig_tornado.pdf"); plt.close(fig)

json.dump(num, open("numbers.json", "w", encoding="utf-8"), indent=1,
          ensure_ascii=False, default=float)
print("\n=== NUMEROS ===")
for k, v in num.items():
    if k not in ("elasticities", "mesh_table"):
        print(f"{k:14s} = {v}")
print("\nelasticities:")
for e, S in num["elasticities"]:
    print(f"  {e:34s} {S if S is None else round(S, 3)}")
print("\nPDF figures written")
