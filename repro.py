"""Reproduce Fig. 1-4 of Zhang et al., Nat. Commun. 11, 1706 (2020) in Python.

Run:  python src/repro.py           -> fig/fig1..fig4 png, fig/figS1_ard_restarts.png, fig/repro_metrics.json
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import eis_data as D
import models as M
from gp_jax import fit_ard_restarts
from common import (stack, rul_stack, save, FIG, BLUE, RED, GREEN, PINK, LGREEN, NAVY, GREY,
                    TRAIN_25, TEST_25, TRAIN_MT, TEST_MT)

PAPER = {"fig1a_25C05": 0.88, "fig2": {"25C05": 0.96, "25C06": 0.73, "25C07": 0.68, "25C08": 0.81},
         "fig3": {"35C02": 0.81, "45C02": 0.72}, "fig4": {"25C05": 0.87, "35C02": 0.75, "45C02": 0.92},
         "eol": {"25C05": 150, "25C06": 120, "25C07": 30, "25C08": 38, "35C02": 252, "45C02": 396}}
metrics = {"paper": PAPER}


def zs(X, mu, sd):
    return (X - mu) / sd


def cap_panel(ax, d, mu_hat, sd_hat, title=None, r2=None, xlim=None):
    x = d["battery_cycle"]
    c0, p0 = d["cap"][0], mu_hat[0]
    ax.fill_between(x, (mu_hat - sd_hat) / p0, (mu_hat + sd_hat) / p0, color=PINK, lw=0)
    ax.plot(x, d["cap"] / c0, "x", ms=3.5, color=BLUE, label="Measured")
    ax.plot(x, mu_hat / p0, "+", ms=4, color=RED, label="Predicted")
    ax.set_xlabel("Cycle number"); ax.set_ylabel("Capacity (normalised)")
    if title: ax.set_title(title)
    if r2 is not None:
        r2n = M.r2(d["cap"] / c0, mu_hat / p0)
        ax.text(0.05, 0.06, f"$R^2$ = {r2:.2f} (mAh)\n$R^2$ = {r2n:.2f} (normalised curves)", transform=ax.transAxes, fontsize=9.5)
    if xlim: ax.set_xlim(*xlim)


def ard_panel(ax, w, freq, color, title=None, top=2):
    idx = np.arange(1, 121)
    ax.semilogx(idx, w, "o", ms=4, color=color)
    for i in np.argsort(-w)[:top]:
        part = "Im" if i >= 60 else "Re"
        ax.annotate(f"{i+1}th\n{part} {freq[i % 60]:.3g} Hz", (i + 1, w[i]), textcoords="offset points",
                    xytext=(-62, -6), fontsize=9)
    ax.set_xlabel("Predictor index"); ax.set_ylabel("Predictor weight")
    ax.set_ylim(-0.02, max(0.25, w.max() * 1.15))
    if title: ax.set_title(title)


def rul_panel(ax, actual, mu, sd, title, r2):
    o = np.argsort(actual)
    ax.fill_between(actual[o], (mu - sd)[o], (mu + sd)[o], color=LGREEN, lw=0)
    ax.plot(actual, mu, "o", ms=3.5, color=GREEN)
    lim = [0, max(actual.max(), mu.max()) * 1.05]
    ax.plot(lim, lim, ":", color=GREY, lw=0.8)
    ax.set_xlabel("Actual RUL"); ax.set_ylabel("Predicted RUL"); ax.set_title(title)
    ax.text(0.62, 0.06, f"$R^2$ = {r2:.2f}", transform=ax.transAxes)


def test_rul(cell):
    d = D.paired(cell, truncate=False)
    r = D.rul_labels(d)
    return d["X"][r["idx"]], r["rul"], int(r["rul"][0])


# ---------------------------------------------------------------- Fig. 1 (25 °C capacity)
X25, y25, _, _ = stack(TRAIN_25)
mu25, sd25 = D.zscore_fit(X25)
iso25 = M.gpr_se_iso().fit(zs(X25, mu25, sd25), y25)
fig1 = {}
for c in TEST_25:
    d = D.paired(c)
    m, s = iso25.predict(zs(d["X"], mu25, sd25), return_std=True)
    fig1[c] = (d, m, s, M.r2(d["cap"], m))
metrics["fig1_r2_mAh"] = {c: round(v[3], 3) for c, v in fig1.items()}
metrics["fig1_r2_normalised"] = {c: round(M.r2(v[0]["cap"] / v[0]["cap"][0], v[1] / v[1][0]), 3) for c, v in fig1.items()}

ard25, ard25_all = fit_ard_restarts(zs(X25, mu25, sd25), y25, n_restarts=6, seed=1)
FREQ = D.FREQ

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
d, m, s, r = fig1["25C05"]
cap_panel(ax[0], d, m, s, "a   25C05 (state V)", r, xlim=(0, 300)); ax[0].legend(loc="upper right")
cols = {"25C05": "#E4572E", "25C06": BLUE, "25C07": "#7B3FA0", "25C08": GREEN}
for c, (d, m, s, r) in fig1.items():
    keep = d["battery_cycle"] <= 300                  # same cycle window as panel a / the paper
    ax[1].plot((d["cap"] / d["cap"][0])[keep], (m / m[0])[keep], ".", ms=4, color=cols[c], label=c)
ax[1].plot([0.7, 1.0], [0.7, 1.0], ":", color=GREY, lw=0.8)
ax[1].set_xlabel("Measured capacity"); ax[1].set_ylabel("Predicted capacity"); ax[1].legend(loc="lower right")
ax[1].set_title("b   25 °C test cells (cycles ≤ 300)")
ard_panel(ax[2], ard25.ard_weights(), FREQ, "#D63AF9", "c   ARD (best of 6 restarts)")
save(fig, "fig1_capacity_25C.png")

# ---------------------------------------------------------------- Fig. 2 (25 °C RUL)
Xr, yr, gr = rul_stack(TRAIN_25)                       # 25C04 never reaches EoL -> excluded (as in RUL.txt)
lin25 = M.gpr_lin().fit(zs(Xr, mu25, sd25), yr)
fig, ax = plt.subplots(2, 2, figsize=(10, 8.4)); ax = ax.ravel()
metrics["fig2_r2"], metrics["eol_cycles_ours"] = {}, {}
for a, c, lab in zip(ax, TEST_25, "abcd"):
    Xt, rt, eol = test_rul(c)
    m, s = lin25.predict(zs(Xt, mu25, sd25), return_std=True)
    r = M.r2(rt, m); metrics["fig2_r2"][c] = round(r, 3); metrics["eol_cycles_ours"][c] = eol
    rul_panel(a, rt, m, s, f"{lab}   {c}  (EoL {eol} cycles)", r)
fig.tight_layout(); save(fig, "fig2_rul_25C.png")

# ---------------------------------------------------------------- Fig. 3 (multi-T capacity)
Xm, ym, _, _ = stack(TRAIN_MT)
mum, sdm = D.zscore_fit(Xm)
isoM = M.gpr_se_iso().fit(zs(Xm, mum, sdm), ym)
fig, ax = plt.subplots(2, 2, figsize=(10, 8.4))
metrics["fig3_r2_mAh"] = {}
for j, c in enumerate(TEST_MT):
    d = D.paired(c)
    m, s = isoM.predict(zs(d["X"], mum, sdm), return_std=True)
    r = M.r2(d["cap"], m); metrics["fig3_r2_mAh"][c] = round(r, 3)
    metrics.setdefault("fig3_r2_normalised", {})[c] = round(M.r2(d["cap"] / d["cap"][0], m / m[0]), 3)
    cap_panel(ax[0, j], d, m, s, f"{'ab'[j]}   {c}", r)
ard_single = {}
for j, (c, col) in enumerate([("35C01", BLUE), ("45C01", GREEN)]):
    X1, y1, _, _ = stack([c])
    mu1, sd1 = D.zscore_fit(X1)
    best, allf = fit_ard_restarts(zs(X1, mu1, sd1), y1, n_restarts=8, seed=2)
    ard_single[c] = (best, allf)
    ard_panel(ax[1, j], best.ard_weights(), FREQ, col, f"{'cd'[j]}   ARD, {c[:2]} °C ({c})")
ax[0, 0].legend(loc="upper right")
fig.tight_layout(); save(fig, "fig3_capacity_multiT.png")

# ---------------------------------------------------------------- Fig. 4 (multi-T RUL)
XrM, yrM, _ = rul_stack(TRAIN_MT)
linM = M.gpr_lin().fit(zs(XrM, mum, sdm), yrM)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
metrics["fig4_r2"] = {}
for a, c, lab in zip(ax, ["25C05", "35C02", "45C02"], "abc"):
    Xt, rt, eol = test_rul(c)
    m, s = linM.predict(zs(Xt, mum, sdm), return_std=True)
    r = M.r2(rt, m); metrics["fig4_r2"][c] = round(r, 3); metrics["eol_cycles_ours"][c] = eol
    rul_panel(a, rt, m, s, f"{lab}   {c}  (EoL {eol} cycles)", r)
fig.tight_layout(); save(fig, "fig4_rul_multiT.png")

# ---------------------------------------------------------------- Fig. S1: ARD stability across restarts
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
runs = [("25 °C set (4 cells)", ard25_all), ("35C01", ard_single["35C01"][1]), ("45C01", ard_single["45C01"][1])]
metrics["ard_top_feature_per_restart"] = {}
for a, (name, fits) in zip(ax, runs):
    fits = sorted(fits, key=lambda m: m.nlml_)
    W = np.array([m.ard_weights() for m in fits])
    a.imshow(W, aspect="auto", cmap="magma_r", interpolation="nearest", extent=[0.5, 120.5, len(fits) + 0.5, 0.5])
    a.axvline(60.5, color="w", lw=1)
    a.axvline(91, color=BLUE, lw=0.8, ls="--"); a.axvline(100, color=BLUE, lw=0.8, ls=":")
    a.set_xlabel("Predictor index  (1–60 Re, 61–120 −Im)")
    a.set_title(name)
    metrics["ard_top_feature_per_restart"][name] = [
        {"nlml": round(float(m.nlml_), 1), "top": int(m.ard_ranking()[0] + 1),
         "part": "Im" if m.ard_ranking()[0] >= 60 else "Re",
         "freq_Hz": round(float(FREQ[m.ard_ranking()[0] % 60]), 3)} for m in fits]
ax[0].set_ylabel("Restart (sorted by marginal likelihood, best on top)")
fig.suptitle("ARD weights from different starting points — dashed: paper's 91st (−Im 17.8 Hz), dotted: 100th (−Im 2.16 Hz)", y=1.02)
save(fig, "figS1_ard_restarts.png")

metrics["n_train_rows"] = {"25C_capacity": int(len(y25)), "multiT_capacity": int(len(ym)),
                           "25C_rul": int(len(yr)), "multiT_rul": int(len(yrM))}
metrics["kernels"] = {"iso25": str(iso25.kernel_), "isoMT": str(isoM.kernel_),
                      "lin25": str(lin25.kernel_), "linMT": str(linM.kernel_)}
np.savez(FIG / "repro_arrays.npz", ard25=ard25.ard_weights(), ard35=ard_single["35C01"][0].ard_weights(),
         ard45=ard_single["45C01"][0].ard_weights(), freq=FREQ)
(FIG / "repro_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in metrics.items() if k != "ard_top_feature_per_restart"}, indent=1, ensure_ascii=False))
